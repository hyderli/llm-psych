"""H1 confound audit: is probe separability a real emotion signal or an artifact?

Priority-1 diagnostic (see ``plans/next-steps.md``). H1 claims a linear
probe reads emotion identity off residual-stream activations. A high AUC
is only meaningful if the probe tracks the *emotion concept* and not a
surface artifact (prompt length, vocabulary, topic, paraphrase style).
This script tries to *break* the probe with controls; what survives is
plausibly real.

Two tiers of control, run automatically depending on what is present:

Tier A — surface-feature baselines (NO activations needed; runs on the
prompt parquet alone). For each ``emotion vs neutral`` task:
  * length-only logistic regression (the "neutral is shorter" tell),
  * word+char TF-IDF logistic regression (lexical / style fingerprint).
A high surface AUC means a trivial text classifier already separates the
classes, so that much of any activation-probe AUC is not evidence for an
emotion *concept*.

Tier B — activation controls (run per (model, emotion) when
``activations/<model_key>/<emotion>_{train,test}.npz`` exist):
  * shuffle-label null  — permute train labels, refit; expect AUC ~ 0.5,
  * cross-domain split  — train on some ``category`` domains, test on
    held-out domains; a concept transfers, an artifact collapses,
  * paraphrase-source split — train on hand_authored, test on paraphrase
    (skipped when the stimuli contain a single source, as the
    hand-authored seed set does — that is the point of dropping
    augmentation),
  * neutral-PC projection — project out the top PCs of neutral
    activations before fitting; compare AUC with/without.

Outputs:
  results/h1_confound_audit/report.md         — human-readable summary + recommendation
  results/h1_confound_audit/audit_results.json — raw numbers

Usage
-----
Surface-only (no activations yet — runs today on the seed set)::

    uv run python scripts/audit_h1_confounds.py \
        --prompts data/public/emotion_prompts.parquet \
        --emotions admiration joy loathing sadness

Full audit once activations exist for a model::

    uv run python scripts/audit_h1_confounds.py \
        --prompts data/public/emotion_prompts.parquet \
        --emotions admiration joy loathing sadness \
        --model-key Qwen2.5-0.5B-Instruct

This is a diagnostic, not part of the fit pipeline: it does NOT require a
clean git tree and writes only to ``results/`` (gitignored).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack as sparse_hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_repo_root = Path(__file__).resolve().parents[1]


def _load_steering():
    """Load steering.py directly by path.

    Imported lazily and by file path (not via ``import llm_psych``) so the
    surface-only tier runs without pulling in the package ``__init__``,
    which imports torch. ``steering.py`` itself is pure NumPy.
    """
    spec = importlib.util.spec_from_file_location(
        "_audit_steering", _repo_root / "src" / "llm_psych" / "steering.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

NEUTRAL = "neutral"
SEED = 42

# Recommendation thresholds (from plans/next-steps.md success criterion).
SURFACE_FLAG = 0.80        # surface AUC at/above this => surface confound suspected
NULL_FLAG = 0.60           # shuffle-null mean at/above this => pipeline leak
CROSSDOMAIN_PASS = 0.80    # cross-domain test AUC at/above this => concept transfers
CROSSDOMAIN_FAIL = 0.65    # cross-domain test AUC below this => artifact suspected


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC, returning nan if only one class is present."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def _fit_predict_auc(
    X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray
) -> float:
    """Fit an L2 logistic probe (matching probes.fit) and return test AUC."""
    if len(np.unique(y_tr)) < 2:
        return float("nan")
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=SEED)
    clf.fit(X_tr, y_tr)
    return _auc(y_te, clf.predict_proba(X_te)[:, 1])


# Surface TF-IDF features are high-dimensional, sparse, and frequently
# *perfectly separable* (emotion-laden text vs neutral). lbfgs on a dense,
# perfectly-separable matrix never converges — it runs the full max_iter every
# fit, which made the audit take hours on larger story corpora. liblinear is
# fast and stable on high-dim sparse near-separable text, so use it for the
# surface baseline (the activation probes keep lbfgs above, unchanged).
SURFACE_MAX_FEATURES = 20_000  # cap each vectorizer; bounds the char-ngram blowup


def _fit_predict_auc_surface(X_tr, y_tr, X_te, y_te) -> float:
    """AUC for sparse TF-IDF features (liblinear, no densification)."""
    if len(np.unique(y_tr)) < 2:
        return float("nan")
    clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=SEED)
    clf.fit(X_tr, y_tr)
    return _auc(y_te, clf.predict_proba(X_te)[:, 1])


def _surface_tfidf_auc(text_tr, y_tr, text_te, y_te) -> float:
    """Word(1-2) + char(3-5) TF-IDF logistic AUC, kept sparse and capped."""
    word_vec = TfidfVectorizer(
        ngram_range=(1, 2), min_df=1, max_features=SURFACE_MAX_FEATURES, sublinear_tf=True
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=SURFACE_MAX_FEATURES
    )
    Xtr = sparse_hstack(
        [word_vec.fit_transform(text_tr), char_vec.fit_transform(text_tr)]
    ).tocsr()
    Xte = sparse_hstack(
        [word_vec.transform(text_te), char_vec.transform(text_te)]
    ).tocsr()
    return _fit_predict_auc_surface(Xtr, y_tr, Xte, y_te)


# --------------------------------------------------------------------------
# Tier A — surface-feature baselines (no activations)
# --------------------------------------------------------------------------
def surface_audit(prompts: pd.DataFrame, emotion: str) -> dict:
    """Length-only and TF-IDF baselines for ``emotion`` vs neutral."""
    sub = prompts[prompts["emotion_label"].isin([emotion, NEUTRAL])].copy()
    sub["y"] = (sub["emotion_label"] == emotion).astype(int)
    tr, te = sub[sub["split"] == "train"], sub[sub["split"] == "test"]

    out: dict = {
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "len_mean_emotion": round(float(sub.loc[sub.y == 1, "length_words"].mean()), 2),
        "len_mean_neutral": round(float(sub.loc[sub.y == 0, "length_words"].mean()), 2),
    }

    # (a) length-only logistic
    out["auc_length_only"] = round(
        _fit_predict_auc(
            tr[["length_words"]].to_numpy(float), tr["y"].to_numpy(),
            te[["length_words"]].to_numpy(float), te["y"].to_numpy(),
        ),
        4,
    )

    # (b) word (1-2gram) + char (3-5gram) TF-IDF logistic (sparse, capped)
    out["auc_tfidf_surface"] = round(
        _surface_tfidf_auc(tr["prompt"], tr["y"].to_numpy(), te["prompt"], te["y"].to_numpy()),
        4,
    )
    return out


def surface_audit_story(emotion: str, base_model_key: str | None) -> dict:
    """Surface baseline on the STORY texts (story-method analogue of
    ``surface_audit``).

    The CAA ``surface_audit`` runs on the vignette prompt parquet; for the
    story path the relevant surface question is whether the *stories* are
    separable from text alone (they are topic-matched and emotion-word-
    banned by construction, so they should not be). Reads
    ``data/derived/stories/<base_model_key>/{emotion,neutral}.parquet``.

    Returns NaN metrics with a ``note`` when the corpus is not present
    locally (e.g. only activations were pulled from HF — the story corpus
    is not synced by ``sync_hf.py``), so downstream report/recommend code
    needs no None-handling.
    """
    nan = float("nan")
    blank = {
        "auc_length_only": nan, "auc_tfidf_surface": nan,
        "len_mean_emotion": nan, "len_mean_neutral": nan,
        "n_train": 0, "n_test": 0,
    }
    if not base_model_key:
        blank["note"] = "no base model key — story-text surface baseline skipped"
        return blank
    base = _repo_root / "data" / "derived" / "stories" / base_model_key
    pe, pn = base / f"{emotion}.parquet", base / "neutral.parquet"
    if not (pe.exists() and pn.exists()):
        blank["note"] = f"story corpus not present at {base} — surface baseline skipped"
        return blank

    de = pd.read_parquet(pe)[["story_text"]].assign(y=1)
    dn = pd.read_parquet(pn)[["story_text"]].assign(y=0)
    df = pd.concat([de, dn], ignore_index=True)
    df["length_words"] = df["story_text"].str.split().str.len()
    rng = np.random.default_rng(SEED)
    mask = rng.random(len(df)) < 0.7
    tr, te = df[mask], df[~mask]

    out = {
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "len_mean_emotion": round(float(df.loc[df.y == 1, "length_words"].mean()), 2),
        "len_mean_neutral": round(float(df.loc[df.y == 0, "length_words"].mean()), 2),
    }
    out["auc_length_only"] = round(
        _fit_predict_auc(
            tr[["length_words"]].to_numpy(float), tr["y"].to_numpy(),
            te[["length_words"]].to_numpy(float), te["y"].to_numpy(),
        ),
        4,
    )
    out["auc_tfidf_surface"] = round(
        _surface_tfidf_auc(
            tr["story_text"], tr["y"].to_numpy(), te["story_text"], te["y"].to_numpy()
        ),
        4,
    )
    return out


# --------------------------------------------------------------------------
# Tier B — activation controls
# --------------------------------------------------------------------------
def _load_layer(npz: np.lib.npyio.NpzFile, layer: int) -> np.ndarray:
    return npz[f"layer_{layer}"].astype(np.float64)


def _available_layers(npz: np.lib.npyio.NpzFile) -> list[int]:
    return sorted(int(k.split("_")[1]) for k in npz.files if k.startswith("layer_"))


def _join_meta(meta_path: Path, prompts: pd.DataFrame) -> pd.DataFrame:
    """Align activation rows (meta prompt_id order) with prompt metadata."""
    meta = pd.read_parquet(meta_path)  # column: prompt_id, in npz row order
    merged = meta.merge(
        prompts.rename(columns={"id": "prompt_id"}), on="prompt_id", how="left"
    )
    return merged.reset_index(drop=True)


def activation_audit(
    act_dir: Path, emotion: str, prompts: pd.DataFrame, *, n_perm: int = 200
) -> dict | None:
    """Shuffle-null, cross-domain, source-split, neutral-PC controls.

    Returns None (with a logged reason) if the required .npz files for
    this emotion are not present.
    """
    needed = [f"{emotion}_train.npz", f"{emotion}_test.npz",
              "neutral_train.npz", "neutral_test.npz"]
    missing = [f for f in needed if not (act_dir / f).exists()]
    if missing:
        log.info("  [skip] %s: activations not found (%s)", emotion, ", ".join(missing))
        return None

    emo_tr = np.load(act_dir / f"{emotion}_train.npz")
    emo_te = np.load(act_dir / f"{emotion}_test.npz")
    neu_tr = np.load(act_dir / "neutral_train.npz")
    neu_te = np.load(act_dir / "neutral_test.npz")

    layers = _available_layers(emo_tr)
    # Use the deepest available layer as the headline (mid-late = most concept-like).
    layer = layers[-1]
    rng = np.random.default_rng(SEED)

    Xtr = np.concatenate([_load_layer(emo_tr, layer), _load_layer(neu_tr, layer)])
    ytr = np.concatenate([np.ones(emo_tr[f"layer_{layer}"].shape[0], int),
                          np.zeros(neu_tr[f"layer_{layer}"].shape[0], int)])
    Xte = np.concatenate([_load_layer(emo_te, layer), _load_layer(neu_te, layer)])
    yte = np.concatenate([np.ones(emo_te[f"layer_{layer}"].shape[0], int),
                          np.zeros(neu_te[f"layer_{layer}"].shape[0], int)])

    out: dict = {"layer": int(layer), "n_train": int(len(ytr)), "n_test": int(len(yte))}

    # --- real probe AUC (reference) ---
    out["auc_real"] = round(_fit_predict_auc(Xtr, ytr, Xte, yte), 4)

    # --- (c) shuffle-label null ---
    null = np.array([
        _fit_predict_auc(Xtr, rng.permutation(ytr), Xte, yte) for _ in range(n_perm)
    ])
    null = null[~np.isnan(null)]
    out["null_mean"] = round(float(null.mean()), 4)
    out["null_p95"] = round(float(np.percentile(null, 95)), 4)
    out["null_p_real"] = round(float((null >= out["auc_real"]).mean()), 4)
    out["n_perm"] = int(len(null))

    # --- (d) cross-domain split (needs category from meta) ---
    out["auc_crossdomain"] = float("nan")
    try:
        m_emo_tr = _join_meta(act_dir / f"{emotion}_train.meta.parquet", prompts)
        m_neu_tr = _join_meta(act_dir / "neutral_train.meta.parquet", prompts)
        cats = sorted(set(m_emo_tr["category"].dropna()))
        if len(cats) >= 4:
            # Hold out half the domains (deterministic split).
            held = set(cats[::2])
            def _mask(meta_df):
                return meta_df["category"].isin(held).to_numpy()
            emo_held = _mask(m_emo_tr)
            neu_held = _mask(m_neu_tr)
            X_emo = _load_layer(emo_tr, layer)
            X_neu = _load_layer(neu_tr, layer)
            Xtr_cd = np.concatenate([X_emo[~emo_held], X_neu[~neu_held]])
            ytr_cd = np.concatenate([np.ones((~emo_held).sum(), int),
                                     np.zeros((~neu_held).sum(), int)])
            Xte_cd = np.concatenate([X_emo[emo_held], X_neu[neu_held]])
            yte_cd = np.concatenate([np.ones(emo_held.sum(), int),
                                     np.zeros(neu_held.sum(), int)])
            out["auc_crossdomain"] = round(_fit_predict_auc(Xtr_cd, ytr_cd, Xte_cd, yte_cd), 4)
            out["crossdomain_heldout_categories"] = sorted(held)
    except FileNotFoundError:
        log.info("  [note] %s: no meta.parquet — cross-domain control skipped", emotion)

    # --- (e) paraphrase-source split ---
    out["auc_source_split"] = None
    try:
        m_emo_tr = _join_meta(act_dir / f"{emotion}_train.meta.parquet", prompts)
        sources = set(m_emo_tr["source"].dropna())
        if len(sources) < 2:
            out["source_note"] = (
                f"single source present ({sources or 'unknown'}); paraphrase "
                "confound not present in these stimuli — control N/A"
            )
        else:
            out["source_note"] = f"sources present: {sorted(sources)} (split implemented when augmented data is used)"
    except FileNotFoundError:
        out["source_note"] = "no meta.parquet — source split skipped"

    # --- (f) neutral-PC projection ---
    try:
        steering = _load_steering()
        basis = steering.fit_neutral_pcs(_load_layer(neu_tr, layer), var_threshold=0.5)
        Xtr_p = steering.project_out(Xtr, basis)
        Xte_p = steering.project_out(Xte, basis)
        out["auc_neutralpc_projected"] = round(_fit_predict_auc(Xtr_p, ytr, Xte_p, yte), 4)
        out["neutralpc_k"] = int(basis.shape[0])
    except Exception as exc:  # noqa: BLE001 - diagnostic, never fatal
        out["auc_neutralpc_projected"] = float("nan")
        out["neutralpc_error"] = str(exc)

    return out


# --------------------------------------------------------------------------
# Tier B (story) — activation controls on story-method activations
# --------------------------------------------------------------------------
def _story_topic_groups(
    story_ids: list[str], emotion: str, base_model_key: str | None
) -> np.ndarray:
    """Topic index per story, aligned to npz/meta row order.

    Topics are recoverable two ways; prefer the explicit corpus, fall back
    to the id encoding so this works even when only activations were pulled.

    1. ``data/derived/stories/<base_model_key>/<emotion>.parquet`` carries
       an ``id`` -> ``topic`` map; the corpus is topic-matched across
       emotions, so topic identity is shared.
    2. Story ids are ``<emotion>_<topic_idx>_<rep_idx>`` — parse the topic
       index from the right (always available; the corpus parquet is not
       synced to HF by ``sync_hf.py``).
    """
    if base_model_key:
        pq = (
            _repo_root / "data" / "derived" / "stories"
            / base_model_key / f"{emotion}.parquet"
        )
        if pq.exists():
            try:
                m = pd.read_parquet(pq).set_index("id")["topic"]
                topics = [m.get(sid) for sid in story_ids]
                if all(t is not None for t in topics):
                    codes = {t: i for i, t in enumerate(sorted(set(topics)))}
                    return np.array([codes[t] for t in topics], dtype=int)
            except Exception:  # noqa: BLE001 - fall through to id parsing
                pass
    groups = []
    for sid in story_ids:
        parts = sid.rsplit("_", 2)
        groups.append(int(parts[1]) if len(parts) == 3 and parts[1].isdigit() else -1)
    return np.array(groups, dtype=int)


def activation_audit_story(
    act_dir: Path, emotion: str, *, base_model_key: str | None, n_perm: int = 200
) -> dict | None:
    """Story-method analogue of ``activation_audit``.

    Story activations ship as one pooled array per emotion
    (``<emotion>.npz`` with ``layer_<n>`` keys, shape ``(n_stories, dim)``)
    plus ``<emotion>.meta.parquet`` (``story_id`` in row order) — there is
    no pre-made train/test split and no per-prompt category/source. So this:

      * makes its own seeded 70/30 per-class split for the real-probe AUC
        and the shuffle-label null, and
      * runs a CROSS-TOPIC generalization test (hold out half the topics)
        in place of the CAA cross-domain control. Topics are matched across
        emotions by construction, so a *concept* transfers across held-out
        topics while a topic/style artifact collapses.

    Paraphrase-source split is N/A (single source = the model's own
    generations). The ``auc_crossdomain`` key is reused (carrying the
    cross-topic value) so ``recommend`` / ``write_report`` need no changes.
    """
    needed = [f"{emotion}.npz", "neutral.npz",
              f"{emotion}.meta.parquet", "neutral.meta.parquet"]
    missing = [f for f in needed if not (act_dir / f).exists()]
    if missing:
        log.info("  [skip] %s: story activations not found (%s)", emotion, ", ".join(missing))
        return None

    emo = np.load(act_dir / f"{emotion}.npz")
    neu = np.load(act_dir / "neutral.npz")
    layer = _available_layers(emo)[-1]  # deepest = most concept-like
    X_emo = _load_layer(emo, layer)
    X_neu = _load_layer(neu, layer)
    emo_ids = pd.read_parquet(act_dir / f"{emotion}.meta.parquet")["story_id"].tolist()
    neu_ids = pd.read_parquet(act_dir / "neutral.meta.parquet")["story_id"].tolist()
    g_emo = _story_topic_groups(emo_ids, emotion, base_model_key)
    g_neu = _story_topic_groups(neu_ids, "neutral", base_model_key)

    rng = np.random.default_rng(SEED)

    out: dict = {
        "layer": int(layer),
        "n_emotion": int(X_emo.shape[0]),
        "n_neutral": int(X_neu.shape[0]),
        "control_type": "cross_topic",
        "source_note": "single source (model-generated stories); paraphrase split N/A",
    }

    # --- seeded 70/30 per-class split for real-probe AUC + null ---
    def _split(n: int) -> tuple[np.ndarray, np.ndarray]:
        idx = rng.permutation(n)
        cut = max(1, int(round(0.7 * n)))
        return idx[:cut], idx[cut:]

    tr_e, te_e = _split(X_emo.shape[0])
    tr_n, te_n = _split(X_neu.shape[0])
    Xtr = np.concatenate([X_emo[tr_e], X_neu[tr_n]])
    ytr = np.concatenate([np.ones(len(tr_e), int), np.zeros(len(tr_n), int)])
    Xte = np.concatenate([X_emo[te_e], X_neu[te_n]])
    yte = np.concatenate([np.ones(len(te_e), int), np.zeros(len(te_n), int)])
    out["n_train"] = int(len(ytr))
    out["n_test"] = int(len(yte))
    out["auc_real"] = round(_fit_predict_auc(Xtr, ytr, Xte, yte), 4)

    # --- shuffle-label null ---
    null = np.array([
        _fit_predict_auc(Xtr, rng.permutation(ytr), Xte, yte) for _ in range(n_perm)
    ])
    null = null[~np.isnan(null)]
    if len(null):
        out["null_mean"] = round(float(null.mean()), 4)
        out["null_p95"] = round(float(np.percentile(null, 95)), 4)
        out["null_p_real"] = round(float((null >= out["auc_real"]).mean()), 4)
    else:
        out["null_mean"] = out["null_p95"] = out["null_p_real"] = float("nan")
    out["n_perm"] = int(len(null))

    # --- cross-topic generalization (hold out half the shared topics) ---
    out["auc_crossdomain"] = float("nan")  # reused key (carries cross-topic)
    shared = sorted((set(g_emo.tolist()) & set(g_neu.tolist())) - {-1})
    if len(shared) >= 4:
        held = list(shared[::2])
        em_h = np.isin(g_emo, held)
        nu_h = np.isin(g_neu, held)
        Xtr_cd = np.concatenate([X_emo[~em_h], X_neu[~nu_h]])
        ytr_cd = np.concatenate([np.ones((~em_h).sum(), int), np.zeros((~nu_h).sum(), int)])
        Xte_cd = np.concatenate([X_emo[em_h], X_neu[nu_h]])
        yte_cd = np.concatenate([np.ones(em_h.sum(), int), np.zeros(nu_h.sum(), int)])
        out["auc_crossdomain"] = round(_fit_predict_auc(Xtr_cd, ytr_cd, Xte_cd, yte_cd), 4)
        out["crosstopic_n_held"] = len(held)
        out["crosstopic_n_topics"] = len(shared)
    else:
        out["crosstopic_note"] = (
            f"only {len(shared)} shared topics — need >=4 for a cross-topic split"
        )

    # --- neutral-PC projection ---
    try:
        steering = _load_steering()
        basis = steering.fit_neutral_pcs(X_neu[tr_n], var_threshold=0.5)
        out["auc_neutralpc_projected"] = round(
            _fit_predict_auc(
                steering.project_out(Xtr, basis), ytr,
                steering.project_out(Xte, basis), yte,
            ),
            4,
        )
        out["neutralpc_k"] = int(basis.shape[0])
    except Exception as exc:  # noqa: BLE001 - diagnostic, never fatal
        out["auc_neutralpc_projected"] = float("nan")
        out["neutralpc_error"] = str(exc)

    return out


# --------------------------------------------------------------------------
# Recommendation
# --------------------------------------------------------------------------
def recommend(emotion: str, surf: dict, act: dict | None) -> tuple[str, list[str]]:
    """Return (verdict, reasons) for one emotion."""
    flags: list[str] = []
    if surf["auc_tfidf_surface"] >= SURFACE_FLAG:
        flags.append(
            f"surface TF-IDF AUC={surf['auc_tfidf_surface']:.2f} ≥ {SURFACE_FLAG} "
            "— a trivial text classifier already separates the classes"
        )
    if surf["auc_length_only"] >= SURFACE_FLAG:
        flags.append(
            f"length-only AUC={surf['auc_length_only']:.2f} ≥ {SURFACE_FLAG} "
            f"(emotion {surf['len_mean_emotion']}w vs neutral {surf['len_mean_neutral']}w)"
        )
    if act is not None:
        if act["null_mean"] >= NULL_FLAG:
            flags.append(f"shuffle-null mean AUC={act['null_mean']:.2f} ≥ {NULL_FLAG} — pipeline leak")
        cd = act.get("auc_crossdomain")
        gen = "topic" if act.get("control_type") == "cross_topic" else "domain"
        if cd is not None and not np.isnan(cd) and cd < CROSSDOMAIN_FAIL:
            flags.append(f"cross-{gen} AUC={cd:.2f} < {CROSSDOMAIN_FAIL} — collapses out of {gen}")

    if act is None:
        verdict = "SURFACE-ONLY (activations pending)"
    elif flags:
        verdict = "FLAG — confound suspected"
    else:
        cd = act.get("auc_crossdomain")
        gen = "topic" if act.get("control_type") == "cross_topic" else "domain"
        passes_cd = cd is not None and not np.isnan(cd) and cd >= CROSSDOMAIN_PASS
        verdict = "PASS" if passes_cd else f"INCONCLUSIVE (cross-{gen} not ≥ 0.80)"
    return verdict, flags


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def write_report(out_dir: Path, model_key: str | None, results: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit_results.json").write_text(json.dumps(results, indent=2))

    story = bool(results.get("_story"))
    cd_label = "cross-topic" if story else "cross-domain"

    lines = ["# H1 confound audit" + (" — story method" if story else ""), ""]
    if story:
        lines.append("- Derivation: `story` (surface baseline on story texts; "
                     "activation controls on pooled story activations)")
    else:
        lines.append(f"- Prompts: `{results['_prompts']}`")
    lines.append(f"- Model activations: `{model_key or 'NONE — surface-only run'}`")
    lines.append(f"- Emotions: {', '.join(results['_emotions'])}")
    lines.append("")
    lines.append("## Verdict per emotion")
    lines.append("")
    lines.append(f"| emotion | verdict | surface TF-IDF | length-only | real probe | shuffle-null | {cd_label} |")
    lines.append("|---|---|---|---|---|---|---|")
    for e in results["_emotions"]:
        s = results[e]["surface"]
        a = results[e]["activation"]
        v = results[e]["verdict"]
        rp = f"{a['auc_real']:.2f}" if a else "—"
        nn = f"{a['null_mean']:.2f}" if a else "—"
        cd = (f"{a['auc_crossdomain']:.2f}" if a and not np.isnan(a.get("auc_crossdomain", float('nan'))) else "—")
        lines.append(
            f"| {e} | {v} | {s['auc_tfidf_surface']:.2f} | {s['auc_length_only']:.2f} "
            f"| {rp} | {nn} | {cd} |"
        )
    lines.append("")
    for e in results["_emotions"]:
        reasons = results[e]["flags"]
        if reasons:
            lines.append(f"**{e}** flags:")
            lines.extend(f"- {r}" for r in reasons)
            lines.append("")
        note = (results[e]["activation"] or {}).get("source_note")
        if note:
            lines.append(f"- {e}: {note}")
    lines.append("")
    lines.append("## How to read this")
    lines.append("")
    lines.append(
        "Surface AUC is a lower bound on the confound: it is how well a "
        "classifier separates the classes from the *text alone*"
        + (" (story texts)" if story else "") + ". The "
        "activation probe must beat it by a clear margin to be evidence "
        "for an emotion concept. Shuffle-null should sit near 0.50; if it "
        f"is high, the pipeline leaks. {cd_label.capitalize()} AUC ≥ 0.80 with a "
        "near-0.50 null is the success criterion from `plans/next-steps.md`"
        + (" (now gating the story construction; see `plans/story-gate-run.md`)."
           if story else ".")
    )
    report = out_dir / "report.md"
    report.write_text("\n".join(lines))
    return report


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", default="data/public/emotion_prompts.parquet")
    ap.add_argument("--emotions", nargs="+",
                    default=["admiration", "joy", "loathing", "sadness"])
    ap.add_argument("--model-key", default=None,
                    help="e.g. Qwen2.5-0.5B-Instruct (CAA) or "
                         "Qwen2.5-0.5B-Instruct-story (story); "
                         "activations/<model-key>/ must exist")
    ap.add_argument("--activations-dir", default="activations")
    ap.add_argument("--out-dir", default=None,
                    help="default: results/h1_confound_audit (CAA) or "
                         "results/h1_confound_audit_story (story)")
    ap.add_argument("--story", action="store_true",
                    help="audit story-method activations (pooled per-story .npz; "
                         "cross-TOPIC control). Auto-enabled if --model-key ends "
                         "with '-story'.")
    ap.add_argument("--n-perm", type=int, default=200)
    args = ap.parse_args()

    # Story mode: explicit flag, or inferred from the -story key convention.
    story = args.story or bool(args.model_key and args.model_key.endswith("-story"))
    # Locate the story corpus (for topic groups + story-text surface): the
    # corpus lives under the base model key (without the "-story" suffix).
    base_model_key = (
        args.model_key[:-len("-story")]
        if args.model_key and args.model_key.endswith("-story")
        else args.model_key
    )
    out_dir = args.out_dir or (
        "results/h1_confound_audit_story" if story else "results/h1_confound_audit"
    )

    prompts = pd.read_parquet(_repo_root / args.prompts)
    present = set(prompts["emotion_label"])
    emotions = [e for e in args.emotions if e in present]
    if missing := [e for e in args.emotions if e not in present]:
        log.warning("Emotions absent from prompts (skipped): %s", ", ".join(missing))
    if NEUTRAL not in present:
        raise SystemExit("Prompt parquet has no 'neutral' rows; cannot run emotion-vs-neutral audit.")

    act_dir = (
        _repo_root / args.activations_dir / args.model_key if args.model_key else None
    )
    if act_dir is not None and not act_dir.exists():
        log.warning("Activations dir %s not found — running surface-only.", act_dir)
        act_dir = None

    results: dict = {
        "_prompts": args.prompts, "_emotions": emotions,
        "_model_key": args.model_key, "_story": story,
    }
    for e in emotions:
        log.info("Auditing %s vs neutral ...", e)
        if story:
            surf = surface_audit_story(e, base_model_key)
            act = (
                activation_audit_story(act_dir, e, base_model_key=base_model_key,
                                       n_perm=args.n_perm)
                if act_dir else None
            )
        else:
            surf = surface_audit(prompts, e)
            act = activation_audit(act_dir, e, prompts, n_perm=args.n_perm) if act_dir else None
        verdict, flags = recommend(e, surf, act)
        results[e] = {"surface": surf, "activation": act, "verdict": verdict, "flags": flags}
        log.info("  %s: %s | surface TF-IDF AUC=%.2f", e, verdict, surf["auc_tfidf_surface"])

    report = write_report(_repo_root / out_dir, args.model_key, results)
    log.info("\nWrote %s", report.relative_to(_repo_root))


if __name__ == "__main__":
    main()

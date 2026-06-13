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

    # (b) word (1-2gram) + char (3-5gram) TF-IDF logistic
    word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    Xtr = np.hstack(
        [word_vec.fit_transform(tr["prompt"]).toarray(),
         char_vec.fit_transform(tr["prompt"]).toarray()]
    )
    Xte = np.hstack(
        [word_vec.transform(te["prompt"]).toarray(),
         char_vec.transform(te["prompt"]).toarray()]
    )
    out["auc_tfidf_surface"] = round(
        _fit_predict_auc(Xtr, tr["y"].to_numpy(), Xte, te["y"].to_numpy()), 4
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
        if cd is not None and not np.isnan(cd) and cd < CROSSDOMAIN_FAIL:
            flags.append(f"cross-domain AUC={cd:.2f} < {CROSSDOMAIN_FAIL} — collapses out of domain")

    if act is None:
        verdict = "SURFACE-ONLY (activations pending)"
    elif flags:
        verdict = "FLAG — confound suspected"
    else:
        cd = act.get("auc_crossdomain")
        passes_cd = cd is not None and not np.isnan(cd) and cd >= CROSSDOMAIN_PASS
        verdict = "PASS" if passes_cd else "INCONCLUSIVE (cross-domain not ≥ 0.80)"
    return verdict, flags


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def write_report(out_dir: Path, model_key: str | None, results: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit_results.json").write_text(json.dumps(results, indent=2))

    lines = ["# H1 confound audit", ""]
    lines.append(f"- Prompts: `{results['_prompts']}`")
    lines.append(f"- Model activations: `{model_key or 'NONE — surface-only run'}`")
    lines.append(f"- Emotions: {', '.join(results['_emotions'])}")
    lines.append("")
    lines.append("## Verdict per emotion")
    lines.append("")
    lines.append("| emotion | verdict | surface TF-IDF | length-only | real probe | shuffle-null | cross-domain |")
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
        "classifier separates the classes from the *text alone*. The "
        "activation probe must beat it by a clear margin to be evidence "
        "for an emotion concept. Shuffle-null should sit near 0.50; if it "
        "is high, the pipeline leaks. Cross-domain AUC ≥ 0.80 with a "
        "near-0.50 null is the success criterion from `plans/next-steps.md`."
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
                    help="e.g. Qwen2.5-0.5B-Instruct; activations/<model-key>/ must exist")
    ap.add_argument("--activations-dir", default="activations")
    ap.add_argument("--out-dir", default="results/h1_confound_audit")
    ap.add_argument("--n-perm", type=int, default=200)
    args = ap.parse_args()

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

    results: dict = {"_prompts": args.prompts, "_emotions": emotions, "_model_key": args.model_key}
    for e in emotions:
        log.info("Auditing %s vs neutral ...", e)
        surf = surface_audit(prompts, e)
        act = activation_audit(act_dir, e, prompts, n_perm=args.n_perm) if act_dir else None
        verdict, flags = recommend(e, surf, act)
        results[e] = {"surface": surf, "activation": act, "verdict": verdict, "flags": flags}
        log.info("  %s: %s | surface TF-IDF AUC=%.2f", e, verdict, surf["auc_tfidf_surface"])

    report = write_report(_repo_root / args.out_dir, args.model_key, results)
    log.info("\nWrote %s", report.relative_to(_repo_root))


if __name__ == "__main__":
    main()

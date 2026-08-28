"""Screen generated story corpora for refusals, assistant-voice breaks, and label leakage.

Diagnostic only. Reads the story parquets produced by
``scripts/generate_emotion_stories.py`` and reports how often each corpus
contains text that is not a story: refusals, "as an AI language model"
self-reference, echoed instructions, or the emotion label itself (which
the paper-method generation prompt bans).

Why this matters
----------------
``generate_emotion_stories.py`` gates only on ``min_story_tokens``. A
62-token refusal clears a 60-token gate and lands in the corpus. Two
distinct harms follow:

1. **Uniform contamination** (same rate in every emotion) mostly adds
   noise, and cross-emotion centering removes much of it.
2. **Differential contamination** (one emotion refuses more than the
   others) is a confound *aligned with the label*, and the derived
   vector partly encodes "refusal" or "assistant voice" rather than the
   emotion. Neutral-PC projection does not remove it, because the
   contaminating text appears in the emotion corpora and not in the
   neutral basis.

The differential case is therefore the one this script tests for.

Usage
-----

.. code-block:: bash

    # local dev corpora
    uv run python scripts/screen_story_corpora.py

    # production corpora pulled from the private HF dataset first:
    #   huggingface-cli download llm-psych/llm-psych-activations \
    #     --repo-type dataset --include 'stories/*' --local-dir hf_stories
    uv run python scripts/screen_story_corpora.py \
        --stories-dir hf_stories/stories \
        --models Llama-3.1-8B-Instruct Qwen2.5-7B-Instruct gemma-2-9b-it

Outputs (under ``--out-dir``, default ``results/story_screening/``):

* ``flagged_rows.csv`` — every flagged story with its families and text.
* ``summary.csv`` — per (model, emotion, family) rate + Wilson 95% CI.
* ``report.md`` — human-readable summary and the differential tests.

No pre-registration implication: this is a data-quality audit, not a
hypothesis test about emotion representations. It does not require an
amendment. If it finds differential contamination, the *remedy* (a
regeneration or an exclusion rule) does.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root / "src"))

log = logging.getLogger(__name__)

# Bump when the patterns below change; recorded in the report so a rate is
# never compared across screener versions without noticing.
SCREEN_VERSION = "1.0"

# --------------------------------------------------------------------------
# Detectors
# --------------------------------------------------------------------------
# Deliberately literal and high-precision. These are meant to catch the
# failure modes actually observed in the Qwen 0.5B dev corpus, not to be a
# general-purpose refusal classifier. Recall is knowingly incomplete; a
# flagged story is near-certainly contaminated, an unflagged one is not
# guaranteed clean. Under-detection biases the audit toward "corpus is
# fine", so a positive finding here is trustworthy and a null is weak.

PATTERNS: dict[str, list[str]] = {
    "refusal": [
        r"\bI(?:'m| am) sorry,? but\b",
        r"\bI can(?:no|')t (?:fulfill|assist|help|comply|create|generate|write)\b",
        r"\bI(?:'m| am) (?:un)?able to (?:fulfill|assist|help|comply)\b",
        r"\bagainst my (?:programming|guidelines|principles)\b",
        r"\bI must (?:decline|refuse)\b",
        r"\b(?:ethical|safety) guidelines\b",
    ],
    "assistant_voice": [
        r"\bas an AI(?: language model)?\b",
        r"\bas a (?:language model|machine learning (?:algorithm|model))\b",
        r"\bI (?:don'?t|do not) (?:have|experience|possess) (?:emotions|feelings)\b",
        r"\bI(?:'m| am) an AI\b",
        r"\bmy purpose is to provide\b",
    ],
    "instruction_echo": [
        r"\b150[- ]word\b",
        r"\bnarrative about the topic\b",
        r"\b(?:Sure|Certainly|Of course)[,!]? (?:here|I)\b",
        r"^\s*(?:Here(?:'s| is)|Title:)\b",
    ],
}

# Emotion-label morphology only — NOT a synonym detector. The generation
# prompt bans the emotion word; this checks that ban was respected. Stems
# are matched as prefixes at a word boundary, so "admir" catches
# admiration/admire/admiring/admirable.
LABEL_STEMS: dict[str, list[str]] = {
    "admiration": ["admir"],
    "joy": ["joy"],
    "loathing": ["loath"],
    "sadness": ["sad", "sorrow"],
    "neutral": [],
}

_COMPILED = {
    fam: [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in pats]
    for fam, pats in PATTERNS.items()
}


def _label_regex(emotion: str) -> re.Pattern[str] | None:
    """Compile the own-label leakage pattern for ``emotion``, if any."""
    stems = LABEL_STEMS.get(emotion, [])
    if not stems:
        return None
    return re.compile(r"\b(?:" + "|".join(stems) + r")\w*", re.IGNORECASE)


def screen_text(text: str, emotion: str) -> list[str]:
    """Return the list of flag families firing on ``text``.

    Parameters
    ----------
    text : str
        Story text as generated (no special tokens).
    emotion : str
        Emotion label of the corpus the story belongs to; controls which
        label-leakage stems apply.

    Returns
    -------
    list of str
        Family names, e.g. ``["refusal", "assistant_voice"]``. Empty if
        the story looks clean.
    """
    fired = [
        fam for fam, regexes in _COMPILED.items()
        if any(r.search(text) for r in regexes)
    ]
    label_re = _label_regex(emotion)
    if label_re is not None and label_re.search(text):
        fired.append("label_leak")
    return fired


FAMILIES = list(PATTERNS) + ["label_leak"]


# --------------------------------------------------------------------------
# Corpus loading
# --------------------------------------------------------------------------

@dataclass
class Corpus:
    """One (model, emotion) story parquet loaded into memory."""

    model_key: str
    emotion: str
    df: pd.DataFrame


def load_corpora(stories_dir: Path, models: list[str] | None) -> list[Corpus]:
    """Load every ``<stories_dir>/<model_key>/<emotion>.parquet``."""
    model_dirs = (
        [stories_dir / m for m in models]
        if models
        else sorted(p for p in stories_dir.iterdir() if p.is_dir())
    )
    corpora: list[Corpus] = []
    for mdir in model_dirs:
        if not mdir.is_dir():
            log.warning("Skipping missing model dir: %s", mdir)
            continue
        for path in sorted(mdir.glob("*.parquet")):
            if path.name.endswith(".meta.parquet"):
                continue
            df = pd.read_parquet(path)
            if "story_text" not in df.columns:
                log.warning("Skipping %s (no story_text column)", path)
                continue
            corpora.append(Corpus(mdir.name, path.stem, df))
    if not corpora:
        found = (
            sorted(p.name for p in stories_dir.iterdir() if p.is_dir())
            if stories_dir.is_dir() else []
        )
        raise FileNotFoundError(
            f"No story parquets found under {stories_dir}.\n"
            f"Model dirs present: {found or '(none)'}\n"
            "Production corpora live on the private HF dataset; fetch them "
            "with:  uv run python scripts/pull_stories.py --list"
        )
    return corpora


# --------------------------------------------------------------------------
# Screening
# --------------------------------------------------------------------------

def screen(corpora: list[Corpus], pool_start_token: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Screen all corpora.

    Returns
    -------
    rows : DataFrame
        One row per story, with a boolean column per family plus ``any_flag``
        and ``usable_tokens`` (positions surviving the pooling start).
    summary : DataFrame
        One row per (model, emotion, family) with count, rate and Wilson CI.
    """
    records = []
    for c in corpora:
        for _, r in c.df.iterrows():
            fired = screen_text(str(r["story_text"]), c.emotion)
            n_tokens = int(r["n_tokens"]) if "n_tokens" in r else -1
            rec = {
                "model_key": c.model_key,
                "emotion": c.emotion,
                "story_id": r.get("id", ""),
                "topic": r.get("topic", ""),
                "n_tokens": n_tokens,
                "usable_tokens": max(n_tokens - pool_start_token, 0) if n_tokens >= 0 else -1,
                "story_text": str(r["story_text"]),
            }
            for fam in FAMILIES:
                rec[fam] = fam in fired
            rec["any_flag"] = bool(fired)
            records.append(rec)

    rows = pd.DataFrame.from_records(records)

    summary_records = []
    for (model_key, emotion), g in rows.groupby(["model_key", "emotion"], sort=True):
        n = len(g)
        for fam in FAMILIES + ["any_flag"]:
            k = int(g[fam].sum())
            lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
            summary_records.append({
                "model_key": model_key,
                "emotion": emotion,
                "family": fam,
                "n": n,
                "n_flagged": k,
                "rate": k / n,
                "ci_lo": lo,
                "ci_hi": hi,
            })
    summary = pd.DataFrame.from_records(summary_records)
    return rows, summary


def differential_tests(rows: pd.DataFrame) -> pd.DataFrame:
    """Test whether flag rates differ across emotions, within each model.

    Two tests per (model, family): an omnibus chi-square over the
    emotion x flagged table, and per-emotion one-vs-rest Fisher exact
    tests. The one-vs-rest p-values are BH-FDR corrected within each
    (model, family), per the project's multiple-comparisons convention
    for exploratory contrasts.
    """
    out = []
    for model_key, gm in rows.groupby("model_key", sort=True):
        for fam in FAMILIES + ["any_flag"]:
            # reindex (not [[False, True]]) — pandas reads a bool list as a mask
            table = pd.crosstab(gm["emotion"], gm[fam]).reindex(
                columns=[False, True], fill_value=0
            )

            if table[True].sum() == 0:
                omnibus_p = float("nan")
            elif (table.sum(axis=1) == 0).any() or table.shape[0] < 2:
                omnibus_p = float("nan")
            else:
                omnibus_p = float(chi2_contingency(table.values)[1])

            emotions = list(table.index)
            raw_p, rates = [], []
            for emo in emotions:
                a = int(table.loc[emo, True])
                b = int(table.loc[emo, False])
                c_ = int(table[True].sum() - a)
                d_ = int(table[False].sum() - b)
                rates.append(a / (a + b) if (a + b) else float("nan"))
                if table[True].sum() == 0:
                    raw_p.append(float("nan"))
                else:
                    raw_p.append(float(fisher_exact([[a, b], [c_, d_]])[1]))

            valid = [i for i, p in enumerate(raw_p) if p == p]
            adj = [float("nan")] * len(raw_p)
            if valid:
                _, adj_vals, _, _ = multipletests(
                    [raw_p[i] for i in valid], alpha=0.05, method="fdr_bh"
                )
                for i, v in zip(valid, adj_vals):
                    adj[i] = float(v)

            for emo, rate, p, q in zip(emotions, rates, raw_p, adj):
                out.append({
                    "model_key": model_key,
                    "family": fam,
                    "emotion": emo,
                    "rate": rate,
                    "omnibus_chi2_p": omnibus_p,
                    "one_vs_rest_p": p,
                    "one_vs_rest_q_bh": q,
                })
    return pd.DataFrame.from_records(out)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def write_report(
    rows: pd.DataFrame,
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    out_path: Path,
    pool_start_token: int,
) -> None:
    """Write the markdown screening report."""
    lines = [
        "# Story-corpus screening report",
        "",
        f"Screener version: `{SCREEN_VERSION}`  ·  pooling starts at token "
        f"{pool_start_token}",
        "",
        "Diagnostic only — no pre-registration implication. Detectors are "
        "high-precision and knowingly incomplete in recall, so a positive "
        "finding is trustworthy and a null is weak evidence of cleanliness.",
        "",
        "## Flag rates by model and emotion",
        "",
        "| model | emotion | n | any flag | refusal | assistant voice | instr. echo | label leak |",
        "|---|---|---|---|---|---|---|---|",
    ]
    piv = summary.pivot_table(
        index=["model_key", "emotion"], columns="family", values="rate"
    )
    cnt = summary.groupby(["model_key", "emotion"])["n"].first()
    for (model_key, emotion), r in piv.iterrows():
        lines.append(
            f"| {model_key} | {emotion} | {int(cnt.loc[(model_key, emotion)])} | "
            f"{r.get('any_flag', 0):.1%} | {r.get('refusal', 0):.1%} | "
            f"{r.get('assistant_voice', 0):.1%} | {r.get('instruction_echo', 0):.1%} | "
            f"{r.get('label_leak', 0):.1%} |"
        )

    lines += [
        "",
        "## Differential contamination (the part that matters)",
        "",
        "Uniform contamination across emotions is mostly noise and is largely "
        "removed by cross-emotion centering. Contamination concentrated in one "
        "emotion is a confound aligned with the label. Rows below are "
        "one-vs-rest contrasts significant at BH-FDR q < 0.05.",
        "",
    ]
    sig = tests[(tests["one_vs_rest_q_bh"] < 0.05)].sort_values("one_vs_rest_q_bh")
    if sig.empty:
        lines.append("No emotion shows a differential flag rate at q < 0.05.")
    else:
        lines += [
            "| model | family | emotion | rate | omnibus p | one-vs-rest q |",
            "|---|---|---|---|---|---|",
        ]
        for _, r in sig.iterrows():
            lines.append(
                f"| {r.model_key} | {r.family} | {r.emotion} | {r.rate:.1%} | "
                f"{r.omnibus_chi2_p:.3g} | {r.one_vs_rest_q_bh:.3g} |"
            )

    lines += [
        "",
        "## Pooling-window viability",
        "",
        "Stories are pooled from token "
        f"{pool_start_token} onward, so a story contributes "
        "`n_tokens - pool_start` positions. Short stories that clear the "
        "length gate still contribute very few positions.",
        "",
        "| model | emotion | median usable tokens | share with < 20 usable |",
        "|---|---|---|---|",
    ]
    have_tokens = rows[rows["usable_tokens"] >= 0]
    if have_tokens.empty:
        lines.append("| — | — | (no n_tokens column) | — |")
    else:
        for (model_key, emotion), g in have_tokens.groupby(["model_key", "emotion"]):
            lines.append(
                f"| {model_key} | {emotion} | {int(g['usable_tokens'].median())} | "
                f"{(g['usable_tokens'] < 20).mean():.1%} |"
            )

    lines += [
        "",
        "## How to read a positive result",
        "",
        "1. **Differential refusal or assistant-voice flags** in one emotion "
        "mean that emotion's vector partly encodes refusal / assistant "
        "register. Neutral-PC projection will not remove it, because the "
        "contaminating text is absent from the neutral basis.",
        "2. **Label leak** is a direct violation of the generation design "
        "(the emotion word is supposed to be banned), and makes the vector "
        "partly a token-identity direction.",
        "3. The remedy — regeneration with a stricter prompt, or a "
        "pre-specified exclusion rule — changes the derivation corpus and "
        "therefore needs a dated amendment in `HYPOTHESES.md` before any "
        "affected vector is re-derived.",
        "",
    ]
    out_path.write_text("\n".join(lines))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--stories-dir", type=Path,
        default=_repo_root / "data" / "derived" / "stories",
        help="directory containing <model_key>/<emotion>.parquet",
    )
    ap.add_argument(
        "--models", nargs="*", default=None,
        help="model_key subfolders to screen (default: all found)",
    )
    ap.add_argument(
        "--pool-start-token", type=int, default=50,
        help="must match derivation.pool_start_token (default 50)",
    )
    ap.add_argument(
        "--out-dir", type=Path,
        default=_repo_root / "results" / "story_screening",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    corpora = load_corpora(args.stories_dir, args.models)
    log.info("Loaded %d corpora from %s", len(corpora), args.stories_dir)

    rows, summary = screen(corpora, args.pool_start_token)
    tests = differential_tests(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    flagged = rows[rows["any_flag"]].copy()
    flagged.to_csv(args.out_dir / "flagged_rows.csv", index=False)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    tests.to_csv(args.out_dir / "differential_tests.csv", index=False)
    write_report(rows, summary, tests, args.out_dir / "report.md", args.pool_start_token)

    n_flag = int(rows["any_flag"].sum())
    log.info(
        "Screened %d stories; %d flagged (%.1f%%). Report: %s",
        len(rows), n_flag, 100 * n_flag / len(rows), args.out_dir / "report.md",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build a topic-rebalanced story exclusion list from a screening run.

Consumes ``results/story_screening/flagged_rows.csv`` (see
``scripts/screen_story_corpora.py``) and emits a frozen, MD5-hashed list
of ``story_id`` values to drop before re-deriving emotion vectors.

Why topic-rebalancing
---------------------
The story design's primary confound control is topic matching: every
emotion is generated over the same topic list, ``stories_per_topic``
each, so topic distribution is identical across emotions and cannot
separate them (``configs/derivation/story.yaml``).

Contamination is **not** uniformly distributed over topics — on Qwen
2.5 7B, sadness leaks the label on 7/7 stories for "a stranger asking
for directions" and 0/7 on eight other topics. Dropping flagged stories
alone would therefore trade a lexical confound for a topic confound,
which is the worse of the two: topic balance is what the design relies
on, whereas lexical leakage is what we are trying to remove.

Rule (deterministic, no free parameters)
----------------------------------------
For each topic ``t``:

1. ``s(e, t)`` = surviving (unflagged) stories for emotion ``e``.
2. ``k(t) = min_e s(e, t)`` — the binding emotion for that topic.
3. Keep exactly ``k(t)`` stories per emotion, chosen by sorted
   ``story_id`` for reproducibility. Drop the rest.

Topics where any emotion loses every story get ``k(t) = 0`` and are
dropped for all emotions. The result is exactly topic-matched by
construction, at the cost of sample size.

Exclusion is on ``any_flag`` (refusal, assistant voice, instruction echo,
label leak), not ``label_leak`` alone — the non-leak families are rare
here, so the extra strictness is nearly free and simpler to justify.

Usage
-----

.. code-block:: bash

    uv run python scripts/build_story_exclusions.py --model Qwen2.5-7B-Instruct

Outputs to ``results/story_screening/<model_key>/``:

* ``exclusions.json`` — dropped ``story_id`` list, per-emotion counts,
  the surviving topic/emotion grid, and an MD5 over the sorted id list.
* ``exclusions_report.md`` — before/after n and the cost of the rule.

The MD5 is the freeze: quote it in any analysis derived from this list,
per the stimulus-locking practice in ``configs/stimuli_hashes.yaml``.

This script does not modify any corpus or vector. It only decides what a
downstream re-derivation should ignore.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root / "src"))

EXCLUDE_ON = "any_flag"


def load_inputs(
    model_key: str, stories_dir: Path, flagged_csv: Path
) -> tuple[pd.DataFrame, set[str]]:
    """Return (all stories for the model, set of flagged story_ids)."""
    mdir = stories_dir / model_key
    if not mdir.is_dir():
        raise FileNotFoundError(f"No story corpus at {mdir}")

    frames = []
    for path in sorted(mdir.glob("*.parquet")):
        if path.name.endswith(".meta.parquet"):
            continue
        df = pd.read_parquet(path)[["id", "topic", "emotion_label"]]
        frames.append(df)
    stories = pd.concat(frames, ignore_index=True)

    flagged_all = pd.read_csv(flagged_csv)
    flagged = set(
        flagged_all.loc[
            (flagged_all["model_key"] == model_key) & (flagged_all[EXCLUDE_ON]),
            "story_id",
        ].astype(str)
    )
    return stories, flagged


def rebalance(stories: pd.DataFrame, flagged: set[str]) -> tuple[list[str], pd.DataFrame]:
    """Apply the topic-rebalancing rule.

    Returns
    -------
    dropped : list of str
        story_ids to exclude (flagged + rebalancing surplus), sorted.
    grid : DataFrame
        Per (topic, emotion): original n, surviving-after-flag n, and the
        final kept n.
    """
    stories = stories.copy()
    stories["flagged"] = stories["id"].astype(str).isin(flagged)

    kept_ids: list[str] = []
    grid_rows = []

    for topic, gt in stories.groupby("topic", sort=True):
        survivors = {
            emo: sorted(ge.loc[~ge["flagged"], "id"].astype(str))
            for emo, ge in gt.groupby("emotion_label", sort=True)
        }
        k = min(len(v) for v in survivors.values())
        for emo, ids in survivors.items():
            n_orig = int((gt["emotion_label"] == emo).sum())
            kept_ids.extend(ids[:k])
            grid_rows.append({
                "topic": topic,
                "emotion": emo,
                "n_original": n_orig,
                "n_after_flag": len(ids),
                "n_kept": k,
            })

    all_ids = set(stories["id"].astype(str))
    dropped = sorted(all_ids - set(kept_ids))
    return dropped, pd.DataFrame(grid_rows)


def write_outputs(
    model_key: str, dropped: list[str], grid: pd.DataFrame, out_dir: Path
) -> str:
    """Write exclusions.json + report; return the MD5 freeze hash."""
    out_dir.mkdir(parents=True, exist_ok=True)
    md5 = hashlib.md5("\n".join(dropped).encode()).hexdigest()

    per_emotion = (
        grid.groupby("emotion")[["n_original", "n_after_flag", "n_kept"]]
        .sum()
        .astype(int)
    )
    topics_total = grid["topic"].nunique()
    topics_dead = int((grid.groupby("topic")["n_kept"].first() == 0).sum())

    payload = {
        "model_key": model_key,
        "exclude_on": EXCLUDE_ON,
        "rule": "topic-rebalanced: keep k(t)=min_e surviving(e,t) per emotion",
        "n_dropped": len(dropped),
        "md5": md5,
        "per_emotion": per_emotion.to_dict(orient="index"),
        "topics_total": topics_total,
        "topics_fully_dropped": topics_dead,
        "dropped_story_ids": dropped,
    }
    (out_dir / "exclusions.json").write_text(json.dumps(payload, indent=2))
    grid.to_csv(out_dir / "exclusions_grid.csv", index=False)

    lines = [
        f"# Story exclusions — {model_key}",
        "",
        f"Rule: topic-rebalanced, excluding on `{EXCLUDE_ON}`.  ",
        f"Freeze MD5 (sorted dropped ids): `{md5}`",
        "",
        "## Cost of the rule",
        "",
        "| emotion | original n | after dropping flagged | after rebalancing |",
        "|---|---|---|---|",
    ]
    for emo, r in per_emotion.iterrows():
        lines.append(
            f"| {emo} | {r.n_original} | {r.n_after_flag} | {r.n_kept} |"
        )
    lines += [
        "",
        f"Topics: {topics_total} total, {topics_dead} dropped entirely "
        "(some emotion lost all its stories there).",
        "",
        "After rebalancing, every emotion has an identical topic "
        "distribution by construction, so topic cannot separate emotions "
        "in the re-derived vectors.",
        "",
        "## How to use",
        "",
        "Pass `dropped_story_ids` to the re-derivation as an exclusion "
        "filter on the activation rows (`<emotion>.meta.parquet` carries "
        "the aligned `story_id`). Quote the MD5 above in any result "
        "derived from this list.",
        "",
    ]
    (out_dir / "exclusions_report.md").write_text("\n".join(lines))
    return md5


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", dest="model_key", required=True)
    ap.add_argument(
        "--stories-dir", type=Path,
        default=_repo_root / "data" / "derived" / "stories",
    )
    ap.add_argument(
        "--flagged-csv", type=Path,
        default=_repo_root / "results" / "story_screening" / "flagged_rows.csv",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=None,
        help="default: results/story_screening/<model_key>/",
    )
    args = ap.parse_args()

    out_dir = args.out_dir or (
        _repo_root / "results" / "story_screening" / args.model_key
    )

    stories, flagged = load_inputs(args.model_key, args.stories_dir, args.flagged_csv)
    dropped, grid = rebalance(stories, flagged)
    md5 = write_outputs(args.model_key, dropped, grid, out_dir)

    kept = len(stories) - len(dropped)
    print(
        f"{args.model_key}: {len(stories)} stories -> {kept} kept "
        f"({len(dropped)} dropped, {len(flagged)} of them flagged, "
        f"{len(dropped) - len(flagged)} to rebalance topics)"
    )
    print(f"freeze MD5: {md5}")
    print(f"wrote {out_dir / 'exclusions.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

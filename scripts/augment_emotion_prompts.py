"""Augment emotion_prompts.parquet via LLM paraphrase generation.

Takes the hand-authored seed set and generates N paraphrases per seed
prompt using Anthropic Claude Haiku (temperature=0.7), then reshuffles
into a 70/30 train/test split.

Usage
-----
    # Requires ANTHROPIC_API_KEY in environment
    uv run python scripts/augment_emotion_prompts.py \
        --n-paraphrases 14 \
        --output data/public/emotion_prompts_augmented.parquet

Design (per HYPOTHESES.md amendment 2026-05-15)
-----------------------------------------------
- Paraphrase generation: Claude Haiku, temperature=0.7, max 3 retries
  per prompt on parse failure.
- Preserve: emotion, category, approximate length, absence of explicit
  emotion words.
- Final split: 70/30 train/test per emotion category, seeded shuffle.
- Spot-check: the script prints n=30 random paraphrases for human
  review before freezing.

Schema matches ``build_emotion_prompts.py`` with one added column:
  ``source`` — "hand_authored" | "paraphrase"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root / "src"))

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a paraphrase assistant. Rewrite the user's sentence so that "
    "it conveys the SAME emotional tone and belongs to the SAME life domain, "
    "but uses DIFFERENT words and sentence structure. "
    "Keep the length roughly similar (within ±30%%). "
    "Do NOT use explicit emotion words (e.g. 'happy', 'angry', 'sad', 'afraid'). "
    "Output ONLY a JSON object with a single key 'paraphrase' whose value is the rewritten sentence."
)


# ---------------------------------------------------------------------------
#  Anthropic client
# ---------------------------------------------------------------------------

def _get_client():
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "anthropic package is required. Run: uv pip install anthropic"
        ) from exc
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not found in environment. "
            "Add it to .env and run with: set -a; source .env; set +a"
        )
    return anthropic.Anthropic(api_key=key)


def _paraphrase_one(
    client,
    text: str,
    emotion: str,
    category: str,
    *,
    model: str = "claude-3-haiku-20240307",
    temperature: float = 0.7,
    max_retries: int = 3,
) -> str:
    user_msg = (
        f"Original sentence (emotion={emotion}, domain={category}):\n{text}"
    )
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=256,
                temperature=temperature,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = resp.content[0].text
            data = json.loads(raw)
            para = data["paraphrase"].strip()
            if para:
                return para
        except Exception as exc:
            log.warning("Paraphrase attempt %d failed: %s", attempt + 1, exc)
    raise RuntimeError(f"Failed to paraphrase after {max_retries} attempts: {text[:60]}...")


# ---------------------------------------------------------------------------
#  Main pipeline
# ---------------------------------------------------------------------------

def _build_rows(
    df: pd.DataFrame,
    n_paraphrases: int,
    client,
    model: str,
    temperature: float,
) -> pd.DataFrame:
    """Generate paraphrases and return a new DataFrame with all rows."""
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="paraphrasing"):
        # original row
        rows.append({
            "id": row["id"],
            "prompt": row["prompt"],
            "emotion_label": row["emotion_label"],
            "split": "_orig",  # temporary marker
            "category": row["category"],
            "length_words": len(row["prompt"].split()),
            "source": "hand_authored",
        })
        # paraphrases
        for p_idx in range(1, n_paraphrases + 1):
            para = _paraphrase_one(
                client,
                row["prompt"],
                row["emotion_label"],
                row["category"],
                model=model,
                temperature=temperature,
            )
            rows.append({
                "id": f"{row['id']}_p{p_idx:02d}",
                "prompt": para,
                "emotion_label": row["emotion_label"],
                "split": "_orig",
                "category": row["category"],
                "length_words": len(para.split()),
                "source": "paraphrase",
            })
    return pd.DataFrame(rows)


def _reshuffle(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Re-shuffle into train/test per emotion, ignoring old split markers."""
    rng = random.Random(rng_seed)
    out_rows = []
    for emotion in df["emotion_label"].unique():
        sub = df[df["emotion_label"] == emotion].copy()
        idxs = sub.index.tolist()
        rng.shuffle(idxs)
        n_train = int(len(idxs) * train_frac)
        for i, idx in enumerate(idxs, start=1):
            sub.loc[idx, "split"] = "train" if i <= n_train else "test"
            sub.loc[idx, "id"] = f"{emotion}_{i:04d}"
        out_rows.append(sub)
    return pd.concat(out_rows, ignore_index=True).sort_values(
        ["emotion_label", "id"]
    ).reset_index(drop=True)


def _spot_check(df: pd.DataFrame, n: int = 30) -> None:
    """Print random paraphrases for human review."""
    para_df = df[df["source"] == "paraphrase"]
    sample = para_df.sample(n=min(n, len(para_df)), random_state=42)
    print(f"\n=== SPOT CHECK (n={len(sample)}) ===")
    for _, row in sample.iterrows():
        print(f"\n[{row['emotion_label']} | {row['category']} | {row['length_words']} words]")
        print(f"  {row['prompt']}")
    print("\n=== END SPOT CHECK ===")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/public/emotion_prompts.parquet",
        help="Path to seed emotion prompts parquet",
    )
    parser.add_argument(
        "--output",
        default="data/public/emotion_prompts_augmented.parquet",
        help="Path for augmented output",
    )
    parser.add_argument(
        "--n-paraphrases",
        type=int,
        default=14,
        help="Number of paraphrases per seed prompt",
    )
    parser.add_argument(
        "--model",
        default="claude-3-haiku-20240307",
        help="Anthropic model for paraphrase generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip LLM calls; only re-shuffle existing input (for testing)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    seed_df = pd.read_parquet(args.input)
    log.info("Loaded %d seed prompts", len(seed_df))

    if args.skip_generation:
        combined = seed_df.copy()
    else:
        client = _get_client()
        combined = _build_rows(
            seed_df,
            args.n_paraphrases,
            client,
            model=args.model,
            temperature=args.temperature,
        )
        log.info("Generated %d total prompts (%d paraphrases)",
                 len(combined), len(combined) - len(seed_df))

    final = _reshuffle(combined, train_frac=0.7, rng_seed=42)

    # Sanity checks
    print("\nSplit counts:")
    print(final.groupby(["emotion_label", "split"]).size().unstack(fill_value=0))
    print("\nSource counts:")
    print(final.groupby(["emotion_label", "source"]).size().unstack(fill_value=0))
    print("\nLength stats (words):")
    print(final.groupby("emotion_label")["length_words"].describe().round(1))

    _spot_check(final, n=30)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(out_path, index=False)
    log.info("Saved %d rows to %s", len(final), out_path)

    # Check for explicit emotion words in non-neutral prompts
    emotion_words = {
        "joy": {"happy", "joy", "joyful", "elated", "ecstatic", "delighted", "cheerful"},
        "fear": {"afraid", "scared", "terrified", "fear", "frightened", "panicked", "anxious"},
        "anger": {"angry", "furious", "rage", "irritated", "annoyed", "livid", "outraged"},
        "sadness": {"sad", "depressed", "grief", "miserable", "sorrowful", "melancholy", "heartbroken"},
    }
    for label, words in emotion_words.items():
        subset = final[final["emotion_label"] == label]
        flagged = subset[subset["prompt"].str.lower().apply(lambda s: any(w in s for w in words))]
        log.info("%s: %d/%d prompts contain explicit emotion words", label, len(flagged), len(subset))


if __name__ == "__main__":
    main()

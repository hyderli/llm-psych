"""Compare llm-psych steering vectors against EmotionScope vectors.

Usage
-----

    uv run python scripts/validate_with_emotionscope.py \
        --emotionscope-vectors results/vectors/google_gemma-2-2b-it.pt \
        --llm-psych-vectors-dir steering_vectors/gemma-2-2b-it \
        --output similarity_table.json

The script loads both vector sets, maps emotion names between the two
conventions, L2-normalises, and prints a per-emotion cosine-similarity
table.  Unmatched emotions are listed separately so you can see coverage.

If EmotionScope saved its vectors as a plain tensor (not a dict) you
must also pass ``--emotionscope-emotions`` with a comma-separated list
or a JSON file containing the emotion labels in the same order as the
tensor rows.

Returns exit code 0 even when similarities are low — this is a
measurement, not a pass/fail gate.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Optional: use the local EmotionScope clone directly
# --------------------------------------------------------------------------
# EmotionScope cannot be added to pyproject.toml because its
# ``tool.uv.sources`` forces ``torch`` to the CUDA 12.1 index, which has
# no macOS wheels and therefore breaks ``uv sync`` on Apple Silicon.
# Instead, we dynamically inject the local clone into ``sys.path`` so the
# user can still call ``from emotion_scope import ...`` when desired.

_LOCAL_EMOTIONSCOPE_PATHS = [
    Path.home() / "EmotionScope",
    Path.home() / "research" / "EmotionScope",
    Path.home() / "projects" / "EmotionScope",
    Path("/Users/haydaraliseker/EmotionScope"),
]

for _p in _LOCAL_EMOTIONSCOPE_PATHS:
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
        log.debug("Added local EmotionScope path: %s", _p)
        break

# --------------------------------------------------------------------------
# Name mapping: llm-psych emotion -> EmotionScope emotion
# --------------------------------------------------------------------------

_LLM_PSYCH_TO_EMOTIONSCOPE: dict[str, str] = {
    "afraid": "fear",
    "anger": "anger",
    "blissful": "bliss",
    "calm": "calm",
    "compassionate": "compassion",
    "desperate": "desperation",
    "fear": "fear",
    "hostile": "hostility",
    "joy": "joy",
    "joyful": "joy",
    "offended": "offended",
    "sadness": "sadness",
    "upset": "upset",
}

# Some EmotionScope names that do not appear in llm-psych (informational)
_KNOWN_EMOTIONSCOPE_ONLY = {
    "frustration",
    "guilt",
    "gratitude",
    "curiosity",
    "disgust",
    "surprise",
    "anticipation",
    "trust",
    "envy",
    "pride",
    "shame",
    "hope",
    "love",
    "hate",
    "relief",
    "boredom",
    "excitement",
    "loneliness",
    "contentment",
    "embarrassment",
    "jealousy",
    "worry",
    "annoyance",
    "disappointment",
    "confusion",
}


def _parse_emotionscope_labels(arg: str) -> list[str]:
    """Return a list of emotion labels from a JSON file or a comma-separated string."""
    p = Path(arg)
    if p.exists():
        with p.open() as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return [str(x).lower().strip() for x in data]
        if isinstance(data, dict):
            # Assume keys are emotion names
            return [str(k).lower().strip() for k in data.keys()]
        raise ValueError(f"Unexpected JSON type in {arg}: {type(data)}")
    # Treat as comma-separated list
    return [x.strip().lower() for x in arg.split(",") if x.strip()]


def _load_llm_psych_vectors(vectors_dir: str | Path) -> dict[str, np.ndarray]:
    """Load vectors from ``steering_vectors/<model>/<emotion>_layer<L>.pt`` files."""
    vectors_dir = Path(vectors_dir)
    if not vectors_dir.is_dir():
        raise FileNotFoundError(f"llm-psych vectors directory not found: {vectors_dir}")

    vectors: dict[str, np.ndarray] = {}
    for pt_file in sorted(vectors_dir.glob("*_layer*.pt")):
        stem = pt_file.stem  # e.g. "anger_layer18"
        emotion = stem.split("_layer")[0]
        vec = torch.load(pt_file, map_location="cpu", weights_only=True)
        if isinstance(vec, torch.Tensor):
            vec = vec.cpu().numpy()
        elif isinstance(vec, np.ndarray):
            pass
        else:
            log.warning("Skipping %s — unexpected type %s", pt_file.name, type(vec))
            continue
        vectors[emotion] = vec.squeeze().astype(np.float32)
        log.debug("Loaded llm-psych %s: shape=%s", emotion, vectors[emotion].shape)

    if not vectors:
        raise RuntimeError(f"No *_layer*.pt files found in {vectors_dir}")
    return vectors


def _load_emotionscope_vectors(
    pt_path: str | Path,
    labels: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Load EmotionScope vectors, returning {emotion_name: vector}."""
    pt_path = Path(pt_path)
    if not pt_path.exists():
        raise FileNotFoundError(f"EmotionScope vectors file not found: {pt_path}")

    raw = torch.load(pt_path, map_location="cpu", weights_only=True)

    # Case 1: dict mapping emotion -> tensor
    if isinstance(raw, dict):
        return {
            k.lower().strip(): v.cpu().numpy().squeeze().astype(np.float32)
            for k, v in raw.items()
            if isinstance(v, (torch.Tensor, np.ndarray))
        }

    # Case 2: plain tensor -> need labels
    if isinstance(raw, torch.Tensor):
        tensor = raw.cpu().numpy().astype(np.float32)
        if tensor.ndim == 1:
            tensor = tensor.reshape(1, -1)
        if labels is None:
            raise ValueError(
                "EmotionScope vectors file is a plain tensor. "
                "Pass --emotionscope-emotions with the emotion label list."
            )
        if len(labels) != tensor.shape[0]:
            raise ValueError(
                f"EmotionScope tensor has {tensor.shape[0]} rows but "
                f"{len(labels)} labels were provided."
            )
        return {lbl: tensor[i].squeeze() for i, lbl in enumerate(labels)}

    # Case 3: custom object with .vectors or .emotion_vectors
    for attr in ("vectors", "emotion_vectors", "data"):
        if hasattr(raw, attr):
            data = getattr(raw, attr)
            if isinstance(data, dict):
                return {
                    k.lower().strip(): v.cpu().numpy().squeeze().astype(np.float32)
                    for k, v in data.items()
                    if isinstance(v, (torch.Tensor, np.ndarray))
                }
            if isinstance(data, torch.Tensor):
                tensor = data.cpu().numpy().astype(np.float32)
                if labels is None:
                    raise ValueError(
                        f"EmotionScope object.{attr} is a tensor; need --emotionscope-emotions."
                    )
                if len(labels) != tensor.shape[0]:
                    raise ValueError(
                        f"Object tensor has {tensor.shape[0]} rows but {len(labels)} labels."
                    )
                return {lbl: tensor[i].squeeze() for i, lbl in enumerate(labels)}

    raise TypeError(
        f"Could not interpret EmotionScope .pt content (type {type(raw)}). "
        "Expected dict, Tensor, or object with .vectors/.emotion_vectors."
    )


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalise a 1-D vector."""
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v
    return v / norm


def _compute_similarity_table(
    llm_vectors: dict[str, np.ndarray],
    es_vectors: dict[str, np.ndarray],
) -> list[dict]:
    """Return rows of the comparison table."""
    rows = []
    for lp_emo, lp_vec in sorted(llm_vectors.items()):
        es_emo = _LLM_PSYCH_TO_EMOTIONSCOPE.get(lp_emo, lp_emo)
        if es_emo not in es_vectors:
            rows.append(
                {
                    "llm_psych_emotion": lp_emo,
                    "emotionscope_emotion": None,
                    "cosine_similarity": None,
                    "status": "unmapped",
                }
            )
            continue

        es_vec = es_vectors[es_emo]
        if lp_vec.shape != es_vec.shape:
            rows.append(
                {
                    "llm_psych_emotion": lp_emo,
                    "emotionscope_emotion": es_emo,
                    "cosine_similarity": None,
                    "status": f"shape_mismatch ({lp_vec.shape} vs {es_vec.shape})",
                }
            )
            continue

        sim = float(np.dot(_l2_normalize(lp_vec), _l2_normalize(es_vec)))
        rows.append(
            {
                "llm_psych_emotion": lp_emo,
                "emotionscope_emotion": es_emo,
                "cosine_similarity": sim,
                "status": "ok",
            }
        )
    return rows


def _print_table(rows: list[dict]) -> None:
    """Print a human-readable similarity table."""
    header = f"{'llm-psych emotion':<18} {'EmotionScope':<18} {'cos_sim':>10}  {'status'}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    matched_sims: list[float] = []
    for r in rows:
        emo = r["llm_psych_emotion"]
        es = r["emotionscope_emotion"] or "—"
        sim = r["cosine_similarity"]
        status = r["status"]
        if sim is not None:
            matched_sims.append(sim)
            print(f"{emo:<18} {es:<18} {sim:>+10.4f}  {status}")
        else:
            print(f"{emo:<18} {es:<18} {'—':>10}  {status}")

    print("-" * len(header))
    if matched_sims:
        print(f"Matched emotions : {len(matched_sims)}")
        print(f"Mean similarity  : {np.mean(matched_sims):+.4f}")
        print(f"Median similarity: {np.median(matched_sims):+.4f}")
        print(f"Std similarity   : {np.std(matched_sims):.4f}")
    else:
        print("No matched emotions found.")
    print("=" * len(header))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare llm-psych steering vectors with EmotionScope vectors."
    )
    parser.add_argument(
        "--emotionscope-vectors",
        required=True,
        help="Path to EmotionScope .pt file (e.g. results/vectors/gemma-2-2b-it.pt).",
    )
    parser.add_argument(
        "--emotionscope-emotions",
        default=None,
        help="Comma-separated emotion names, or path to JSON list/dict. Required if .pt is a plain tensor.",
    )
    parser.add_argument(
        "--llm-psych-vectors-dir",
        required=True,
        help="Directory containing llm-psych steering vectors (e.g. steering_vectors/Llama-3.1-8B-Instruct).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON file to write the similarity table.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # ------------------------------------------------------------------
    # Load vectors
    # ------------------------------------------------------------------
    es_labels = None
    if args.emotionscope_emotions:
        es_labels = _parse_emotionscope_labels(args.emotionscope_emotions)

    log.info("Loading EmotionScope vectors from %s", args.emotionscope_vectors)
    es_vectors = _load_emotionscope_vectors(args.emotionscope_vectors, labels=es_labels)
    log.info("Loaded %d EmotionScope emotions: %s", len(es_vectors), sorted(es_vectors))

    log.info("Loading llm-psych vectors from %s", args.llm_psych_vectors_dir)
    lp_vectors = _load_llm_psych_vectors(args.llm_psych_vectors_dir)
    log.info("Loaded %d llm-psych emotions: %s", len(lp_vectors), sorted(lp_vectors))

    # ------------------------------------------------------------------
    # Compute & print
    # ------------------------------------------------------------------
    rows = _compute_similarity_table(lp_vectors, es_vectors)
    _print_table(rows)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fh:
            json.dump(rows, fh, indent=2)
        log.info("Wrote table to %s", out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Extract per-layer emotion vectors from story activations (Anthropic method).

See plans/story_emotion_vectors.md for the full specification.

Algorithm per transformer layer l (0-indexed, 0..33 for Gemma 3 4B):
  1. Per story: skip prompt tokens + STORY_SKIP_TOKENS, mean remaining story
     tokens → story_vec [hidden_dim].
  2. Per emotion: mean over all 20 stories (4 topics × 5) → mean_emotion.
  3. Global mean centering: subtract mean of the 4 emotion means (neutral
     excluded from centering).
  4. PCA neutralization: fit PCA on all neutral story tokens at this layer,
     keep top-k components explaining ≥ PCA_VARIANCE_THRESHOLD of variance,
     project those components out of each centered emotion vector.
  5. Save one .pt per emotion (dict {layer_idx: tensor}) + metadata JSON.

Usage (run on cloud after .pt activation files are in HF):
    python scripts/extract_story_emotion_vectors.py \
        --hf-activations-prefix activations/gemma3-4b-story \
        --hf-output-prefix steering_vectors/gemma3-4b-story

Local stories/ meta JSONs must be present (committed to repo).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download
from sklearn.decomposition import PCA
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMOTIONS = ["joy", "sadness", "admiration", "loathing"]
NEUTRAL = "neutral"
STORY_SKIP_TOKENS = 50
PCA_VARIANCE_THRESHOLD = 0.50
MODEL_ID = "google/gemma-3-4b-it"
N_TRANSFORMER_LAYERS = 34  # hidden_states indices 1..34
DEFAULT_REPO_ID = "llm-psych/llm-psych-activations"

TOPIC_KEYS = [
    "a_grain_of_sand",
    "a_red_cone_in_a_beige_room",
    "making_a_pot_of_coffee",
    "twilight_in_a_small_town",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_env_token() -> str:
    """Resolve HF_TOKEN from .env or environment."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN=") and not line.startswith("#"):
                val = line[len("HF_TOKEN="):].strip()
                if val:
                    os.environ.setdefault("HF_TOKEN", val)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        sys.exit(
            "ERROR: HF_TOKEN not found. Set it in .env or the environment."
        )
    return token


def get_meta_entries(
    stories_dir: Path, topic_key: str
) -> list[dict]:
    """Load and return story metadata for a topic."""
    path = stories_dir / f"{topic_key}_meta.json"
    if not path.exists():
        sys.exit(f"ERROR: meta file not found: {path}")
    with path.open() as f:
        return json.load(f)


def compute_prompt_len(
    tokenizer: AutoTokenizer, prompt: str
) -> int:
    """Tokenize the prompt (with chat template + 'Story:') → token count.

    This is the exact boundary between prompt tokens and story tokens in
    the saved full-sequence activation tensor.
    """
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "Story:"},
    ]
    ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, return_tensors="pt"
    )
    return int(ids.shape[1])


def project_out_components(
    vector: np.ndarray, components: np.ndarray
) -> np.ndarray:
    """Remove each principal component from vector.

    Parameters
    ----------
    vector
        Shape (hidden_dim,).
    components
        Shape (k, hidden_dim), unit-norm rows.

    Returns
    -------
    np.ndarray
        Shape (hidden_dim,), float32.
    """
    v = vector.copy().astype(np.float64)
    for pc in components:
        v -= np.dot(v, pc) * pc
    return v.astype(np.float32)


# ---------------------------------------------------------------------------
# Core accumulation
# ---------------------------------------------------------------------------

def accumulate_topic(
    geom: dict[str, list[torch.Tensor]],
    meta_entries: list[dict],
    tokenizer: AutoTokenizer,
    story_means: dict[str, list[np.ndarray]],
    neutral_token_acts: dict[int, list[np.ndarray]],
) -> None:
    """Process one topic's geometry tensor.

    Accumulates:
    - story_means[emotion]: list of [N_LAYERS, hidden_dim] arrays (per story)
    - neutral_token_acts[layer]: list of [n_tokens, hidden_dim] arrays

    Parameters
    ----------
    geom
        Loaded .pt dict: {emotion: [tensor_0, ..., tensor_4]}.
        Each tensor is [hidden_states_count, seq_len, hidden_dim].
    meta_entries
        Loaded meta JSON for this topic.
    tokenizer
        Gemma 3 4B tokenizer (for prompt length computation).
    story_means
        Running accumulator — modified in-place.
    neutral_token_acts
        Running accumulator for neutral tokens — modified in-place.
    """
    # Build a quick lookup: (emotion, story_idx) → meta entry
    meta_lookup: dict[tuple[str, int], dict] = {
        (e["emotion"], e["story_idx"]): e for e in meta_entries
    }

    all_emotions = EMOTIONS + [NEUTRAL]
    for emo in all_emotions:
        tensors = geom.get(emo)
        if tensors is None:
            sys.exit(f"ERROR: emotion '{emo}' not found in geometry file.")
        for story_idx, tensor in enumerate(tensors):
            entry = meta_lookup.get((emo, story_idx))
            if entry is None:
                sys.exit(
                    f"ERROR: no meta entry for ({emo}, story_idx={story_idx})."
                )

            prompt_len = compute_prompt_len(tokenizer, entry["prompt"])
            story_start = prompt_len + STORY_SKIP_TOKENS

            # tensor shape: [hidden_states_count, seq_len, hidden_dim]
            # Transformer layer l (0-indexed) → hidden_states index l+1
            seq_len = tensor.shape[1]
            if story_start >= seq_len:
                print(
                    f"  WARNING: story_start={story_start} >= seq_len={seq_len} "
                    f"for ({emo}, story_idx={story_idx}). Skipping story."
                )
                continue

            # Extract story portion: shape [N_TRANSFORMER_LAYERS, story_tokens, hidden_dim]
            # hidden_states indices 1..N_TRANSFORMER_LAYERS (skip embedding at index 0)
            story_slice = tensor[1:, story_start:, :].float().numpy()
            # shape: [N_TRANSFORMER_LAYERS, story_tokens, hidden_dim]

            if story_slice.shape[1] == 0:
                print(
                    f"  WARNING: zero story tokens for ({emo}, story_idx={story_idx}). "
                    "Skipping."
                )
                continue

            # Per-story mean over tokens → [N_TRANSFORMER_LAYERS, hidden_dim]
            story_mean = story_slice.mean(axis=1)  # [N_TRANSFORMER_LAYERS, hidden_dim]

            if emo != NEUTRAL:
                story_means[emo].append(story_mean)
            else:
                story_means[NEUTRAL].append(story_mean)
                # Also accumulate per-layer token activations for PCA
                for l in range(N_TRANSFORMER_LAYERS):
                    neutral_token_acts[l].append(story_slice[l])  # [story_tokens, hidden_dim]


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_emotion_vectors(
    story_means: dict[str, list[np.ndarray]],
    neutral_token_acts: dict[int, list[np.ndarray]],
) -> tuple[dict[str, dict[int, np.ndarray]], dict[int, dict]]:
    """Compute final per-layer emotion vectors.

    Steps: mean → center → PCA neutralize.

    Returns
    -------
    vectors
        {emotion: {layer_idx: np.ndarray[hidden_dim]}} for EMOTIONS.
    pca_meta
        {layer_idx: {"n_components": int, "explained_variance": float}}
    """
    # Step 2: per-emotion mean across stories → [N_LAYERS, hidden_dim]
    emo_means: dict[str, np.ndarray] = {}
    for emo in EMOTIONS:
        stories = story_means[emo]
        if len(stories) == 0:
            sys.exit(f"ERROR: no story means accumulated for emotion '{emo}'.")
        emo_means[emo] = np.stack(stories, axis=0).mean(axis=0)  # [N_LAYERS, hidden_dim]

    # Step 3: global mean centering (4 emotions only, no neutral)
    global_mean = np.stack(
        [emo_means[e] for e in EMOTIONS], axis=0
    ).mean(axis=0)  # [N_LAYERS, hidden_dim]

    centered: dict[str, np.ndarray] = {
        e: emo_means[e] - global_mean for e in EMOTIONS
    }

    # Step 4: PCA neutralization, per layer
    vectors: dict[str, dict[int, np.ndarray]] = {e: {} for e in EMOTIONS}
    pca_meta: dict[int, dict] = {}

    # Upper bound on components needed for 50% variance. Residual-stream
    # activations in 7-8B models typically reach 50% variance well within
    # 300 components; 500 is a safe ceiling that still uses randomized SVD
    # (sklearn switches to full SVD only when n_components >= min(n, p)).
    PCA_MAX_COMPONENTS = 500

    for l in range(N_TRANSFORMER_LAYERS):
        neutral_acts_l = np.vstack(neutral_token_acts[l]).astype(np.float64)
        # neutral_acts_l: [total_neutral_tokens, hidden_dim]

        max_components = min(neutral_acts_l.shape[0], neutral_acts_l.shape[1], PCA_MAX_COMPONENTS)
        pca = PCA(n_components=max_components)
        pca.fit(neutral_acts_l)

        cumvar = np.cumsum(pca.explained_variance_ratio_)
        if cumvar[-1] < PCA_VARIANCE_THRESHOLD:
            # Rare: 500 components not enough; warn and use all available
            print(
                f"  WARNING layer {l}: {max_components} components explain only "
                f"{cumvar[-1]:.1%} variance (< {PCA_VARIANCE_THRESHOLD:.0%}). "
                "Using all available components."
            )
        n_keep = int(np.searchsorted(cumvar, PCA_VARIANCE_THRESHOLD) + 1)
        n_keep = min(n_keep, max_components)
        top_components = pca.components_[:n_keep]  # [n_keep, hidden_dim]

        explained = float(cumvar[n_keep - 1])
        pca_meta[l] = {
            "n_components": n_keep,
            "explained_variance": round(explained, 4),
        }

        for emo in EMOTIONS:
            v = centered[emo][l].astype(np.float64)
            v_projected = project_out_components(v, top_components)
            vectors[emo][l] = v_projected

        if (l + 1) % 10 == 0 or l == N_TRANSFORMER_LAYERS - 1:
            print(
                f"  Layer {l:2d}: neutral PCA kept {n_keep} components "
                f"({explained:.1%} variance)"
            )

    return vectors, pca_meta


# ---------------------------------------------------------------------------
# Save and upload
# ---------------------------------------------------------------------------

def save_and_upload(
    vectors: dict[str, dict[int, np.ndarray]],
    pca_meta: dict[int, dict],
    story_means: dict[str, list[np.ndarray]],
    hf_output_prefix: str,
    repo_id: str,
    token: str,
) -> None:
    """Write per-emotion .pt files + metadata, then upload to HF dataset."""
    with tempfile.TemporaryDirectory(prefix="llm-psych-vectors-") as tmpdir:
        out_dir = Path(tmpdir)

        # Save per-emotion vector dicts
        for emo in EMOTIONS:
            layer_dict = {
                l: torch.tensor(vectors[emo][l], dtype=torch.float32)
                for l in range(N_TRANSFORMER_LAYERS)
            }
            torch.save(layer_dict, out_dir / f"{emo}_all_layers.pt")
            print(f"  Saved {emo}_all_layers.pt ({N_TRANSFORMER_LAYERS} layers)")

        # Save extraction metadata
        n_stories_per_emotion = {e: len(story_means[e]) for e in EMOTIONS}
        meta = {
            "model_id": MODEL_ID,
            "n_transformer_layers": N_TRANSFORMER_LAYERS,
            "emotions": EMOTIONS,
            "n_topics": len(TOPIC_KEYS),
            "topics": TOPIC_KEYS,
            "n_stories_per_emotion": n_stories_per_emotion,
            "story_skip_tokens": STORY_SKIP_TOKENS,
            "pca_variance_threshold": PCA_VARIANCE_THRESHOLD,
            "pca_per_layer": pca_meta,
            "date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        meta_path = out_dir / "extraction_meta.json"
        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2)

        # Upload to HF
        api = HfApi(token=token)
        print(f"\nUploading to {repo_id}/{hf_output_prefix} ...")
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(out_dir),
            path_in_repo=hf_output_prefix,
            commit_message=f"Add Gemma 3 4B story emotion vectors ({datetime.utcnow().strftime('%Y-%m-%d')})",
            ignore_patterns=[".DS_Store", "*.tmp"],
        )
        print("Upload complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stories-dir",
        type=Path,
        default=Path(__file__).parent.parent / "stories",
        help="Local directory containing <topic>_meta.json files.",
    )
    p.add_argument(
        "--hf-activations-prefix",
        default="",
        help="Path prefix inside the HF dataset where .pt activation files live "
             "(e.g. 'activations/gemma3-4b-story'). Leave empty for root level.",
    )
    p.add_argument(
        "--hf-output-prefix",
        default="steering_vectors/gemma3-4b-story",
        help="Path prefix inside the HF dataset for output vectors.",
    )
    p.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="HF dataset repo id.",
    )
    p.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Local directory for downloading .pt files. "
             "Defaults to a system temp dir.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    token = load_env_token()

    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Accumulators
    story_means: dict[str, list[np.ndarray]] = {
        e: [] for e in EMOTIONS + [NEUTRAL]
    }
    neutral_token_acts: dict[int, list[np.ndarray]] = {
        l: [] for l in range(N_TRANSFORMER_LAYERS)
    }

    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="llm-psych-acts-"))
    work_dir.mkdir(parents=True, exist_ok=True)

    # --- Process each topic ---
    for topic_key in TOPIC_KEYS:
        print(f"\nProcessing topic: {topic_key}")

        # Download .pt activation file from HF
        prefix = args.hf_activations_prefix.strip("/")
        filename = f"{topic_key}_geometry.pt"
        hf_file_path = f"{prefix}/{filename}" if prefix else filename
        print(f"  Downloading {hf_file_path} from {args.repo_id} ...")
        local_pt = hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            filename=hf_file_path,
            local_dir=str(work_dir),
            token=token,
        )
        print(f"  Downloaded to {local_pt}")

        # Load geometry
        geom: dict[str, list[torch.Tensor]] = torch.load(
            local_pt, map_location="cpu", weights_only=True
        )

        # Load meta JSON (local)
        meta_entries = get_meta_entries(args.stories_dir, topic_key)

        accumulate_topic(
            geom=geom,
            meta_entries=meta_entries,
            tokenizer=tokenizer,
            story_means=story_means,
            neutral_token_acts=neutral_token_acts,
        )

        del geom
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Verify story counts ---
    for emo in EMOTIONS:
        n = len(story_means[emo])
        expected = len(TOPIC_KEYS) * 5
        if n != expected:
            print(
                f"WARNING: expected {expected} stories for {emo}, got {n}."
            )
    print(f"\nStory counts: { {e: len(story_means[e]) for e in EMOTIONS} }")

    # --- Extract vectors ---
    print("\nExtracting emotion vectors (centering + PCA neutralization) ...")
    vectors, pca_meta = extract_emotion_vectors(story_means, neutral_token_acts)

    # --- Save and upload ---
    print("\nSaving and uploading ...")
    save_and_upload(
        vectors=vectors,
        pca_meta=pca_meta,
        story_means=story_means,
        hf_output_prefix=args.hf_output_prefix,
        repo_id=args.repo_id,
        token=token,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()

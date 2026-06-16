#!/usr/bin/env python3
"""Measure emotion vector activations on implicit-emotion and intensity-template prompts.

See plans/prompt_activations.md for full specification.

Two experiments:

  Part A — Implicit emotion prompts
      Run ~20 diverse prompts through the model. Measure cosine similarity between
      the residual-stream activation at the last prompt token (layer 22) and each
      emotion vector. Checks that vectors activate appropriately on real scenarios
      (e.g. "dog passed away" → high sadness score).

  Part B — Intensity templates
      Same measurement across parameterised templates where a numerical value X
      modulates emotional intensity. Checks that vectors track semantic content
      rather than surface form (e.g. higher tylenol dose → higher sadness/fear).

Usage (needs GPU):
    python scripts/prompt_activations.py \
        --hf-prefix steering_vectors/gemma3-4b-story

Output:
    results/prompt_activations/activation_scores.parquet
    figures/prompt_activations/heatmap_implicit.png
    figures/prompt_activations/intensity_templates.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMOTIONS = ["joy", "sadness", "admiration", "loathing"]
PRIMARY_LAYER = 22
MODEL_ID = "google/gemma-3-4b-it"
DEFAULT_REPO_ID = "llm-psych/llm-psych-activations"

EMOTION_COLORS = {
    "joy": "#F4A261",
    "admiration": "#2A9D8F",
    "sadness": "#457B9D",
    "loathing": "#E76F51",
}

# ---------------------------------------------------------------------------
# Prompt set — Part A
# ---------------------------------------------------------------------------

IMPLICIT_PROMPTS: list[dict] = [
    # Positive events
    {"id": "p01", "category": "positive_event", "expected": "joy",
     "text": "My daughter just took her first steps today! What are some ways to capture more of these precious moments?"},
    {"id": "p02", "category": "positive_event", "expected": "joy",
     "text": "I just found out I got into my dream PhD programme. I've been working towards this for years."},
    {"id": "p03", "category": "positive_event", "expected": "admiration",
     "text": "My colleague stayed late every night for a month to help the whole team hit our deadline. I'd like to thank them properly."},
    {"id": "p04", "category": "positive_event", "expected": "admiration",
     "text": "I just read that a surgeon performed a 20-hour operation to separate conjoined twins successfully. How do people develop that kind of skill?"},
    # Loss / threat
    {"id": "p05", "category": "loss", "expected": "sadness",
     "text": "My dog passed away this morning after fourteen years together. I don't know what to do with all his things."},
    {"id": "p06", "category": "loss", "expected": "sadness",
     "text": "My mum was just diagnosed with early-stage dementia. She doesn't fully understand what it means yet."},
    {"id": "p07", "category": "threat", "expected": "sadness",
     "text": "I think I'm being let go at the end of the month. I have two kids and a mortgage. What should I do first?"},
    {"id": "p08", "category": "threat", "expected": "loathing",
     "text": "Someone left rotting food in the shared fridge at work for three weeks. The smell is unbearable and nobody will claim it."},
    # Betrayal / violation
    {"id": "p09", "category": "betrayal", "expected": "loathing",
     "text": "My best friend has been lying to me about where she works for two years. I found out by accident. I feel sick."},
    {"id": "p10", "category": "betrayal", "expected": "loathing",
     "text": "I just found out my business partner has been skimming money from our accounts. We've worked together for six years."},
    # Milestone moments
    {"id": "p11", "category": "milestone", "expected": "joy",
     "text": "I finished paying off my student loans today after ten years. It doesn't feel real yet."},
    {"id": "p12", "category": "milestone", "expected": "joy",
     "text": "My parents are celebrating their 40th wedding anniversary next week. I want to do something special."},
    # Neutral / mundane
    {"id": "p13", "category": "neutral", "expected": "none",
     "text": "What is the best way to organise a bookshelf — by author, genre, or colour?"},
    {"id": "p14", "category": "neutral", "expected": "none",
     "text": "How long does it take to hard-boil an egg?"},
    {"id": "p15", "category": "neutral", "expected": "none",
     "text": "What are the main differences between Python lists and tuples?"},
    # Medical / concern
    {"id": "p16", "category": "medical", "expected": "sadness",
     "text": "My test results came back and the doctor wants to discuss them in person. My appointment is in three days."},
    {"id": "p17", "category": "medical", "expected": "sadness",
     "text": "I've had a persistent headache for five days now. Painkillers aren't helping much."},
]

# ---------------------------------------------------------------------------
# Intensity templates — Part B
# ---------------------------------------------------------------------------

INTENSITY_TEMPLATES: list[dict] = [
    {
        "name": "tylenol_mg",
        "template": "I just took {X} mg of tylenol for my back pain.",
        "x_values": [200, 500, 1000, 2000, 4000],
        "x_label": "Tylenol dose (mg)",
        "direction": "increasing danger",
    },
    {
        "name": "sister_age",
        "template": "My sister lived until the age of {X}.",
        "x_values": [90, 70, 50, 30, 10, 5, 1],
        "x_label": "Sister's age at death",
        "direction": "decreasing (younger → more tragic)",
    },
    {
        "name": "dog_missing",
        "template": "My dog is missing for {X} days now.",
        "x_values": [1, 3, 7, 14, 30, 90],
        "x_label": "Days missing",
        "direction": "increasing duration",
    },
]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_env_token() -> str | None:
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN=") and not line.startswith("#"):
                val = line[len("HF_TOKEN="):].strip()
                if val:
                    os.environ.setdefault("HF_TOKEN", val)
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def load_emotion_vectors(
    hf_prefix: str, repo_id: str, token: str | None, work_dir: Path
) -> dict[str, np.ndarray]:
    """Load emotion vectors at PRIMARY_LAYER only."""
    vectors: dict[str, np.ndarray] = {}
    for emo in EMOTIONS:
        prefix = hf_prefix.strip("/")
        filename = f"{emo}_all_layers.pt"
        hf_path = f"{prefix}/{filename}" if prefix else filename
        local = hf_hub_download(
            repo_id=repo_id, repo_type="dataset",
            filename=hf_path, local_dir=str(work_dir), token=token,
        )
        layer_dict = torch.load(local, map_location="cpu", weights_only=True)
        vectors[emo] = layer_dict[PRIMARY_LAYER].float().numpy()
    return vectors


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def get_last_token_activation(
    model,
    tokenizer,
    prompt: str,
    layer: int,
) -> np.ndarray:
    """Run prompt through model, return hidden state at last token, given layer.

    Parameters
    ----------
    model
        Loaded Gemma 3 4B model.
    tokenizer
        Corresponding tokenizer.
    prompt
        Raw user message text.
    layer
        Transformer layer index (0-indexed). Corresponds to hidden_states[layer+1].

    Returns
    -------
    np.ndarray of shape (hidden_dim,)
    """
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    # Newer transformers returns BatchEncoding; older returns a raw tensor.
    input_ids = (encoded["input_ids"] if hasattr(encoded, "keys") else encoded).to(model.device)

    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)

    # hidden_states: tuple of [batch, seq_len, hidden_dim], length = n_layers + 1
    # hidden_states[0] = embedding, hidden_states[layer+1] = after transformer block `layer`
    hs = out.hidden_states[layer + 1]  # [1, seq_len, hidden_dim]
    return hs[0, -1, :].detach().float().cpu().numpy()  # last token


def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))


# ---------------------------------------------------------------------------
# Part A — implicit prompts
# ---------------------------------------------------------------------------

def run_implicit_prompts(
    model, tokenizer, emotion_vectors: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows = []
    for i, entry in enumerate(IMPLICIT_PROMPTS):
        print(f"  [{i+1}/{len(IMPLICIT_PROMPTS)}] {entry['id']} — {entry['category']}")
        h = get_last_token_activation(model, tokenizer, entry["text"], PRIMARY_LAYER)
        for emo in EMOTIONS:
            rows.append({
                "prompt_id": entry["id"],
                "prompt_text": entry["text"][:60] + "…",
                "category": entry["category"],
                "expected_emotion": entry["expected"],
                "emotion": emo,
                "cos_sim": cosine_sim(h, emotion_vectors[emo]),
            })
    return pd.DataFrame(rows)


def plot_implicit_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    pivot = df.pivot_table(index="prompt_id", columns="emotion", values="cos_sim")
    # Reorder columns and add readable row labels
    pivot = pivot[EMOTIONS]
    labels = {e["id"]: f"{e['id']}: {e['text'][:50]}…" for e in IMPLICIT_PROMPTS}
    pivot.index = [labels[i] for i in pivot.index]

    fig, ax = plt.subplots(figsize=(8, len(IMPLICIT_PROMPTS) * 0.42 + 1.5))
    sns.heatmap(
        pivot, ax=ax, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
        linewidths=0.4, cbar_kws={"shrink": 0.6},
    )
    ax.set_title(f"Emotion vector activations on implicit-emotion prompts\n(layer {PRIMARY_LAYER}, cosine similarity)", pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    path = out_dir / "heatmap_implicit.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Part B — intensity templates
# ---------------------------------------------------------------------------

def run_intensity_templates(
    model, tokenizer, emotion_vectors: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows = []
    for tmpl in INTENSITY_TEMPLATES:
        print(f"  Template: {tmpl['name']}")
        for x in tmpl["x_values"]:
            prompt = tmpl["template"].format(X=x)
            h = get_last_token_activation(model, tokenizer, prompt, PRIMARY_LAYER)
            for emo in EMOTIONS:
                rows.append({
                    "template_name": tmpl["name"],
                    "x_value": x,
                    "prompt": prompt,
                    "emotion": emo,
                    "cos_sim": cosine_sim(h, emotion_vectors[emo]),
                })
    return pd.DataFrame(rows)


def plot_intensity_templates(df: pd.DataFrame, out_dir: Path) -> None:
    n = len(INTENSITY_TEMPLATES)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, tmpl in zip(axes, INTENSITY_TEMPLATES):
        sub = df[df["template_name"] == tmpl["name"]]
        for emo in EMOTIONS:
            emo_sub = sub[sub["emotion"] == emo].sort_values("x_value")
            ax.plot(
                emo_sub["x_value"], emo_sub["cos_sim"],
                label=emo, color=EMOTION_COLORS[emo],
                marker="o", linewidth=2, markersize=5,
            )
        ax.set_xlabel(tmpl["x_label"], fontsize=9)
        ax.set_ylabel("Cosine similarity", fontsize=9)
        template_display = tmpl["template"].replace("{X}", "X")
        ax.set_title(f'"{template_display}"\n({tmpl["direction"]})', fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
        ax.legend(frameon=False, fontsize=8)
        sns.despine(ax=ax)

    fig.suptitle(
        f"Emotion vector activations across intensity templates — layer {PRIMARY_LAYER}",
        fontsize=11,
    )
    fig.tight_layout()
    path = out_dir / "intensity_templates.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hf-prefix", default="steering_vectors/gemma3-4b-story")
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--work-dir", type=Path,
                   default=Path(__file__).parent.parent / "steering_vectors" / "gemma3-4b-story")
    p.add_argument("--figures-dir", type=Path,
                   default=Path(__file__).parent.parent / "figures" / "prompt_activations")
    p.add_argument("--results-dir", type=Path,
                   default=Path(__file__).parent.parent / "results" / "prompt_activations")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    token = load_env_token()
    if not token:
        sys.exit("ERROR: HF_TOKEN not found in .env or environment.")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading emotion vectors ...")
    emotion_vectors = load_emotion_vectors(args.hf_prefix, args.repo_id, token, args.work_dir)

    print(f"\nLoading {MODEL_ID} (4-bit) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map="auto",
    )
    model.eval()

    # --- Part A ---
    print("\nPart A — implicit emotion prompts ...")
    implicit_df = run_implicit_prompts(model, tokenizer, emotion_vectors)
    plot_implicit_heatmap(implicit_df, args.figures_dir)

    # --- Part B ---
    print("\nPart B — intensity templates ...")
    intensity_df = run_intensity_templates(model, tokenizer, emotion_vectors)
    plot_intensity_templates(intensity_df, args.figures_dir)

    # Save results
    all_df = pd.concat([
        implicit_df.assign(experiment="implicit"),
        intensity_df.rename(columns={"template_name": "prompt_id", "x_value": "x_value"}).assign(experiment="intensity"),
    ], ignore_index=True)
    all_df.to_parquet(args.results_dir / "activation_scores.parquet", index=False)

    meta = {
        "model_id": MODEL_ID,
        "primary_layer": PRIMARY_LAYER,
        "n_implicit_prompts": len(IMPLICIT_PROMPTS),
        "n_intensity_templates": len(INTENSITY_TEMPLATES),
        "date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with (args.results_dir / "prompt_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()

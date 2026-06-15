"""Logit-lens validation of emotion vectors (C2; Sofroniew et al. 2026 §2).

Maps each story-derived emotion vector into vocabulary space through the
model's final norm + unembedding and reports the top ``+k`` (and bottom
``-k``) tokens. If the vector encodes an emotion *concept*, those tokens
are emotion-congruent (e.g. loathing -> disgust/contempt-laden tokens;
joy -> delight/celebration tokens). This is **stimulus-free and
surface-independent** — it reads what the direction points at in output
space, so it cannot be a stimulus-lexis artifact, which is exactly why it
is load-bearing after the gate found within-corpus probing
surface-saturated (RESEARCH_LOG 2026-06-14; HYPOTHESES.md 2026-06-14).

Runs on the vectors already in ``steering_vectors/<model_key>-story/`` —
no new forward passes, no GPU required for the dev fleet. For the
primaries, run on the pod or pass ``--device cuda --dtype bfloat16``.

Usage
-----
    # dev fleet on the Mac (CPU is plenty for the unembed matmul)
    uv run python scripts/validate_logit_lens.py --model qwen25_05b
    uv run python scripts/validate_logit_lens.py --model llama32_1b
    uv run python scripts/validate_logit_lens.py --model gemma2_2b

    # a primary on a pod
    uv run python scripts/validate_logit_lens.py --model gemma2_9b \
        --device cuda --dtype bfloat16

Caveat: logit lens is approximate for mid-layers (it skips the
computation between the vector's layer and the unembed); treat it as
corroborating, with the layer reported. Pull the vectors first if needed:
``uv run python scripts/sync_hf.py pull steering_vectors --model <key>-story``.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root / "src"))

from dotenv import load_dotenv  # noqa: E402

from llm_psych.models import load_model  # noqa: E402


def _read_model_cfg(model_config: str) -> dict:
    path = _repo_root / "configs" / "model" / f"{model_config}.yaml"
    if not path.exists():
        raise SystemExit(f"model config not found: {path}")
    return yaml.safe_load(path.read_text())


def _discover_vectors(story_dir: Path) -> dict[str, dict[int, Path]]:
    """emotion -> {layer: path} from <emotion>_layer<L>.npy files."""
    out: dict[str, dict[int, Path]] = defaultdict(dict)
    for f in sorted(story_dir.glob("*_layer*.npy")):
        stem = f.stem  # e.g. "admiration_layer12"
        emotion, _, layer_s = stem.rpartition("_layer")
        if emotion and layer_s.isdigit():
            out[emotion][int(layer_s)] = f
    return dict(out)


def _pick_layer(
    vectors: dict[str, dict[int, Path]], requested: int | None, n_layers: int | None = None
) -> int:
    """A layer present for every emotion.

    Default = the shared layer nearest **~2/3 model depth** (the paper's
    "mid-late" analysis layer), NOT the deepest. The sweep (2026-06-14)
    showed the deepest layers drift toward token-level read-out and lose the
    abstract-concept signal (admiration garbles at Qwen-0.5B L22; Gemma joy
    drifts to "relaxed" at L24), while ~2/3 depth is clean for all four
    emotions. Falls back to the deepest shared layer if n_layers is unknown.
    """
    shared = sorted(set.intersection(*(set(layers) for layers in vectors.values())))
    if not shared:
        raise SystemExit("no layer is shared across all emotions")
    if requested is not None:
        if requested not in shared:
            raise SystemExit(
                f"layer {requested} not present for all emotions; shared = {shared}"
            )
        return requested
    if n_layers:
        target = round(2 * n_layers / 3)
        return min(shared, key=lambda layer: abs(layer - target))
    return max(shared)


def _final_norm(model) -> torch.nn.Module | None:
    """The model's final pre-unembed norm (Llama/Qwen/Gemma2: model.model.norm)."""
    inner = getattr(model, "model", None)
    return getattr(inner, "norm", None) if inner is not None else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", dest="model_config", required=True,
                    help="Hydra model config name, e.g. qwen25_05b, gemma2_9b")
    ap.add_argument("--layer", type=int, default=None,
                    help="vector layer to read (default: deepest shared layer)")
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--device", default="cpu", help="cpu | cuda | mps")
    ap.add_argument("--dtype", default="bfloat16",
                    help="bfloat16 (default; model-faithful). float32 for a CPU-only Mac run "
                         "if bf16 is unsupported; avoid float16 (overflow on Gemma's outliers).")
    ap.add_argument("--vectors-dir", default="steering_vectors")
    ap.add_argument("--out-dir", default="results/vector_validation")
    ap.add_argument("--no-final-norm", action="store_true",
                    help="unembed the raw vector without applying the final norm")
    ap.add_argument("--sweep", action="store_true",
                    help="report top-k per emotion across ALL shared layers "
                         "(diagnose layer dependence, e.g. admiration on 0.5B)")
    args = ap.parse_args()

    load_dotenv(_repo_root / ".env")  # HF_TOKEN for gated models (e.g. Gemma)
    cfg = _read_model_cfg(args.model_config)
    hf_model_id = cfg["hf_model_id"]
    model_key = hf_model_id.split("/")[-1]
    story_dir = _repo_root / args.vectors_dir / f"{model_key}-story"
    if not story_dir.exists():
        raise SystemExit(
            f"no story vectors at {story_dir}. Pull them first:\n"
            f"  uv run python scripts/sync_hf.py pull steering_vectors "
            f"--model {model_key}-story"
        )

    vectors = _discover_vectors(story_dir)
    if not vectors:
        raise SystemExit(f"no <emotion>_layer<L>.npy files in {story_dir}")
    shared = sorted(set.intersection(*(set(layers) for layers in vectors.values())))
    if not shared:
        raise SystemExit("no layer is shared across all emotions")
    if args.sweep:
        layers_to_run = shared
    else:
        layers_to_run = [_pick_layer(vectors, args.layer, n_layers=cfg.get("n_layers"))]

    dtype = getattr(torch, args.dtype)
    lm = load_model(
        hf_model_id,
        revision=cfg.get("hf_revision"),
        torch_dtype=dtype,
        device_map=args.device,
    )
    model, tok = lm.model, lm.tokenizer
    model.eval()

    W = model.get_output_embeddings().weight  # [vocab, hidden]
    norm = None if args.no_final_norm else _final_norm(model)
    norm_note = "final-norm + unembed" if norm is not None else "raw unembed (no final norm)"

    def top_bottom(path: Path) -> tuple[list[str], list[str]]:
        vt = torch.tensor(np.load(path), dtype=W.dtype, device=W.device)
        h = norm(vt) if norm is not None else vt
        logits = h @ W.T
        dec = lambda idx: [repr(tok.decode([int(i)])) for i in idx]  # noqa: E731
        return (dec(torch.topk(logits, args.top_k).indices),
                dec(torch.topk(-logits, args.top_k).indices))

    layer_desc = (f"sweep over shared layers {shared}" if args.sweep
                  else f"**{layers_to_run[0]}** (~2/3 depth, paper convention; unless --layer given)")
    lines = [
        f"# Logit-lens validation — {model_key}" + (" (layer sweep)" if args.sweep else ""),
        "",
        f"- Vectors: `{story_dir.relative_to(_repo_root)}` (story method)",
        f"- Layer: {layer_desc}",
        f"- Read-out: {norm_note}; top ±{args.top_k} tokens",
        "",
        "Emotion-congruent top tokens (and opposite-valence bottom tokens) are "
        "evidence the vector is a concept, not stimulus lexis.",
        "",
    ]

    with torch.no_grad():
        for emotion in sorted(vectors):
            lines.append(f"## {emotion}")
            lines.append("")
            if args.sweep:
                for L in layers_to_run:
                    if L not in vectors[emotion]:
                        continue
                    top, _ = top_bottom(vectors[emotion][L])
                    lines.append(f"- **L{L}:** " + ", ".join(top))
                lines.append("")
                print(f"[{emotion}] swept {len(layers_to_run)} layers")
            else:
                top, bot = top_bottom(vectors[emotion][layers_to_run[0]])
                lines.append(f"- **top +{args.top_k}:** " + ", ".join(top))
                lines.append(f"- **bottom −{args.top_k}:** " + ", ".join(bot))
                lines.append("")
                print(f"[{emotion}] top: {', '.join(top[:8])}")

    out_dir = _repo_root / args.out_dir / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / ("logit_lens_sweep.md" if args.sweep else "logit_lens.md")
    report.write_text("\n".join(lines))
    print(f"\nWrote {report.relative_to(_repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

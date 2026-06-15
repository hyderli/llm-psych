"""Implicit-emotion scenario validation (C2; Sofroniew et al. 2026 §2).

Each scenario evokes one of the four emotions *without naming it*. We read
the residual stream at the **Assistant-colon** analog (the last prompt token
after the chat template's ``add_generation_prompt``), neutral-center it, and
take cosine similarity with each story-derived emotion vector. If the vector
that wins (argmax) is the scenario's intended emotion, the vector activates
in the semantically correct context — and since the scenarios are a totally
different surface form from the narrative stories, a pass is genuine
cross-context generalization (surface lexis cannot bridge it).

Runs on the dev `-story` vectors on the Mac (no GPU needed). Pull them first:
``uv run python scripts/sync_hf.py pull steering_vectors --model <key>-story``.

Usage:  uv run python scripts/validate_implicit_scenarios.py --model qwen25_05b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml

_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root / "src"))

from dotenv import load_dotenv  # noqa: E402

from llm_psych.hooks import ResidualStreamRecorder  # noqa: E402
from llm_psych.models import load_model  # noqa: E402

EMOTIONS = ["admiration", "joy", "loathing", "sadness"]
STIMULI = _repo_root / "data" / "public" / "implicit_emotion_scenarios.jsonl"


def _read_model_cfg(name: str) -> dict:
    path = _repo_root / "configs" / "model" / f"{name}.yaml"
    if not path.exists():
        raise SystemExit(f"model config not found: {path}")
    return yaml.safe_load(path.read_text())


def _check_hash(path: Path) -> None:
    reg = _repo_root / "configs" / "stimuli_hashes.yaml"
    if not reg.exists():
        return
    expected = (yaml.safe_load(reg.read_text()) or {}).get(path.name)
    if expected:
        actual = hashlib.md5(path.read_bytes()).hexdigest()
        print(f"[stimuli] {path.name} MD5 {'OK' if actual == expected else 'MISMATCH (frozen set changed!)'}")


def _discover_vectors(story_dir: Path) -> dict[str, dict[int, Path]]:
    out: dict[str, dict[int, Path]] = defaultdict(dict)
    for f in sorted(story_dir.glob("*_layer*.npy")):
        emotion, _, layer_s = f.stem.rpartition("_layer")
        if emotion and layer_s.isdigit():
            out[emotion][int(layer_s)] = f
    return dict(out)


def _pick_layer(vectors: dict[str, dict[int, Path]], requested: int | None, n_layers: int | None) -> int:
    shared = sorted(set.intersection(*(set(v) for v in vectors.values())))
    if not shared:
        raise SystemExit("no layer shared across all emotions")
    if requested is not None:
        if requested not in shared:
            raise SystemExit(f"layer {requested} not in shared {shared}")
        return requested
    if n_layers:  # ~2/3 depth (paper's mid-late analysis layer), not deepest
        target = round(2 * n_layers / 3)
        return min(shared, key=lambda L: abs(L - target))
    return max(shared)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", dest="model_config", required=True)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bfloat16",
                    help="bfloat16 (default; model-faithful, avoids fp16 overflow, half the "
                         "memory of fp32). Use float32 for a slow CPU-only Mac run if bf16 is unsupported.")
    ap.add_argument("--vectors-dir", default="steering_vectors")
    ap.add_argument("--out-dir", default="results/vector_validation")
    args = ap.parse_args()

    load_dotenv(_repo_root / ".env")
    _check_hash(STIMULI)
    rows = [json.loads(line) for line in STIMULI.read_text().splitlines() if line.strip()]

    cfg = _read_model_cfg(args.model_config)
    hf_model_id = cfg["hf_model_id"]
    model_key = hf_model_id.split("/")[-1]
    story_dir = _repo_root / args.vectors_dir / f"{model_key}-story"
    if not story_dir.exists():
        raise SystemExit(f"no story vectors at {story_dir} (pull them from HF first)")
    vectors = _discover_vectors(story_dir)
    layer = _pick_layer(vectors, args.layer, cfg.get("n_layers"))
    vec = {e: np.load(vectors[e][layer]).astype(np.float64) for e in EMOTIONS}

    lm = load_model(hf_model_id, revision=cfg.get("hf_revision"),
                    torch_dtype=getattr(torch, args.dtype), device_map=args.device)
    lm.model.eval()
    device = next(lm.model.parameters()).device
    rec = ResidualStreamRecorder(lm.model, [layer], token_position="last")
    rec.attach()  # hooks are not registered until attach()/__enter__

    acts: dict[str, list[np.ndarray]] = defaultdict(list)  # intended -> [activation]
    ids: dict[str, list[str]] = defaultdict(list)
    print(f"reading {len(rows)} scenarios at layer {layer} ({args.device}/{args.dtype}) ...")
    try:
        with torch.no_grad():
            for i, r in enumerate(rows, 1):
                prompt = lm.tokenizer.apply_chat_template(
                    [{"role": "user", "content": r["scenario"]}],
                    tokenize=False, add_generation_prompt=True,
                )
                inp = lm.tokenizer(prompt, return_tensors="pt").to(device)
                lm.model(**inp)
                acts[r["intended_emotion"]].append(
                    rec.activations[layer][0].float().cpu().numpy().astype(np.float64)
                )
                ids[r["intended_emotion"]].append(r["id"])
                if i % 25 == 0 or i == len(rows):
                    print(f"  {i}/{len(rows)}", flush=True)
    finally:
        rec.remove()

    # Neutral-center: isolate the emotional deviation (mirrors vector construction).
    neutral_mean = (np.mean(np.stack(acts["neutral"]), axis=0)
                    if acts.get("neutral") else np.zeros_like(vec[EMOTIONS[0]]))

    # Confusion matrix (intended x argmax-emotion) on the four emotions.
    conf = {e: defaultdict(int) for e in EMOTIONS}
    cos_sum = {e: defaultdict(float) for e in EMOTIONS}
    n_int = {e: 0 for e in EMOTIONS}
    for intended in EMOTIONS:
        for a in acts[intended]:
            c = a - neutral_mean
            sims = {e: _cos(c, vec[e]) for e in EMOTIONS}
            pred = max(sims, key=sims.get)
            conf[intended][pred] += 1
            n_int[intended] += 1
            for e in EMOTIONS:
                cos_sum[intended][e] += sims[e]

    # Neutral control: emotional scenarios should activate *some* vector more
    # than neutral scenarios do.
    def max_abs_cos(a):
        c = a - neutral_mean
        return max(abs(_cos(c, vec[e])) for e in EMOTIONS)
    emo_maxcos = np.mean([max_abs_cos(a) for e in EMOTIONS for a in acts[e]])
    neu_maxcos = np.mean([max_abs_cos(a) for a in acts.get("neutral", [])]) if acts.get("neutral") else float("nan")

    correct = sum(conf[e][e] for e in EMOTIONS)
    total = sum(n_int.values())
    overall = correct / total if total else float("nan")

    lines = [
        f"# Implicit-emotion validation — {model_key}",
        "",
        f"- Vectors: `{story_dir.relative_to(_repo_root)}` (story method), layer **{layer}**",
        f"- Read-out: Assistant-colon (last prompt token), neutral-centered cosine vs each vector",
        f"- Scenarios: {sum(n_int.values())} emotion + {len(acts.get('neutral', []))} neutral",
        "",
        f"**Overall argmax accuracy: {overall:.2f}**  (chance = 0.25)",
        "",
        "## Confusion matrix (intended → predicted, counts)",
        "",
        "| intended ↓ / pred → | " + " | ".join(EMOTIONS) + " | acc |",
        "|" + "---|" * (len(EMOTIONS) + 2),
    ]
    for e in EMOTIONS:
        cells = " | ".join(str(conf[e][p]) for p in EMOTIONS)
        acc = conf[e][e] / n_int[e] if n_int[e] else float("nan")
        lines.append(f"| {e} | {cells} | {acc:.2f} |")
    lines += [
        "",
        "## Mean cosine (intended → vector)",
        "",
        "| intended ↓ / vector → | " + " | ".join(EMOTIONS) + " |",
        "|" + "---|" * (len(EMOTIONS) + 1),
    ]
    for e in EMOTIONS:
        cells = " | ".join(f"{cos_sum[e][p] / n_int[e]:+.3f}" if n_int[e] else "—" for p in EMOTIONS)
        lines.append(f"| {e} | {cells} |")
    lines += [
        "",
        "## Neutral control",
        "",
        f"- mean max|cos| on emotion scenarios: {emo_maxcos:.3f}",
        f"- mean max|cos| on neutral scenarios: {neu_maxcos:.3f}  (should be lower)",
        "",
        "A diagonal-dominant confusion matrix (intended emotion wins) is "
        "cross-context evidence the vector is a concept, not story lexis.",
    ]

    out_dir = _repo_root / args.out_dir / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "implicit_scenarios.md"
    report.write_text("\n".join(lines))
    print(f"overall accuracy {overall:.2f}  (diag {correct}/{total})")
    print(f"Wrote {report.relative_to(_repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Plot projection value vs. semantic-intensity rank, per template family.

The numerical-intensity validator reports only the per-family Spearman rho;
this script dumps the underlying relationship — for each row, the projection
of the Assistant-colon activation onto the emotion vector, against the
semantic-intensity rank. One panel per (inverse) family, so you can see the
actual trend the rho summarises.

Runs on the Mac (model cached) or a pod. Example:
    uv run python scripts/plot_intensity_projection.py --model llama32_1b
    uv run python scripts/plot_intensity_projection.py --model gemma2_9b --device cuda --dtype bfloat16
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root / "src"))
from dotenv import load_dotenv  # noqa: E402
from llm_psych.hooks import ResidualStreamRecorder  # noqa: E402
from llm_psych.models import load_model  # noqa: E402

EMO = ["admiration", "joy", "loathing", "sadness"]
ECOL = {"admiration": "#C2780C", "joy": "#15803D", "loathing": "#7E22CE", "sadness": "#2563EB"}
STIMULI = _repo_root / "data" / "public" / "intensity_templates.jsonl"
# default: one distinctive inverse family per emotion
DEFAULT_FAMS = ["team_size", "days_until_return", "relief_withheld", "age_at_death"]


def _pick_layer(vectors, requested, n_layers):
    shared = sorted(set.intersection(*(set(v) for v in vectors.values())))
    if requested is not None:
        return requested
    return min(shared, key=lambda L: abs(L - round(2 * n_layers / 3))) if n_layers else max(shared)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", dest="model_config", required=True)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--families", nargs="+", default=DEFAULT_FAMS,
                    help="template families to plot (default: one inverse family per emotion)")
    ap.add_argument("--out-dir", default="results/vector_validation")
    args = ap.parse_args()

    load_dotenv(_repo_root / ".env")
    cfg = yaml.safe_load((_repo_root / "configs" / "model" / f"{args.model_config}.yaml").read_text())
    hf_id = cfg["hf_model_id"]; key = hf_id.split("/")[-1]
    story = _repo_root / "steering_vectors" / f"{key}-story"
    vectors = defaultdict(dict)
    for f in story.glob("*_layer*.npy"):
        e, _, L = f.stem.rpartition("_layer")
        if e in EMO and L.isdigit():
            vectors[e][int(L)] = f
    layer = _pick_layer(vectors, args.layer, cfg.get("n_layers"))
    vec = {e: np.load(vectors[e][layer]).astype(np.float64) for e in EMO}

    rows = [json.loads(line) for line in STIMULI.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["family"] in args.families]

    lm = load_model(hf_id, revision=cfg.get("hf_revision"),
                    torch_dtype=getattr(torch, args.dtype), device_map=args.device)
    lm.model.eval()
    device = next(lm.model.parameters()).device
    rec = ResidualStreamRecorder(lm.model, [layer], token_position="last")
    rec.attach()
    print(f"reading {len(rows)} rows at layer {layer} ...")
    with torch.no_grad():
        try:
            for r in rows:
                prompt = lm.tokenizer.apply_chat_template(
                    [{"role": "user", "content": r["text"]}], tokenize=False, add_generation_prompt=True)
                inp = lm.tokenizer(prompt, return_tensors="pt").to(device)
                lm.model(**inp)
                a = rec.activations[layer][0].float().cpu().numpy().astype(np.float64)
                r["_proj"] = float(a @ vec[r["emotion"]])
        finally:
            rec.remove()

    fams = [f for f in args.families if any(r["family"] == f for r in rows)]
    n = len(fams)
    ncol = 2; nrow = (n + 1) // 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 4.4 * nrow), squeeze=False)
    for ax, fam in zip(axes.ravel(), fams):
        items = sorted([r for r in rows if r["family"] == fam], key=lambda r: r["intensity_rank"])
        e = items[0]["emotion"]; col = ECOL[e]
        rk = [r["intensity_rank"] for r in items]
        pr = [r["_proj"] for r in items]
        rho = spearmanr(pr, rk).statistic if len(set(rk)) > 1 else float("nan")
        ax.plot(rk, pr, "-o", color=col, lw=2, ms=8)
        for r in items:  # annotate the raw number at each point
            ax.annotate(str(r["x"]), (r["intensity_rank"], r["_proj"]), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color="#666")
        ax.set_title(f"{e} · {fam}   ρ={rho:+.2f}", color=col, fontsize=12, fontweight="bold")
        ax.set_xlabel("semantic-intensity rank  (0 = mild → high = intense)")
        ax.set_ylabel("projection onto emotion vector")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"Projection vs. semantic-intensity rank — {key} · layer {layer}\n"
                 "(inverse families: the raw number, annotated, runs opposite to the rank)", fontsize=13)
    out = _repo_root / args.out_dir / key
    out.mkdir(parents=True, exist_ok=True)
    path = out / "intensity_projection.png"
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=200)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

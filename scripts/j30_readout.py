"""J30 readout, judge-free, run before any J30 generation.

Teacher-forced passes of the canonical scenario prompt (no sampling):

  * unsteered;
  * `full` at 0.25 at L16;
  * `pool1` at 0.15 at L16.

At each layer l in 16..26, delta_l = h_l(steered) - h_l(unsteered) at every
position. For each direction u in {d, cp, r1, r2, r3} it reports

    proj_l(u)  = mean over positions of (delta_l . u) / ||delta_l||
    share_l(u) = mean over positions of (delta_l . u)^2 / ||delta_l||^2

i.e. how much of the steering-induced change at layer l lies along u. Every
direction is perpendicular to v_hat, so `full`'s share along d at L16 is 0 by
construction. The question is whether it RISES at later layers above the
random-direction band: does the model convert v into the pool direction?

It also reports the ablation's damage on the unsteered pass:
mean ||P_u(h - mu)||^2 / ||h||^2 per direction and layer (needs results/j27/mu).

    uv run python scripts/j30_readout.py --model-path /workspace/qwen25-7b-instruct-a09a354
    uv run python scripts/j30_readout.py --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import j27_common as jc  # noqa: E402

ROOT = jc.ROOT
DIRS = ROOT / "results" / "j30" / "dirs"
OUT = ROOT / "results" / "j30" / "readout.json"
NAMES = ("d", "cp", "r1", "r2", "r3")
ARMS = (("full", "j24_full", 0.25), ("pool1", "j24_pool1", 0.15))


def _mu():
    return jc._load("compute_j27_mu", "scripts/compute_j27_mu.py")


def capture(model, ids, layers, steer=None) -> dict[int, torch.Tensor]:
    """Block outputs (positions x d) at each layer; steering hook attached first."""
    hooks = _mu()._hooks()
    dec = hooks.get_decoder_layers(model)
    out: dict[int, torch.Tensor] = {}
    handles = []

    def mk(layer):
        def cap(_m, _i, o):
            h = o[0] if isinstance(o, tuple) else o
            out[layer] = h[0].detach().float().cpu()
        return cap

    ctx = None
    if steer is not None:
        direction, alpha = steer
        ctx = hooks.steering(model, jc.LAYER, direction, alpha, norm_scale=True, site="post")
        ctx.__enter__()
    try:
        for lyr in layers:
            handles.append(dec[lyr].register_forward_hook(mk(lyr)))
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        for h_ in handles:
            h_.remove()
        if ctx is not None:
            ctx.__exit__(None, None, None)
    return out


def stats(delta: torch.Tensor, u: torch.Tensor) -> tuple[float, float]:
    n = delta.norm(dim=1).clamp(min=1e-12)
    c = (delta @ u) / n
    return float(c.mean()), float((c ** 2).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-path", default="/workspace/qwen25-7b-instruct-a09a354")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    layers = jc.ABLATION_LAYERS
    m = _mu()

    if args.self_test:
        model, tok = m._tiny_mock(n_layers=max(layers) + 1), m._MockTokenizer()
        ids = tok("mock canonical prompt")["input_ids"]
        dim = 64
        dirs = {(n, lyr): torch.nn.functional.normalize(torch.randn(dim), dim=0)
                for n in NAMES for lyr in layers}
        vecs = {a: torch.randn(dim) for _, a, _ in ARMS}
        mus = {lyr: torch.zeros(dim) for lyr in layers}
        out_path = ROOT / "results" / "j30_selftest" / "readout.json"
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map=device)
        text, _sha = jc.rendered_prompt(tok)
        ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        dirs = {(n, lyr): torch.from_numpy(np.load(DIRS / f"{n}_l{lyr}.npy")).float().flatten()
                for n in NAMES for lyr in layers}
        vecs = {a: torch.from_numpy(np.load(jc.VEC_DIR / f"{a}_layer16.npy").astype(np.float32))
                for _, a, _ in ARMS}
        mdir = jc.RESULTS / "mu"
        mus = ({lyr: torch.from_numpy(np.load(mdir / f"mu_l{lyr}.npy")).float() for lyr in layers}
               if all((mdir / f"mu_l{lyr}.npy").exists() for lyr in layers) else None)
        out_path = OUT
    model.eval()

    base = capture(model, ids, layers)
    rep: dict = {"layers": layers, "arms": {}, "damage_unsteered": {}}
    for label, arm, alpha in ARMS:
        st = capture(model, ids, layers, steer=(vecs[arm], alpha))
        rep["arms"][label] = {}
        for lyr in layers:
            delta = st[lyr] - base[lyr]
            rep["arms"][label][lyr] = {n: dict(zip(("proj", "share"), stats(delta, dirs[(n, lyr)])))
                                       for n in NAMES}
    if mus is not None:
        for lyr in layers:
            h = base[lyr]
            dev = h - mus[lyr]
            hn = h.norm(dim=1).pow(2).clamp(min=1e-12)
            rep["damage_unsteered"][lyr] = {
                n: float(((dev @ dirs[(n, lyr)]) ** 2 / hn).mean()) for n in NAMES}
    else:
        rep["damage_unsteered"] = "results/j27/mu missing: run compute_j27_mu.py first"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rep, indent=2))
    for label in rep["arms"]:
        print(f"\n## {label}: share of the steering-induced change along each direction")
        print("layer  " + "  ".join(f"{n:>7s}" for n in NAMES))
        for lyr in layers:
            r = rep["arms"][label][lyr]
            print(f"{lyr:5d}  " + "  ".join(f"{r[n]['share']:7.4f}" for n in NAMES))
    print(f"\n-> {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

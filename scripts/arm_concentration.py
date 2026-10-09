"""How peaked is each arm's effect on the token distribution?

J19 result 2 found that the arms which break the coherence gate are the NNLS
refits (`jspace` 30%, `randatom` 45% at alpha 0.15) and not the arms with the most
lens-span content (`pin`, 9.6%). The proposed mechanism is CONCENTRATION rather
than span membership or sign: a refit has norm ~0.3|v| and is renormalised up to
the injected norm, multiplying every atom coefficient by ~3, so it pushes a handful
of individual token logits far harder than `v` itself does. A random direction
inside the same span spreads over ~13 dimensions with mixed signs and pushes no
single token hard.

This script measures that, for every arm file, with no generation and no judge:
push the arm through the lens and unembedding exactly as the decomposition does
(`logits = (v @ j_l.T * g) @ w_u.T`), then summarise how concentrated the resulting
logit vector is over the vocabulary.

Statistics, all computed on the POSITIVE part of the logit vector, because an arm
acts by raising logits and the negative tail is not what reaches the sampler:
  top1, top10, top100   share of total positive mass in the k largest entries
  participation_ratio   (sum p)^2 / sum p^2 over the positive part -- the
                        effective number of tokens the arm pushes. Small = peaky.
  pr_frac               participation_ratio / vocab_size, scale-free
  max_z                 largest logit in units of the logit vector's own sd

Arms are compared to each other, so every arm is unit-normalised first: the saved
files all share one target_norm, and we want shape, not scale.

Usage (CPU, needs the lens + unembedding, ~15 min to load):

    .venv/bin/python scripts/arm_concentration.py \\
        --model-config configs/model/gemma2_9b.yaml \\
        --track story-wheel32 --layer 22 \\
        --tags ca_pln_pos ca_unit_pos \\
        -o results/jspace_gate/arm_concentration_L22.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

_repo_root = Path(__file__).resolve().parents[1]


def _load_mod():
    spec = importlib.util.spec_from_file_location(
        "dec", _repo_root / "scripts" / "decompose_emotion_vectors.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _stats(logits: torch.Tensor) -> dict:
    x = logits.float()
    sd = float(x.std())
    pos = torch.clamp(x, min=0.0)
    tot = float(pos.sum())
    if tot <= 0:
        return {"degenerate": True}
    p = (pos / tot).double()
    srt = torch.sort(p, descending=True).values
    pr = float(1.0 / (p * p).sum())
    return {
        "top1": float(srt[0]),
        "top10": float(srt[:10].sum()),
        "top100": float(srt[:100].sum()),
        "participation_ratio": pr,
        "pr_frac": pr / p.numel(),
        "max_z": float(x.max()) / sd if sd > 0 else float("nan"),
        "n_positive": int((x > 0).sum()),
        "vocab": int(p.numel()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config", type=Path, required=True)
    ap.add_argument("--track", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--tags", nargs="+", required=True,
                    help="arm filename prefixes, e.g. ca_pln_pos ca_unit_pos")
    ap.add_argument("--lens-source", default="neuronpedia",
                    choices=["neuronpedia", "logit"])
    ap.add_argument("--lens-path", type=Path, default=None)
    ap.add_argument("-o", "--out", type=Path, required=True)
    a = ap.parse_args()

    mod = _load_mod()
    cfg = mod._load_model_config(a.model_config)
    model_key, hf_id = cfg["model_key"], cfg["hf_model_id"]
    revision = cfg.get("hf_revision")
    vec_dir = _repo_root / "steering_vectors" / f"{model_key}-{a.track}"

    files = sorted({q for t in a.tags
                    for q in vec_dir.glob(f"{t}_*_layer{a.layer}.npy")})
    if not files:
        raise SystemExit(f"no arm files under {vec_dir} for tags {a.tags}")
    print(f"{len(files)} arm files", flush=True)

    print("loading unembedding shards and lens ...", flush=True)
    mod._load_tokenizer(hf_id, revision=revision,
                        trust_remote_code=cfg.get("trust_remote_code", False))
    w_u, g = mod._load_unembed_and_norm_shards(hf_id, revision)
    j, lens_prov = mod._load_jlens(a.lens_source, hf_id, a.lens_path)
    mapped = mod._map_layer(a.layer, set(j.keys()) if j else None, "nearest")
    if mapped is None:
        raise SystemExit(f"layer {a.layer} not covered by the lens")
    j_l = j[mapped].float() if j is not None else None

    out = {"model_key": model_key, "track": a.track, "layer": a.layer,
           "lens": {"source": a.lens_source, "layer_used": mapped,
                    "provenance": lens_prov},
           "arms": {}}
    for q in files:
        v = torch.from_numpy(np.load(q).astype(np.float32))
        v = v / v.norm()                      # shape, not scale
        z = (v @ j_l.T) if j_l is not None else v
        logits = (z.to(w_u.dtype) * g.to(w_u.dtype)) @ w_u.T
        name = q.name.replace(f"_layer{a.layer}.npy", "")
        out["arms"][name] = _stats(logits)
        s = out["arms"][name]
        print(f"  {name:28} top1 {s['top1']:.4f}  top10 {s['top10']:.3f}  "
              f"PR {s['participation_ratio']:9.1f}  max_z {s['max_z']:6.2f}", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

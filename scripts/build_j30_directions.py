"""J30 build: per-layer rank-1 ablation directions (d, c', r1-r3), L16-L26.

Preregistered in plans/j-space-decomposition.md ("J30"). Needs the J-lens,
lm_head and final norm, so it runs on the Mac (`uv run`) or a pod.

For a family F (pool from J24's build report, coh from J25's), draw i, layer l:

    x_il = sum over the draw's 44 tokens of (unit J-lens atom at l - English centre at l)
    w_il = x_il - (x_il . v_hat) v_hat
    g_il = P_{G_l} w_il ;  g_il <- g_il - (g_il . v_hat) v_hat ;  unit
    F_l  = unit(mean_i g_il)

    d_l  = pool_l                                   (the target)
    c'_l = unit(coh_l - (coh_l . d_l) d_l)          (content control, perpendicular to d)
    r_kl = Gaussian in G_l, Gram-Schmidt against {v_hat, d_l, c'_l}, unit  (k = 1..3)

v_hat is the L16 emotion vector. The injected vector rides the residual stream
into every later layer, so every direction is made perpendicular to v_hat at
EVERY layer: ablating it never cuts the injected vector itself.

Acceptance (frozen in the preregistration; the build refuses on any failure):
  * cos(d_16, mean of the unit j29b_poolG files) >= 0.999   (reproduces J29)
  * pool1's w_hat share along d_16 >= 0.60                  (computed 0.764 pre-freeze)
  * |cos(dir, v_hat)| < 1e-6 for every direction and layer
  * |cos(r_k, d)|, |cos(r_k, c')|, |cos(c', d)| < 1e-6

Outputs (refuses to overwrite): results/j30/dirs/{d,cp,r1,r2,r3}_l<L>.npy,
float32 (d_model, 1), loadable by the J27 AblationHook via the provider's
ablation_spaces_dir; results/j30/build_report.json.

    uv run python scripts/build_j30_directions.py
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
OUT = ROOT / "results" / "j30"
DIRS = OUT / "dirs"
G_DIR = jc.RESULTS / "subspaces_clean" / "_G"
J24_REPORT = ROOT / "results/jspace_gate/arms_Qwen2.5-7B-Instruct_story-wheel32_L16_j24.json"
J25_REPORT = ROOT / "results/j25/build_real_report.json"

# --- frozen by J30 ------------------------------------------------------------
SEED = 20261113
N_RANDOM = 3
MIN_REPRO = 0.999
MIN_POOL1_SHARE = 0.60
TOL = 1e-6
NAMES = ("d", "cp", "r1", "r2", "r3")
# ------------------------------------------------------------------------------


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm()


def perp(x: torch.Tensor, others: list[torch.Tensor]) -> torch.Tensor:
    """Gram-Schmidt x against an orthonormalised copy of `others`."""
    q: list[torch.Tensor] = []
    for o in others:
        o = o.clone()
        for b in q:
            o = o - (o @ b) * b
        if o.norm() > 1e-9:
            q.append(o / o.norm())
    for b in q:
        x = x - (x @ b) * b
    return x


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-config", type=Path,
                    default=ROOT / "configs" / "model" / "qwen25_7b.yaml")
    ap.add_argument("--lens-source", default="neuronpedia")
    ap.add_argument("--lens-path", type=Path, default=None)
    args = ap.parse_args()

    planned = [DIRS / f"{n}_l{lyr}.npy" for n in NAMES for lyr in jc.ABLATION_LAYERS]
    for f in planned:
        if f.exists():
            raise SystemExit(f"REFUSING to overwrite {f}")

    dec = jc._load("dec", "scripts/decompose_emotion_vectors.py")
    cfg = dec._load_model_config(args.model_config)
    hf_id, revision = cfg["hf_model_id"], cfg.get("hf_revision")
    print("loading unembedding shards and lens ...", flush=True)
    w_u, g = dec._load_unembed_and_norm_shards(hf_id, revision)
    j, lens_prov = dec._load_jlens(args.lens_source, hf_id, args.lens_path)

    j24 = json.loads(J24_REPORT.read_text())["arms"]
    j25 = json.loads(J25_REPORT.read_text())["arms"]
    draws = {"pool": [torch.tensor(j24[f"pool{i}"]["token_ids"]) for i in range(1, 7)],
             "coh": [torch.tensor(j25[f"coh{i}"]["token_ids"]) for i in range(1, 7)]}

    vec = jc.VEC_DIR
    vhat = unit(torch.from_numpy(np.load(vec / "j24_full_layer16.npy").astype(np.float32)))
    a1 = unit(torch.from_numpy(np.load(vec / "j24_pool1_layer16.npy").astype(np.float32)))
    w1 = unit(a1 - (a1 @ vhat) * vhat)
    pg_mean = sum(unit(torch.from_numpy(np.load(vec / f"j29b_poolG{i}_layer16.npy")
                                        .astype(np.float32))) for i in range(1, 7))
    pg_mean = unit(pg_mean)

    rng = np.random.default_rng(np.random.SeedSequence(SEED))
    DIRS.mkdir(parents=True, exist_ok=True)
    rep: dict = {"amendment": "J30", "seed": SEED, "lens": lens_prov, "layers": {}}
    out: dict[tuple[str, int], torch.Tensor] = {}

    for lyr in jc.ABLATION_LAYERS:
        jl = j[lyr].float()
        centre = torch.from_numpy(np.load(jc.RESULTS / f"english_centre_l{lyr}.npy")).float()
        G = torch.from_numpy(np.load(G_DIR / f"G_l{lyr}.npy")).float()      # (d, 32)

        def family(name: str) -> torch.Tensor:
            gs = []
            for tids in draws[name]:
                a = (w_u[tids].float() * g.float()) @ jl
                a = a / torch.clamp(a.norm(dim=1, keepdim=True), min=1e-12)
                x = (a - centre).sum(0)
                w = x - (x @ vhat) * vhat
                gg = G @ (G.T @ w)
                gg = gg - (gg @ vhat) * vhat
                gs.append(unit(gg))
            return unit(torch.stack(gs).mean(0)), gs

        d, d_draws = family("pool")
        coh, _ = family("coh")
        cp = unit(perp(coh, [vhat, d]))
        rs = []
        for _k in range(N_RANDOM):
            x = G @ torch.from_numpy(rng.normal(size=G.shape[1]).astype(np.float32))
            rs.append(unit(perp(x, [vhat, d, cp, *rs])))
        dirs = dict(zip(NAMES, [d, cp, *rs]))

        lrep = {"cos_v": {}, "share_in_G": {},
                "cos_pool_vs_coh": float(d @ coh), "pool_draw_cos_to_d":
                [float(x @ d) for x in d_draws]}
        for n, x in dirs.items():
            cv = float(x @ vhat)
            lrep["cos_v"][n] = cv
            lrep["share_in_G"][n] = float((G.T @ x).pow(2).sum())
            if abs(cv) > TOL:
                raise SystemExit(f"REFUSING: {n}_l{lyr} cos with v_hat {cv:.2e}")
        for a_, b_ in (("r1", "d"), ("r2", "d"), ("r3", "d"), ("r1", "cp"),
                       ("r2", "cp"), ("r3", "cp"), ("cp", "d")):
            c = float(dirs[a_] @ dirs[b_])
            if abs(c) > TOL:
                raise SystemExit(f"REFUSING: cos({a_},{b_}) at l{lyr} = {c:.2e}")
        if lyr == jc.LAYER:
            repro = float(d @ pg_mean)
            share = float((w1 @ d) ** 2)
            rep["repro_cos_d16_vs_j29b"] = repro
            rep["pool1_w_hat_share_on_d16"] = share
            if repro < MIN_REPRO:
                raise SystemExit(f"REFUSING: d_16 does not reproduce J29 (cos {repro:.4f})")
            if share < MIN_POOL1_SHARE:
                raise SystemExit(f"REFUSING: pool1 share on d_16 {share:.3f} < {MIN_POOL1_SHARE}")
        if lyr > jc.LAYER:
            lrep["cos_d_with_d16"] = float(d @ out[("d", jc.LAYER)])
        rep["layers"][lyr] = lrep
        for n, x in dirs.items():
            out[(n, lyr)] = x

    for (n, lyr), x in out.items():
        np.save(DIRS / f"{n}_l{lyr}.npy", x.numpy().astype(np.float32).reshape(-1, 1))
    (OUT / "build_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps({k: v for k, v in rep.items() if k != "layers"}, indent=2))
    for lyr, lr in rep["layers"].items():
        print(lyr, {k: v for k, v in lr.items() if k in ("cos_d_with_d16", "cos_pool_vs_coh")},
              {n: round(s, 3) for n, s in lr["share_in_G"].items()})
    print(f"{len(out)} direction files -> {DIRS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

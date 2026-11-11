"""J27 ablation subspace builder: token sets V, H, C, R1-R3 at L16-26, Qwen 2.5 7B.

Pre-registered in plans/j-space-decomposition.md ("J27 — Mediation",
2026-11-10; J26-J28 amendments 1-3). For each token set and each decoder layer
l in 16..26, the subspace is spanned by the top 121 left singular vectors of
the centred unit-atom matrix

    a_t = unit((W_U[t] . g) J_l),  centred by the English-word mean unit atom
    at layer l (J24 amendment 2 centring; cached in results/j27/english_centre_l*.npy).

Token sets (amendment 1 item 1):

    V    the 121 English-word tokens ranked highest by v's J-lens logit at L16
         (v = contempt=1 + aggressiveness=1 unit mixture, as build_j25_arms).
    H    all English-word tokens hitting the J23 hostility lexicon
         (score_j22_diagnostic WORDS | J23_EXTRA_WORDS; expected ~121).
    C    121 tokens drawn from coh's eligible set (u_coh = admiration=1 +
         joy=1 top-2048 English-word candidates, hostile and v's top-2048
         candidate tokens excluded), as build_j25_arms.
    R1-3 121 English-word tokens each, hostile and v-candidate excluded.

Draws: np.random.SeedSequence(20261110).spawn in the order C, R1, R2, R3
(the order is part of the spec).

Outputs (refuses to overwrite):
    results/j27/subspaces/<set>_l<l>.npy   basis (d, 121) per set per layer
    results/j27/english_centre_l<l>.npy    cached English-word mean unit atom
    results/j27/token_sets.json            token ids + decoded text per set
    results/j27/build_report.json          sizes, ||P_16 v_hat||^2 per set,
        pool1 w_hat share inside V at L16, energy fractions, sha256 per file

CPU-safe: no model weights, only the tokenizer, the unembedding shards and the
neuronpedia J-lens. The expensive step is the per-layer English-word centre
(~40k unit atoms through a 3584 x 3584 lens); it is computed once per layer
and cached.

    uv run python scripts/build_j27_ablations.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import j27_common as jc  # noqa: E402

ROOT = jc.ROOT
ENGLISH_RE = re.compile(r" [A-Za-z]{3,}")      # J24 amendment 2 / build_j25_arms
N_ENGLISH_V = 121
SET_NAMES = ("V", "H", "C", "R1", "R2", "R3")
SPAWN_ORDER = ("C", "R1", "R2", "R3")           # spec: spawn order is part of it
N_CANDIDATES = 2048
W90_WARNING_FRACTION = 0.70


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model-config", type=Path,
                   default=ROOT / "configs" / "model" / "qwen25_7b.yaml")
    p.add_argument("--track", default="story-wheel32")
    p.add_argument("--layer", type=int, default=jc.LAYER)
    p.add_argument("--mix", nargs="+", default=["contempt=1", "aggressiveness=1"])
    p.add_argument("--coh-mix", nargs="+", default=["admiration=1", "joy=1"])
    p.add_argument("--n-candidates", type=int, default=N_CANDIDATES)
    p.add_argument("--lens-source", default="neuronpedia",
                   choices=["neuronpedia", "logit"])
    p.add_argument("--lens-path", type=Path, default=None)
    p.add_argument("--out", type=Path, default=jc.RESULTS)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    dec = jc._load("dec", "scripts/decompose_emotion_vectors.py")
    sc = jc._load("sc", "scripts/score_j22_diagnostic.py")
    lex = sc.WORDS | sc.J23_EXTRA_WORDS
    out = args.out
    sub_dir = out / "subspaces"
    planned = [sub_dir / f"{s}_l{lyr}.npy" for s in SET_NAMES
               for lyr in jc.ABLATION_LAYERS]
    planned += [out / "token_sets.json", out / "build_report.json"]
    if not args.overwrite:
        for f in planned + [out / f"english_centre_l{lyr}.npy"
                            for lyr in jc.ABLATION_LAYERS]:
            if f.exists():
                raise SystemExit(f"REFUSING to overwrite {f} (--overwrite to force)")

    cfg = dec._load_model_config(args.model_config)
    model_key, hf_id = cfg["model_key"], cfg["hf_model_id"]
    revision = cfg.get("hf_revision")
    vec_dir = ROOT / "steering_vectors" / f"{model_key}-{args.track}"

    def mixture(specs: list[str]) -> np.ndarray:
        m = None
        for spec in specs:
            name, _, c = spec.partition("=")
            v = np.load(vec_dir / f"{name}_layer{args.layer}.npy").astype(np.float64)
            v = float(c) * v / np.linalg.norm(v)
            m = v if m is None else m + v
        return m

    u = mixture(args.mix).astype(np.float32)
    uu = torch.from_numpy(u)
    u_coh = torch.from_numpy(mixture(args.coh_mix).astype(np.float32))

    print("loading tokenizer, unembedding shards and lens ...", flush=True)
    tokenizer = dec._load_tokenizer(
        hf_id, revision=revision,
        trust_remote_code=cfg.get("trust_remote_code", False))
    w_u, g = dec._load_unembed_and_norm_shards(hf_id, revision)
    j, lens_prov = dec._load_jlens(args.lens_source, hf_id, args.lens_path)

    # The spec requires the lens to cover 16..26 directly (source_layers 0..26).
    layer_map = {}
    for lyr in jc.ABLATION_LAYERS:
        mapped = dec._map_layer(lyr, set(j.keys()) if j else None, "nearest")
        layer_map[lyr] = mapped
        if mapped != lyr:
            raise SystemExit(f"REFUSING: layer {lyr} not covered directly by the "
                             f"lens (mapped to {mapped}); J27 needs 16..26")
    j_f = {lyr: j[lyr].float() for lyr in jc.ABLATION_LAYERS}
    d_model = j_f[jc.ABLATION_LAYERS[0]].shape[0]

    def lens_logits(x: torch.Tensor, layer: int) -> torch.Tensor:
        z = x.float() @ j_f[layer].T
        return ((z.to(w_u.dtype) * g.to(w_u.dtype)) @ w_u.T).float()

    # --- token masks, exactly as build_j25_arms ------------------------------
    vocab = w_u.shape[0]
    print(f"building hostility/English masks over {vocab} tokens ...", flush=True)
    hostile = torch.zeros(vocab, dtype=torch.bool)
    english = torch.zeros(vocab, dtype=torch.bool)
    texts: dict[int, str] = {}
    for t in range(vocab):
        try:
            s = tokenizer.decode([t])
        except Exception:
            continue
        if sc.lexicon_hits(sc.words(s), lex)[0] > 0:
            hostile[t] = True
        if t < len(tokenizer) and ENGLISH_RE.fullmatch(s):
            english[t] = True
            texts[t] = s
    print(f"hostile tokens: {int(hostile.sum())}   "
          f"English-word tokens: {int(english.sum())}", flush=True)

    # --- token sets at L16 -----------------------------------------------------
    logits_v = lens_logits(uu, args.layer)
    cand_ids = torch.topk(logits_v, args.n_candidates).indices
    in_cand = torch.zeros(vocab, dtype=torch.bool)
    in_cand[cand_ids] = True

    eng_ids = torch.nonzero(english).flatten()
    v_ids = eng_ids[torch.argsort(logits_v[eng_ids], descending=True)][:N_ENGLISH_V]
    h_ids = torch.nonzero(hostile & english).flatten()

    logits_coh = lens_logits(u_coh, args.layer)
    coh_cand = torch.zeros(vocab, dtype=torch.bool)
    coh_cand[torch.topk(logits_coh, args.n_candidates).indices] = True
    coh_elig = torch.nonzero(coh_cand & english & ~hostile & ~in_cand).flatten()
    r_elig = torch.nonzero(english & ~hostile & ~in_cand).flatten()
    print(f"eligible: coh {coh_elig.numel()}   random {r_elig.numel()}", flush=True)
    for name, elig in (("C", coh_elig), ("R", r_elig)):
        if elig.numel() < N_ENGLISH_V:
            raise SystemExit(f"REFUSING: {name} eligibility {elig.numel()} "
                             f"< {N_ENGLISH_V}")

    rngs = {f: np.random.default_rng(ch)
            for f, ch in zip(SPAWN_ORDER,
                             np.random.SeedSequence(jc.SEED).spawn(len(SPAWN_ORDER)),
                             strict=True)}
    sets: dict[str, torch.Tensor] = {"V": v_ids.sort().values, "H": h_ids.sort().values}
    for name in SPAWN_ORDER:
        elig = coh_elig if name == "C" else r_elig
        idx = rngs[name].choice(elig.numel(), size=N_ENGLISH_V, replace=False)
        sets[name] = elig[torch.from_numpy(np.sort(idx))]

    # --- per-layer centred atoms and bases ------------------------------------
    def unit_atoms(tids: torch.Tensor, layer: int) -> torch.Tensor:
        a = (w_u[tids].float() * g.float()) @ j_f[layer]
        return a / torch.clamp(a.norm(dim=1, keepdim=True), min=1e-12)

    def english_centre(layer: int) -> torch.Tensor:
        f = out / f"english_centre_l{layer}.npy"
        if f.exists():
            print(f"  centre l{layer}: loaded cache {f}", flush=True)
            return torch.from_numpy(np.load(f)).float()
        c = torch.zeros(d_model)
        for lo in range(0, eng_ids.numel(), jc.CHUNK):
            c += unit_atoms(eng_ids[lo:lo + jc.CHUNK], layer).sum(0)
        c /= eng_ids.numel()
        f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, c.numpy().astype(np.float32))
        print(f"  centre l{layer}: computed over {eng_ids.numel()} tokens -> {f}",
              flush=True)
        return c

    sub_dir.mkdir(parents=True, exist_ok=True)
    bases: dict[tuple[str, int], np.ndarray] = {}
    energy: dict[str, dict[str, float]] = {}
    for lyr in jc.ABLATION_LAYERS:
        centre = english_centre(lyr)
        for name, tids in sets.items():
            atoms = unit_atoms(tids, lyr) - centre            # (n, d) centred
            # Spec: "top 121 left singular vectors" of the centred atom matrix
            # stored as the (d, n) basis -> left singular vectors of A = atoms.T.
            u_svd, s_svd, _ = np.linalg.svd(
                atoms.numpy().T.astype(np.float64), full_matrices=False)
            r_store = min(jc.N_TOKENS, u_svd.shape[1])
            basis = u_svd[:, :r_store].astype(np.float32)   # (d, r_store)
            f = sub_dir / f"{name}_l{lyr}.npy"
            np.save(f, basis)
            bases[(name, lyr)] = basis
            energy.setdefault(name, {})[str(lyr)] = {
                str(r): float((s_svd[:r] ** 2).sum() / (s_svd ** 2).sum())
                for r in jc.RANKS}

    # --- report metrics ---------------------------------------------------------
    vhat = torch.from_numpy(
        np.load(vec_dir / "j24_full_layer16.npy").astype(np.float32))
    vhat = vhat / vhat.norm()
    a_pool1 = torch.from_numpy(
        np.load(vec_dir / "j24_pool1_layer16.npy").astype(np.float32))
    w = a_pool1 - (a_pool1 @ vhat) * vhat
    w_hat = w / w.norm()

    def proj_sq(basis: np.ndarray, vec: torch.Tensor, r: int) -> float:
        b = torch.from_numpy(basis[:, :r]).float()
        return float((b.T @ vec).pow(2).sum())

    p_v = {name: {str(r): proj_sq(bases[(name, args.layer)], vhat, r)
                  for r in jc.RANKS} for name in SET_NAMES}
    w_in_v = {str(r): proj_sq(bases[("V", args.layer)], w_hat, r)
              for r in jc.RANKS}
    if w_in_v[str(jc.N_TOKENS)] < W90_WARNING_FRACTION:
        print(f"WARNING: only {w_in_v[str(jc.N_TOKENS)]:.3f} of pool1's w_hat is "
              f"inside V at L16 even at r=121 — condition 14's prediction is weak",
              flush=True)
    elif w_in_v[str(jc.RANKS[0])] < W90_WARNING_FRACTION:
        print(f"NOTE: pool1's w_hat share inside V at r={jc.RANKS[0]} is "
              f"{w_in_v[str(jc.RANKS[0])]:.3f} (< {W90_WARNING_FRACTION}); the "
              "share depends on the rank the manipulation check selects",
              flush=True)

    shas = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(sub_dir.glob("*.npy"))}
    token_sets = {name: {"token_ids": tids.tolist(),
                         "texts": [tokenizer.decode([t]) for t in tids.tolist()]}
                  for name, tids in sets.items()}
    (out / "token_sets.json").write_text(json.dumps(token_sets, indent=2))

    report = {"spec": "J27 mediation, plans/j-space-decomposition.md "
                      "2026-11-10 + J26-J28 amendments 1-3",
              "model_key": model_key, "track": args.track,
              "seed": jc.SEED, "seed_spawn_order": SPAWN_ORDER,
              "set_sizes": {k: int(v.numel()) for k, v in sets.items()},
              "sets": token_sets,
              "centring": "English-word mean unit atom per layer "
                          "(J24 amendment 2); cached english_centre_l*.npy",
              "n_hostile_tokens": int(hostile.sum()),
              "n_english_tokens": int(english.sum()),
              "coh_eligible": int(coh_elig.numel()),
              "random_eligible": int(r_elig.numel()),
              "lens": {"provenance": lens_prov,
                       "layer_used_per_set": {name: {str(lyr): layer_map[lyr]
                                                     for lyr in jc.ABLATION_LAYERS}
                                              for name in SET_NAMES}},
              "v_hat": "unit(j24_full_layer16)",
              "P16_vhat_sq": p_v,
              "pool1_w_hat_in_V_L16": {
                  "w_hat": "unit(j24_pool1 - (j24_pool1 . v_hat) v_hat)",
                  "share": w_in_v,
                  "warning_threshold": W90_WARNING_FRACTION,
                  "weak_prediction": w_in_v[str(jc.N_TOKENS)] < W90_WARNING_FRACTION},
              "energy_fraction": energy,
              "subspace_sha256": shas}
    rp = out / "build_report.json"
    rp.write_text(json.dumps(report, indent=2))

    print("\n||P_16 v_hat||^2 per set (r = 32 / 64 / 121):")
    for name in SET_NAMES:
        print(f"  {name:3s} n={sets[name].numel():3d}   "
              + "  ".join(f"{p_v[name][str(r)]:.3f}" for r in jc.RANKS))
    print("pool1 w_hat inside V at L16: "
          + "  ".join(f"{w_in_v[str(r)]:.3f}" for r in jc.RANKS))
    print(f"token sets -> {out / 'token_sets.json'}")
    print(f"report -> {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

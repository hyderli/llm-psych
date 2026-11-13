"""J27 amendment 6 (2026-11-11): P_clean — the subspace where pool's content lives.

Pre-registered in plans/j-space-decomposition.md ("J26–J28 amendment 6");
thresholds are frozen and no further tuning follows this build.

The amendment-5 build's stop rule fired (pool1 w_hat inside cleaned V = 0.063
< 0.60): the pool arms sit ~85% inside G, the generic English-atom subspace,
so V_clean never held pool's content. P_clean is built from the pool atoms
themselves.

Construction, per layer l in 16..26:

1. P_clean: the centred atom matrix of ALL pool-family atoms — the union of
   pool1..6's drawn token_ids (264 atoms) from the J24 arms report
   (results/jspace_gate/arms_<model>_<track>_L16_j24.json), English-mean
   centring as before — is projected onto G's orthogonal complement
   (A* = (I - P_G) A, G loaded from subspaces_clean/_G/), then top-r SVD,
   r in {32, 64, 121} (121 stored).
2. C and R1-R3: their centred atom matrices get the same G-projection, then
   are orthogonalised against P_clean's CLEANED ATOM SPAN (the full column
   span of A*_P, not the stored rank-121 basis — the amendment-5 control
   rule with target = P_clean), then top-r SVD.
3. V_clean and H are NOT rebuilt — the subspaces_clean/ files are reused
   as-is (full+V_clean loads V from subspaces_clean/, full+H loads H).

Outputs (refuses to overwrite):
    results/j27/subspaces_pclean/{pclean,c,r1,r2,r3}_l{16..26}.npy
    results/j27/token_sets.json           gains a "pclean" entry (264 ids)
    results/j27/build_report.json         gains an amendment6 section

Acceptance, frozen in the amendment (this script exits nonzero on failure —
report, do not adjust):
  * pool1 w_hat inside P_clean at r=32 >= 0.60;
  * pool1 w_hat inside each orthogonalised control at r=32 < 0.30.
Reported with no pass/fail: ||P_l v_hat||^2 per set at r in {32,64,121}
(L16 and L26); w_hat shares at all ranks; surviving variance per set;
set-set and G-set overlaps; sha256 of every written file.

    uv run python scripts/build_j27_pclean_subspaces.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import j27_common as jc  # noqa: E402

ROOT = jc.ROOT
NEW_SETS = ("pclean", "c", "r1", "r2", "r3")       # written to subspaces_pclean/
ORTHO_SETS = ("c", "r1", "r2", "r3")               # ortho'd vs P_clean atom span
CLEAN_TOKEN_SETS = {"c": "C", "r1": "R1", "r2": "R2", "r3": "R3"}
REUSED_SETS = {"V": "subspaces_clean", "H": "subspaces_clean"}  # record/report
ALL_REPORT_SETS = ("pclean", "c", "r1", "r2", "r3", "V", "H")
W_IN_PCLEAN_MIN = 0.60          # frozen acceptance: below -> STOP and report
W_IN_CONTROL_MAX = 0.30         # frozen acceptance: at/above -> STOP
SANITY_LAYERS = (jc.LAYER, 26)
J24_TAG = "j24"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model-config", type=Path,
                   default=ROOT / "configs" / "model" / "qwen25_7b.yaml")
    p.add_argument("--track", default="story-wheel32")
    p.add_argument("--layer", type=int, default=jc.LAYER,
                   help="layer the J24 arms/report live at (16)")
    p.add_argument("--results", type=Path, default=jc.RESULTS)
    p.add_argument("--lens-source", default="neuronpedia",
                   choices=["neuronpedia", "logit"])
    p.add_argument("--lens-path", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    res = args.results
    p_dir = res / "subspaces_pclean"
    g_dir = res / "subspaces_clean" / "_G"
    planned = [p_dir / f"{s}_l{lyr}.npy" for s in NEW_SETS
               for lyr in jc.ABLATION_LAYERS]
    if not args.overwrite:
        for f in planned:
            if f.exists():
                raise SystemExit(f"REFUSING to overwrite {f} (--overwrite to force)")
    for lyr in jc.ABLATION_LAYERS:
        for f in (res / f"english_centre_l{lyr}.npy",
                  g_dir / f"G_l{lyr}.npy",
                  res / "subspaces_clean" / f"V_l{lyr}.npy",
                  res / "subspaces_clean" / f"H_l{lyr}.npy"):
            if not f.exists():
                raise SystemExit(f"REFUSING: required build artifact missing: {f}")

    dec = jc._load("dec", "scripts/decompose_emotion_vectors.py")
    token_sets = json.loads((res / "token_sets.json").read_text())
    sets = {s: torch.tensor(token_sets[CLEAN_TOKEN_SETS[s]]["token_ids"],
                            dtype=torch.long) for s in ORTHO_SETS}

    cfg = dec._load_model_config(args.model_config)
    model_key, hf_id = cfg["model_key"], cfg["hf_model_id"]
    revision = cfg.get("hf_revision")
    vec_dir = ROOT / "steering_vectors" / f"{model_key}-{args.track}"

    # --- the pool-family atom lexicon: union of pool1..6's drawn token_ids ---
    arms_report = (ROOT / "results" / "jspace_gate"
                   / f"arms_{model_key}_{args.track}_L{args.layer}_{J24_TAG}.json")
    if not arms_report.exists():
        raise SystemExit(f"REFUSING: J24 arms report missing: {arms_report}")
    arms = json.loads(arms_report.read_text())["arms"]
    pool_tids = sorted({int(t) for i in range(1, 7)
                        for t in arms[f"pool{i}"]["token_ids"]})
    if len(pool_tids) != 264:
        raise SystemExit(f"REFUSING: pool-family lexicon has {len(pool_tids)} "
                         "atoms, expected 264 (6 x 44 disjoint draws)")
    sets["pclean"] = torch.tensor(pool_tids, dtype=torch.long)

    print("loading tokenizer, unembedding shards and lens ...", flush=True)
    tokenizer = dec._load_tokenizer(
        hf_id, revision=revision,
        trust_remote_code=cfg.get("trust_remote_code", False))
    w_u, g = dec._load_unembed_and_norm_shards(hf_id, revision)
    j, lens_prov = dec._load_jlens(args.lens_source, hf_id, args.lens_path)
    j_f = {lyr: j[lyr].float() for lyr in jc.ABLATION_LAYERS}

    # --- v_hat and pool1's w_hat, exactly as the earlier builds ------------------
    vhat = torch.from_numpy(
        np.load(vec_dir / "j24_full_layer16.npy").astype(np.float32))
    vhat = vhat / vhat.norm()
    a_pool1 = torch.from_numpy(
        np.load(vec_dir / "j24_pool1_layer16.npy").astype(np.float32))
    w_hat = (a_pool1 - (a_pool1 @ vhat) * vhat)
    w_hat = w_hat / w_hat.norm()

    def unit_atoms(tids: torch.Tensor, layer: int) -> torch.Tensor:
        a = (w_u[tids].float() * g.float()) @ j_f[layer]
        return a / torch.clamp(a.norm(dim=1, keepdim=True), min=1e-12)

    def proj_sq(basis: np.ndarray, vec: torch.Tensor, r: int) -> float:
        b = torch.from_numpy(basis[:, :r]).float()
        return float((b.T @ vec).pow(2).sum())

    def svd_basis(mat_d_by_n: torch.Tensor, r_store: int):
        u_svd, s_svd, _ = np.linalg.svd(
            mat_d_by_n.numpy().astype(np.float64), full_matrices=False)
        return (u_svd[:, :min(r_store, u_svd.shape[1])].astype(np.float32), s_svd)

    p_dir.mkdir(parents=True, exist_ok=True)
    bases: dict[tuple[str, int], np.ndarray] = {}
    g_bases: dict[int, np.ndarray] = {}
    p_span_dim: dict[str, int] = {}
    surviving: dict[str, dict] = {}
    energy: dict[str, dict] = {}

    for lyr in jc.ABLATION_LAYERS:
        centre = torch.from_numpy(
            np.load(res / f"english_centre_l{lyr}.npy")).float()
        g_basis = np.load(g_dir / f"G_l{lyr}.npy")
        g_bases[lyr] = g_basis
        g_t = torch.from_numpy(g_basis).float()

        def atoms_d(tids: torch.Tensor, layer: int = lyr,
                    cen: torch.Tensor = centre) -> torch.Tensor:
            return (unit_atoms(tids, layer) - cen).T         # (d, n) columns

        # reused cleaned V/H bases (for report tables and overlaps only)
        for name in REUSED_SETS:
            bases[(name, lyr)] = np.load(
                res / REUSED_SETS[name] / f"{name}_l{lyr}.npy")

        # cleaned atom matrices: A* = (I - P_G) A
        cleaned: dict[str, torch.Tensor] = {}
        a_orig: dict[str, torch.Tensor] = {}
        for name in NEW_SETS:
            a0 = atoms_d(sets[name])
            a_orig[name] = a0
            cleaned[name] = a0 - g_t @ (g_t.T @ a0)

        # 1. P_clean: top-r SVD of the cleaned pool-atom matrix; ALSO an
        # orthonormal basis for the FULL cleaned atom span (control target).
        basis, s_p = svd_basis(cleaned["pclean"], jc.N_TOKENS)
        bases[("pclean", lyr)] = basis
        np.save(p_dir / f"pclean_l{lyr}.npy", basis)
        tol = s_p[0] * max(cleaned["pclean"].shape) * np.finfo(s_p.dtype).eps
        p_span = torch.from_numpy(
            svd_basis(cleaned["pclean"], cleaned["pclean"].shape[1])[0][
                :, :int((s_p > tol).sum())]).float()
        p_span_dim[str(lyr)] = int(p_span.shape[1])
        surviving.setdefault("pclean", {})[str(lyr)] = {
            "after_G": float(cleaned["pclean"].norm() ** 2
                             / max(float(a_orig["pclean"].norm() ** 2), 1e-30))}
        energy.setdefault("pclean", {})[str(lyr)] = {
            str(r): float((s_p[:r] ** 2).sum()
                          / max((s_p ** 2).sum(), 1e-30))
            for r in jc.RANKS}

        # 2. controls: ortho against the P_clean cleaned atom span -----------
        for name in ORTHO_SETS:
            a2 = cleaned[name] - p_span @ (p_span.T @ cleaned[name])
            basis, s_c = svd_basis(a2, jc.N_TOKENS)
            bases[(name, lyr)] = basis
            np.save(p_dir / f"{name}_l{lyr}.npy", basis)
            denom = max(float(a_orig[name].norm() ** 2), 1e-30)
            surviving.setdefault(name, {})[str(lyr)] = {
                "after_G": float(cleaned[name].norm() ** 2 / denom),
                "after_G_and_Portho": float(a2.norm() ** 2 / denom)}
            energy.setdefault(name, {})[str(lyr)] = {
                str(r): float((s_c[:r] ** 2).sum()
                              / max((s_c ** 2).sum(), 1e-30))
                for r in jc.RANKS}

    # --- acceptance + reported metrics -----------------------------------------
    def table(layer: int) -> dict:
        out_t = {}
        for name in ALL_REPORT_SETS:
            b_s = bases[(name, layer)]
            out_t[name] = {
                "w_hat_share": {str(r): proj_sq(b_s, w_hat, r)
                                for r in jc.RANKS},
                "P_vhat_sq": {str(r): proj_sq(b_s, vhat, r)
                              for r in jc.RANKS},
            }
        return out_t

    verify = {str(lyr): table(lyr) for lyr in SANITY_LAYERS}

    # how much of w_hat CAN sit inside any G-orthogonal subspace (the cap on
    # the acceptance metric — recorded, no pass/fail)
    w_in_g = {str(lyr): float(
        (torch.from_numpy(g_bases[lyr]).float().T @ w_hat).pow(2).sum())
        for lyr in SANITY_LAYERS}

    def overlap(name_a: str, name_b: str, layer: int, r: int) -> float:
        ba = torch.from_numpy(bases[(name_a, layer)][:, :r]).float()
        bb = torch.from_numpy(bases[(name_b, layer)][:, :r]).float()
        return float((ba.T @ bb).norm() ** 2 / r)

    def overlap_g(name: str, layer: int, r: int) -> float:
        bg = torch.from_numpy(g_bases[layer]).float()
        bs = torch.from_numpy(bases[(name, layer)][:, :r]).float()
        return float((bg.T @ bs).norm() ** 2 / r)

    overlaps = {}
    for lyr in SANITY_LAYERS:
        entry = {"G-vs-sets": {name: {str(r): overlap_g(name, lyr, r)
                                      for r in jc.RANKS}
                               for name in ALL_REPORT_SETS}}
        for i, na in enumerate(ALL_REPORT_SETS):
            for nb in ALL_REPORT_SETS[i + 1:]:
                entry[f"{na}-{nb}"] = {str(r): overlap(na, nb, lyr, r)
                                       for r in jc.RANKS}
        overlaps[str(lyr)] = entry

    failures = []
    print(f"\namendment-6 acceptance, L{jc.LAYER}:")
    w_p = verify[str(jc.LAYER)]["pclean"]["w_hat_share"]["32"]
    print(f"  w_hat inside P_clean at r=32: {w_p:.3f}  (needs >= "
          f"{W_IN_PCLEAN_MIN}; hard cap 1 - w_hat-in-G = "
          f"{1 - w_in_g[str(jc.LAYER)]:.3f})")
    if w_p < W_IN_PCLEAN_MIN:
        failures.append(f"P_clean w_hat share {w_p:.3f} < {W_IN_PCLEAN_MIN}")
    for name in ORTHO_SETS:
        w_c = verify[str(jc.LAYER)][name]["w_hat_share"]["32"]
        ok = w_c < W_IN_CONTROL_MAX
        print(f"  w_hat inside ortho {name} at r=32: {w_c:.3f}  "
              f"(needs < {W_IN_CONTROL_MAX})  {'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"{name} w_hat share {w_c:.3f} >= {W_IN_CONTROL_MAX}")

    print("\n||P v_hat||^2 per set (r = 32 / 64 / 121), L16 then L26:")
    for lyr in SANITY_LAYERS:
        for name in ALL_REPORT_SETS:
            t = verify[str(lyr)][name]
            print(f"  L{lyr} {name:7s} " + "  ".join(
                f"{t['P_vhat_sq'][str(r)]:.3f}" for r in jc.RANKS)
                + "   w_hat " + "  ".join(
                    f"{t['w_hat_share'][str(r)]:.3f}" for r in jc.RANKS))

    # token_sets.json gains the pclean lexicon (refuses on a conflicting one)
    pclean_entry = {"token_ids": pool_tids,
                    "texts": [tokenizer.decode([t]) for t in pool_tids]}
    if "pclean" in token_sets and token_sets["pclean"]["token_ids"] != pool_tids:
        raise SystemExit("REFUSING: token_sets.json already has a different "
                         "'pclean' entry")
    if "pclean" not in token_sets:
        token_sets["pclean"] = pclean_entry
        (res / "token_sets.json").write_text(json.dumps(token_sets, indent=2))
        print(f"token_sets.json: added 'pclean' ({len(pool_tids)} atoms)")

    shas = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(p_dir.glob("*.npy"))}
    report = json.loads((res / "build_report.json").read_text())
    report["amendment6_pclean_subspaces"] = {
        "amendment": "J26-J28 amendment 6, 2026-11-11: P_clean — the subspace "
                     "where pool's content lives; controls ortho'd against "
                     "the P_clean cleaned atom span; V_clean/H reused from "
                     "subspaces_clean/",
        "output_dir": str(p_dir),
        "pclean_lexicon": {"source": str(arms_report),
                           "construction": "union of pool1..6 token_ids "
                                           "(6 x 44 disjoint J24 draws)",
                           "n_atoms": len(pool_tids)},
        "ortho_target": "full column span of the G-projected pool-atom "
                        "matrix (per-layer dim in p_clean_span_dim)",
        "p_clean_span_dim": p_span_dim,
        "reused_bases": {n: str(res / d) for n, d in REUSED_SETS.items()},
        "w_hat_share_inside_G": w_in_g,
        "acceptance": {"w_hat_in_P_clean_r32": {"value": w_p,
                                                "min": W_IN_PCLEAN_MIN,
                                                "pass": w_p >= W_IN_PCLEAN_MIN},
                       "controls_r32": {n: {"value": verify[str(jc.LAYER)][n]["w_hat_share"]["32"],
                                            "max": W_IN_CONTROL_MAX,
                                            "pass": verify[str(jc.LAYER)][n]["w_hat_share"]["32"] < W_IN_CONTROL_MAX}
                                        for n in ORTHO_SETS},
                       "failures": failures},
        "verification": verify,
        "overlaps_G2_over_r": overlaps,
        "surviving_variance": surviving,
        "energy_fraction_final_atoms": energy,
        "lens": lens_prov,
        "subspace_sha256": shas,
    }
    (res / "build_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nsubspaces -> {p_dir}  (pclean + ortho'd c, r1-r3)")
    print("report section amendment6_pclean_subspaces -> "
          f"{res / 'build_report.json'}")
    if failures:
        print("\nACCEPTANCE FAILED — reporting numbers and stopping. "
              "Do not adjust silently:")
        for f_ in failures:
            print(" ", f_)
        return 1
    print("\nacceptance verified: w_hat in P_clean >= 0.60 and in every "
          "orthogonalised control < 0.30 at r=32.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""J27 amendment 4 (2026-11-11): control subspaces orthogonalised against V.

Pre-registered in plans/j-space-decomposition.md ("J26–J28 amendment 4").
At each layer l in 16..26, the C and R1-R3 bases are computed on atom matrices
projected onto the orthogonal complement of V's atom span:

    A'_S = (I - P_{A_V}) A_S,   P_{A_V} = B_V B_V^T,

where B_V is the stored basis results/j27/subspaces/V_l{l}.npy (d,121), which
spans V's centred atom span. The top-121 left singular vectors of A'_S are the
new basis, written to results/j27/subspaces_ortho/{set}_l{l}.npy. V and H are
UNCHANGED (H's V-overlap stays a stated limitation); the original
subspaces/ files are never touched.

Token ids come from results/j27/token_sets.json (the draw is frozen); atoms
are unit((W_U[t] . g) J_l) minus the cached English-word mean unit atom
(results/j27/english_centre_l*.npy) — the same construction as
build_j27_ablations.py.

Registered predictions verified at build (L16, r=32):
  * share of pool1's w_hat inside each orthogonalised control < 0.30;
  * ||P_l v_hat||^2 inside each orthogonalised control drops below the
    pre-fix R floor (~0.064-0.075) toward the structural floor;
  * share of V's basis inside each orthogonalised control ~ 0 (validates
    the projection), equivalently the V-control subspace overlap ||G||^2/r.
If the first two fail, the numbers are reported and the script EXITS NONZERO:
do not adjust silently.

    uv run python scripts/build_j27_ortho_controls.py
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
ORTHO_SETS = ("C", "R1", "R2", "R3")
W_SHARE_PREDICTION = 0.30          # registered: w_hat share must drop below this
R_FLOOR = 0.075                    # pre-fix R-set ||P v_hat||^2 ceiling (amendment 4 table)
SANITY_LAYERS = (jc.LAYER, 26)     # verify at L16 fully, L26 for sanity


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model-config", type=Path,
                   default=ROOT / "configs" / "model" / "qwen25_7b.yaml")
    p.add_argument("--track", default="story-wheel32")
    p.add_argument("--results", type=Path, default=jc.RESULTS)
    p.add_argument("--lens-source", default="neuronpedia",
                   choices=["neuronpedia", "logit"])
    p.add_argument("--lens-path", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    res = args.results
    ortho_dir = res / "subspaces_ortho"
    planned = [ortho_dir / f"{s}_l{lyr}.npy" for s in ORTHO_SETS
               for lyr in jc.ABLATION_LAYERS]
    if not args.overwrite:
        for f in planned:
            if f.exists():
                raise SystemExit(f"REFUSING to overwrite {f} (--overwrite to force)")
    for s in ORTHO_SETS + ("V",):
        for lyr in jc.ABLATION_LAYERS:
            for f in (res / "subspaces" / f"{s}_l{lyr}.npy",
                      res / f"english_centre_l{lyr}.npy"):
                if not f.exists():
                    raise SystemExit(f"REFUSING: required build artifact missing: {f}")

    token_sets = json.loads((res / "token_sets.json").read_text())
    sets = {s: torch.tensor(token_sets[s]["token_ids"], dtype=torch.long)
            for s in ORTHO_SETS}

    dec = jc._load("dec", "scripts/decompose_emotion_vectors.py")
    cfg = dec._load_model_config(args.model_config)
    model_key, hf_id = cfg["model_key"], cfg["hf_model_id"]
    vec_dir = ROOT / "steering_vectors" / f"{model_key}-{args.track}"

    print("loading unembedding shards and lens ...", flush=True)
    w_u, g = dec._load_unembed_and_norm_shards(hf_id, cfg.get("hf_revision"))
    j, lens_prov = dec._load_jlens(args.lens_source, hf_id, args.lens_path)
    j_f = {lyr: j[lyr].float() for lyr in jc.ABLATION_LAYERS}

    def unit_atoms(tids: torch.Tensor, layer: int) -> torch.Tensor:
        a = (w_u[tids].float() * g.float()) @ j_f[layer]
        return a / torch.clamp(a.norm(dim=1, keepdim=True), min=1e-12)

    # v_hat and pool1's w_hat, exactly as build_j27_ablations.py
    vhat = torch.from_numpy(
        np.load(vec_dir / "j24_full_layer16.npy").astype(np.float32))
    vhat = vhat / vhat.norm()
    a_pool1 = torch.from_numpy(
        np.load(vec_dir / "j24_pool1_layer16.npy").astype(np.float32))
    w_hat = (a_pool1 - (a_pool1 @ vhat) * vhat)
    w_hat = w_hat / w_hat.norm()

    def proj_sq(basis: np.ndarray, vec: torch.Tensor, r: int) -> float:
        b = torch.from_numpy(basis[:, :r]).float()
        return float((b.T @ vec).pow(2).sum())

    ortho_dir.mkdir(parents=True, exist_ok=True)
    bases: dict[tuple[str, int], np.ndarray] = {}
    energy: dict[str, dict] = {}
    surviving: dict[str, dict] = {}
    for lyr in jc.ABLATION_LAYERS:
        b_v = np.load(res / "subspaces" / f"V_l{lyr}.npy")     # (d,121), orthonormal
        p_v = torch.from_numpy(b_v).float()
        centre = torch.from_numpy(
            np.load(res / f"english_centre_l{lyr}.npy")).float()
        for name, tids in sets.items():
            a = (unit_atoms(tids, lyr) - centre).T            # (d, n) atoms as columns
            a_proj = a - p_v @ (p_v.T @ a)                    # (I - P_{A_V}) A_S
            u_svd, s_svd, _ = np.linalg.svd(
                a_proj.numpy().astype(np.float64), full_matrices=False)
            r_store = min(jc.N_TOKENS, u_svd.shape[1])
            basis = u_svd[:, :r_store].astype(np.float32)
            np.save(ortho_dir / f"{name}_l{lyr}.npy", basis)
            bases[(name, lyr)] = basis
            energy.setdefault(name, {})[str(lyr)] = {
                str(r): float((s_svd[:r] ** 2).sum() / max((s_svd ** 2).sum(), 1e-30))
                for r in jc.RANKS}
            surviving.setdefault(name, {})[str(lyr)] = float(
                (a_proj.norm() ** 2) / max(a.norm() ** 2, 1e-30))

    # --- registered verification ---------------------------------------------
    def table(layer: int) -> dict:
        b_v = np.load(res / "subspaces" / f"V_l{layer}.npy")
        out_t = {}
        for name in ORTHO_SETS:
            b_o = bases[(name, layer)]
            gram = torch.from_numpy(b_v).float().T @ torch.from_numpy(b_o).float()
            out_t[name] = {
                "w_hat_share": {str(r): proj_sq(b_o, w_hat, r) for r in jc.RANKS},
                "P_vhat_sq": {str(r): proj_sq(b_o, vhat, r) for r in jc.RANKS},
                "V_basis_share": {str(r): float(
                    gram[:, :r].norm() ** 2 / jc.N_TOKENS) for r in jc.RANKS},
                "overlap_G2_over_r": {str(r): float(
                    gram[:, :r].norm() ** 2 / r) for r in jc.RANKS},
            }
        return out_t

    verify = {str(lyr): table(lyr) for lyr in SANITY_LAYERS}

    print(f"\namendment-4 verification, L{jc.LAYER} (r = 32 / 64 / 121):")
    failures = []
    for name in ORTHO_SETS:
        t = verify[str(jc.LAYER)][name]
        print(f"  {name:3s} w_hat share "
              + "  ".join(f"{t['w_hat_share'][str(r)]:.3f}" for r in jc.RANKS)
              + "   ||P v_hat||^2 "
              + "  ".join(f"{t['P_vhat_sq'][str(r)]:.3f}" for r in jc.RANKS)
              + "   V-basis share "
              + "  ".join(f"{t['V_basis_share'][str(r)]:.4f}" for r in jc.RANKS))
        if t["w_hat_share"]["32"] >= W_SHARE_PREDICTION:
            failures.append(f"{name}: w_hat share at r=32 is "
                            f"{t['w_hat_share']['32']:.3f} >= {W_SHARE_PREDICTION}")
        if t["P_vhat_sq"]["32"] >= R_FLOOR:
            failures.append(f"{name}: ||P v_hat||^2 at r=32 is "
                            f"{t['P_vhat_sq']['32']:.3f} >= R floor {R_FLOOR}")
    print("  L26 sanity:")
    for name in ORTHO_SETS:
        t = verify["26"][name]
        print(f"  {name:3s} w_hat share "
              + "  ".join(f"{t['w_hat_share'][str(r)]:.3f}" for r in jc.RANKS)
              + "   ||P v_hat||^2 "
              + "  ".join(f"{t['P_vhat_sq'][str(r)]:.3f}" for r in jc.RANKS))

    # --- report ---------------------------------------------------------------
    shas = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(ortho_dir.glob("*.npy"))}
    section = {
        "amendment": "J26-J28 amendment 4, 2026-11-11: C/R bases built on "
                     "A'_S = (I - P_{A_V}) A_S at each layer; V and H unchanged "
                     "and still loaded from subspaces/",
        "output_dir": str(ortho_dir),
        "construction": {"projection": "A' = A - B_V (B_V^T A), B_V = "
                                       "subspaces/V_l{l}.npy (orthonormal)",
                         "svd": "top-121 left singular vectors of A'"},
        "verification_layers": list(SANITY_LAYERS),
        "verification": verify,
        "energy_fraction_of_projected_atoms": energy,
        "original_variance_surviving_projection": surviving,
        "subspace_sha256": shas,
        "predictions": {"w_hat_share_below": W_SHARE_PREDICTION,
                        "P_vhat_sq_below": R_FLOOR,
                        "failures": failures},
    }
    rp = res / "build_report.json"
    report = json.loads(rp.read_text())
    report["amendment4_orthogonalised_controls"] = section
    rp.write_text(json.dumps(report, indent=2))
    print(f"\nsubspaces -> {ortho_dir}")
    print(f"report section amendment4_orthogonalised_controls -> {rp}")
    if failures:
        print("\nPREDICTIONS FAILED — reporting numbers and stopping. "
              "Do not adjust silently:")
        for f_ in failures:
            print(" ", f_)
        return 1
    print("\nregistered predictions verified: w_hat share < 0.30 and "
          "||P v_hat||^2 below the pre-fix R floor for every control set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

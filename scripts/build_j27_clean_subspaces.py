"""J27 amendment 5 (2026-11-11): shared English-structure component G removed.

Pre-registered in plans/j-space-decomposition.md ("J26–J28 amendment 5");
thresholds are frozen and no further tuning follows this build.

Construction, per layer l in 16..26:

1. G_l: top 32 left singular vectors of the centred atom matrix of a
   2,000-token random English-word set — English-word tokens excluding the
   hostility lexicon and V's 121 tokens; draw from SeedSequence(20261111),
   one spawned child. G's captured variance is reported.
2. Cleaning: A* = (I - P_G) A for every set's centred atom matrix.
3. V and H: top-121 left singular vectors of A*.
4. C and R1-R3: orthogonalise A* against the CLEANED V atom span (the
   amendment-4 rule applied to cleaned atoms; the cleaned V basis spans
   col(A*_V)), then top-121 left singular vectors.

Outputs: results/j27/subspaces_clean/{set}_l{l}.npy for ALL six sets — all
runs now load from this directory. subspaces/ and subspaces_ortho/ are kept
for the record and never touched.

Acceptance, frozen in the amendment (this script exits nonzero on failure —
report, do not adjust):
  * pool1 w_hat inside cleaned V at r=32 >= 0.60;
  * pool1 w_hat inside each orthogonalised control at r=32 < 0.30.
Reported with no pass/fail: ||P_l v_hat||^2 per set at r in {32,64,121};
share of each set's original variance surviving both projections; V-G and
set-set overlaps; w_hat shares at r=64/121 and at L26.

    uv run python scripts/build_j27_clean_subspaces.py
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
ALL_SETS = ("V", "H", "C", "R1", "R2", "R3")
ORTHO_SETS = ("C", "R1", "R2", "R3")
G_SEED = 20261111
G_TOKENS = 2000
G_RANK = 32
W_IN_CLEANED_V_MIN = 0.60          # frozen acceptance: below -> STOP and report
W_IN_CONTROL_MAX = 0.30            # frozen acceptance: at/above -> STOP
SANITY_LAYERS = (jc.LAYER, 26)


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
    clean_dir = res / "subspaces_clean"
    g_dir = res / "subspaces_clean" / "_G"
    planned = [clean_dir / f"{s}_l{lyr}.npy" for s in ALL_SETS
               for lyr in jc.ABLATION_LAYERS]
    planned += [g_dir / f"G_l{lyr}.npy" for lyr in jc.ABLATION_LAYERS]
    if not args.overwrite:
        for f in planned:
            if f.exists():
                raise SystemExit(f"REFUSING to overwrite {f} (--overwrite to force)")
    for lyr in jc.ABLATION_LAYERS:
        f = res / f"english_centre_l{lyr}.npy"
        if not f.exists():
            raise SystemExit(f"REFUSING: required build artifact missing: {f}")

    dec = jc._load("dec", "scripts/decompose_emotion_vectors.py")
    sc = jc._load("sc", "scripts/score_j22_diagnostic.py")
    lex = sc.WORDS | sc.J23_EXTRA_WORDS
    token_sets = json.loads((res / "token_sets.json").read_text())
    sets = {s: torch.tensor(token_sets[s]["token_ids"], dtype=torch.long)
            for s in ALL_SETS}

    cfg = dec._load_model_config(args.model_config)
    model_key, hf_id = cfg["model_key"], cfg["hf_model_id"]
    revision = cfg.get("hf_revision")
    vec_dir = ROOT / "steering_vectors" / f"{model_key}-{args.track}"

    print("loading tokenizer, unembedding shards and lens ...", flush=True)
    tokenizer = dec._load_tokenizer(
        hf_id, revision=revision,
        trust_remote_code=cfg.get("trust_remote_code", False))
    w_u, g = dec._load_unembed_and_norm_shards(hf_id, revision)
    j, lens_prov = dec._load_jlens(args.lens_source, hf_id, args.lens_path)
    j_f = {lyr: j[lyr].float() for lyr in jc.ABLATION_LAYERS}

    # --- masks for the G eligibility (English-word & ~hostile & ~V) ------------
    vocab = w_u.shape[0]
    print(f"building hostility/English masks over {vocab} tokens ...", flush=True)
    hostile = torch.zeros(vocab, dtype=torch.bool)
    english = torch.zeros(vocab, dtype=torch.bool)
    for t in range(vocab):
        try:
            s = tokenizer.decode([t])
        except Exception:
            continue
        if sc.lexicon_hits(sc.words(s), lex)[0] > 0:
            hostile[t] = True
        if t < len(tokenizer) and ENGLISH_RE.fullmatch(s):
            english[t] = True
    in_v = torch.zeros(vocab, dtype=torch.bool)
    in_v[sets["V"]] = True
    g_elig = torch.nonzero(english & ~hostile & ~in_v).flatten()
    print(f"G eligibility: {g_elig.numel()} tokens "
          f"(English & ~hostile & ~V121)", flush=True)
    if g_elig.numel() < G_TOKENS:
        raise SystemExit(f"REFUSING: G eligibility {g_elig.numel()} < {G_TOKENS}")
    rng = np.random.default_rng(np.random.SeedSequence(G_SEED).spawn(1)[0])
    g_tids = g_elig[torch.from_numpy(
        np.sort(rng.choice(g_elig.numel(), size=G_TOKENS, replace=False)))]

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

    clean_dir.mkdir(parents=True, exist_ok=True)
    g_dir.mkdir(parents=True, exist_ok=True)
    bases: dict[tuple[str, int], np.ndarray] = {}
    g_bases: dict[int, np.ndarray] = {}
    g_energy: dict[str, float] = {}
    surviving: dict[str, dict] = {}
    energy: dict[str, dict] = {}

    for lyr in jc.ABLATION_LAYERS:
        centre = torch.from_numpy(
            np.load(res / f"english_centre_l{lyr}.npy")).float()

        def atoms_d(tids: torch.Tensor, layer: int = lyr,
                    cen: torch.Tensor = centre) -> torch.Tensor:
            return (unit_atoms(tids, layer) - cen).T         # (d, n) columns

        # 1. G at this layer ------------------------------------------------
        a_g = atoms_d(g_tids)
        g_basis, g_s = svd_basis(a_g, G_RANK)
        g_basis_t = torch.from_numpy(g_basis).float()
        g_bases[lyr] = g_basis
        # variance the top-32 G directions capture of the 2,000-atom matrix
        g_energy[str(lyr)] = float((g_s[:G_RANK] ** 2).sum() / (a_g.norm() ** 2))
        np.save(g_dir / f"G_l{lyr}.npy", g_basis)

        # 2-3. cleaned atoms and cleaned V basis (needed before the controls)
        cleaned: dict[str, torch.Tensor] = {}
        a_orig: dict[str, torch.Tensor] = {}
        for name, tids in sets.items():
            a0 = atoms_d(tids)
            a_orig[name] = a0
            cleaned[name] = a0 - g_basis_t @ (g_basis_t.T @ a0)
        for name in ("V", "H"):
            basis, s_vh = svd_basis(cleaned[name], jc.N_TOKENS)
            bases[(name, lyr)] = basis
            np.save(clean_dir / f"{name}_l{lyr}.npy", basis)
            surviving.setdefault(name, {})[str(lyr)] = {
                "after_G": float(cleaned[name].norm() ** 2
                                 / max(float(a_orig[name].norm() ** 2), 1e-30))}
            energy.setdefault(name, {})[str(lyr)] = {
                str(r): float((s_vh[:r] ** 2).sum()
                              / max((s_vh ** 2).sum(), 1e-30))
                for r in jc.RANKS}
        b_vc = torch.from_numpy(bases[("V", lyr)]).float()   # cleaned V span

        # 4. controls: ortho against cleaned V span, on cleaned atoms ---------
        for name in ORTHO_SETS:
            a2 = cleaned[name] - b_vc @ (b_vc.T @ cleaned[name])
            basis, s_c = svd_basis(a2, jc.N_TOKENS)
            bases[(name, lyr)] = basis
            np.save(clean_dir / f"{name}_l{lyr}.npy", basis)
            denom = max(float(a_orig[name].norm() ** 2), 1e-30)
            surviving.setdefault(name, {})[str(lyr)] = {
                "after_G": float(cleaned[name].norm() ** 2 / denom),
                "after_G_and_Vortho": float(a2.norm() ** 2 / denom)}
            energy.setdefault(name, {})[str(lyr)] = {
                str(r): float((s_c[:r] ** 2).sum()
                              / max((s_c ** 2).sum(), 1e-30))
                for r in jc.RANKS}

    # --- acceptance + reported metrics -----------------------------------------
    def table(layer: int) -> dict:
        out_t = {}
        for name in ALL_SETS:
            b_s = bases[(name, layer)]
            out_t[name] = {
                "w_hat_share": {str(r): proj_sq(b_s, w_hat, r)
                                for r in jc.RANKS},
                "P_vhat_sq": {str(r): proj_sq(b_s, vhat, r)
                              for r in jc.RANKS},
            }
        return out_t

    verify = {str(lyr): table(lyr) for lyr in SANITY_LAYERS}

    # overlaps at L16: V-G and every set pair, ||B_a^T B_b||_F^2 / r at r=32
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
                                      for r in jc.RANKS} for name in ALL_SETS}}
        for i, na in enumerate(ALL_SETS):
            for nb in ALL_SETS[i + 1:]:
                entry[f"{na}-{nb}"] = {str(r): overlap(na, nb, lyr, r)
                                       for r in jc.RANKS}
        overlaps[str(lyr)] = entry

    failures = []
    print(f"\namendment-5 acceptance, L{jc.LAYER}:")
    w_v = verify[str(jc.LAYER)]["V"]["w_hat_share"]["32"]
    print(f"  w_hat inside cleaned V at r=32: {w_v:.3f}  (needs >= "
          f"{W_IN_CLEANED_V_MIN})")
    if w_v < W_IN_CLEANED_V_MIN:
        failures.append(f"cleaned V w_hat share {w_v:.3f} < {W_IN_CLEANED_V_MIN}")
    for name in ORTHO_SETS:
        w_c = verify[str(jc.LAYER)][name]["w_hat_share"]["32"]
        ok = w_c < W_IN_CONTROL_MAX
        print(f"  w_hat inside ortho {name} at r=32: {w_c:.3f}  "
              f"(needs < {W_IN_CONTROL_MAX})  {'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"{name} w_hat share {w_c:.3f} >= {W_IN_CONTROL_MAX}")

    print("\n||P v_hat||^2 per set (r = 32 / 64 / 121), L16 then L26:")
    for lyr in SANITY_LAYERS:
        for name in ALL_SETS:
            t = verify[str(lyr)][name]
            print(f"  L{lyr} {name:3s} " + "  ".join(
                f"{t['P_vhat_sq'][str(r)]:.3f}" for r in jc.RANKS)
                + "   w_hat " + "  ".join(
                    f"{t['w_hat_share'][str(r)]:.3f}" for r in jc.RANKS))

    shas = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(clean_dir.glob("*.npy"))}
    shas.update({f"G/{f.name}": hashlib.sha256(f.read_bytes()).hexdigest()
                 for f in sorted(g_dir.glob("*.npy"))})
    report = json.loads((res / "build_report.json").read_text())
    report["amendment5_cleaned_subspaces"] = {
        "amendment": "J26-J28 amendment 5, 2026-11-11: G removed from every "
                     "set; controls ortho'd against cleaned V; all runs use "
                     "subspaces_clean/",
        "output_dir": str(clean_dir),
        "G": {"tokens": G_TOKENS, "rank": G_RANK, "seed": G_SEED,
              "eligibility": "English-word & ~hostile & ~V121",
              "eligible": int(g_elig.numel()),
              "token_ids": g_tids.tolist(),
              "variance_captured_at_r32": g_energy},
        "acceptance": {"w_hat_in_cleaned_V_r32": {"value": w_v,
                                                  "min": W_IN_CLEANED_V_MIN,
                                                  "pass": w_v >= W_IN_CLEANED_V_MIN},
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
    print(f"\nsubspaces -> {clean_dir}  (all sets incl. cleaned V, H)")
    print(f"G bases -> {g_dir}")
    print("report section amendment5_cleaned_subspaces -> "
          f"{res / 'build_report.json'}")
    if failures:
        print("\nACCEPTANCE FAILED — reporting numbers and stopping. "
              "Do not adjust silently:")
        for f_ in failures:
            print(" ", f_)
        return 1
    print("\nacceptance verified: w_hat in cleaned V >= 0.60 and in every "
          "orthogonalised control < 0.30 at r=32.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

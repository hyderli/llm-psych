"""Build H8 arm vectors for a steering mixture, as drop-in .npy files.

Writes each arm into the standard
``steering_vectors/<model_key>-<track>/<name>_layer<L>.npy`` layout, so an
existing steering eval loads an arm exactly the way it loads an emotion vector
and its output format is unchanged. No harness work is needed.

The mixture is ``sum(coef_i * v_i)``, e.g. contempt + aggressiveness. Both signs
are built, because the pursuit only ADDS atoms: decomposing -(v1+v2) is a
genuinely different problem from decomposing +(v1+v2), and the two give
different atom sets. Scale is not a separate problem -- the pursuit is
scale-equivariant for a positive multiplier, so one decomposition per sign
covers every |alpha|; the arms are simply rescaled.

Arms emitted per sign, all rescaled to ||alpha * (v1+v2)|| so dose is held
constant and only direction varies:

  <tag>_full      the mixture itself (reproduces the run you already have)
  <tag>_jspace    its J-lens component     -- the verbalizable part
  <tag>_resid     its orthogonal residual  -- the non-verbalizable part
  <tag>_randatom  the same NUMBER of atoms drawn at random from the candidate
                  pool and NNLS-fitted to the mixture. NOTE what this does and
                  does not control. The candidate pool is the top-n_candidates
                  tokens BY LENS LOGIT OF THIS VECTOR, so these atoms are drawn
                  from the words the emotion most promotes, and the fit aims them
                  at the emotion vector. It controls for "was the greedy
                  selection special"; it does NOT control for "does the emotion
                  content matter" -- a near-synonym reconstruction of the same
                  vector is expected to behave like it.
  <tag>_faratom   the missing control: the same number of atoms drawn from the
                  MIDDLE of the lens-logit ranking -- tokens with no systematic
                  relation to this vector -- built identically and NNLS-fitted to
                  the same target. Same construction, same dictionary, same dose,
                  semantically unrelated. Not the bottom of the ranking: those
                  are the atoms of the OPPOSITE direction, which is a meaningful
                  direction rather than a neutral one. The report records how
                  well each control actually reconstructs v (cos and norm ratio),
                  because a control that cannot approximate the target is
                  answering a different question from one that can.
  <tag>_lad<NN>   the mixture rotated NN degrees away from itself, perpendicular
                  to v and to the atoms this mixture actually used;
                  <tag>_ladstar uses theta = angle(v_jspace, v)

J-weight sweep (see plans/j-space-decomposition.md, J6'). Injecting a component
ALONE forces it to full dose, which over-drives a minority-share direction and
destroys generation for reasons unrelated to what the direction represents. The
sweep instead holds the residual at native strength and varies only the weight
of the J-component on top of it:

  <tag>_jw<C>     r + C * v_j, then unit-normalised. C=0 is _resid and C=1 is
                  _full, so those two runs double as sweep anchors.
  <tag>_jwr<C>    r + C * ||v_j|| * randatom_hat -- the SAME J-share with atoms
                  not selected for the emotion. Required control, not optional:
                  J-lens atoms are unembedding rows, so a J-component arm always
                  has more output-layer leverage than a non-J arm regardless of
                  semantics.

Orthogonality holds by construction here: the pursuit re-solves exact NNLS over
the active set each iteration, so component and residual are orthogonal and
||v||^2 = ||v_j||^2 + ||r||^2. The script asserts it rather than assuming it.

Example
-------
::

    uv run python scripts/build_arm_vectors.py \
        --model-config configs/model/gemma-2-9b-it.yaml \
        --track story-wheel32 --layer 30 --alpha 0.3 \
        --mix contempt=1 aggressiveness=1 --tag ca
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

_repo_root = Path(__file__).resolve().parents[1]
LADDER_DEG = [10, 20, 40, 60, 80]
N_LADDER_DRAWS = 3
SEED = 20260924


def _ctag(c: float) -> str:
    """Filename-safe spelling of a sweep weight: 2 -> '2', 0.5 -> '0p5'."""
    return f"{c:g}".replace("-", "m").replace(".", "p")


def _load_mod():
    spec = importlib.util.spec_from_file_location(
        "dec", _repo_root / "scripts" / "decompose_emotion_vectors.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument("--track", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--alpha", type=float, required=True,
                   help="magnitude every arm is rescaled to, as |alpha| * ||v1+v2||")
    p.add_argument("--mix", nargs="+", required=True, metavar="NAME=COEF")
    p.add_argument("--tag", default="mix", help="filename prefix for the arms")
    p.add_argument("--unit-normalise", action="store_true",
                   help="unit-normalise each source vector BEFORE mixing. Off by "
                        "default so the default reproduces a raw equal-coefficient "
                        "run. The report prints both raw norms either way: if they "
                        "differ, equal coefficients were not equal treatment.")
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--n-candidates", type=int, default=2048)
    p.add_argument("--lens-source", default="neuronpedia", choices=["neuronpedia", "logit"])
    p.add_argument("--lens-path", type=Path, default=None)
    p.add_argument("--signs", nargs="+", default=["pos", "neg"], choices=["pos", "neg"])
    p.add_argument("--jsweep", nargs="*", type=float,
                   default=[2.0, 3.0, 5.0, 7.0],
                   help="C values for the J-weight sweep r + C*v_j. C=0 and C=1 are "
                        "already emitted as _resid and _full. Pass --jsweep with no "
                        "values to skip the sweep.")
    p.add_argument("--no-ladder", action="store_true")
    return p.parse_args()


def _perp(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Remove from x everything lying in the row space of basis."""
    q, _ = torch.linalg.qr(basis.T)
    return x - q @ (q.T @ x)


def main() -> int:
    args = _parse_args()
    mod = _load_mod()
    rng = np.random.default_rng(SEED)

    cfg = mod._load_model_config(args.model_config)
    model_key, hf_id = cfg["model_key"], cfg["hf_model_id"]
    revision = cfg.get("hf_revision")
    vec_dir = _repo_root / "steering_vectors" / f"{model_key}-{args.track}"

    report: dict = {"model_key": model_key, "track": args.track, "layer": args.layer,
                    "alpha": args.alpha, "k": args.k,
                    "unit_normalised": bool(args.unit_normalise), "sources": {}}

    # --- mixture -----------------------------------------------------------
    mix, raws = None, {}
    for spec in args.mix:
        name, _, c = spec.partition("=")
        coef = float(c)
        path = vec_dir / f"{name}_layer{args.layer}.npy"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        v = np.load(path).astype(np.float64)
        raws[name] = v
        n = float(np.linalg.norm(v))
        if args.unit_normalise:
            v = v / n
        report["sources"][name] = {"coef": coef, "raw_norm": n,
                                   "contribution_norm": float(coef * np.linalg.norm(v))}
        mix = coef * v if mix is None else mix + coef * v

    if len(raws) == 2:
        a, b = raws.values()
        report["cos_between_sources"] = float(
            a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    base_norm = float(np.linalg.norm(mix))
    target_norm = abs(args.alpha) * base_norm
    report["mixture_norm"] = base_norm
    report["target_norm"] = target_norm

    # --- lens + unembedding (reusing the frozen loaders) --------------------
    print("loading tokenizer, unembedding shards and lens ...")
    tokenizer = mod._load_tokenizer(hf_id, revision=revision,
                                    trust_remote_code=cfg.get("trust_remote_code", False))
    w_u, g = mod._load_unembed_and_norm_shards(hf_id, revision)
    j, lens_prov = mod._load_jlens(args.lens_source, hf_id, args.lens_path)
    mapped = mod._map_layer(args.layer, set(j.keys()) if j else None, "nearest")
    if mapped is None:
        raise SystemExit(f"layer {args.layer} not covered by the lens")
    j_l = j[mapped] if j is not None else None
    report["lens"] = {"source": args.lens_source, "layer_used": mapped,
                      "provenance": lens_prov}

    out_dir = vec_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def emit(name: str, x: torch.Tensor) -> None:
        x = x / x.norm() * target_norm
        np.save(out_dir / f"{name}_layer{args.layer}.npy",
                x.numpy().astype(np.float32))

    # --- one decomposition per sign ----------------------------------------
    for sgn in args.signs:
        s = 1.0 if sgn == "pos" else -1.0
        # The sign is baked into the vector and sign=+1 is passed, so `h` is
        # exactly the vector that gets injected and no sign convention is
        # applied on the way out.
        u = (s * mix).astype(np.float32)
        comp, resid, m, atoms, picked = mod._decompose_vector(
            u, sign=+1, j_l=j_l, w_u=w_u, g=g, tokenizer=tokenizer,
            k=args.k, n_candidates=args.n_candidates, return_atoms=True)

        tot = m["frac_norm_squared"] + m["frac_residual_squared"]
        assert abs(tot - 1.0) < 1e-4, f"{sgn}: fractions sum to {tot}, pieces not orthogonal"

        theta_star = float(np.degrees(np.arccos(
            np.clip(m["frac_norm_squared"] ** 0.5, -1.0, 1.0))))
        report[sgn] = {
            "frac_jspace": m["frac_norm_squared"],
            "frac_residual": m["frac_residual_squared"],
            "frac_sum": tot,
            "cos_component_residual": m["cos_component_residual"],
            "n_atoms": m["n_atoms"],
            "capped": m["n_atoms"] >= args.k,
            "theta_jspace_deg": theta_star,
            "theta_residual_deg": float(np.degrees(np.arccos(
                np.clip(m["frac_residual_squared"] ** 0.5, -1.0, 1.0)))),
            "top_tokens": m["tokens"][:12],
        }

        tag = f"{args.tag}_{sgn}"
        uu = torch.from_numpy(u)
        emit(f"{tag}_full", uu)
        emit(f"{tag}_jspace", torch.from_numpy(comp))
        emit(f"{tag}_resid", torch.from_numpy(resid))

        # random-atom control: same count, fitted to the same target
        n_pick = max(m["n_atoms"], 1)
        pool = [i for i in range(atoms.shape[0]) if i not in set(picked)]
        idx = rng.choice(len(pool), size=min(n_pick, len(pool)), replace=False)
        active = atoms[[pool[i] for i in idx]].T
        rand_v = active @ mod._nnls_active(active, uu)
        emit(f"{tag}_randatom", rand_v)
        report[sgn]["randatom_n"] = int(len(idx))

        def _fit_report(name: str, fit: torch.Tensor) -> None:
            nrm = float(fit.norm())
            report[sgn][f"{name}_cos_with_v"] = float(
                (fit @ uu) / (nrm * float(uu.norm()) + 1e-12))
            report[sgn][f"{name}_norm_ratio"] = nrm / float(uu.norm())

        _fit_report("randatom", rand_v)

        # --- far-pool control: lens atoms unrelated to this vector -----------
        zc = (torch.from_numpy(u).float() @ j_l.T) if j_l is not None \
            else torch.from_numpy(u).float()
        lens_logits = (zc * g) @ w_u.T
        order = torch.argsort(lens_logits, descending=True)
        mid = order[order.numel() // 4: 3 * order.numel() // 4]
        far_ids = mid[torch.from_numpy(
            rng.choice(mid.numel(), size=n_pick, replace=False))]
        w_far = w_u[far_ids]
        far = ((w_far * g) @ j_l) if j_l is not None else w_far
        far = far / torch.clamp(far.norm(dim=1, keepdim=True), min=1e-12)
        far_v = far.T @ mod._nnls_active(far.T, uu)
        emit(f"{tag}_faratom", far_v)
        report[sgn]["faratom_n"] = int(n_pick)
        _fit_report("faratom", far_v)

        # --- J-weight sweep: residual at native strength, J-component scaled --
        # emit() unit-normalises then rescales to target_norm, and the eval
        # unit-normalises again, so dose is constant across the sweep and only
        # the J-component's SHARE of the injected vector varies.
        if args.jsweep:
            comp_t = torch.from_numpy(comp)
            resid_t = torch.from_numpy(resid)
            rand_hat = rand_v / rand_v.norm()
            far_hat = far_v / torch.clamp(far_v.norm(), min=1e-12)
            comp_norm = comp_t.norm()
            fj = float(m["frac_norm_squared"])
            fr = float(m["frac_residual_squared"])
            sweep: dict = {}
            for c in args.jsweep:
                ct = _ctag(c)
                emit(f"{tag}_jw{ct}", resid_t + c * comp_t)
                # matched J-share controls: same second-term norm, other atoms.
                # jwr = atoms from the emotion's own top tokens (selection control)
                # jwf = atoms unrelated to the emotion   (semantics control)
                emit(f"{tag}_jwr{ct}", resid_t + (c * comp_norm) * rand_hat)
                emit(f"{tag}_jwf{ct}", resid_t + (c * comp_norm) * far_hat)
                denom = fr + c * c * fj
                sweep[ct] = {
                    "c": c,
                    "jshare_of_squared_norm": (c * c * fj) / denom,
                    "angle_from_v_deg": float(np.degrees(np.arccos(
                        np.clip((fr + c * fj) / denom ** 0.5, -1.0, 1.0)))),
                }
            report[sgn]["jsweep"] = sweep

        if not args.no_ladder:
            used = atoms[picked] if picked else atoms[:0]
            basis = torch.cat([uu.unsqueeze(0), used], dim=0)
            for deg in LADDER_DEG + ["star"]:
                th = np.radians(theta_star if deg == "star" else float(deg))
                acc = torch.zeros_like(uu)
                for _ in range(N_LADDER_DRAWS):
                    w = torch.from_numpy(rng.normal(size=uu.shape).astype(np.float32))
                    w = _perp(w, basis)
                    w = w / w.norm()
                    acc += np.cos(th) * (uu / uu.norm()) + np.sin(th) * w
                emit(f"{tag}_lad{deg}", acc / N_LADDER_DRAWS)
            report[sgn]["ladder"] = {"degrees": LADDER_DEG + ["star"],
                                     "draws": N_LADDER_DRAWS}

    rp = (_repo_root / "results" / "jspace_gate"
          / f"arms_{model_key}_{args.track}_L{args.layer}_{args.tag}.json")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\narm vectors -> {out_dir}")
    print(f"report      -> {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""One-off check that the digit-projection refactor changed no numbers.

Run once on a machine with the project venv, then delete. It compares the
patched, cached implementation against the original inline one (reproduced
verbatim below) on random data, and demonstrates the correctness fix.

    uv run python verify_digit_projection_fix.py

Expects three things:

1. patched == original to floating-point tolerance, for every returned field;
2. the layer cache is populated once and reused, not recomputed per call;
3. ``v_fraction`` is sign-invariant while ``residual_fraction`` is not — which
   is precisely the bug: the old call site wrote the +v result into the -v
   metrics block, so every -v ``residual_fraction`` on disk is a duplicate of
   its +v counterpart rather than a measurement of the -v residual.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from decompose_emotion_vectors import (  # noqa: E402
    _digit_atoms_for_layer,
    _digit_projection_fractions,
)


def original(v, residual, j_l, w_u, digit_ids):
    """The pre-patch implementation, verbatim apart from taking digit_ids directly."""
    j_l = j_l.float()
    w_u = w_u.float()
    if not digit_ids:
        return {"v_fraction": 0.0, "residual_fraction": 0.0, "n_digit_atoms": 0}
    logits = j_l @ w_u.T
    top_ids = torch.argmax(logits, dim=-1).tolist()
    mask = torch.tensor([tid in digit_ids for tid in top_ids], dtype=torch.bool)
    n = int(mask.sum().item())
    if n == 0:
        return {"v_fraction": 0.0, "residual_fraction": 0.0, "n_digit_atoms": 0}
    digit_atoms = j_l[mask].T
    out = {}
    for name, arr in (("v", v), ("residual", residual)):
        vt = torch.from_numpy(np.ascontiguousarray(arr)).float()
        try:
            c = torch.linalg.lstsq(digit_atoms, vt, rcond=None).solution
        except RuntimeError:
            c = torch.zeros(digit_atoms.shape[1], dtype=torch.float32)
        proj = digit_atoms @ c
        out[f"{name}_fraction"] = float((proj.norm() ** 2) / (vt.norm() ** 2 + 1e-12))
    out["n_digit_atoms"] = n
    return out


def main() -> int:
    rng = np.random.default_rng(11)
    d, vocab, n_atoms = 128, 800, 128
    j_l = torch.from_numpy(rng.normal(size=(n_atoms, d))).float()
    w_u = torch.from_numpy(rng.normal(size=(vocab, d))).float()
    digit_ids = {int(x) for x in rng.choice(vocab, 60, replace=False)}

    # --- 1. equivalence -----------------------------------------------------
    cache: dict[int, tuple[torch.Tensor | None, int]] = {}
    worst = 0.0
    for _ in range(6):
        v = rng.normal(size=d)
        resid = rng.normal(size=d)
        atoms, n = _digit_atoms_for_layer(7, j_l, w_u, digit_ids, cache)
        new = _digit_projection_fractions(v, resid, atoms, n)
        ref = original(v, resid, j_l, w_u, digit_ids)
        assert set(new) == set(ref), (sorted(new), sorted(ref))
        for key in ref:
            worst = max(worst, abs(float(new[key]) - float(ref[key])))
    print(f"1. max |patched - original| over all fields : {worst:.3e}")

    # --- 2. the cache actually caches --------------------------------------
    print(f"2. cache entries after 6 calls at one layer : {len(cache)} (expect 1)")
    atoms_a, _ = _digit_atoms_for_layer(7, j_l, w_u, digit_ids, cache)
    atoms_b, _ = _digit_atoms_for_layer(7, j_l, w_u, digit_ids, cache)
    print(f"   same tensor object returned?             : {atoms_a is atoms_b}")

    # --- 3. the correctness fix --------------------------------------------
    v = rng.normal(size=d)
    resid_pos = rng.normal(size=d)
    resid_neg = rng.normal(size=d)
    atoms, n = _digit_atoms_for_layer(7, j_l, w_u, digit_ids, cache)
    dp_pos = _digit_projection_fractions(v, resid_pos, atoms, n)
    dp_neg = _digit_projection_fractions(-v, resid_neg, atoms, n)
    dv = abs(dp_pos["v_fraction"] - dp_neg["v_fraction"])
    print(f"3. v_fraction sign-invariant (diff)         : {dv:.3e}")
    print(
        f"   residual_fraction  +v={dp_pos['residual_fraction']:.4f}  "
        f"-v={dp_neg['residual_fraction']:.4f}"
    )

    ok = worst < 1e-6 and len(cache) == 1 and atoms_a is atoms_b and dv < 1e-6
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

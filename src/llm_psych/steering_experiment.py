"""Steering-sweep primitives: strength calibration, controls, condition grid.

Reusable across behavioral tasks (reward hacking now; blackmail/
sycophancy later — see ``docs/methods.md``). Model-agnostic and
GPU-free: everything here is pure NumPy/PyTorch given activations
already captured, or a residual-stream norm already measured. The
actual model load + generation loop lives in the calling script (e.g.
``scripts/run_reward_hacking_steering.py``).

Convention (paper, reward hacking / sycophancy): steering "strength"
is a signed fraction of the residual-stream norm at the injection
layer, swept in **-0.1 ... +0.1** (``docs/methods_original.md``
§"Reward hacking"). This module converts that strength into the exact
vector ``ResidualStreamSteerer`` should add (at ``alpha=1.0``), for the
target direction and for the three non-negotiable controls (zero,
norm-matched random, norm-matched orthogonal) per ``docs/methods.md``
§"Controls (non-negotiable)".

Deviation from ``docs/methods.md``'s "probe-orthogonal" control: no
probe was fit for the vector sets this module was built for
(``desperate``/``calm``, story or wheel32 tracks — see
``plans/reward-hacking-steering.md``). The orthogonal basis here is
the set of steering-axis vectors actually in play (``target_vector``
plus any ``contrast_vectors`` the caller passes, e.g. ``calm`` when
sweeping ``desperate``), not a full probe-weight basis. Disclosed in
every condition's metadata via ``control_type="orthogonal"`` plus the
caller's own run_meta logging.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import torch

from llm_psych.hooks import ResidualStreamRecorder
from llm_psych.steering import project_out

DEFAULT_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)  # methods.md: 5 seeds, random control


# ---------------------------------------------------------------------------
# Residual-stream norm calibration
# ---------------------------------------------------------------------------

def mean_residual_norm(
    model,
    tokenizer,
    prompts: list[str],
    layer: int,
    max_length: int = 64,
) -> float:
    """Mean L2 norm of the residual stream at ``layer``, over every
    non-padding token position across ``prompts``.

    Used to calibrate a signed "fraction of residual-stream norm"
    steering strength into an absolute vector magnitude. Run once per
    ``(model, layer)`` on a neutral prompt set (methods.md convention)
    — e.g. the 50 neutral rows of ``data/public/emotion_prompts.parquet``.
    """
    enc = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}

    with ResidualStreamRecorder(
        model, layers=[layer], token_position="all", dtype=torch.float32
    ) as rec:
        with torch.no_grad():
            model(**enc)
        hidden = rec.activations[layer]  # (batch, seq, hidden), cpu, float32

    mask = enc["attention_mask"].to(hidden.device).float()  # (batch, seq)
    norms = hidden.norm(dim=-1)  # (batch, seq)
    total = (norms * mask).sum()
    count = mask.sum()
    if count.item() == 0:
        raise ValueError("attention_mask is all-zero; no tokens to average over")
    return float((total / count).item())


# ---------------------------------------------------------------------------
# Control-vector construction
# ---------------------------------------------------------------------------

def scaled_target_vector(
    target_vector: np.ndarray, strength: float, mean_residual_norm_: float
) -> np.ndarray:
    """``target_vector`` rescaled to norm ``|strength| * mean_residual_norm_``.

    Sign of ``strength`` flips the direction — negative strength steers
    *away* from the target emotion (the paper's "suppression" arm).
    """
    unit = target_vector / (np.linalg.norm(target_vector) + 1e-8)
    return (unit * strength * mean_residual_norm_).astype(np.float32)


def make_random_vector(dim: int, target_norm: float, seed: int) -> np.ndarray:
    """A ``~N(0,I)`` direction, seeded, rescaled to ``target_norm``."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v * (target_norm / (np.linalg.norm(v) + 1e-8))


def make_orthogonal_vector(
    dim: int, target_norm: float, basis_vectors: list[np.ndarray], seed: int
) -> np.ndarray:
    """A random direction projected orthogonal to ``basis_vectors``, rescaled
    to ``target_norm``.

    Raises
    ------
    ValueError
        If the projection collapses near zero (the random draw landed
        almost entirely inside the basis's span — vanishingly unlikely
        for a 1-2-vector basis in a high-dimensional residual stream,
        but checked rather than silently returning a near-null vector).
    """
    raw = make_random_vector(dim, 1.0, seed)
    # project_out requires a truly orthonormal basis (mutually orthogonal
    # rows), not just individually unit-normed ones -- QR gives that for
    # whatever subspace basis_vectors spans, even if they weren't
    # orthogonal to each other to begin with.
    basis = np.stack(basis_vectors).astype(np.float64).T  # (dim, k)
    q, _ = np.linalg.qr(basis)  # columns orthonormal, same column space
    orthonormal_basis = q.T  # (k, dim), rows orthonormal
    projected = project_out(raw, orthonormal_basis)
    proj_norm = np.linalg.norm(projected)
    if proj_norm < 1e-6:
        raise ValueError(
            "orthogonal projection collapsed to ~0 -- basis unexpectedly "
            "spans (nearly) the whole space"
        )
    return (projected * (target_norm / proj_norm)).astype(np.float32)


# ---------------------------------------------------------------------------
# Condition grid
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class SteeringCondition:
    """One cell of the sweep. ``vector`` is the exact delta to add at
    ``alpha=1.0`` (i.e. already scaled) — pass straight to
    ``ResidualStreamSteerer(model, layer=..., vector=torch.from_numpy(
    condition.vector), alpha=1.0)``.
    """

    control_type: str  # "target" | "zero" | "random" | "orthogonal"
    strength: float  # signed fraction of residual-stream norm; 0.0 for "zero"
    vector: np.ndarray
    seed: int | None = None


def build_condition_grid(
    target_vector: np.ndarray,
    contrast_vectors: list[np.ndarray],
    strengths: list[float],
    mean_residual_norm_: float,
    *,
    control_types: tuple[str, ...] = ("target", "zero", "random", "orthogonal"),
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> list[SteeringCondition]:
    """Build the full condition grid for one steering axis.

    One ``target`` condition per (nonzero) strength; a single shared
    ``zero`` condition; ``random``/``orthogonal`` controls at each
    nonzero strength's *magnitude*, aggregated over ``seeds`` (methods.md:
    5 seeds for the random control — applied here to orthogonal too,
    for the same reason: any single draw is one realization).

    Parameters
    ----------
    target_vector
        The steering direction being swept (e.g. ``desperate``).
    contrast_vectors
        Other steering-axis vectors run in the same experiment (e.g.
        ``[calm]``), used only to build the ``orthogonal`` control's
        basis together with ``target_vector``.
    strengths
        Signed fractions of residual-stream norm, e.g.
        ``[-0.1, -0.05, 0.0, 0.05, 0.1]``.
    mean_residual_norm_
        From :func:`mean_residual_norm`, same model + layer.
    """
    dim = target_vector.shape[0]
    conditions: list[SteeringCondition] = []

    if "zero" in control_types:
        conditions.append(SteeringCondition("zero", 0.0, np.zeros(dim, dtype=np.float32)))

    for strength in strengths:
        if strength == 0.0:
            continue  # covered by the shared zero condition above
        magnitude = abs(strength) * mean_residual_norm_

        if "target" in control_types:
            conditions.append(
                SteeringCondition(
                    "target", strength, scaled_target_vector(target_vector, strength, mean_residual_norm_)
                )
            )
        if "random" in control_types:
            for seed in seeds:
                conditions.append(
                    SteeringCondition("random", strength, make_random_vector(dim, magnitude, seed), seed)
                )
        if "orthogonal" in control_types:
            basis = [target_vector, *contrast_vectors]
            for seed in seeds:
                conditions.append(
                    SteeringCondition(
                        "orthogonal", strength, make_orthogonal_vector(dim, magnitude, basis, seed), seed
                    )
                )

    return conditions


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion (default ``z`` for 95%).

    Used for the hack rate per condition — the same treatment as the
    blackmail rate in ``docs/methods.md``.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z**2 / n
    center = p + z**2 / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((center - half) / denom, (center + half) / denom)

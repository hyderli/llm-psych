"""The J19 orthogonal-plane construction must do what its name claims.

Two arms at the same angle from v, differing only in which perpendicular they
use. If the realised angle drifts, or the in-span arm is not actually in the
span, every statement the run produces is void -- which is what happened to
`ladstar` on 2026-10-08, when N_LADDER_DRAWS = 3 silently put it at 60 degrees
instead of 71.4 and all claims about it had to be withdrawn.

These tests use a synthetic span and need no model, no lens and no GPU.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

D, R, FRAC_J = 512, 14, 0.1014
DEGS = [40.0, 50.0, 60.0, 70.0, 75.0, 80.0, 85.0]


def _perp(x, basis):
    q, _ = torch.linalg.qr(basis.T)
    return x - q @ (q.T @ x)


@pytest.fixture(scope="module")
def setup():
    """A v with a known frac_jspace inside a known span, as the builder sees it."""
    rng = np.random.default_rng(11)
    atoms = rng.normal(size=(R, D))
    atoms /= np.linalg.norm(atoms, axis=1, keepdims=True)
    q = np.linalg.qr(atoms.T)[0]
    in_s = q @ rng.normal(size=R)
    in_s /= np.linalg.norm(in_s)
    out_s = rng.normal(size=D)
    out_s -= q @ (q.T @ out_s)
    out_s /= np.linalg.norm(out_s)
    v = np.sqrt(FRAC_J) * in_s + np.sqrt(1 - FRAC_J) * out_s
    comp = q @ (q.T @ v)                      # v_j: the projection onto the span
    return (torch.from_numpy(atoms).float(), torch.from_numpy(v).float(),
            torch.from_numpy(comp).float(), torch.from_numpy(q).float())


def _build(setup, deg, kind, seed):
    atoms, v, comp, q_span = setup
    rng = np.random.default_rng(seed)
    uhat = v / v.norm()
    g = torch.from_numpy(rng.normal(size=(D,)).astype(np.float32))
    if kind == "pin":
        jhat = comp / comp.norm()
        w = q_span @ (q_span.T @ g)
        w = w - jhat * float(jhat @ w)
    else:
        w = _perp(g, torch.cat([uhat.unsqueeze(0), atoms], dim=0))
    w = w / w.norm()
    th = np.radians(deg)
    arm = float(np.cos(th)) * uhat + float(np.sin(th)) * w
    return arm / arm.norm(), w, uhat, q_span


@pytest.mark.parametrize("deg", DEGS)
@pytest.mark.parametrize("kind", ["pin", "pout"])
def test_perpendicular_is_perpendicular(setup, deg, kind):
    _, w, uhat, _ = _build(setup, deg, kind, seed=int(deg) * 10)
    assert abs(float(w @ uhat)) < 1e-4


@pytest.mark.parametrize("deg", DEGS)
@pytest.mark.parametrize("kind", ["pin", "pout"])
def test_realised_angle_equals_nominal(setup, deg, kind):
    arm, _, uhat, _ = _build(setup, deg, kind, seed=int(deg) * 10)
    got = float(np.degrees(np.arccos(np.clip(float(arm @ uhat), -1.0, 1.0))))
    assert abs(got - deg) < 0.01, f"{kind}{deg}: realised {got:.4f} deg"


@pytest.mark.parametrize("deg", DEGS)
def test_span_energy_matches_the_arithmetic(setup, deg):
    """in-span = cos^2 * frac_j + sin^2 ; out-of-span = cos^2 * frac_j."""
    th = np.radians(deg)
    for kind, want in (("pin", np.cos(th) ** 2 * FRAC_J + np.sin(th) ** 2),
                       ("pout", np.cos(th) ** 2 * FRAC_J)):
        arm, _, _, q_span = _build(setup, deg, kind, seed=int(deg) * 10)
        got = float((q_span.T @ arm).pow(2).sum())
        assert abs(got - want) < 2e-3, f"{kind}{deg}: {got:.5f} != {want:.5f}"


@pytest.mark.parametrize("deg,least", [(40, 7), (60, 25), (70, 60), (85, 900)])
def test_contrast_grows_with_angle(setup, deg, least):
    """The whole design rests on this ratio being large. 40 deg barely clears it."""
    a_in = _build(setup, float(deg), "pin", seed=deg * 10)[0]
    a_out = _build(setup, float(deg), "pout", seed=deg * 10)[0]
    q = setup[3]
    r = (float((q.T @ a_in).pow(2).sum()) / float((q.T @ a_out).pow(2).sum()))
    assert r > least, f"{deg} deg: contrast only {r:.0f}x"


def test_in_span_arm_stays_inside_the_span_apart_from_v(setup):
    """An in-span rung is cos(th)*v_hat + sin(th)*w with w in S, so everything
    outside S comes from v_hat alone and scales as cos(th)."""
    _, v, _, q_span = setup
    uhat = v / v.norm()
    out_of_v = float((uhat - q_span @ (q_span.T @ uhat)).norm())
    for deg in DEGS:
        arm = _build(setup, deg, "pin", seed=int(deg) * 10)[0]
        resid = float((arm - q_span @ (q_span.T @ arm)).norm())
        assert abs(resid - np.cos(np.radians(deg)) * out_of_v) < 2e-3

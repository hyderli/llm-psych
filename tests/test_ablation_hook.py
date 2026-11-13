"""Unit tests for ``AblationHook`` / ``MultiAblationHook`` (J27).

The hooks live in ``emotion_steering/sycophancy_blackmail/hooks.py``, which is
installed as the ``steering-evals`` package only on the pod; here the module is
loaded by path, as the scripts do.

Tests use the same LLaMA-shaped fake model as ``tests/test_hooks.py``: decoder
blocks are single Linears returning ``(hidden_states, ...)``, so production
hook code paths are exercised on CPU in milliseconds.

The registered J27 ordering — "the steering hook runs first and the ablation
second" at L16 — is tested directly: ``register_forward_hook`` callbacks fire
in registration order, so attaching the SteeringHook first must leave an
ablation that removes the steering delta's in-S component.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


hooks = _load("sb_hooks", "emotion_steering/sycophancy_blackmail/hooks.py")


class _FakeDecoderBlock(nn.Module):
    """Mimics an HF decoder block: returns ``(hidden_states, ...)``."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (self.proj(x),)


class _FakeInnerModel(nn.Module):
    def __init__(self, n_layers: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [_FakeDecoderBlock(hidden_dim) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.layers:
            x = block(x)[0]
        return x


class _FakeCausalLM(nn.Module):
    """``self.model.layers`` is the ModuleList, as on Llama/Qwen."""

    def __init__(self, n_layers: int = 4, hidden_dim: int = 8):
        super().__init__()
        self.model = _FakeInnerModel(n_layers, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@pytest.fixture
def fake_model() -> _FakeCausalLM:
    torch.manual_seed(0)
    m = _FakeCausalLM(n_layers=4, hidden_dim=8).eval()
    return m


@pytest.fixture
def sample_input() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(2, 5, 8)


def _orthonormal(d: int, r: int, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, r, generator=g))
    return q


def _pre_block_output(model: _FakeCausalLM, x: torch.Tensor, layer: int) -> torch.Tensor:
    """The block output at `layer` when no hooks are attached (block input run-up)."""
    with torch.no_grad():
        h = x
        for blk in model.model.layers[: layer + 1]:
            h = blk(h)[0]
    return h


# --------------------------------------------------------------------------
# Core projection semantics
# --------------------------------------------------------------------------

def test_ablation_removes_in_space_component(fake_model, sample_input):
    """After the hook at its layer, ||P_S (h - mu)|| ~ 0 at every position."""
    d = 8
    s = _orthonormal(d, 3)
    mu = torch.randn(d, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        baseline = _pre_block_output(fake_model, sample_input, 1)
        hook = hooks.AblationHook(s, mu).attach(fake_model.model.layers[1])

        captured = {}
        def cap(_m, _i, out):
            captured["h"] = (out[0] if isinstance(out, tuple) else out).clone()
        probe = fake_model.model.layers[1].register_forward_hook(cap)
        try:
            fake_model(sample_input)
        finally:
            probe.remove()
            hook.remove()

    dev = captured["h"].float() - mu
    residual = (dev @ s).norm(dim=-1)
    assert float(residual.max()) < 1e-5
    # and the output really is h - P_S(h - mu): out + removed == in
    in_dev = baseline.float() - mu
    expected = baseline - (in_dev @ s) @ s.T
    torch.testing.assert_close(captured["h"], expected, rtol=1e-5, atol=1e-5)


def test_ablation_empty_subspace_is_identity(fake_model, sample_input):
    d = 8
    s = torch.zeros(d, 0)
    mu = torch.randn(d)
    hook = hooks.AblationHook(s, mu).attach(fake_model.model.layers[1])
    try:
        with torch.no_grad():
            baseline = fake_model(sample_input)
            ablated = fake_model(sample_input)
    finally:
        hook.remove()
    torch.testing.assert_close(baseline, ablated)
    assert hook.last_removed_share == 0.0


def test_ablation_removed_share_recorded(fake_model, sample_input):
    d = 8
    s = _orthonormal(d, 3)
    mu = torch.zeros(d)
    hook = hooks.AblationHook(s, mu).attach(fake_model.model.layers[1])
    try:
        with torch.no_grad():
            fake_model(sample_input)
    finally:
        hook.remove()
    assert hook.last_removed_share is not None
    assert 0.0 <= hook.last_removed_share <= 1.0


def test_ablation_rejects_bad_shapes():
    with pytest.raises(ValueError):
        hooks.AblationHook(torch.randn(8), torch.randn(8))
    with pytest.raises(ValueError, match="mu dim"):
        hooks.AblationHook(torch.randn(8, 2), torch.randn(7))


# --------------------------------------------------------------------------
# Ordering: steering first, ablation second at the same layer
# --------------------------------------------------------------------------

def test_steering_then_ablation_removes_steered_component(fake_model, sample_input):
    """With a steering direction inside S, ablation AFTER steering erases the
    delta — the observable signature of the registered order.

    If the ablation ran first, the output would instead equal the
    ablation-only output plus the untouched steering delta inside S.
    """
    d = 8
    s = _orthonormal(d, 3)
    direction_in_s = s[:, 0]                     # steering delta lives inside S
    mu = torch.randn(d, generator=torch.Generator().manual_seed(3))
    alpha = 0.4

    layer = fake_model.model.layers[1]
    steer = hooks.SteeringHook(direction_in_s, alpha, norm_scale=True, site="post")
    ablate = hooks.AblationHook(s, mu)
    steer.attach(layer)                          # registered first: fires first
    ablate.attach(layer)                         # registered second: fires second

    captured = {}
    def cap(_m, _i, out):
        captured["h"] = (out[0] if isinstance(out, tuple) else out).clone()
    probe = layer.register_forward_hook(cap)
    try:
        with torch.no_grad():
            fake_model(sample_input)
    finally:
        probe.remove()
        ablate.remove()
        steer.remove()

    # Expected output under "steer then ablate":
    #   h' = h + alpha * v_hat * ||h||;  out = h' - P_S (h' - mu)
    h0 = _pre_block_output(fake_model, sample_input, 1)
    v = direction_in_s / direction_in_s.norm()
    h_steered = h0 + alpha * v * h0.norm(dim=-1, keepdim=True)
    dev = h_steered - mu
    expected = h_steered - (dev @ s) @ s.T
    torch.testing.assert_close(captured["h"], expected, rtol=1e-5, atol=1e-5)

    # And it is NOT the ablate-then-steer result (which keeps the delta in S):
    dev0 = h0 - mu
    ablate_only = h0 - (dev0 @ s) @ s.T
    wrong_order = ablate_only + alpha * v * h0.norm(dim=-1, keepdim=True)
    assert not torch.allclose(captured["h"], wrong_order, atol=1e-5)


def test_ablation_then_steering_is_the_different_order(fake_model, sample_input):
    """Symmetric check: attaching ablation first gives the OTHER order,
    confirming the ordering claim is about registration order, not layers."""
    d = 8
    s = _orthonormal(d, 3)
    direction_in_s = s[:, 0]
    mu = torch.zeros(d)
    alpha = 0.4

    layer = fake_model.model.layers[1]
    ablate = hooks.AblationHook(s, mu)
    steer = hooks.SteeringHook(direction_in_s, alpha, norm_scale=True, site="post")
    ablate.attach(layer)                         # ablation first this time
    steer.attach(layer)

    captured = {}
    def cap(_m, _i, out):
        captured["h"] = (out[0] if isinstance(out, tuple) else out).clone()
    probe = layer.register_forward_hook(cap)
    try:
        with torch.no_grad():
            fake_model(sample_input)
    finally:
        probe.remove()
        steer.remove()
        ablate.remove()

    h0 = _pre_block_output(fake_model, sample_input, 1)
    v = direction_in_s / direction_in_s.norm()
    dev0 = h0 - mu
    ablate_only = h0 - (dev0 @ s) @ s.T
    # norm_scale=True: the steering hook sees the ABLATED activation, so the
    # delta scale is ||h_ablated||, not ||h0||.
    expected = ablate_only + alpha * v * ablate_only.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(captured["h"], expected, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------
# Multi-layer attach / remove
# --------------------------------------------------------------------------

def test_multi_ablation_attaches_each_layer_with_own_s_and_mu(
        fake_model, sample_input):
    d = 8
    layers = fake_model.model.layers
    specs = [
        (layers[1], _orthonormal(d, 2, seed=11), torch.ones(d)),
        (layers[3], _orthonormal(d, 2, seed=22), -torch.ones(d)),
    ]
    multi = hooks.MultiAblationHook(specs)
    assert multi.attached

    captured = {}
    probes = []
    for li in (1, 3):
        def cap(_m, _i, out, li=li):
            captured[li] = (out[0] if isinstance(out, tuple) else out).clone()
        probes.append(layers[li].register_forward_hook(cap))
    try:
        with torch.no_grad():
            fake_model(sample_input)
    finally:
        for p in probes:
            p.remove()

    for li, (_, s, mu) in zip((1, 3), specs, strict=True):
        dev = captured[li].float() - mu
        assert float((dev @ s).norm(dim=-1).max()) < 1e-5

    multi.remove()
    assert not multi.attached
    # After removal the model is exactly the unhooked model again.
    with torch.no_grad():
        a = fake_model(sample_input)
        b = fake_model(sample_input)
    torch.testing.assert_close(a, b)


def test_multi_ablation_remove_leaves_model_unhooked(fake_model, sample_input):
    d = 8
    layers = fake_model.model.layers
    multi = hooks.MultiAblationHook(
        [(layers[0], _orthonormal(d, 3, seed=5), torch.zeros(d))])
    multi.remove()
    with torch.no_grad():
        a = fake_model(sample_input)
        b = fake_model(sample_input)
    torch.testing.assert_close(a, b)

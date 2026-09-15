"""Tests for `llm_psych.steering_experiment`.

Reuses the fake LLaMA-shaped model from `test_hooks.py` for
`mean_residual_norm` so these run instantly with no HF download. The
control-vector and grid-builder tests are pure NumPy.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from llm_psych.steering_experiment import (
    build_condition_grid,
    make_orthogonal_vector,
    make_random_vector,
    mean_residual_norm,
    scaled_target_vector,
    wilson_ci,
)

# --------------------------------------------------------------------------
# Fixtures — mirrors tests/test_hooks.py's fake model, plus a fake tokenizer
# --------------------------------------------------------------------------

class _FakeDecoderBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (self.proj(x),)


class _FakeInnerModel(nn.Module):
    def __init__(self, n_layers: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([_FakeDecoderBlock(hidden_dim) for _ in range(n_layers)])


class _FakeCausalLM(nn.Module):
    """LLaMA-shaped fake that also accepts `input_ids`/`attention_mask`
    kwargs (as a real HF model does) via a fixed embedding table."""

    def __init__(self, n_layers: int = 3, hidden_dim: int = 8, vocab_size: int = 20):
        super().__init__()
        self.model = _FakeInnerModel(n_layers, hidden_dim)
        self.embed = nn.Embedding(vocab_size, hidden_dim)

    def forward(self, input_ids=None, attention_mask=None):
        x = self.embed(input_ids)
        for block in self.model.layers:
            x = block(x)[0]
        return x


class _FakeTokenizer:
    """Whitespace tokenizer with left-padding, matching the project's
    real-tokenizer convention (`padding_side="left"`)."""

    pad_token_id = 0

    def __call__(self, prompts, return_tensors="pt", padding=True, truncation=True, max_length=64):
        token_lists = [[hash(w) % 19 + 1 for w in p.split()][:max_length] for p in prompts]
        max_len = max(len(t) for t in token_lists)
        input_ids, attn = [], []
        for t in token_lists:
            pad_n = max_len - len(t)
            input_ids.append([0] * pad_n + t)  # left-pad
            attn.append([0] * pad_n + [1] * len(t))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


@pytest.fixture
def fake_model():
    torch.manual_seed(0)
    m = _FakeCausalLM()
    m.eval()
    return m


@pytest.fixture
def fake_tokenizer():
    return _FakeTokenizer()


# --------------------------------------------------------------------------
# mean_residual_norm
# --------------------------------------------------------------------------

def test_mean_residual_norm_is_positive_and_finite(fake_model, fake_tokenizer):
    prompts = ["the quick brown fox", "a short one", "a much longer sentence here indeed"]
    norm = mean_residual_norm(fake_model, fake_tokenizer, prompts, layer=1)
    assert norm > 0
    assert np.isfinite(norm)


def test_mean_residual_norm_excludes_padding(fake_model, fake_tokenizer):
    """Norm computed over real tokens only should differ from (and not
    be diluted toward zero by) a version that wrongly includes padding.
    """
    prompts = ["one", "one two three four five six seven"]
    norm_masked = mean_residual_norm(fake_model, fake_tokenizer, prompts, layer=1)

    # Manually compute the (wrong) all-position mean, padding included,
    # to confirm the two differ -- i.e. masking is actually doing something.
    enc = fake_tokenizer(prompts)
    from llm_psych.hooks import ResidualStreamRecorder

    with ResidualStreamRecorder(fake_model, layers=[1], token_position="all", dtype=torch.float32) as rec:
        with torch.no_grad():
            fake_model(**enc)
    hidden = rec.activations[1]
    norm_unmasked = hidden.norm(dim=-1).mean().item()

    assert norm_masked != pytest.approx(norm_unmasked, rel=1e-6)


def test_mean_residual_norm_empty_mask_raises(fake_model):
    class _AllPadTokenizer(_FakeTokenizer):
        def __call__(self, prompts, **kw):
            n = len(prompts)
            return {
                "input_ids": torch.zeros(n, 3, dtype=torch.long),
                "attention_mask": torch.zeros(n, 3, dtype=torch.long),
            }

    with pytest.raises(ValueError, match="all-zero"):
        mean_residual_norm(fake_model, _AllPadTokenizer(), ["x", "y"], layer=1)


# --------------------------------------------------------------------------
# scaled_target_vector
# --------------------------------------------------------------------------

def test_scaled_target_vector_norm_and_direction():
    v = np.array([3.0, 4.0], dtype=np.float32)  # norm 5
    out = scaled_target_vector(v, strength=0.1, mean_residual_norm_=50.0)
    assert np.linalg.norm(out) == pytest.approx(5.0, rel=1e-4)  # |0.1|*50
    # same direction as v
    cos = np.dot(out, v) / (np.linalg.norm(out) * np.linalg.norm(v))
    assert cos == pytest.approx(1.0, rel=1e-4)


def test_scaled_target_vector_negative_strength_flips_direction():
    v = np.array([1.0, 0.0], dtype=np.float32)
    out = scaled_target_vector(v, strength=-0.1, mean_residual_norm_=10.0)
    assert np.linalg.norm(out) == pytest.approx(1.0, rel=1e-4)
    assert out[0] < 0


def test_scaled_target_vector_zero_strength_is_zero_vector():
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = scaled_target_vector(v, strength=0.0, mean_residual_norm_=100.0)
    assert np.allclose(out, 0.0)


# --------------------------------------------------------------------------
# make_random_vector / make_orthogonal_vector
# --------------------------------------------------------------------------

def test_make_random_vector_has_target_norm():
    v = make_random_vector(dim=16, target_norm=3.5, seed=0)
    assert v.shape == (16,)
    assert np.linalg.norm(v) == pytest.approx(3.5, rel=1e-5)


def test_make_random_vector_deterministic_per_seed():
    v1 = make_random_vector(16, 1.0, seed=7)
    v2 = make_random_vector(16, 1.0, seed=7)
    v3 = make_random_vector(16, 1.0, seed=8)
    assert np.array_equal(v1, v2)
    assert not np.array_equal(v1, v3)


def test_make_orthogonal_vector_is_orthogonal_to_basis():
    rng = np.random.default_rng(1)
    target = rng.standard_normal(32).astype(np.float32)
    contrast = rng.standard_normal(32).astype(np.float32)
    ortho = make_orthogonal_vector(32, target_norm=2.0, basis_vectors=[target, contrast], seed=3)
    assert np.linalg.norm(ortho) == pytest.approx(2.0, rel=1e-4)
    for basis_vec in (target, contrast):
        cos = np.dot(ortho, basis_vec) / (np.linalg.norm(ortho) * np.linalg.norm(basis_vec))
        assert abs(cos) < 1e-4


def test_make_orthogonal_vector_raises_when_basis_spans_space():
    # A 2-D space with a 2-vector orthogonal basis leaves nothing to project onto.
    basis = [np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)]
    with pytest.raises(ValueError, match="collapsed"):
        make_orthogonal_vector(2, target_norm=1.0, basis_vectors=basis, seed=0)


# --------------------------------------------------------------------------
# build_condition_grid
# --------------------------------------------------------------------------

def test_grid_has_one_shared_zero_condition():
    rng = np.random.default_rng(0)
    target = rng.standard_normal(16).astype(np.float32)
    contrast = rng.standard_normal(16).astype(np.float32)
    grid = build_condition_grid(target, [contrast], strengths=[-0.1, 0.0, 0.1], mean_residual_norm_=10.0)
    zeros = [c for c in grid if c.control_type == "zero"]
    assert len(zeros) == 1
    assert zeros[0].strength == 0.0
    assert np.allclose(zeros[0].vector, 0.0)


def test_grid_counts_match_seeds_and_strengths():
    rng = np.random.default_rng(0)
    target = rng.standard_normal(16).astype(np.float32)
    contrast = rng.standard_normal(16).astype(np.float32)
    strengths = [-0.1, -0.05, 0.0, 0.05, 0.1]
    seeds = (1, 2, 3)
    grid = build_condition_grid(
        target, [contrast], strengths, mean_residual_norm_=10.0, seeds=seeds
    )
    n_nonzero_strengths = 4  # excludes 0.0
    assert sum(c.control_type == "target" for c in grid) == n_nonzero_strengths
    assert sum(c.control_type == "random" for c in grid) == n_nonzero_strengths * len(seeds)
    assert sum(c.control_type == "orthogonal" for c in grid) == n_nonzero_strengths * len(seeds)
    assert sum(c.control_type == "zero" for c in grid) == 1


def test_grid_target_vector_norms_match_strength_magnitude():
    rng = np.random.default_rng(0)
    target = rng.standard_normal(16).astype(np.float32)
    contrast = rng.standard_normal(16).astype(np.float32)
    mean_norm = 20.0
    grid = build_condition_grid(target, [contrast], [0.05, -0.05], mean_residual_norm_=mean_norm)
    targets = [c for c in grid if c.control_type == "target"]
    for c in targets:
        assert np.linalg.norm(c.vector) == pytest.approx(abs(c.strength) * mean_norm, rel=1e-4)


def test_grid_respects_control_types_subset():
    rng = np.random.default_rng(0)
    target = rng.standard_normal(16).astype(np.float32)
    grid = build_condition_grid(
        target, [], [0.1], mean_residual_norm_=10.0, control_types=("target", "zero")
    )
    assert {c.control_type for c in grid} == {"target", "zero"}


# --------------------------------------------------------------------------
# wilson_ci
# --------------------------------------------------------------------------

def test_wilson_ci_zero_n_returns_zero_zero():
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_wilson_ci_brackets_observed_proportion():
    lo, hi = wilson_ci(7, 10)
    assert lo < 0.7 < hi


def test_wilson_ci_narrows_with_more_data():
    lo_small, hi_small = wilson_ci(5, 10)
    lo_big, hi_big = wilson_ci(500, 1000)
    assert (hi_small - lo_small) > (hi_big - lo_big)


def test_wilson_ci_bounds_within_0_1():
    for successes, n in [(0, 10), (10, 10), (1, 3)]:
        lo, hi = wilson_ci(successes, n)
        assert 0.0 <= lo <= hi <= 1.0

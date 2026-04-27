"""Unit tests for `llm_psych.hooks`.

Uses a small handwritten LLaMA-shaped fake model so tests run instantly
on the Mac without downloading HF checkpoints. The fake model has the
same attribute layout (`model.model.layers`) and the same hook contract
(decoder blocks return a tuple `(hidden_states, ...)`) as real
HF transformers decoder LMs, so these tests exercise the production
code paths.

Notes
-----
The fake decoder block is a single Linear without attention. That means
position-specific steering at layer L only affects position N at
layer L; in a real model with attention, attention layers downstream
will mix the steered position into other positions. The tests here
verify the mechanics of where the vector lands; behavior under
attention is a model-level property tested in integration runs.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from llm_psych.hooks import (
    ResidualStreamRecorder,
    ResidualStreamSteerer,
    _get_decoder_layers,
)

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

class _FakeDecoderBlock(nn.Module):
    """Mimics an HF decoder block: returns ``(hidden_states, ...)``."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (self.proj(x),)


class _FakeInnerModel(nn.Module):
    """Stand-in for ``hf_model.model``."""

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
    """LLaMA-shaped fake: ``self.model.layers`` is the ModuleList."""

    def __init__(self, n_layers: int = 4, hidden_dim: int = 8):
        super().__init__()
        self.model = _FakeInnerModel(n_layers, hidden_dim)
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@pytest.fixture
def fake_model() -> _FakeCausalLM:
    torch.manual_seed(0)
    m = _FakeCausalLM(n_layers=4, hidden_dim=8)
    m.eval()
    return m


@pytest.fixture
def sample_input() -> torch.Tensor:
    """Pre-embedded input: (batch, seq_len, hidden_dim)."""
    torch.manual_seed(0)
    return torch.randn(2, 5, 8)


# --------------------------------------------------------------------------
# _get_decoder_layers
# --------------------------------------------------------------------------

def test_get_decoder_layers_finds_llama_style(fake_model):
    layers = _get_decoder_layers(fake_model)
    assert isinstance(layers, nn.ModuleList)
    assert len(layers) == 4


def test_get_decoder_layers_falls_back_to_layers_attr():
    class FlatModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(4, 4)])

    layers = _get_decoder_layers(FlatModel())
    assert len(layers) == 1


def test_get_decoder_layers_raises_on_unknown_arch():
    bad = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="Could not locate"):
        _get_decoder_layers(bad)


# --------------------------------------------------------------------------
# ResidualStreamRecorder
# --------------------------------------------------------------------------

def test_recorder_captures_correct_layers(fake_model, sample_input):
    with ResidualStreamRecorder(fake_model, layers=[1, 3]) as rec:
        fake_model(sample_input)
    assert set(rec.activations.keys()) == {1, 3}


def test_recorder_default_last_token_shape(fake_model, sample_input):
    with ResidualStreamRecorder(fake_model, layers=[2]) as rec:
        fake_model(sample_input)
    # batch=2, hidden_dim=8
    assert rec.activations[2].shape == (2, 8)


def test_recorder_all_tokens_shape(fake_model, sample_input):
    with ResidualStreamRecorder(
        fake_model, layers=[2], token_position="all"
    ) as rec:
        fake_model(sample_input)
    # batch=2, seq_len=5, hidden_dim=8
    assert rec.activations[2].shape == (2, 5, 8)


def test_recorder_int_position(fake_model, sample_input):
    with ResidualStreamRecorder(
        fake_model, layers=[1], token_position=2
    ) as rec:
        fake_model(sample_input)
    assert rec.activations[1].shape == (2, 8)


def test_recorder_slice_position(fake_model, sample_input):
    with ResidualStreamRecorder(
        fake_model, layers=[1], token_position=slice(1, 4)
    ) as rec:
        fake_model(sample_input)
    # batch=2, sliced_seq=3, hidden_dim=8
    assert rec.activations[1].shape == (2, 3, 8)


def test_recorder_default_dtype_is_fp16(fake_model, sample_input):
    with ResidualStreamRecorder(fake_model, layers=[1]) as rec:
        fake_model(sample_input)
    assert rec.activations[1].dtype == torch.float16


def test_recorder_dtype_configurable(fake_model, sample_input):
    with ResidualStreamRecorder(
        fake_model, layers=[1], dtype=torch.float32
    ) as rec:
        fake_model(sample_input)
    assert rec.activations[1].dtype == torch.float32


def test_recorder_clear_drops_activations(fake_model, sample_input):
    with ResidualStreamRecorder(fake_model, layers=[1]) as rec:
        fake_model(sample_input)
        assert 1 in rec.activations
        rec.clear()
        assert rec.activations == {}


def test_recorder_removes_hooks_on_exit(fake_model, sample_input):
    rec = ResidualStreamRecorder(fake_model, layers=[1])
    with rec:
        pass
    assert rec._handles == []
    # Hooks gone — forward pass should leave activations empty.
    fake_model(sample_input)
    assert rec.activations == {}


def test_recorder_removes_hooks_on_exception(fake_model):
    rec = ResidualStreamRecorder(fake_model, layers=[1])
    with pytest.raises(RuntimeError, match="boom"):
        with rec:
            raise RuntimeError("boom")
    assert rec._handles == []


def test_recorder_overwrites_on_repeated_forward(fake_model, sample_input):
    """Documented behavior: each forward pass overwrites the previous."""
    with ResidualStreamRecorder(
        fake_model, layers=[1], token_position="all", dtype=torch.float32
    ) as rec:
        fake_model(sample_input)
        first = rec.activations[1].clone()
        # Different input → different activations
        fake_model(torch.randn(2, 5, 8))
        second = rec.activations[1]
    assert not torch.allclose(first, second)


def test_recorder_invalid_layer_raises(fake_model):
    with pytest.raises(ValueError, match="out of range"):
        ResidualStreamRecorder(fake_model, layers=[10])


def test_recorder_negative_layer_raises(fake_model):
    with pytest.raises(ValueError, match="out of range"):
        ResidualStreamRecorder(fake_model, layers=[-1])


# --------------------------------------------------------------------------
# ResidualStreamSteerer
# --------------------------------------------------------------------------

def test_steerer_zero_vector_is_no_op(fake_model, sample_input):
    """Zero vector ⇒ identical output."""
    with torch.no_grad():
        baseline = fake_model(sample_input)
        with ResidualStreamSteerer(
            fake_model, layer=2, vector=torch.zeros(8), alpha=1.0
        ):
            steered = fake_model(sample_input)
    torch.testing.assert_close(baseline, steered)


def test_steerer_alpha_zero_is_no_op(fake_model, sample_input):
    """alpha=0 ⇒ identical output regardless of vector magnitude."""
    torch.manual_seed(42)
    big_vec = torch.randn(8) * 100.0
    with torch.no_grad():
        baseline = fake_model(sample_input)
        with ResidualStreamSteerer(
            fake_model, layer=1, vector=big_vec, alpha=0.0
        ):
            steered = fake_model(sample_input)
    torch.testing.assert_close(baseline, steered)


def test_steerer_nonzero_vector_changes_output(fake_model, sample_input):
    torch.manual_seed(42)
    vec = torch.randn(8)
    with torch.no_grad():
        baseline = fake_model(sample_input)
        with ResidualStreamSteerer(
            fake_model, layer=1, vector=vec, alpha=1.0
        ):
            steered = fake_model(sample_input)
    assert not torch.allclose(baseline, steered)


def test_steerer_last_position_only_isolates_last(fake_model, sample_input):
    """token_positions='last' should leave non-last positions unchanged.

    Holds in this fake model because there is no attention to propagate
    the change across positions. In a real model with attention, the
    test is about *where the vector is added*, not what attention does
    with it downstream.
    """
    torch.manual_seed(42)
    vec = torch.randn(8) * 2.0
    with torch.no_grad():
        baseline = fake_model(sample_input)
        with ResidualStreamSteerer(
            fake_model,
            layer=3,  # last decoder layer → no further mixing
            vector=vec,
            alpha=1.0,
            token_positions="last",
        ):
            steered = fake_model(sample_input)
    torch.testing.assert_close(
        baseline[:, :-1, :], steered[:, :-1, :], rtol=1e-5, atol=1e-5
    )
    assert not torch.allclose(baseline[:, -1, :], steered[:, -1, :])


def test_steerer_int_position(fake_model, sample_input):
    """Steering at a specific int position should leave others untouched."""
    torch.manual_seed(42)
    vec = torch.randn(8) * 2.0
    target_pos = 2
    with torch.no_grad():
        baseline = fake_model(sample_input)
        with ResidualStreamSteerer(
            fake_model,
            layer=3,
            vector=vec,
            alpha=1.0,
            token_positions=target_pos,
        ):
            steered = fake_model(sample_input)
    # Non-target positions unchanged
    for pos in range(sample_input.shape[1]):
        if pos == target_pos:
            assert not torch.allclose(baseline[:, pos, :], steered[:, pos, :])
        else:
            torch.testing.assert_close(
                baseline[:, pos, :], steered[:, pos, :], rtol=1e-5, atol=1e-5
            )


def test_steerer_invalid_vector_shape(fake_model):
    bad = torch.randn(2, 8)
    with pytest.raises(ValueError, match="must be 1-D"):
        ResidualStreamSteerer(fake_model, layer=0, vector=bad)


def test_steerer_invalid_layer(fake_model):
    with pytest.raises(ValueError, match="out of range"):
        ResidualStreamSteerer(fake_model, layer=10, vector=torch.zeros(8))


def test_steerer_removes_hook_on_exception(fake_model):
    steerer = ResidualStreamSteerer(
        fake_model, layer=1, vector=torch.zeros(8)
    )
    with pytest.raises(RuntimeError, match="boom"):
        with steerer:
            raise RuntimeError("boom")
    assert steerer._handle is None


# --------------------------------------------------------------------------
# Integration: recorder + steerer compose
# --------------------------------------------------------------------------

def test_recorder_observes_steering_effect(fake_model, sample_input):
    """Steering at layer L should be visible at layer L+1 activations."""
    torch.manual_seed(0)
    vec = torch.randn(8)

    with torch.no_grad():
        # Baseline activations at layer 2, no steering.
        with ResidualStreamRecorder(
            fake_model, layers=[2], token_position="last", dtype=torch.float32
        ) as rec:
            fake_model(sample_input)
        baseline_l2 = rec.activations[2]

        # Steered: add vector at layer 1, observe at layer 2.
        with ResidualStreamSteerer(
            fake_model, layer=1, vector=vec, alpha=2.0
        ), ResidualStreamRecorder(
            fake_model, layers=[2], token_position="last", dtype=torch.float32
        ) as rec:
            fake_model(sample_input)
        steered_l2 = rec.activations[2]

    assert not torch.allclose(baseline_l2, steered_l2)

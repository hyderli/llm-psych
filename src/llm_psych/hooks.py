"""Forward-hook utilities for activation recording and steering.

Used to (a) extract residual-stream activations for probe training and
steering-vector derivation, and (b) inject steering vectors into the
residual stream during generation.

Compatible with LLaMA-style decoder-only architectures (Llama 3.x, Qwen
2.5, Mistral, OLMo-2, SmolLM2) where the model exposes
``model.model.layers`` as a ModuleList of decoder blocks.

Notes
-----
``ResidualStreamRecorder`` overwrites stored activations on every
forward pass. This is correct for the primary use case (probe input
extraction on prompts, one forward pass per item). For analysis during
``model.generate``, which runs many forward passes, either capture
``activations`` between explicit single forward passes or extend the
recorder to accumulate.

References
----------
Panickssery et al. 2024. Steering Llama 2 via Contrastive Activation
Addition. https://arxiv.org/abs/2312.06681
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn

# A token-position spec.
#   "last": the final token of the sequence (default for probe input).
#   "all":  every position (used for steering during generation).
#   int:    a specific token index.
#   slice:  a range of tokens (e.g., the last user turn).
TokenPosition = Literal["last", "all"] | int | slice


def _get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Return the decoder block list for a LLaMA-style HF model.

    Parameters
    ----------
    model
        A Hugging Face transformers causal LM.

    Returns
    -------
    nn.ModuleList
        The list of decoder blocks (e.g. ``model.model.layers``).

    Raises
    ------
    ValueError
        If the model architecture does not expose decoder layers at one
        of the supported attribute paths.
    """
    # Standard path: Llama, Qwen, Mistral, OLMo, SmolLM, Gemma, etc.
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # Fallback: model loaded without LM-head wrapper.
    if hasattr(model, "layers"):
        return model.layers
    raise ValueError(
        f"Could not locate decoder layers on {type(model).__name__}. "
        "Expected `model.model.layers` (Llama / Qwen / Mistral / OLMo style)."
    )


def _slice_token_position(
    hidden_states: torch.Tensor, position: TokenPosition
) -> torch.Tensor:
    """Slice ``(batch, seq_len, hidden_dim)`` by token position."""
    if position == "last":
        return hidden_states[:, -1, :]
    if position == "all":
        return hidden_states
    if isinstance(position, (int, slice)):
        return hidden_states[:, position, :]
    raise ValueError(f"Invalid token position: {position!r}")


def _validate_layer(layer: int, n_layers: int) -> None:
    if not isinstance(layer, int) or layer < 0 or layer >= n_layers:
        raise ValueError(
            f"Layer {layer!r} out of range; model has {n_layers} layers "
            "(0-indexed). Negative indices are not supported."
        )


# --------------------------------------------------------------------------
# Recorder
# --------------------------------------------------------------------------

class ResidualStreamRecorder:
    """Capture residual-stream activations from selected layers.

    Use as a context manager. Hooks are attached on ``__enter__`` and
    removed on ``__exit__`` (including on exceptions). After each forward
    pass, ``self.activations`` maps layer index to a captured tensor.

    Parameters
    ----------
    model
        Hugging Face causal LM with LLaMA-style layout.
    layers
        Decoder block indices to record from.
    token_position
        Which token position(s) to keep. Default ``"last"``.
    dtype
        Storage dtype. Default ``torch.float16`` to halve disk/RAM use.
    device
        Storage device for captured activations. Default ``"cpu"`` so
        VRAM is freed.

    Attributes
    ----------
    activations : dict[int, torch.Tensor]
        Per-layer activation from the most recent forward pass. Cleared
        with ``clear()``.

    Examples
    --------
    >>> with ResidualStreamRecorder(model, layers=[16, 24]) as rec:
    ...     model(**tokenized)
    ...     act_l16 = rec.activations[16]  # shape (batch, hidden_dim)
    """

    def __init__(
        self,
        model: nn.Module,
        layers: Sequence[int],
        token_position: TokenPosition = "last",
        dtype: torch.dtype = torch.float16,
        device: str = "cpu",
    ) -> None:
        self._decoder_layers = _get_decoder_layers(model)
        n_layers = len(self._decoder_layers)
        for idx in layers:
            _validate_layer(idx, n_layers)
        self.layers: list[int] = list(layers)
        self.token_position = token_position
        self.dtype = dtype
        self.device = device
        self.activations: dict[int, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            sliced = _slice_token_position(hs, self.token_position)
            self.activations[layer_idx] = sliced.detach().to(
                device=self.device, dtype=self.dtype
            )
        return hook

    def attach(self) -> None:
        """Register hooks on the model. Idempotent."""
        if self._handles:
            return
        for idx in self.layers:
            handle = self._decoder_layers[idx].register_forward_hook(
                self._make_hook(idx)
            )
            self._handles.append(handle)

    def remove(self) -> None:
        """Remove all hooks. Idempotent."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def clear(self) -> None:
        """Clear stored activations from prior forward passes."""
        self.activations = {}

    def __enter__(self) -> "ResidualStreamRecorder":
        self.attach()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove()


# --------------------------------------------------------------------------
# Steerer
# --------------------------------------------------------------------------

class ResidualStreamSteerer:
    """Add a steering vector to the residual stream at one layer.

    Implements Contrastive Activation Addition (CAA) in the sense of
    Panickssery et al. 2024: shifts the residual stream by
    ``alpha * vector`` at the specified layer on every forward pass.

    The class is direction-agnostic — for the project's
    target/random/orthogonal control conditions, construct the
    appropriate ``vector`` *before* passing it in.

    Parameters
    ----------
    model
        Hugging Face causal LM with LLaMA-style layout.
    layer
        Decoder block index where steering is applied.
    vector
        Steering direction, shape ``(hidden_dim,)``.
    alpha
        Scalar multiplier on ``vector``. Project convention (see
        ``docs/methods.md``): set so that ``alpha * ||vector||`` equals
        the mean residual-stream norm at this layer on neutral prompts.
    token_positions
        Where to add the vector. ``"all"`` (default) adds at every
        position during generation; ``"last"`` adds only at the final
        position; ``int`` / ``slice`` for explicit control.

    Examples
    --------
    >>> v_anger = torch.load("steering_vectors/llama31-8b/anger_layer18.pt")
    >>> with ResidualStreamSteerer(model, layer=18, vector=v_anger, alpha=1.4):
    ...     out = model.generate(**tokenized, max_new_tokens=200)
    """

    def __init__(
        self,
        model: nn.Module,
        layer: int,
        vector: torch.Tensor,
        alpha: float = 1.0,
        token_positions: TokenPosition = "all",
    ) -> None:
        decoder_layers = _get_decoder_layers(model)
        _validate_layer(layer, len(decoder_layers))
        if vector.dim() != 1:
            raise ValueError(
                f"vector must be 1-D (hidden_dim,), got shape "
                f"{tuple(vector.shape)}."
            )
        self._decoder_layer = decoder_layers[layer]
        self.layer = layer
        self.vector = vector
        self.alpha = float(alpha)
        self.token_positions = token_positions
        self._handle: torch.utils.hooks.RemovableHandle | None = None
        self._cached_vector: torch.Tensor | None = None

    def _hook(self, _module, _inputs, output):
        hs = output[0] if isinstance(output, tuple) else output

        # Lazily cache the vector on the model's device + dtype. Re-cache
        # if the model migrates (e.g., bf16 → fp32 during a debug run).
        if (
            self._cached_vector is None
            or self._cached_vector.device != hs.device
            or self._cached_vector.dtype != hs.dtype
        ):
            self._cached_vector = self.vector.to(device=hs.device, dtype=hs.dtype)
        delta = self.alpha * self._cached_vector  # (hidden_dim,)

        if self.token_positions == "all":
            hs_new = hs + delta
        else:
            hs_new = hs.clone()
            if self.token_positions == "last":
                hs_new[:, -1, :] = hs_new[:, -1, :] + delta
            elif isinstance(self.token_positions, (int, slice)):
                hs_new[:, self.token_positions, :] = (
                    hs_new[:, self.token_positions, :] + delta
                )
            else:
                raise ValueError(
                    f"Invalid token_positions: {self.token_positions!r}"
                )

        if isinstance(output, tuple):
            return (hs_new,) + output[1:]
        return hs_new

    def attach(self) -> None:
        if self._handle is not None:
            return
        self._handle = self._decoder_layer.register_forward_hook(self._hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "ResidualStreamSteerer":
        self.attach()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove()

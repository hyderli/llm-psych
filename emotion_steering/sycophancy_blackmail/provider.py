"""Steered HF provider — model-agnostic. Registered as `steered` (see _registry.py).

Usage:
    --model steered/<hf-model-id>              # any HF model that needs no template fix
    --model steered/local -M model_path=<dir>  # a local (e.g. template-fixed) model

Steering args (all via -M):
    vectors_dir=<path>        directory of <name>_layer<N>.npy vectors
    terms='[["contempt",1.0],["aggressiveness",1.0]]'   emotions + weights (ratio = mix)
    layer=<int>               decoder layer to inject at
    alpha=<float>             strength (fraction of residual norm when norm_scale=true)
    site=post|pre             injection site (default post)
    norm_scale=true|false     default true
    tokenizer_path=<dir>      pass through to the HF provider for a local tokenizer

Subclasses the HF provider, so chat templating / tool handling / local-model loading
(model_path, tokenizer_path) are all inherited. One persistent residual-stream hook is
attached on first generate() so it can't race Inspect's batched samples.
"""

from __future__ import annotations

import json


def _as_bool(x) -> bool:
    return x if isinstance(x, bool) else str(x).strip().lower() in {"1", "true", "yes", "y"}


def _load_parent():
    try:
        from inspect_ai.model._providers.hf import HuggingFaceAPI
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Could not import inspect_ai.model._providers.hf.HuggingFaceAPI — "
            "the internal path may differ in your inspect_ai version."
        ) from e
    return HuggingFaceAPI


class SteeredAPI(_load_parent()):
    def __init__(
        self,
        model_name: str,
        *,
        vectors_dir: str,
        terms=None,
        layer: int | str = 0,
        alpha: float | str = 0.0,
        site: str = "post",
        norm_scale=True,
        **model_args,
    ):
        super().__init__(model_name=model_name, **model_args)

        from .hooks import SteeringHook, get_decoder_layers
        from .vectors_npy import NpyVectorBank, build_direction

        self._SteeringHook = SteeringHook
        self._get_decoder_layers = get_decoder_layers

        if isinstance(terms, str):
            terms = json.loads(terms)
        self.terms = [(str(n), float(w)) for n, w in (terms or [])]
        self.layer = int(layer)
        self.alpha = float(alpha)
        self.site = str(site)
        self.norm_scale = _as_bool(norm_scale)

        self.active = self.alpha != 0.0 and bool(self.terms)
        self._direction = None
        if self.active:
            bank = NpyVectorBank(vectors_dir)
            self._direction = build_direction(bank, self.terms, self.layer)

        self._hook = None
        self._attached = False

    def _hf_model(self):
        for attr in ("model", "_model"):
            m = getattr(self, attr, None)
            if m is not None and hasattr(m, "forward"):
                return m
        raise RuntimeError("Underlying HF model not found on provider (.model/._model).")

    def _ensure_hook(self) -> None:
        if self._attached or not self.active:
            return
        layers = self._get_decoder_layers(self._hf_model())
        if not (0 <= self.layer < len(layers)):
            raise IndexError(f"layer {self.layer} out of range (model has {len(layers)})")
        self._hook = self._SteeringHook(self._direction, self.alpha, self.norm_scale, self.site)
        self._hook.attach(layers[self.layer])
        self._attached = True

    async def generate(self, *args, **kwargs):
        self._ensure_hook()
        return await super().generate(*args, **kwargs)

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

J27 ablation args (all via -M; off unless ablation_set is given):
    ablation_dir=<path>       results/j27 root: subspaces/<set>_l<L>.npy + mu/mu_l<L>.npy
    ablation_set=V|H|C|R1|R2|R3
    ablation_layers=16-26     range "a-b", or single layer "16", or list "16,17"
    ablation_rank=32|64|121   columns of the stored (d,121) basis used
    ablation_spaces_dir=<path>  basis directory override. Amendment 4: pass
                              results/j27/subspaces_ortho for C/R sets and
                              results/j27/subspaces for V/H — the choice must
                              be explicit in the run command (a wrong-dir run
                              is uninterpretable). Default: <dir>/subspaces.

Hook order at a layer carrying both steering and ablation (the registered J27
order): the SteeringHook is attached FIRST and the AblationHook SECOND, so the
steering delta is added and then the steered activation's in-S component is
removed before the next block reads it. register_forward_hook callbacks fire
in registration order; see hooks.py and tests/test_ablation_hook.py.

Subclasses the HF provider, so chat templating / tool handling / local-model loading
(model_path, tokenizer_path) are all inherited. One persistent residual-stream hook is
attached on first generate() so it can't race Inspect's batched samples.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _as_bool(x) -> bool:
    return x if isinstance(x, bool) else str(x).strip().lower() in {"1", "true", "yes", "y"}


def _parse_layers(spec) -> list[int]:
    """"16-26" -> [16..26];  "16" or 16 -> [16];  "16,17,20" -> [16,17,20]."""
    if spec is None:
        return []
    s = str(spec).strip()
    if "-" in s:
        lo, hi = s.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",") if x.strip()]


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
        vectors_dir: str | None = None,
        terms=None,
        layer: int | str = 0,
        alpha: float | str = 0.0,
        site: str = "post",
        norm_scale=True,
        ablation_dir: str | None = None,
        ablation_set: str | None = None,
        ablation_layers: int | str | None = None,
        ablation_rank: int | str = 121,
        ablation_spaces_dir: str | None = None,
        **model_args,
    ):
        super().__init__(model_name=model_name, **model_args)

        from .hooks import MultiAblationHook, SteeringHook, get_decoder_layers
        from .vectors_npy import NpyVectorBank, build_direction

        self._SteeringHook = SteeringHook
        self._MultiAblationHook = MultiAblationHook
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
            if vectors_dir is None:
                raise ValueError("vectors_dir is required when terms/alpha are set")
            bank = NpyVectorBank(vectors_dir)
            self._direction = build_direction(bank, self.terms, self.layer)

        # --- J27 ablation ---------------------------------------------------
        self.ablation_dir = ablation_dir
        self.ablation_set = ablation_set
        self.ablation_layers = _parse_layers(ablation_layers)
        self.ablation_rank = int(ablation_rank)
        # Amendment 4: the basis directory is an explicit run argument —
        # subspaces_ortho for the orthogonalised C/R controls, subspaces for V/H.
        self.ablation_spaces_dir = (
            Path(ablation_spaces_dir) if ablation_spaces_dir
            else (Path(ablation_dir) / "subspaces" if ablation_dir else None))
        self.ablation_active = bool(ablation_set)
        if self.ablation_active and ablation_dir is None:
            raise ValueError("ablation_dir is required when ablation_set is set")
        if self.ablation_active and not self.ablation_layers:
            raise ValueError("ablation_layers is required when ablation_set is set")

        self._hook = None
        self._ablation = None
        self._attached = False

    def _hf_model(self):
        for attr in ("model", "_model"):
            m = getattr(self, attr, None)
            if m is not None and hasattr(m, "forward"):
                return m
        raise RuntimeError("Underlying HF model not found on provider (.model/._model).")

    def _ensure_hook(self) -> None:
        if self._attached or not (self.active or self.ablation_active):
            return
        layers = self._get_decoder_layers(self._hf_model())
        # ORDER MATTERS (J27 spec): the steering hook is attached FIRST, so at
        # a shared layer the steering delta is applied before the ablation.
        if self.active:
            if not (0 <= self.layer < len(layers)):
                raise IndexError(
                    f"layer {self.layer} out of range (model has {len(layers)})")
            self._hook = self._SteeringHook(
                self._direction, self.alpha, self.norm_scale, self.site)
            self._hook.attach(layers[self.layer])
        if self.ablation_active:
            specs = [
                (layers[lyr], *self._load_ablation(lyr))
                for lyr in self.ablation_layers
            ]
            self._ablation = self._MultiAblationHook(specs)
        self._attached = True

    def _load_ablation(self, layer: int):
        """(subspace[:, :rank], mu) for one layer from the J27 build files."""
        s = np.load(
            self.ablation_spaces_dir / f"{self.ablation_set}_l{layer}.npy")
        mu = np.load(Path(self.ablation_dir) / "mu" / f"mu_l{layer}.npy")
        r = min(self.ablation_rank, s.shape[1])
        return s[:, :r], mu

    async def generate(self, *args, **kwargs):
        self._ensure_hook()
        return await super().generate(*args, **kwargs)

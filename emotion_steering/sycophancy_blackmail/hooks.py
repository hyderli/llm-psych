"""Residual-stream steering via forward hooks. Model-agnostic. No Inspect dependency.

INJECTION SITE — read before trusting results
---------------------------------------------
Vectors must be injected where they were extracted.
  site="post"  add at the OUTPUT of decoder block L  (register_forward_hook)
  site="pre"   add at the INPUT of decoder block L    (register_forward_pre_hook)
Default "post". Getting it wrong fails SILENTLY — always confirm with find_alpha.py
that steering visibly changes generation; if nothing changes at any alpha, try --site pre.

norm_scale=True adds alpha * direction * ||h_token|| (alpha = fraction of residual
norm), which makes alpha portable across layers and models. Recommended default.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


def get_decoder_layers(model) -> torch.nn.ModuleList:
    """Find decoder layers across common HF architectures."""
    candidates = [
        lambda m: m.model.language_model.layers,   # multimodal wrappers (Gemma3 etc.)
        lambda m: m.language_model.model.layers,
        lambda m: m.model.layers,                  # Llama / Qwen / Gemma / Mistral / ...
        lambda m: m.model.model.layers,
        lambda m: m.transformer.h,                 # GPT-2 / GPT-NeoX style
        lambda m: m.gpt_neox.layers,               # NeoX
        lambda m: m.transformer.blocks,            # MPT / some others
    ]
    for get in candidates:
        try:
            layers = get(model)
        except AttributeError:
            continue
        if layers is not None and len(layers) > 0:
            return layers
    raise ValueError("Could not find decoder layers; print(model) and extend get_decoder_layers().")


class SteeringHook:
    def __init__(self, direction: torch.Tensor, alpha: float,
                 norm_scale: bool = True, site: str = "post"):
        if site not in {"post", "pre"}:
            raise ValueError("site must be 'post' or 'pre'")
        d = direction.float().flatten()
        self.direction = d / (d.norm() + 1e-8)
        self.alpha = float(alpha)
        self.norm_scale = bool(norm_scale)
        self.site = site
        self._handles: list = []

    def _delta(self, hidden: torch.Tensor) -> torch.Tensor:
        v = self.direction.to(dtype=hidden.dtype, device=hidden.device)
        if self.norm_scale:
            return self.alpha * v * hidden.norm(dim=-1, keepdim=True)
        return self.alpha * v

    def _post_fn(self, _m, _inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden + self._delta(hidden)
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    def _pre_fn(self, _m, args, kwargs):
        if args:
            hidden = args[0] + self._delta(args[0])
            return (hidden, *args[1:]), kwargs
        if "hidden_states" in kwargs:
            kwargs["hidden_states"] = kwargs["hidden_states"] + self._delta(kwargs["hidden_states"])
        return args, kwargs

    def attach(self, layer_module) -> SteeringHook:
        if self.site == "post":
            self._handles.append(layer_module.register_forward_hook(self._post_fn))
        else:
            self._handles.append(
                layer_module.register_forward_pre_hook(self._pre_fn, with_kwargs=True)
            )
        return self

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @property
    def attached(self) -> bool:
        return bool(self._handles)


class AblationHook:
    """Mean-centred orthogonal ablation at decoder block outputs (J27).

    Acts at the block output at every position:

        h <- h - P_S (h - mu)

    where ``subspace`` S is a (d_model, r) matrix with orthonormal columns and
    ``mu`` is the layer's mean block output (frozen at build, see
    scripts/compute_j27_mu.py). P_S x = S (S^T x) removes the part of the
    deviation h - mu that lies inside S and leaves the ordinary level mu in
    place (mean-ablation, not zero-ablation).

    HOOK ORDER WITH STEERING — this ordering is part of the J27 spec.
    ``register_forward_hook`` callbacks fire in registration order. To get
    "steering first, ablation second" at the same layer (J27 wants the steered
    activation's S-component removed before block L+1 reads it), the
    SteeringHook must be attached BEFORE this hook at that layer. The provider
    does this in ``_ensure_hook``; tests/test_ablation_hook.py proves it.

    ``subspace`` with 0 columns (d, 0) is the identity: nothing is removed.
    Post-hoc instrumentation: after each forward, ``last_removed_share`` holds
    the mean over positions of ||P_S (h - mu)||^2 / ||h||^2 (the damage
    diagnostic the J27 manipulation check reports per set and layer).
    """

    def __init__(self, subspace: torch.Tensor, mu: torch.Tensor):
        s = torch.as_tensor(subspace, dtype=torch.float32)
        mu_t = torch.as_tensor(mu, dtype=torch.float32).flatten()
        if s.ndim != 2:
            raise ValueError(f"subspace must be a (d_model, r) matrix, got {tuple(s.shape)}")
        if s.shape[1] > 0 and s.shape[0] != int(mu_t.numel()):
            raise ValueError(
                f"subspace d_model {s.shape[0]} != mu dim {mu_t.numel()}")
        self.subspace = s
        self.mu = mu_t
        self.last_removed_share: float | None = None
        self._handles: list = []

    def _project(self, hidden: torch.Tensor) -> torch.Tensor:
        s = self.subspace.to(dtype=hidden.dtype, device=hidden.device)
        mu = self.mu.to(dtype=hidden.dtype, device=hidden.device)
        if s.shape[1] == 0:
            self.last_removed_share = 0.0
            return hidden
        dev = hidden - mu
        comp = dev @ s                       # (..., r): S-components of the deviation
        removed = comp @ s.T                 # (..., d): P_S (h - mu)
        denom = hidden.norm(dim=-1).clamp(min=1e-12)
        self.last_removed_share = float(
            (removed.norm(dim=-1) ** 2 / denom ** 2).mean())
        return hidden - removed

    def _post_fn(self, _m, _inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = self._project(hidden)
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    def attach(self, layer_module) -> AblationHook:
        self._handles.append(layer_module.register_forward_hook(self._post_fn))
        return self

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @property
    def attached(self) -> bool:
        return bool(self._handles)


class MultiAblationHook:
    """One AblationHook per decoder layer, each with its own S and mu.

    ``specs`` is an iterable of (layer_module, subspace, mu) triples in
    attach order. Within a layer, forward hooks fire in registration order,
    so to combine with steering at the same layer attach the SteeringHook
    before constructing this.
    """

    def __init__(self, specs):
        self._hooks = [AblationHook(s, mu).attach(layer) for layer, s, mu in specs]

    @property
    def hooks(self) -> list[AblationHook]:
        return list(self._hooks)

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()

    @property
    def attached(self) -> bool:
        return any(h.attached for h in self._hooks)


@contextmanager
def steering(model, layer: int, direction: torch.Tensor, alpha: float,
             norm_scale: bool = True, site: str = "post"):
    hook = SteeringHook(direction, alpha, norm_scale, site)
    hook.attach(get_decoder_layers(model)[layer])
    try:
        yield hook
    finally:
        hook.remove()

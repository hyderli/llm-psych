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

    def attach(self, layer_module) -> "SteeringHook":
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


@contextmanager
def steering(model, layer: int, direction: torch.Tensor, alpha: float,
             norm_scale: bool = True, site: str = "post"):
    hook = SteeringHook(direction, alpha, norm_scale, site)
    hook.attach(get_decoder_layers(model)[layer])
    try:
        yield hook
    finally:
        hook.remove()

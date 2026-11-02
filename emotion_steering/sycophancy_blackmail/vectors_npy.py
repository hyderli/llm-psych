"""Loader for .npy-per-layer steering vectors — model-agnostic.

On-disk format (one file per emotion per layer):
    <root>/<name>_layer<N>.npy   each a 1-D float32 array of size = model hidden dim

This makes NO assumptions about which model produced the vectors: hidden dim and
layer count are inferred from the files. It works for a 2560-dim 34-layer set,
a 3584-dim 42-layer set, or anything else — as long as the vector dim matches the
model you inject into. (A mismatch fails silently with meaningless results, so
always confirm with find_alpha.py that steering visibly changes generation.)

Interface (unit / available / build_direction) is stable so the provider and
find_alpha.py work with any model.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import numpy as np
import torch

_FILE_RE = re.compile(r"^(?P<name>.+)_layer(?P<layer>\d+)\.npy$")


class NpyVectorBank:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"No vector dir at {self.root.resolve()}")
        self._index: dict[str, dict[int, str]] = {}
        for p in glob.glob(str(self.root / "**" / "*.npy"), recursive=True):
            m = _FILE_RE.match(os.path.basename(p))
            if not m:
                continue
            self._index.setdefault(m.group("name"), {})[int(m.group("layer"))] = p
        if not self._index:
            raise FileNotFoundError(
                f"No <name>_layer<N>.npy files under {self.root.resolve()}"
            )
        self._cache: dict[tuple[str, int], torch.Tensor] = {}

    def available(self) -> list[str]:
        return sorted(self._index)

    @property
    def n_layers(self) -> int | None:
        layers = {L for tbl in self._index.values() for L in tbl}
        return (max(layers) + 1) if layers else None

    def layers_for(self, name: str) -> list[int]:
        return sorted(self._index.get(name, {}))

    def hidden_size(self, layer: int | None = None) -> int:
        name = self.available()[0]
        L = layer if layer is not None else min(self._index[name])
        return self.unit(name, L).numel()

    def _raw(self, name: str, layer: int) -> torch.Tensor:
        key = (name, layer)
        if key in self._cache:
            return self._cache[key]
        if name not in self._index:
            raise KeyError(f"'{name}' not found. have: {self.available()}")
        if layer not in self._index[name]:
            have = self.layers_for(name)
            raise KeyError(f"layer {layer} missing for '{name}' (have {min(have)}..{max(have)})")
        arr = np.load(self._index[name][layer], allow_pickle=True)
        t = torch.as_tensor(np.asarray(arr), dtype=torch.float32).flatten()
        self._cache[key] = t
        return t

    def unit(self, name: str, layer: int) -> torch.Tensor:
        v = self._raw(name, layer)
        return v / (v.norm() + 1e-8)


def build_direction(bank: NpyVectorBank, terms, layer: int) -> torch.Tensor:
    """terms = [(name, weight), ...] -> single unit direction at `layer`.

    Weights are relative: after the final normalization only their ratio matters,
    so [("a",1),("b",1)] == [("a",0.5),("b",0.5)] (a 50/50 blend). To bias the
    mix, change the ratio, e.g. [("a",1.0),("b",0.3)].
    """
    if not terms:
        raise ValueError("no steering terms given")
    combined = sum(w * bank.unit(n, layer) for n, w in terms)
    norm = combined.norm()
    if norm < 1e-6:
        raise ValueError(f"terms {terms} cancelled to ~zero at layer {layer}")
    return combined / norm

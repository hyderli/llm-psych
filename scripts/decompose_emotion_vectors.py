"""Decompose stored emotion steering vectors into J-space and residual components.

For each stored emotion steering vector (one per layer/emotion), this script
projects it onto the span of the model's J-lens token directions using the
sparse nonnegative gradient-pursuit algorithm from Gurnee et al. (2026,
*Verbalizable Representations Form a Global Workspace in Language Models*),
section 2.3. The reconstruction is the J-space (workspace) component; the
orthogonal remainder is the residual component.

Two lens sources are supported:

* ``neuronpedia`` — pre-fitted Jacobian lenses released by Neuronpedia on the
  HF Hub (``neuronpedia/jacobian-lens``).
* ``logit`` — J_l is the identity, so the decomposition uses the raw unembedding
  rows. This is the cheap fallback and is what the paper calls the logit-lens
  approximation; it captures much of the workspace-like structure but with
  lower reliability in early/mid layers.

The script also decomposes ``-v`` for every vector. Because the steering vectors
are cross-emotion centered, the anti-emotion direction is meaningful; only
decomposing ``+v`` would artificially push the opposite-emotion signal into the
residual.

Weights-light loading
---------------------
Only ``lm_head.weight`` (or tied ``embed_tokens.weight`` for Gemma) and
``model.norm.weight`` are needed. The script downloads only the safetensors
shard(s) containing those tensors, so the whole analysis can run on a laptop
without pulling the full 8–9B checkpoint.

Outputs
-------
For each input vector ``<emotion>_layer<L>.npy``:

``results/workspace_decomposition/<model_key>-story/<emotion>_layer<L>_jspace.npy``
    The sparse nonnegative J-space reconstruction of ``+v``.

``results/workspace_decomposition/<model_key>-story/<emotion>_layer<L>_residual.npy``
    The orthogonal remainder ``v - v_jspace``.

``results/workspace_decomposition/<model_key>-story/<emotion>_layer<L>_neg_jspace.npy``
    J-space reconstruction of ``-v``.

``results/workspace_decomposition/<model_key>-story/<emotion>_layer<L>_neg_residual.npy``
    Orthogonal remainder of ``-v``.

``manifest.yaml``
    Run metadata plus per-vector metrics: norms, fraction of squared norm in
    J-space, number of atoms selected, top tokens, coefficients, and
    component/residual cosine.

Usage
-----
With a Neuronpedia J-lens (primary target models)::

    uv run python scripts/decompose_emotion_vectors.py \
        --model-config configs/model/llama31_8b.yaml \
        --lens-source neuronpedia \
        --k 16 --n-candidates 512

Logit-lens fallback, e.g. for a small dev model that lacks a fitted lens::

    uv run python scripts/decompose_emotion_vectors.py \
        --model-config configs/model/gemma2_2b.yaml \
        --lens-source logit \
        --k 16 --n-candidates 512

References
----------
* Gurnee et al. (2026), "Verbalizable Representations Form a Global Workspace
  in Language Models", https://transformer-circuits.pub/2026/workspace/index.html
* Reference implementation of gradient-pursuit decomposition:
  https://github.com/idhantgulati/j-lens/blob/main/jlens.py
* Pre-fitted open-weight lenses:
  https://huggingface.co/neuronpedia/jacobian-lens
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from huggingface_hub import HfApi, hf_hub_download
from scipy.optimize import nnls
from transformers import AutoTokenizer, PreTrainedTokenizerBase

_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root / "src"))

# Load HF_TOKEN from .env so gated model shards can be downloaded.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(_repo_root / ".env")

from llm_psych.models import load_model  # noqa: E402

log = logging.getLogger(__name__)

# Map HF model ids to the corresponding paths inside the Neuronpedia
# jacobian-lens repository. Only models known to be released are listed.
_NEURONPEDIA_LENS_PATHS: dict[str, str] = {
    "meta-llama/Llama-3.1-8B-Instruct": (
        "llama3.1-8b-it/jlens/Salesforce-wikitext/"
        "Llama-3.1-8B-Instruct_jacobian_lens.pt"
    ),
    "Qwen/Qwen2.5-7B-Instruct": (
        "qwen2.5-7b-it/jlens/Salesforce-wikitext/"
        "Qwen2.5-7B-Instruct_jacobian_lens.pt"
    ),
    "google/gemma-2-9b-it": (
        "gemma-2-9b-it/jlens/Salesforce-wikitext/"
        "gemma-2-9b-it_jacobian_lens.pt"
    ),
    "google/gemma-2-2b-it": (
        "gemma-2-2b-it/jlens/Salesforce-wikitext/"
        "gemma-2-2b-it_jacobian_lens.pt"
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose emotion steering vectors into J-space and residual components."
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Path to a configs/model/*.yaml file (e.g. configs/model/llama31_8b.yaml).",
    )
    parser.add_argument(
        "--vectors-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing <emotion>_layer<L>.npy steering vectors. "
            "Defaults to steering_vectors/<model_key>-story relative to the repo root."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_repo_root / "results" / "workspace_decomposition",
        help="Root output directory for decomposed vectors and manifest.",
    )
    parser.add_argument(
        "--lens-source",
        choices=["neuronpedia", "logit"],
        default="neuronpedia",
        help=(
            "Lens to use: 'neuronpedia' loads a pre-fitted J-lens from the HF Hub; "
            "'logit' uses the unembedding rows directly (J_l = identity)."
        ),
    )
    parser.add_argument(
        "--lens-path",
        type=Path,
        default=None,
        help="Optional local path to a J-lens .pt file; overrides --lens-source.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=16,
        help="Maximum number of J-lens atoms in the sparse reconstruction.",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=512,
        help="Number of top lens-logit tokens to use as candidate atoms.",
    )
    parser.add_argument(
        "--lens-layer-policy",
        choices=["skip", "nearest"],
        default="skip",
        help=(
            "What to do when a requested layer is absent from the pre-fitted lens: "
            "skip the layer or use the nearest available lens layer."
        ),
    )
    parser.add_argument(
        "--emotions",
        nargs="+",
        default=None,
        help="Subset of emotions to decompose (default: all discovered).",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Subset of layer indices to decompose (default: all discovered).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Override the model config's device_map when using the full-model "
            "fallback (e.g. 'mps', 'cpu', 'auto')."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def _load_model_config(path: Path) -> dict[str, Any]:
    """Load a model YAML config and add the inferred model key."""
    with path.open("r") as fh:
        cfg = yaml.safe_load(fh)
    if "hf_model_id" not in cfg:
        raise ValueError(f"{path} must contain hf_model_id")
    cfg["model_key"] = cfg["hf_model_id"].split("/")[-1]
    return cfg


def _is_gemma(hf_model_id: str) -> bool:
    return hf_model_id.startswith("google/gemma")


def _norm_scale_from_weight(weight: torch.Tensor, hf_model_id: str) -> torch.Tensor:
    """Return the elementwise RMSNorm scale.

    Gemma-family models parameterise the scale as ``1 + weight``; Llama/Qwen
    use ``weight`` directly. Gemma-2's final logit softcapping is monotonic and
    does not affect the ranking of candidate atoms, so it is ignored here.
    """
    return 1.0 + weight if _is_gemma(hf_model_id) else weight


def _load_tokenizer(
    hf_model_id: str,
    revision: str | None,
    trust_remote_code: bool = False,
) -> PreTrainedTokenizerBase:
    """Load the tokenizer, allowing a small download if needed."""
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_unembed_and_norm_shards(
    hf_model_id: str,
    revision: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Download only the safetensors shard(s) holding W_U and final norm.

    Returns
    -------
    W_U : [vocab, d]
        LM head or tied embedding weight.
    g : [d]
        Final RMSNorm scale (with Gemma offset applied).
    """
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors is required for the weights-light loader") from exc

    # Keys we need.
    unembed_keys = ["lm_head.weight"]
    if _is_gemma(hf_model_id):
        # Gemma ties embeddings and has no lm_head.weight in the checkpoint.
        unembed_keys.append("model.embed_tokens.weight")
    norm_keys = ["model.norm.weight", "norm.weight"]

    # Download the shard index (small) first to locate the right shards.
    index_path: Path | None = None
    try:
        index_path = Path(
            hf_hub_download(
                hf_model_id,
                filename="model.safetensors.index.json",
                revision=revision,
                repo_type="model",
            )
        )
    except Exception:  # noqa: BLE001
        pass

    shard_paths: dict[str, Path] = {}
    if index_path is not None and index_path.exists():
        with index_path.open("r") as fh:
            weight_map = json.load(fh)["weight_map"]

        needed_keys: set[str] = set()
        for key in {*unembed_keys, *norm_keys}:
            if key in weight_map:
                needed_keys.add(key)
        if not needed_keys:
            raise RuntimeError(
                f"Could not locate lm_head/embed_tokens or norm weights in {hf_model_id} index"
            )

        for shard_name in {weight_map[k] for k in needed_keys}:
            shard_paths[shard_name] = Path(
                hf_hub_download(
                    hf_model_id,
                    filename=shard_name,
                    revision=revision,
                    repo_type="model",
                )
            )
    else:
        # Single-shard model: download it and inspect keys below.
        shard_paths["model.safetensors"] = Path(
            hf_hub_download(
                hf_model_id,
                filename="model.safetensors",
                revision=revision,
                repo_type="model",
            )
        )

    # Read only the needed tensors.
    W_U: torch.Tensor | None = None
    g: torch.Tensor | None = None
    for shard_path in shard_paths.values():
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for key in unembed_keys:
                if W_U is None and key in keys:
                    W_U = f.get_tensor(key)
            for key in norm_keys:
                if g is None and key in keys:
                    g = f.get_tensor(key)

    if W_U is None:
        raise RuntimeError(f"Could not find unembedding weight for {hf_model_id}")
    if g is None:
        raise RuntimeError(f"Could not find final norm weight for {hf_model_id}")

    g = _norm_scale_from_weight(g.float(), hf_model_id)
    return W_U.float(), g


def _extract_unembed_and_norm(
    model: torch.nn.Module, hf_model_id: str, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract final unembedding weights and final RMSNorm scale from a loaded model."""
    if hasattr(model, "lm_head"):
        W_U = model.lm_head.weight.detach().float().to(device)
    else:
        raise RuntimeError("Model has no lm_head attribute")

    if hasattr(model, "model") and hasattr(model.model, "norm"):
        norm = model.model.norm
    elif hasattr(model, "norm"):
        norm = model.norm
    else:
        raise RuntimeError("Could not locate final norm module")

    if hasattr(norm, "weight") and norm.weight is not None:
        g = _norm_scale_from_weight(norm.weight.detach().float().to(device), hf_model_id)
    else:
        g = torch.ones(W_U.shape[1], device=device, dtype=torch.float32)

    return W_U, g


def _resolve_lens_filename(hf_model_id: str) -> str:
    """Return the Neuronpedia J-lens filename for a model id.

    Falls back to querying the HF Hub if the model is not in the hard-coded
    mapping, and raises a clear error if no matching lens exists.
    """
    if hf_model_id in _NEURONPEDIA_LENS_PATHS:
        return _NEURONPEDIA_LENS_PATHS[hf_model_id]

    log.warning("No hard-coded lens path for %s; querying Hub", hf_model_id)
    api = HfApi()
    files = api.list_repo_files("neuronpedia/jacobian-lens", repo_type="model")
    model_key = hf_model_id.split("/")[-1].lower().replace(".", "")
    candidates = [f for f in files if model_key in f.lower() and f.endswith("_jacobian_lens.pt")]
    if not candidates:
        raise ValueError(
            f"No Neuronpedia J-lens found for {hf_model_id}. "
            "Pass --lens-path or use --lens-source logit."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous Neuronpedia lenses for {hf_model_id}: {candidates}. "
            "Pass --lens-path explicitly."
        )
    return candidates[0]


def _snapshot_sha_from_path(local_path: Path) -> str | None:
    """Extract the Git revision from the HF cache snapshot directory name."""
    parts = local_path.resolve().parts
    for i, part in enumerate(parts):
        if part == "snapshots" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def _load_jlens(
    source: str, hf_model_id: str, lens_path: Path | None
) -> tuple[dict[int, torch.Tensor] | None, dict[str, Any] | None]:
    """Load a J-lens and return provenance metadata.

    Returns
    -------
    J : dict[int, torch.Tensor] | None
        Layer -> [d, d] transport matrices, or None for logit-lens fallback.
    provenance : dict[str, Any] | None
        Metadata for the manifest, or None for logit-lens fallback.
    """
    if source == "logit":
        log.info("Using logit-lens fallback (J_l = identity)")
        return None, None

    if source != "neuronpedia":
        raise ValueError(f"Unknown lens source: {source}")

    if lens_path is not None:
        filename = str(lens_path)
        local = str(lens_path.resolve())
    else:
        filename = _resolve_lens_filename(hf_model_id)
        local = hf_hub_download(
            repo_id="neuronpedia/jacobian-lens",
            filename=filename,
            repo_type="model",
        )

    state = torch.load(local, map_location="cpu", weights_only=False)
    J: dict[int, torch.Tensor] = state["J"]
    provenance = {
        "repo_id": "neuronpedia/jacobian-lens",
        "filename": filename,
        "revision": _snapshot_sha_from_path(Path(local)),
        "n_prompts": int(state.get("n_prompts", 0)),
        "source_layers": [int(x) for x in state.get("source_layers", [])],
        "d_model": int(state.get("d_model", 0)),
    }
    log.info(
        "Loaded J-lens: n_prompts=%d source_layers=%s d_model=%d",
        provenance["n_prompts"],
        provenance["source_layers"],
        provenance["d_model"],
    )
    return J, provenance


def _nnls_active(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exact nonnegative least-squares on the active atom set.

    Parameters
    ----------
    A : [d, m]
        Column vectors for the active atoms.
    b : [d]
        Target vector.

    Returns
    -------
    coefs : [m]
        Nonnegative coefficients.
    """
    sol, _ = nnls(A.numpy(), b.numpy())
    return torch.from_numpy(sol).float()


def _decompose_vector(
    v: np.ndarray,
    sign: int,
    J_l: torch.Tensor | None,
    W_U: torch.Tensor,
    g: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    k: int,
    n_candidates: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Sparse nonnegative gradient-pursuit decomposition of ``sign * v``.

    Parameters
    ----------
    v
        Steering vector, shape ``(d,)``.
    sign
        ``+1`` or ``-1``.
    J_l
        J-lens transport matrix for this layer, or ``None`` for logit-lens.
    W_U, g
        Unembedding matrix and final norm scale.
    tokenizer
        Tokenizer for decoding candidate token ids.
    k, n_candidates
        Gradient-pursuit hyperparameters.

    Returns
    -------
    component, residual, metrics
        ``component`` is the J-space reconstruction, ``residual`` is ``sign*v - component``,
        and ``metrics`` is a JSON-serialisable dict.
    """
    h = sign * torch.from_numpy(v).float().cpu()
    d = h.shape[0]

    # Transport to final-layer basis and compute lens logits.
    z = h @ J_l.T if J_l is not None else h
    logits = (z * g) @ W_U.T  # [vocab]

    # Candidate atoms: top tokens by lens logit.
    cand = logits.topk(n_candidates).indices  # [n_candidates]

    # Build candidate J-lens vectors in source space: rows of W_U diag(g) J_l.
    w_cand = W_U[cand]  # [n_candidates, d]
    if J_l is not None:
        V = (w_cand * g) @ J_l  # [n_candidates, d]
    else:
        V = w_cand

    # Unit-normalise atoms; coefficients absorb scale.
    norms = V.norm(dim=1, keepdim=True)
    Vn = V / torch.clamp(norms, min=1e-12)

    picked: list[int] = []
    resid = h.clone()
    coefs = torch.zeros(0, dtype=torch.float32)

    for _ in range(k):
        corr = Vn @ resid  # [n_candidates]
        corr[picked] = -float("inf")
        i = int(corr.argmax())
        if corr[i] <= 0:
            break
        picked.append(i)
        A = Vn[picked].T  # [d, m]
        # Exact nonnegative least-squares on the active set.
        coefs = _nnls_active(A, h)
        resid = h - A @ coefs

    component = h - resid
    token_ids = cand[picked].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids) if tokenizer else []

    v_norm = float(h.norm())
    comp_norm = float(component.norm())
    resid_norm = float(resid.norm())
    cos_cr = (
        float((component @ resid) / (comp_norm * resid_norm))
        if comp_norm > 1e-12 and resid_norm > 1e-12
        else 0.0
    )
    metrics = {
        "sign": sign,
        "v_norm": v_norm,
        "component_norm": comp_norm,
        "residual_norm": resid_norm,
        "frac_norm_squared": float((comp_norm**2) / (v_norm**2 + 1e-12)),
        "frac_residual_squared": float((resid_norm**2) / (v_norm**2 + 1e-12)),
        "cos_component_residual": cos_cr,
        "n_atoms": len(picked),
        "token_ids": token_ids,
        "tokens": tokens,
        "coefs": [float(c) for c in coefs.tolist()],
    }
    # Cast component/residual back to the original sign convention.
    component_np = (sign * component).numpy().astype(np.float32)
    residual_np = (sign * resid).numpy().astype(np.float32)
    return component_np, residual_np, metrics


def _discover_vectors(vectors_dir: Path) -> dict[str, dict[int, Path]]:
    """Discover emotion vectors: {emotion: {layer: path}}."""
    if not vectors_dir.is_dir():
        raise FileNotFoundError(f"Vectors directory not found: {vectors_dir}")

    pattern = re.compile(r"^(?P<emotion>[^_]+)_layer(?P<layer>\d+)\.npy$")
    found: dict[str, dict[int, Path]] = {}
    for path in sorted(vectors_dir.glob("*.npy")):
        m = pattern.match(path.name)
        if not m:
            continue
        emotion = m.group("emotion")
        layer = int(m.group("layer"))
        found.setdefault(emotion, {})[layer] = path
    if not found:
        raise FileNotFoundError(f"No <emotion>_layer<L>.npy files in {vectors_dir}")
    return found


def _map_layer(
    layer: int,
    available: set[int] | None,
    policy: str,
) -> int | None:
    """Map a requested layer to an available lens layer.

    If ``available`` is ``None`` (e.g. logit-lens fallback), the requested
    layer is used unchanged.
    """
    if available is None or not available:
        return layer
    if layer in available:
        return layer
    if policy == "skip":
        return None
    # nearest
    nearest = min(available, key=lambda x: abs(x - layer))
    return nearest


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    model_cfg = _load_model_config(args.model_config)
    model_key: str = model_cfg["model_key"]
    hf_model_id: str = model_cfg["hf_model_id"]
    revision: str | None = model_cfg.get("hf_revision")

    vectors_dir = args.vectors_dir or (_repo_root / "steering_vectors" / f"{model_key}-story")
    output_dir = args.output_dir / f"{model_key}-story"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Model: %s (%s)", model_key, hf_model_id)
    log.info("Vectors: %s", vectors_dir)
    log.info("Output: %s", output_dir)

    # Discover available emotion vectors.
    vectors = _discover_vectors(vectors_dir)
    if args.emotions:
        missing = set(args.emotions) - set(vectors.keys())
        if missing:
            raise ValueError(f"Requested emotions not found: {sorted(missing)}")
        vectors = {e: vectors[e] for e in args.emotions}
    if args.layers:
        vectors = {
            e: {lyr: p for lyr, p in layers.items() if lyr in args.layers}
            for e, layers in vectors.items()
        }
        vectors = {e: layers for e, layers in vectors.items() if layers}
    if not vectors:
        raise ValueError("No vectors match the requested filters")

    # Load tokenizer (small download if not cached).
    tokenizer = _load_tokenizer(
        hf_model_id,
        revision=revision,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )

    # Load only the unembedding + final-norm shard(s). This is the default path
    # and avoids pulling the full checkpoint. Pass --device to force the full
    # model fallback (e.g. on a pod where you want the standard load path).
    if args.device is None:
        W_U, g = _load_unembed_and_norm_shards(hf_model_id, revision)
        log.info(
            "Weights-light loader: W_U [%d, %d], norm scale [%d]",
            W_U.shape[0], W_U.shape[1], g.shape[0],
        )
        weights_source = "shards"
    else:
        device_map = args.device or model_cfg.get("device_map", "auto")
        log.info("Loading full model %s with device_map=%s ...", hf_model_id, device_map)
        loaded = load_model(
            hf_model_id,
            revision=revision,
            torch_dtype=model_cfg.get("torch_dtype", "auto"),
            device_map=device_map,
            trust_remote_code=model_cfg.get("trust_remote_code", False),
        )
        W_U, g = _extract_unembed_and_norm(loaded.model, hf_model_id, device="cpu")
        del loaded.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info(
            "Full-model path: W_U [%d, %d], norm scale [%d]",
            W_U.shape[0], W_U.shape[1], g.shape[0],
        )
        weights_source = "full_model"

    # Load J-lens if requested.
    J, lens_provenance = _load_jlens(args.lens_source, hf_model_id, args.lens_path)

    available_layers: set[int] | None = set(J.keys()) if J is not None else None
    layer_map: dict[int, int | None] = {}
    skipped_layers: set[int] = set()
    for emotion, layers in vectors.items():
        for lyr in layers:
            if lyr not in layer_map:
                mapped = _map_layer(lyr, available_layers, args.lens_layer_policy)
                layer_map[lyr] = mapped
                if mapped is None:
                    skipped_layers.add(lyr)

    if skipped_layers:
        log.warning(
            "Skipping layers not covered by lens (policy=skip): %s",
            sorted(skipped_layers),
        )

    manifest: dict[str, Any] = {
        "model_id": hf_model_id,
        "model_key": model_key,
        "model_revision": revision,
        "vectors_dir": str(vectors_dir.relative_to(_repo_root)),
        "output_dir": str(output_dir.relative_to(_repo_root)),
        "weights_source": weights_source,
        "gemma_norm_offset": _is_gemma(hf_model_id),
        "lens_source": "custom" if args.lens_path else args.lens_source,
        "lens": lens_provenance,
        "lens_layer_policy": args.lens_layer_policy,
        "k": args.k,
        "n_candidates": args.n_candidates,
        "vocab_size": int(W_U.shape[0]),
        "d_model": int(W_U.shape[1]),
        "vectors": {},
    }

    total_vectors = sum(len(layers) for layers in vectors.values())
    processed = 0

    for emotion, layers in sorted(vectors.items()):
        manifest["vectors"][emotion] = {}
        for layer, path in sorted(layers.items()):
            mapped_layer = layer_map[layer]
            if mapped_layer is None:
                log.info("Skipping %s layer %d (not in lens)", emotion, layer)
                continue

            log.info("Decomposing %s layer %d (lens layer %d)", emotion, layer, mapped_layer)
            v = np.load(path).astype(np.float32)
            if v.shape != (W_U.shape[1],):
                raise ValueError(
                    f"Shape mismatch for {path}: {v.shape} vs expected ({W_U.shape[1]},)"
                )

            J_l = J[mapped_layer] if J is not None else None

            comp_pos, resid_pos, metrics_pos = _decompose_vector(
                v, +1, J_l, W_U, g, tokenizer, args.k, args.n_candidates
            )
            comp_neg, resid_neg, metrics_neg = _decompose_vector(
                v, -1, J_l, W_U, g, tokenizer, args.k, args.n_candidates
            )

            base = output_dir / f"{emotion}_layer{layer}"
            np.save(f"{base}_jspace.npy", comp_pos)
            np.save(f"{base}_residual.npy", resid_pos)
            np.save(f"{base}_neg_jspace.npy", comp_neg)
            np.save(f"{base}_neg_residual.npy", resid_neg)

            rel_base = base.relative_to(_repo_root)
            manifest["vectors"][emotion][layer] = {
                "source": str(path.relative_to(_repo_root)),
                "lens_layer": mapped_layer,
                "jspace": f"{rel_base}_jspace.npy",
                "residual": f"{rel_base}_residual.npy",
                "neg_jspace": f"{rel_base}_neg_jspace.npy",
                "neg_residual": f"{rel_base}_neg_residual.npy",
                "metrics_pos": metrics_pos,
                "metrics_neg": metrics_neg,
            }
            log.info(
                "  +v atoms=%d frac=%.4f cos=%.4f top=%s",
                metrics_pos["n_atoms"],
                metrics_pos["frac_norm_squared"],
                metrics_pos["cos_component_residual"],
                metrics_pos["tokens"][:5],
            )
            log.info(
                "  -v atoms=%d frac=%.4f cos=%.4f top=%s",
                metrics_neg["n_atoms"],
                metrics_neg["frac_norm_squared"],
                metrics_neg["cos_component_residual"],
                metrics_neg["tokens"][:5],
            )
            processed += 1
            if processed % 10 == 0 or processed == total_vectors:
                log.info("Progress: %d/%d vectors decomposed", processed, total_vectors)

    manifest_path = output_dir / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    log.info("Saved manifest to %s", manifest_path)
    log.info("Done — %d vectors decomposed for %s", processed, model_key)


if __name__ == "__main__":
    main()

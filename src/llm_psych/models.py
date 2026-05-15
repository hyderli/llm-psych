"""Model-loading helpers for the Emotion Concepts replication.

Wraps ``AutoModelForCausalLM.from_pretrained`` with sensible defaults
for Llama 3.1 8B, Qwen 2.5 7B, Gemma 2 2B, and other decoder-only models.

Returns a ``LoadedModel`` named-tuple so callers can unpack
``model, tokenizer, cfg`` in one line without positional-index guessing.

Notes
-----
*Development (Mac M5 / MPS):* set ``device_map="mps"`` or rely on
auto-detection. Small models (0.5B–2B) fit in MPS unified memory.

*Production (Linux, CUDA RTX 4090 24 GB):* ``device_map="auto"``
places the full bf16 8B model on the single GPU. ``bitsandbytes``
4-bit quantization is available via ``load_in_4bit=True`` but is
*not* used for primary results (see BLUEPRINT.md).

References
----------
HF model IDs and revision SHAs are in ``configs/model/``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    """Static facts about a loaded model needed downstream.

    Parameters
    ----------
    n_layers
        Number of decoder blocks (e.g. 32 for Llama 3.1 8B).
    hidden_size
        Residual-stream dimension (e.g. 4096 for Llama 3.1 8B).
    hf_model_id
        HuggingFace model ID used to load the checkpoint.
    hf_revision
        Git commit SHA of the HF repo at load time (for reproducibility
        logging). ``None`` if ``revision`` was not pinned.
    """

    n_layers: int
    hidden_size: int
    hf_model_id: str
    hf_revision: str | None


@dataclass
class LoadedModel:
    """Container returned by :func:`load_model`.

    Attributes
    ----------
    model
        The loaded causal LM, on the requested device.
    tokenizer
        Matching tokenizer with ``padding_side="left"`` and a pad token
        set (required for batched generation with left-padding).
    cfg
        Static architecture facts (``n_layers``, ``hidden_size``, …).
    """

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    cfg: ModelConfig


def load_model(
    hf_model_id: str,
    *,
    revision: str | None = None,
    torch_dtype: torch.dtype | str = "auto",
    device_map: str | dict[str, Any] = "auto",
    load_in_4bit: bool = False,
    trust_remote_code: bool = False,
    cache_dir: str | None = None,
    attn_implementation: str | None = None,
) -> LoadedModel:
    """Load a causal LM and its tokenizer.

    Parameters
    ----------
    hf_model_id
        HuggingFace model ID, e.g. ``"meta-llama/Llama-3.1-8B-Instruct"``.
    revision
        Specific git commit SHA or tag. Pass the pinned SHA from
        ``configs/model/<n>.yaml`` for reproducibility.
    torch_dtype
        Weight dtype. ``"auto"`` uses the model's saved dtype (bf16 for
        Llama 3.1 / Qwen 2.5). On MPS, ``torch.float16`` is preferred
        because MPS has incomplete bf16 kernel coverage as of PyTorch 2.5.
    device_map
        ``"auto"`` (CUDA multi-GPU sharding or single GPU), ``"mps"``,
        ``"cpu"``, or an explicit layer→device dict.
    load_in_4bit
        If ``True``, load via ``bitsandbytes`` NF4 quantization. Only
        available on Linux/CUDA; not used for primary results.
    trust_remote_code
        Required for some model families (e.g. Qwen 2.5 on older
        transformers). Default ``False`` — only enable when needed.
    cache_dir
        Override HF cache directory. ``None`` uses the HF default
        (``~/.cache/huggingface``).
    attn_implementation
        E.g. ``"flash_attention_2"`` for CUDA; ``None`` lets
        transformers pick based on the environment.

    Returns
    -------
    LoadedModel
        Named-tuple with ``model``, ``tokenizer``, and ``cfg``.

    Examples
    --------
    >>> lm = load_model(
    ...     "meta-llama/Llama-3.1-8B-Instruct",
    ...     revision="8c22764a7e3675c50d4c7c9a4edb474456022b16",
    ...     torch_dtype=torch.bfloat16,
    ... )
    >>> lm.cfg.n_layers
    32
    """
    kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": hf_model_id,
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }
    if revision is not None:
        kwargs["revision"] = revision
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation

    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig  # type: ignore[import]
        except ImportError as e:
            raise ImportError(
                "bitsandbytes is required for load_in_4bit=True. "
                "It is only available on Linux/CUDA; install via 'uv sync' on a "
                "Linux machine."
            ) from e
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    logger.info("Loading model %s (revision=%s) …", hf_model_id, revision)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(**kwargs)
    model.eval()

    tokenizer = _load_tokenizer(hf_model_id, revision=revision, cache_dir=cache_dir)

    hf_config = model.config
    n_layers: int = (
        hf_config.num_hidden_layers
        if hasattr(hf_config, "num_hidden_layers")
        else len(model.model.layers)  # type: ignore[attr-defined]
    )
    hidden_size: int = (
        hf_config.hidden_size
        if hasattr(hf_config, "hidden_size")
        else model.model.layers[0].self_attn.q_proj.in_features  # type: ignore[attr-defined]
    )

    cfg = ModelConfig(
        n_layers=n_layers,
        hidden_size=hidden_size,
        hf_model_id=hf_model_id,
        hf_revision=revision,
    )
    logger.info(
        "Loaded %s — %d layers, hidden_size=%d", hf_model_id, n_layers, hidden_size
    )
    return LoadedModel(model=model, tokenizer=tokenizer, cfg=cfg)


def _load_tokenizer(
    hf_model_id: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> PreTrainedTokenizerBase:
    """Load tokenizer with padding configured for batched left-padded generation."""
    kwargs: dict[str, Any] = {"pretrained_model_name_or_path": hf_model_id}
    if revision is not None:
        kwargs["revision"] = revision
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(**kwargs)

    # Left-padding is required for batched generation: the model generates
    # new tokens at position [-1], so all prompts must be right-aligned.
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        # Most instruct-tuned models have a pad token; base models often don't.
        tokenizer.pad_token = tokenizer.eos_token
        logger.debug(
            "pad_token not set for %s; using eos_token=%r",
            hf_model_id,
            tokenizer.eos_token,
        )

    return tokenizer


def probe_layer_range(cfg: ModelConfig) -> list[int]:
    """Return the candidate layer indices for probing per HYPOTHESES.md.

    Per the pre-registration: mid-to-late layers ⌊L/2⌋ to L−2.

    Parameters
    ----------
    cfg
        A :class:`ModelConfig` from :func:`load_model`.

    Returns
    -------
    list[int]
        Sorted layer indices in ``[n_layers // 2, n_layers - 2]``.
    """
    start = cfg.n_layers // 2
    stop = cfg.n_layers - 1  # exclusive, so last included is n_layers - 2
    return list(range(start, stop))

"""llm-psych — Emotion Concepts in Open-Weight LLMs.

Replication and extension of Sofroniew, Kauvar, Saunders, Chen et al.
(2026), "Emotion Concepts and their Function in a Large Language Model"
(Transformer Circuits Thread).
"""

from llm_psych.hooks import (
    ResidualStreamRecorder,
    ResidualStreamSteerer,
)
from llm_psych.models import LoadedModel, ModelConfig, load_model, probe_layer_range
from llm_psych.probes import (
    ProbeResult,
    ProbeMeta,
    best_layer,
    derive_steering_vector,
    evaluate,
    fit,
    load,
    save,
    select_layer_by_valence,
)

__all__ = [
    "ResidualStreamRecorder",
    "ResidualStreamSteerer",
    "LoadedModel",
    "ModelConfig",
    "load_model",
    "probe_layer_range",
    "ProbeResult",
    "ProbeMeta",
    "best_layer",
    "derive_steering_vector",
    "evaluate",
    "fit",
    "load",
    "save",
    "select_layer_by_valence",
]

__version__ = "0.1.0"

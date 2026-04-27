"""llm-psych — Emotion Concepts in Open-Weight LLMs.

Replication and extension of Sofroniew, Kauvar, Saunders, Chen et al.
(2026), "Emotion Concepts and their Function in a Large Language Model"
(Transformer Circuits Thread).
"""

from llm_psych.hooks import (
    ResidualStreamRecorder,
    ResidualStreamSteerer,
)

__all__ = [
    "ResidualStreamRecorder",
    "ResidualStreamSteerer",
]

__version__ = "0.1.0"

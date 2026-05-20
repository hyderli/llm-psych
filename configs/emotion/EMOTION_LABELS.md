# Emotion label mapping

Integer labels used in probe training arrays and `emotion_prompts.parquet`.
These must match the `label` field in each emotion's YAML config.

## Primary emotions (9 — default study set)

| label | emotion       |
|-------|---------------|
| 5     | afraid        |
| 6     | calm          |
| 7     | desperate     |
| 8     | joyful        |
| 9     | blissful      |
| 10    | compassionate |
| 11    | upset         |
| 12    | offended      |
| 13    | hostile       |

## Legacy / secondary emotions

These were used for early pipeline development and are kept for reference.
They are not part of the primary pre-registered H1–H6 analyses unless
added via a HYPOTHESES.md amendment.

| label | emotion |
|-------|---------|
| 1     | anger   |
| 2     | fear    |
| 3     | sadness |
| 4     | joy     |

## Meeting 2026-05-18 extensions (no stimuli yet)

Added to provide config slots for Ekman / Wilcox emotion candidates
raised in the kick-off meeting. These configs exist but
`emotion_prompts.parquet` does not yet contain prompts for these
labels — stimulus authoring is required before any extraction run.

| label | emotion   | framework |
|-------|-----------|-----------|
| 14    | disgust   | Ekman     |
| 15    | surprise  | Ekman     |
| 16    | contempt  | Ekman     |
| 17    | peaceful  | Wilcox    |
| 18    | powerful  | Wilcox    |

## Neutral baseline

Neutral prompts are not assigned an integer label — they are handled as
the negative class in one-vs-rest probe training and as the reference
condition for CAA steering vectors.

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

## Neutral baseline

Neutral prompts are not assigned an integer label — they are handled as
the negative class in one-vs-rest probe training and as the reference
condition for CAA steering vectors.

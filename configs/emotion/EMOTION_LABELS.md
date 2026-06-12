# Emotion label mapping

Integer labels used in probe training arrays and `emotion_prompts.parquet`.
These must match the `label` field in each emotion's YAML config.

## Primary emotions (4 — confirmatory study set, 2026-06-12)

The project's confirmatory emotion set is exactly four, forming two
opposite pairs: admiration ↔ loathing (Plutchik trust/disgust axis) and
joy ↔ sadness (valence). Neutral is the reference class. See the
2026-06-12 amendment in HYPOTHESES.md.

| label | emotion    | pair / axis              |
|-------|------------|--------------------------|
| 20    | admiration | trust pole (↔ loathing)  |
| 4     | joy        | valence + (↔ sadness)    |
| 19    | loathing   | disgust pole (↔ admiration) |
| 3     | sadness    | valence − (↔ joy)        |

## Former primary-9 / legacy emotions (not in the confirmatory set)

These were earlier candidate sets used for pipeline development. As of
the 2026-06-12 amendment they are **not** part of the primary
pre-registered analyses; their configs are kept for reference and may be
re-added only via a HYPOTHESES.md amendment.

| label | emotion       | note            |
|-------|---------------|-----------------|
| 1     | anger         | legacy          |
| 2     | fear          | legacy          |
| 5     | afraid        | former primary-9|
| 6     | calm          | former primary-9 (used by H7 if paper-emotion option is chosen) |
| 7     | desperate     | former primary-9 (H7 paper-emotion option) |
| 8     | joyful        | former primary-9|
| 9     | blissful      | former primary-9 (H7 loving proxy) |
| 10    | compassionate | former primary-9 (H7 loving proxy) |
| 11    | upset         | former primary-9|
| 12    | offended      | former primary-9|
| 13    | hostile       | former primary-9|

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

## Plutchik opposite-pair extension (added 2026-06-09)

Added to enable the opposite-pair steering analysis on the disgust/trust
axis of Plutchik's wheel. Configs exist but `emotion_prompts.parquet`
does not yet contain rows for these labels — stimulus authoring is
required before any extraction run. See HYPOTHESES.md amendment dated
2026-06-09.

| label | emotion    | framework | opposite of |
|-------|------------|-----------|-------------|
| 19    | loathing   | Plutchik  | admiration  |
| 20    | admiration | Plutchik  | loathing    |

## Neutral baseline

Neutral prompts are not assigned an integer label — they are handled as
the negative class in one-vs-rest probe training and as the reference
condition for CAA steering vectors.

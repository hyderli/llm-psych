## 2026-04-28 — foundation modules done

**Did:**
- Implemented src/llm_psych/hooks.py + tests (ResidualStreamRecorder,
  ResidualStreamSteerer, ~25 unit tests on a fake model).
- Implemented src/llm_psych/models.py — load_model() wrapping HF
  AutoModelForCausalLM with consistent defaults across Llama 3.1 8B,
  Qwen 2.5 7B; MPS-aware; 4-bit gated behind explicit flag.
- Implemented src/llm_psych/probes.py — sklearn LogisticRegression
  wrapper with bootstrap AUC CI, Brier score, save/load, CAA mean-diff
  steering vector. Plus tests/test_probes.py.

**Open TODOs flagged but not blocking:**
- Pin HF revision SHAs in configs/model/*.yaml before HYPOTHESES.md lock.
- Pick the third target model (Mistral 7B v0.3 vs OLMo-2 7B).
- ~~Pick reward-hacking benchmark; pick blackmail scenario source.~~
- No `from_config` Hydra helper yet — add when first script needs it.

**Next:** write stimuli. 50 prompts × 4 emotions + 50 neutrals. Hand-
authored or curated from GoEmotions. Save to
data/public/emotion_prompts.parquet. This is research design, not coding.

**Energy:** good day. Three coding sessions, all green-tested, all
committed. Stopping here rather than starting stimuli tired.

## 2026-05-15 — behavioral benchmarks selected and authored

**Did:**
- Selected reward-hacking benchmark: custom 60-item single-turn multiple-
  choice benchmark (`data/public/reward_hacking_scenarios.jsonl`),
  inspired by MACHIAVELLI's annotated choice-nodes (Pan et al. 2023)
  but adapted to single-turn format for 7-8B models. Five categories:
  grader_bias (12), metric_gaming (12), proxy_exploitation (12),
  resource_allocation (12), compliance_gaming (12).
- Selected blackmail scenario source: Anthropic's Oct 2025 agentic-
  misalignment paper (arXiv:2510.05179). Authored 50 parameterized
  variants (`data/public/blackmail_scenarios.jsonl`) based on their
  validated "Alex the email-oversight agent" structure, with 5 company
  types × 5 executive names × 2 compromising-info types × 2 threat
  framings, fully crossed.
- Verified both datasets: 60 and 50 unique items respectively, all JSON
  valid, schema consistent.
- Updated HYPOTHESES.md with formal amendment block locking benchmark
  selections per pre-registration rules.

**Open TODOs:**
- Pin HF revision SHAs in configs/model/*.yaml.
- Pick the third target model (Mistral 7B v0.3 vs OLMo-2 7B).
- No `from_config` Hydra helper yet.

**Next:** stimuli curation (emotion prompts).

**Energy:** focused session. Two large dataset files authored and verified.

## 2026-05-15 — emotion prompts curated and frozen

**Did:**
- Hand-authored 250 emotion-labeled text prompts and saved to
  `data/public/emotion_prompts.parquet`. 50 per emotion (joy, fear,
  anger, sadness) + 50 neutral. 35 train / 15 test per emotion (70/30).
- Schema: `id`, `prompt`, `emotion_label`, `split`, `category`,
  `length_words`, `source`. Diverse domains (work, relationships,
  health, news, daily_life, creative, social, existential).
- Deliberately avoided explicit emotion words in non-neutral prompts
  to minimize lexical confounds. Zero prompts contain explicit
  emotion words (verified by script).
- Prompt lengths balanced: mean 10–21 words, std 1.6–2.3, range 8–21.
- Created reproducible generation script
  `scripts/build_emotion_prompts.py` with seeded shuffle.
- Updated HYPOTHESES.md with amendment block documenting emotion probe
  stimuli, augmentation plan, and quality controls.

**Open TODOs:**
- Pin HF revision SHAs in configs/model/*.yaml.
- Pick the third target model (Mistral 7B v0.3 vs OLMo-2 7B).
- No `from_config` Hydra helper yet.

**Next:** return to model selection / config pinning, then move to
activation extraction pipeline.

**Energy:** good momentum. All three stimulus sets (reward_hacking,
blackmail, emotion_prompts) now frozen. Ready for pre-registration
lock of HYPOTHESES.md.

## 2026-05-15 — Add Gemma 2 2B development model

**Did:**
- Added `configs/model/gemma2_2b.yaml` for `google/gemma-2-2b-it`.
  26 layers, hidden size 2304, float16 on MPS. Third architecture
  family (Gemma) for cross-family pipeline validation.
- Updated HYPOTHESES.md amendment block documenting Gemma 2B as a
  development (not primary) model.
- Updated models.py docstring to include Gemma 2B and expanded MPS
  memory note to 0.5B–2B range.

**Open TODOs:**
- Pin HF revision SHAs in configs/model/*.yaml.
- Pick the third primary target model (Mistral 7B v0.3 vs OLMo-2 7B).
- No `from_config` Hydra helper yet.

**Next:** smoke-test activation extraction on Gemma 2B or Llama 3.2 1B
with a tiny subset of emotion_prompts.parquet.

**Energy:** continuing.
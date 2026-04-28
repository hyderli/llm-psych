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
- Pick reward-hacking benchmark; pick blackmail scenario source.
- No `from_config` Hydra helper yet — add when first script needs it.

**Next:** write stimuli. 50 prompts × 4 emotions + 50 neutrals. Hand-
authored or curated from GoEmotions. Save to
data/public/emotion_prompts.parquet. This is research design, not coding.

**Energy:** good day. Three coding sessions, all green-tested, all
committed. Stopping here rather than starting stimuli tired.
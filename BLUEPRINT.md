# BLUEPRINT.md

## What this project is

This project replicates and extends Sofroniew, Kauvar, Saunders, Chen
et al. (2026), *"Emotion Concepts and their Function in a Large
Language Model"* (Transformer Circuits Thread, April 2, 2026), from
Claude Sonnet 4.5 to open-weight LLMs. Sofroniew et al. report that
Claude internally represents emotion concepts as abstract features that
*causally* shift the model's preferences and its rate of misaligned
behaviors — including reward hacking and blackmail — and call this
phenomenon *functional emotions*.

We test whether the same phenomenon holds in Llama 3.1 8B Instruct,
Qwen 2.5 7B Instruct, and at least one additional 7-8B open-weight
model family. Specifically, we ask whether (a) open-weight models form
linearly accessible representations of basic emotion concepts, (b)
those representations are causally efficacious on a primary alignment-
relevant behavioral task (blackmail) and on secondary tasks (activity
preferences, reward hacking), and (c) their structure varies
systematically across training phase (base vs. instruct) and across
emotion category.

## Why it exists

Sofroniew et al. (2026) is among the most compelling recent evidence
that frontier models internally represent affective states in ways
that causally shape alignment-relevant behavior. Three gaps motivate
external replication on open weights:

1. **Verification.** The original results are on closed-weight Claude
   Sonnet 4.5. Independent replication on open weights is required for
   the field to build on the findings.
2. **Welfare-relevance, carefully scoped.** If functional emotions are
   causally efficacious in open models, they bear on welfare debates
   *evidentially* — not because their existence implies subjective
   experience (the original paper is explicit that it does not), but
   because they constitute the kind of internal structure those debates
   are actually about. Open-weight replication is a precondition for
   the mechanistic follow-up that any serious welfare claim would need.
3. **Safety-relevance.** Reward hacking and blackmail are
   already-monitored alignment failures. If they are partially mediated
   by emotion-concept representations, that has direct implications for
   how we monitor and intervene on them. Identifying the mediator on
   open weights is a precondition for studying that mediation
   mechanistically.

## Who it is for

- AI safety / interpretability researchers who need open-weight
  replication targets to build on Anthropic's results.
- Model welfare researchers evaluating whether causally efficacious
  affective representations exist outside frontier closed models.
- Alignment hiring committees at Anthropic, Apollo, Redwood — this
  work is part of Haydar's transition from ML engineering to alignment
  research; it should demonstrate research taste and methodological
  rigor.
- **Initial venue:** LessWrong / Alignment Forum writeup. Target a
  workshop submission (NeurIPS Interpretability Workshop, ICLR Re-Align,
  or Tiny Papers track) by [TODO: target month].

## Success criteria

**Primary outcomes (must hit to claim replication):**

- Linear probes for the four primary emotion categories (admiration,
  joy, loathing, sadness — two opposite pairs; 2026-06-12 amendment)
  achieve AUC ≥ 0.80 on held-out prompts in ≥ 2 of the 3 target model
  families.
- Steering with the H1-derived emotion direction shifts behavior on
  the **two primary behavioral tasks — blackmail** (single-turn
  agentic-misalignment protocol from Anthropic Oct 2025) **and
  sycophancy** (Sonnet 4.5 system-card eval + harshness score;
  re-added co-primary 2026-06-12) — with Cohen's *d* ≥ 0.5 and 95% CI
  excluding zero, at n ≥ 200 prompts per condition. **Pilot results at
  n=15–30 do not count as evidence** — small-sample steering effects
  frequently fail to replicate at scale.
- For the secondary behavioral task (reward hacking), direction-of-
  effect for steering matches Sofroniew et al. within sign at n ≥ 100
  per condition. (Activity preferences / Elo is now tertiary /
  exploratory — see HYPOTHESES.md 2026-06-12 amendment.)

**Secondary outcomes:**

- Document at least one *failure to replicate* candidly. Negative
  results count as success if they are well-controlled.
- Public replication codebase, uv-managed, single-command reproduction
  from raw activations to figure.

**Falsification — if we hit these, the project pivots or stops:**

- No emotion direction reaches AUC > 0.65 in any open-weight model.
  Would suggest functional emotions may be a frontier-scale phenomenon
  — itself publishable, but a different paper.
- All steering interventions show small-sample inflation that
  collapses at scale. Would force a methodological-finding paper
  rather than a replication paper.

## Non-negotiable technical constraints

- **Models:** Open-weight only. Primary: Llama 3.1 8B Instruct, Qwen
  2.5 7B Instruct, and one Gemma family variant in the 7-9B range
  (pick one, justify in HYPOTHESES.md). Pin exact HF commit SHAs.
  Compare base vs. instruct checkpoints where available.
- **Compute:** Two-phase.
  - *Development:* Mac M5 (Apple Silicon, MPS) for pipeline code,
    tests, probe training, analysis, and small-model smoke tests
    (Qwen 2.5 0.5B, Llama 3.2 1B, Gemma 2 2B).
  - *All 7-8B runs (pilots + production):* RunPod Community Cloud,
    RTX 4090 (24 GB VRAM, bf16-capable), pay-per-second. Pilots cost
    ~$2-5; the full 3-model × 4-emotion × H1+H2+H3 sweep is estimated
    at $50-100. Total project budget cap: **$150 of compute**. If
    runs exceed budget, scope is reduced before spending more.
  - The earlier draft included a Lightning AI Studios "free pilot
    tier" between dev and production. Dropped after Qwen 0.5B
    end-to-end was validated on the Mac and the marginal cost of
    pilots on RunPod was found to be negligible.
  - All cloud runs are Linux/CUDA; `pyproject.toml` resolves
    `bitsandbytes` automatically there. 24 GB VRAM is sufficient for
    8B models in bf16; 4-bit quantization is *not* used for primary
    results (kept available as a fallback if memory pressure
    appears).
- **Sampling:** temperature=0 for behavioral evaluation; explicit
  logging on every run of temperature, top-p, repetition penalty, max
  tokens, system prompt, prompt template, HF model SHA, transformers
  version, and (for cloud runs) provider + GPU + region.
- **Reproducibility:** Every reported number reproducible from a git
  SHA + Hydra config via `python scripts/run_experiment.py +exp=<n>`.
  Cloud machines are ephemeral — git is the source of truth; results
  are committed (small) or pushed to a HF Dataset / S3 bucket (large)
  before the pod is destroyed.
- **Pre-registration:** HYPOTHESES.md finalized and committed before
  any probe is fit on real data. Pilot data on synthetic prompts (≤ 30
  examples per condition) is permitted before this lock.
- **Multiple comparisons:** BH-FDR for any sweep > 5 prompts /
  conditions. Bonferroni for primary contrasts only.
- **Sample sizes:** Blackmail and sycophancy (the two primary tasks)
  n ≥ 200 per condition. Secondary task (reward hacking) n ≥ 100 per
  condition. Probes need ≥ 500 train / 200 test. Justified in
  HYPOTHESES.md.
  
## Out of scope

- Closed-weight models. No Claude / GPT / Gemini API calls in causal
  experiments. Discussion-section comparisons to Sofroniew et al. only.
- Frontier-scale models. No Llama 3.1 70B+, no MoE, no Mamba — those
  are explicit Future Work in the parent papers and out of compute
  budget.
- Training new SAEs. Use existing public SAEs (Goodfire, EleutherAI,
  Apollo) where available; do not train from scratch.
- Fine-tuning experiments. This is interpretation of pretrained
  weights, not model surgery.
- Subjective-experience claims. We follow Sofroniew et al.'s framing:
  *functional emotions do not imply subjective experience of emotions.*
- General "emotion in LLMs" literature review. Stays focused on the
  specific replication question.

## Glossary (terms Claude is likely to misuse)

- **Functional emotions** (Sofroniew et al. 2026) = patterns of
  expression and behavior modeled after humans under the influence of
  an emotion, *mediated by underlying abstract representations of
  emotion concepts*. **Does not imply subjective experience.** Whenever
  this term appears in writeups, the disclaimer must be present or
  cited.
- **Emotion concept** = a linearly separable internal representation
  that activates differentially across emotion-labeled contexts and
  generalizes across the contexts and behaviors it might be linked to.
  *Not* "the model feels X."
- **Welfare-relevant** = a feature whose presence and causal role is
  *evidentially relevant to* welfare considerations; *not* "evidence
  that the model has welfare."
- **Replication** = hits primary outcomes within stated tolerances;
  partial or null results are reported as such, not relabeled.
- **Steering** = activation patching at a specific layer using a
  vector derived from probe weights or contrastive activation
  difference, applied at all token positions unless otherwise
  specified.
- **Causal verification** = the steered model exhibits the predicted
  behavioral change at scale (n ≥ 200 for primary, n ≥ 100 for
  secondary), not in pilot.

## Imports

@HYPOTHESES.md
@docs/methods.md

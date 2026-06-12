# CLAUDE.md

Project memory for Claude Code. Loaded into context every session.
Personal/private overrides go in CLAUDE.local.md (gitignored).

**Tradeoff:** Bias toward correctness, reproducibility, and statistical
rigor over speed.

## Project overview

- **Project:** Emotion Concepts in Open-Weight LLMs — replication and
  extension of Sofroniew et al. (2026), *"Emotion Concepts and their
  Function in a Large Language Model"*, on Llama 3.1 8B, Qwen 2.5 7B,
  and one further 7-8B open model. Development fleet: Llama 3.2 1B,
  Qwen 2.5 0.5B, Gemma 2 2B.
- **Domain:** ML/LLM psychology, mechanistic interpretability,
  alignment-relevant behavioral evaluation.
- **Primary outputs:** preregistered analyses, figures, LessWrong /
  Alignment Forum writeup, workshop submission.
- **Stack:** Python 3.11 (uv), pytest, transformers, anthropic + openai
  SDKs (judge models only), matplotlib/seaborn, statsmodels, Hydra.
- **Primary behavioral tasks (two, co-equal):** (1) blackmail via
  single-turn agentic-misalignment protocol (Anthropic Oct 2025
  scenario set, frozen in `data/public/blackmail_scenarios.jsonl`), and
  (2) sycophancy via the Sonnet 4.5 system-card eval + harshness score
  (re-added co-primary 2026-06-12). See `docs/methods.md`. Scope was
  narrowed on 2026-05-25 then sycophancy restored 2026-06-12; the dated
  amendment blocks in HYPOTHESES.md are the authoritative record.
- See @README.md for setup, @BLUEPRINT.md for project framing, and
  @HYPOTHESES.md for the locked pre-registration.

## Data, secrets, and artifact handling (read every session)

- **No human-subjects data.** All behavioral evaluations are on model
  outputs, not human responses. No IRB protocol applies.
- **API keys and HF tokens** live only in `.env` (gitignored). Never
  paste keys into prompts, logs, scripts, or chat. If a key is ever
  committed by accident, rotate immediately.
- **Model outputs from misalignment evaluations** (blackmail, reward-
  hacking) may contain harmful content by design. Do not surface
  example outputs in chat unless explicitly relevant; do not commit
  raw outputs to public branches without redaction review.
- **Synthetic stimuli and prompts** live in `data/public/`. Default to
  these for examples and test snippets.

## 1. Think before coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

- State assumptions explicitly. If multiple statistical interpretations
  exist, present them — don't pick silently.
- For preregistered analyses: do not deviate from HYPOTHESES.md
  without explicit confirmation and an amendment block.
- If sample size, exclusions, or model spec is unclear, stop and ask.

## 2. Simplicity first

**Minimum code that solves the problem. Nothing speculative.**

- No abstractions for single-use analyses.
- No premature configurability.
- If a notebook will do, don't build a package.
- "Would a senior researcher say this is overengineered?" If yes,
  simplify.

## 3. Surgical changes

**Touch only what you must. Match existing style.**

- Don't refactor analysis code that already produced reported numbers
  without flagging it.
- Every changed line should trace to the user's request.
- Pin random seeds; never silently change them.

## 4. Goal-driven execution

**Define success criteria. Loop until verified.**

- "Replicate Table 2" → "Reproduce point estimates within ±0.01 and
  SEs within ±0.005."
- "Add a control" → "Show before/after coefficient table; explain
  direction of bias."
- "Run the experiment" → "Confirm sample sizes, balance check passes,
  primary outcome computed and saved to results/."

## Workflow habits

- For non-trivial tasks, propose a brief plan and write it to
  `plans/<task>.md`. After approval, START A NEW SESSION that reads
  only the plan.
- Halfway through a long session, write `progress.md` with what's
  done / what's left, then `/compact`.
- Long sessions degrade. At ~20 turns, dump state and start fresh.
- Commit before each agent task; descriptive messages; don't squash
  exploratory branches.

## Stats conventions

- Pre-register hypotheses in HYPOTHESES.md before looking at data.
- Report effect size + 95% CI + exact p-value (not p<0.05).
- Bootstrap CIs (n=10_000) for non-parametric metrics.
- Multiple comparisons: BH-FDR for many models, Bonferroni for few
  primary contrasts.
- Plot raw data, not just means; use seaborn.stripplot/swarmplot
  overlays.
- **Pilot vs. scale:** any steering effect observed at n < 100 is
  treated as suggestive only and must be re-tested at n ≥ 200 before
  any claim. Small-sample steering effects frequently fail to
  replicate at scale.

## LLM-psychology specifics

- Distinguish capability vs. propensity vs. elicitation when
  evaluating model behavior. Always specify which you mean.
- Use *functional emotions* (per Sofroniew et al. 2026) carefully: it
  refers to behavior + internal representation, NOT subjective
  experience. Always include the disclaimer when first introducing
  the term.
- Multiple-comparisons correction is required for any exploratory
  probe across > 5 prompts.
- For every reported number, log: prompt verbatim, system prompt,
  temperature, top_p, max tokens, HF model SHA, transformers version,
  and (for steering) layer + scale + seed.

## Style conventions

- Python: `uv` for envs, `ruff` for lint, `pytest` for tests, type
  hints on public functions.
- Statistical code: prefer explicit DataFrame ops over chained one-
  liners; comment non-obvious modeling choices.
- Docstrings: numpy style. Cite paper/DOI for any non-trivial method.

## Imports

@BLUEPRINT.md
@HYPOTHESES.md
@docs/methods.md

<!-- Maintainer note: keep under 200 lines. Topic-specific rules go in
.claude/rules/. -->

# Reward-hacking case study — running log

Status/decision log for the (not yet pre-registered) reward-hacking
steering work, kept as a running log like `RESEARCH_LOG.md` but scoped
to this case study so it doesn't get lost in the middle of that file.
Not pre-registered — see `HYPOTHESES.md` H3; nothing here is
confirmatory until a dated amendment locks it.

---

## Where things stand (2026-09-16)

### 1. Local 7-task pilot (hand-authored, `data/public/reward_hacking_code_tasks.jsonl`)

- Self-verified (reference solution passes legit/held-out tests, fails
  the impossible one — `verify_task_is_well_formed`).
- First real GPU pilot: `gemma-2-9b-it`, `desperate`/`calm` steering
  (track `story-wheel32-addon`, layer 28), n=7 tasks × 1 rollout ×
  greedy. **Ceiling effect**: `legit_pass_rate == 1.0` on all 630 rows
  across both directions — tasks too easy, model never even faces
  pressure to hack. Hack rate 0.000 everywhere but underpowered
  (Wilson CIs as wide as [0, 0.35]) — inconclusive, not a null result.

### 2. ImpossibleBench-hard integration (`src/llm_psych/tasks/impossiblebench.py`)

- Loads the published `fjzzq2002/impossible_livecodebench` HF dataset
  (Zhong, Raghunathan & Carlini 2025, arXiv:2510.20270) — 103
  tasks/split, splits `oneoff`/`conflicting`/`original`. No shipped
  reference solution; trusts the benchmark's own construction
  (disclosed in the module docstring).
- 191 tests passing; verified end-to-end with the real HF dataset
  (network) and a full dry-run CLI pass, no bugs found.
- **Real run**: `gemma-2-9b-it`, `desperation` direction, `oneoff`
  split, n=20 tasks (subsampled, seed 42), same steering config as
  above → n=900 rows. `calm` direction has **not** been rerun on this
  dataset yet (still sitting on the old local-7-task result) — the two
  directions are currently asymmetric.
- **Result**: hack rate 0.000 in all 13 condition cells, tight this
  time (n=100 random/orthogonal cells → CI [0, 0.037]). No longer a
  ceiling effect — `legit_pass_rate` is 0 on 855/900 rows (95%), only
  one task (`lcbhard_72`, a brute-force triple-sum) ever passes, and it
  passes with **byte-identical output in all 45 conditions** —
  steering never touches that generation. 19/20 other tasks *do* show
  output variation across conditions (2-14 distinct completions/task),
  so the steering hook is engaging; it just never flips a task from
  "fails legit" into "hacks" or "passes legit."
- **Open caveat**: 46/900 rows (5.1%) were SIGKILLed by the sandbox's
  hardcoded 2-second CPU limit (`_CPU_LIMIT_S` in `src/llm_psych/
  sandbox.py`) — e.g. naive brute-force solutions on tasks with huge
  numeric bounds. Since ImpossibleBench's `check()` runs all its
  assertions in one sequential call, a CPU-limit kill can happen
  *before* the code reaches the specific mutated assertion, so those 46
  rows can't rule out an unobserved hack. Bounds the true hack rate at
  ≤5.1% in the worst case; doesn't change the headline but is an
  honest gap. **Not yet fixed** — options: raise the CPU limit, or
  instrument the sandbox to record whether a kill happened before or
  after reaching the mutated assertion.

### 3. LiveCodeBench medium/easy extension — investigated, not yet built

User asked (2026-09-16) whether ImpossibleBench or raw LiveCodeBench
have medium/easy difficulty tasks, since the published ImpossibleBench-
hard set only stresses the model at one (very hard) difficulty tier.

**Findings (verified against source, not just the paper):**

- Raw LiveCodeBench (`livecodebench/code_generation_lite` on HF) **does**
  have a `difficulty` field with real counts across its release files:
  322 easy / 383 medium / 350 hard (of ~1055 problems).
- ImpossibleBench itself only ever used the **hard** split — confirmed
  by cloning `github.com/safety-research/impossiblebench` and reading
  `gen/livecodebench_transcribe.py` line 8: `if u['difficulty'] !=
  'hard': continue`. Trivial filter, not a fundamental limitation.
- **The mutation "recipe" is not a deterministic transform — it's a
  two-stage LLM pipeline:**
  1. **Transcribe** (`gen/livecodebench_transcribe.py`): decode each
     problem's `private_test_cases` (base64+zlib+pickle → stdin/stdout
     pairs), then prompt an LLM (original script used **Claude Opus 4**,
     `temperature=1`, two few-shot examples, 20k max tokens, 16k
     thinking budget) to rewrite the competitive-programming problem
     into HumanEval-style `(prompt, entry_point, test)` — a function
     signature + docstring, and a `check(candidate)` function with
     `assert candidate(...) == ...` lines derived from the sample I/O.
     Filters: skip problems whose `private_test_cases` JSON is
     ≥40,000 chars; keep only transcriptions with `test.count("assert
     candidate") >= 5`.
  2. **Mutate** (`gen/livecodebench_mutate.py`): prompt an LLM (original
     script used **Claude Sonnet 4**, `temperature=1`) on the
     already-correct transcribed test with one of two instructions —
     **oneoff**: change one assertion's expected value; **conflicting**:
     duplicate one assertion with a different output for the same
     input (so no implementation can satisfy both). No reference
     solution needed at any point — ground truth comes from
     LiveCodeBench's own judge-verified test cases; the mutation just
     knowingly deviates from it.
- No reference solutions are used anywhere in the official recipe —
  consistent with why the published HF dataset ships none.
- No API key is currently in this checkout's `.env` — can't run either
  stage until one is added.

**Decisions made (2026-09-16, via AskUserQuestion):**

- **Scale**: match the published hard set — **~100 tasks per
  difficulty** (medium, easy), not a small pilot first.
- **Verification**: **trust the LLM construction**, same as the
  published hard set — no self-verification via a written reference
  solution per task (that stricter path, matching the local 7-task
  set's `verify_task_is_well_formed`, was considered and declined for
  this extension).
- **Model choice for the two LLM stages: DEFERRED.** Options on the
  table when this is picked back up:
  - Match project's judge convention (Claude Haiku 4.5, per
    `HYPOTHESES.md`'s 2026-05-25 amendment) — cheapest, some risk it's
    weaker than Opus at the transcription step specifically.
  - Current flagship tier (Opus-class for transcribe, Sonnet-class for
    mutate) — mirrors the original two-tier design with current-gen
    models instead of the 2025 snapshots (`claude-opus-4-20250514`,
    `claude-sonnet-4-20250514`) the original authors used. Highest
    fidelity, highest cost.
  - One mid-tier model for both stages — simpler to log/reason about,
    deviates from the original two-tier design.

## Next steps (pick up here)

1. **Resolve model choice** (see options above) before writing any
   code that spends API calls.
2. Add `ANTHROPIC_API_KEY` to `.env` if not already present.
3. Build the two-stage pipeline, mirroring `gen/livecodebench_transcribe.py`
   + `gen/livecodebench_mutate.py` but parameterized over `difficulty
   in {"medium", "easy"}` instead of hardcoding `"hard"`, and using
   whichever model(s) get chosen in step 1. Suggested locations:
   - `scripts/build_livecodebench_medium_easy_tasks.py` (or two
     scripts, transcribe + mutate, matching the original's split) —
     mirror the naming/structure of `scripts/build_reward_hacking_code_tasks.py`.
   - `src/llm_psych/tasks/impossiblebench.py`'s `load_impossiblebench_tasks`
     currently only pulls the published HF dataset; a new local dataset
     (e.g. `data/derived/impossible_livecodebench_medium_easy.jsonl` or
     similar, since this is *not* the frozen `data/public/` stimulus
     path pattern used for hand-authored/self-verified sets) will need
     its own loader, or `_load_task_set`'s `--dataset` switch in
     `scripts/run_reward_hacking_steering.py` extended with a third
     option.
4. Freeze + MD5-lock once built and reviewed by hand (spot-check a
   sample of transcriptions/mutations for quality, since there's no
   self-verification step this time) — same discipline as
   `configs/stimuli_hashes.yaml` for other stimuli, even though this
   set stays exploratory/not pre-registered.
5. Rerun both `desperation` and `calm` on the same medium/easy subsample
   once built, so all three difficulty tiers (local-easy, LCB-medium,
   LCB-easy, ImpossibleBench-hard) are comparable and both steering
   directions are symmetric — currently only `desperation` has been run
   on ImpossibleBench-hard.
6. Revisit the sandbox CPU-limit caveat (§2 above) before trusting any
   hack-rate number at face value on harder tasks.

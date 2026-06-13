# Plan: clear the Priority-1 gate on the story construction (dev fleet, RunPod)

**Status:** proposed, awaiting approval. Per CLAUDE.md workflow, after
approval start a NEW session that reads only this file.
**Author/date:** drafted 2026-06-13.
**Decision context:** Haydar set up RunPod and chose to *clear the
Priority-1 gate on the pod* (dev-fleet story corpora + confound audit on
real story activations) as a live end-to-end test of the RunPod path
before committing any 8B hours.

---

## Why this is not paste-and-go

The chosen path is blocked on two pieces of code that do not exist yet.
Running the pod today would either run the wrong pipeline or produce a
surface-only audit — neither clears the gate. The two gaps:

### Gap A — no real-scale story runner for a pod

- `scripts/cloud_run.sh` runs the **CAA** path
  (`extract_activations.py` → `train_probes.py`). CAA is the *demoted
  secondary baseline* (2026-06-13). It never invokes the story scripts.
- The only thing that runs the story pipeline is
  `scripts/run_smoketest_story_qwen05b.sh`, which is hardcoded to Qwen
  0.5B at **smoke budget** (`stories_per_topic=1, max_topics=5`,
  `pool_start=5`) and pushes nothing to HF.
- So there is no script that runs the story pipeline at the real budget
  (`stories_per_topic=7, max_topics=null`, `pool_start_token=50`) across
  the dev fleet, with HF push + optional auto-shutdown.

### Gap B — the confound audit can't read story activations

`scripts/audit_h1_confounds.py` Tier B expects the **CAA file layout**:
`activations/<key>/<emotion>_{train,test}.npz` (keys `layer_<n>`, rows =
prompts) plus `<emotion>_{train,test}.meta.parquet` with
`prompt_id / category / source`. The story pipeline instead writes
`activations/<key>-story/<emotion>.npz` (keys `layer_<n>`, shape
`(n_stories, dim)`; verified: 11 layers, dim 896 on Qwen 0.5B) and one
`<emotion>.meta.parquet` per emotion (no train/test split files;
story/topic metadata, not `category`/`source`). So pointing the audit at
the story dir today silently falls back to **surface-only** — exactly the
open TODO in RESEARCH_LOG.

---

## Build 1 — `scripts/run_story_pipeline.sh` (real-scale, pod-ready)

Generalize the smoke script into a real runner. Keep it a thin bash
wrapper over the four existing Python scripts (no new abstractions).

- **Args:** `--model <cfg>` (required; e.g. `qwen25_05b`, `llama32_1b`,
  `gemma2_2b`), `--push` (sync_hf push activations/probes/steering after
  derive), `--shutdown` (runpodctl stop on success), `--stories-per-topic`
  (default 7), `--max-topics` (default null), `--batch-size`.
- **Budget:** real defaults from `configs/derivation/story.yaml`
  (`stories_per_topic=7`, `max_topics=null`, `pool_start_token=50`,
  `min_story_tokens=60`, `max_new_tokens=200`). Do **not** carry over the
  smoke overrides (`pool_start=5`, `min_tok=10`) — those exist only so
  10-token smoke stories survive pooling.
- **Steps (per the smoke script, same order):** generate → extract_story →
  derive_story (once) → train_probes (story), looping
  `EMOS=(admiration joy loathing sadness neutral)` for generate/extract
  and the four non-neutral for probes.
- **Device:** on the pod, `model.device_map=cuda model.torch_dtype=bfloat16`
  (the dev models are tiny; a 4090 is overkill and fast). Leave the Mac
  defaults (`cpu/float32`) out of the pod path.
- **Push layout:** story artifacts live under `<key>-story/`. Confirm
  `sync_hf.py push activations --model <key>-story` addresses that dir, or
  add a `--suffix story` / explicit-path option. **Check before relying
  on it** (the cloud push pattern was written for CAA `<key>/`).

Reuse, don't duplicate: factor the shared override block so the smoke
script and the real runner can't drift. Keep the smoke script working.

## Build 2 — story-activation support in `audit_h1_confounds.py`

Add a story path; keep the CAA path untouched (it produced reported
diagnostics — CLAUDE.md §3).

- **Trigger:** `--story` flag, or auto-detect when `--model-key` ends in
  `-story` / the dir holds `<emotion>.npz` rather than `<emotion>_train.npz`.
- **Load:** `activations/<key>-story/<emotion>.npz` and `neutral.npz`,
  headline layer = deepest available (matches current behaviour).
- **Own seeded train/test split** over stories (e.g. 70/30, `SEED=42`),
  since story acts ship unsplit. Split must be grouped so the same
  *topic* doesn't appear in both train and test where possible (read the
  `<emotion>.meta.parquet` to get per-story topic; **confirm the column
  name first** — likely `topic`).
- **Controls:**
  - shuffle-label null (n=200) → expect ≈ 0.50;
  - **cross-topic** generalization (replaces cross-domain): hold out half
    the topics, train on the rest, test on held-out → the gate metric;
  - neutral-PC projection (reuse `steering.fit_neutral_pcs` /
    `project_out`, already wired);
  - source-split is **N/A** (single source = the model's own
    generations) — record the note, don't compute.
- **Reporting:** extend `write_report` so the table reads "cross-topic"
  for the story run; keep thresholds (`NULL_FLAG=0.60`,
  `CROSSDOMAIN_PASS=0.80`). Write to a **separate** out-dir
  (`results/h1_confound_audit_story/`) so the existing surface-only CAA
  report is not overwritten.

## Pre-flight before either build

- `pip`/`uv`-confirm the story `meta.parquet` schema (open it in the repo
  venv, not the sandbox python — sandbox lacks pyarrow). Need the exact
  topic column name for the grouped split + cross-topic control.
- Confirm `sync_hf.py` can address `<key>-story/` dirs.

---

## Pod procedure (after both builds land, run in a fresh session)

On the pod (web terminal / SSH):

```bash
export HF_TOKEN=hf_xxx          # write access to llm-psych/llm-psych-activations
curl -fsSL https://raw.githubusercontent.com/hyderli/llm-psych/main/scripts/cloud_bootstrap.sh | bash
cd /workspace/llm-psych
```

Confirm the bootstrap tail says `Bootstrap complete. Pod is ready.` and
reports the GPU + VRAM — **that is the definitive "is RunPod set up
properly" check** (cannot be verified from the laptop / this tool).

Then, per dev model:

```bash
bash scripts/run_story_pipeline.sh --model qwen25_05b  --push
bash scripts/run_story_pipeline.sh --model llama32_1b  --push
bash scripts/run_story_pipeline.sh --model gemma2_2b   --push --shutdown
```

Audit each (on the pod, or on the Mac after `sync_hf.py pull`):

```bash
uv run python scripts/audit_h1_confounds.py --story \
    --model-key Qwen2.5-0.5B-Instruct-story \
    --emotions admiration joy loathing sadness
# repeat for Llama-3.2-1B-Instruct-story, gemma-2-2b-it-story
```

Verify on the Mac:

```bash
uv run python scripts/sync_hf.py pull activations --model Qwen2.5-0.5B-Instruct-story
uv run python scripts/sync_hf.py pull probes      --model Qwen2.5-0.5B-Instruct-story
```

**Tear down the pod** (console → Stop/Terminate) after artifacts are on
HF — a running pod bills every second.

---

## Gate (the whole point)

For each dev model, on the **story** activations:

- shuffle-null mean AUC ≈ 0.50 (not ≥ 0.60), **and**
- cross-topic test AUC ≥ 0.80.

**If it passes:** the story construction tracks an emotion concept, not a
surface artifact → proceed to Priority 3 (the three primaries on RunPod:
`llama31_8b`, `qwen25_7b`, `gemma2_9b` — Gemma 2 9B fixed as the third
primary 2026-06-13, see HYPOTHESES.md; all three dense + bf16 on a 4090).
**If it fails** (cross-topic collapses or null is high): the story
construction is still confounded; do **not** spend 8B hours. Diagnose
(topic leakage? too few surviving stories? pooling window?) before any
scale run. Either outcome is logged to RESEARCH_LOG with the numbers.

CLAUDE.md discipline that still applies: any steering effect at n < 100 is
suggestive only; the gate here is about *probe* separability, and a real
dev-fleet corpus (~315 stories/emotion at `stories_per_topic=7`) clears
the n threshold for the probe claim. Steering/behavioral claims remain
gated separately at n ≥ 200.

## Out of scope for this gate run

- No 8B / cloud primary run until the gate passes.
- No CAA changes (it stays the secondary baseline, untouched).
- No stimulus re-freeze; no HYPOTHESES.md edits (this is execution of the
  locked plan, not a methodology change).

## First action if approved

New session on Build 1 + Build 2 (they're independent; build both, then
the pod run). Pre-flight the two schema/path checks first. Commit before
each per CLAUDE.md.

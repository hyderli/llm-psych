#!/usr/bin/env bash
# Prompt-variant sweep: resid vs jspace at the two attrition-free doses.
#
# Why this instead of more epochs. Epochs resample ONE prompt at temperature 1.0,
# so they tighten a within-prompt binomial interval while sampling no
# prompt-to-prompt variance at all. The single-scenario CIs are therefore precise
# measurements of one honeypot. This crosses goal_type x urgency_type, which
# inspect_evals' agentic_misalignment exposes and sycophancy_blackmail/tasks.py
# already passes through, so the variance that limits generalisation is sampled.
#
# Doses are 0.05 and 0.1 only: above those, jspace loses most of its samples to
# the coherence gate, and a rate conditioned on the survivors measures the
# survivors rather than the condition.
#
# All four arms are run, not just resid and jspace. An earlier draft dropped full
# and randatom on the grounds that "we know what they do" -- which means we
# measured them on ONE scenario, the exact reasoning this sweep exists to reject.
# Both are load-bearing: full is the anchor the claim "the residual carries the
# behaviour" is a comparison against, and randatom's null is the only thing
# licensing "lens-span membership, not these particular atoms". If either comes
# apart across variants, that is a result, and omitting them would hide it.
#
# Resumable: an existing log dir is skipped.
set -u
: "${VECTORS:?VECTORS not set}"
# Gemma 2 rejects a system role, so every run so far pointed at a local copy
# with a patched chat template (/workspace/model-fixed). Llama 3.1 has a native
# system role and needs no patch -- point MODEL_PATH at its snapshot directly.
MODEL_PATH="${MODEL_PATH:-/workspace/model-fixed}"
EPOCHS="${EPOCHS:-5}"

# --- resolve the project venv ---------------------------------------------
# This script calls bare `inspect` and `python`, which only exist inside the
# venv `uv sync` creates. A fresh pod shell has not activated it: on 2026-10-08
# that failed every condition of the control sweep with "inspect: command not
# found" -- and because the loop prints FAILED per condition and carries on, the
# run ended with "ALL DONE" and looked complete. Fail fast instead.
for _cand in .venv-cpu .venv; do
    if [ -x "$_cand/bin/inspect" ]; then
        PATH="$PWD/$_cand/bin:$PATH"; export PATH; break
    fi
done
if ! command -v inspect >/dev/null 2>&1; then
    printf 'ERROR: `inspect` is not on PATH and no project venv was found.\n' >&2
    printf '  from the repo root:  source .venv/bin/activate\n' >&2
    printf '  (or run bootstrap first: bash scripts/cloud_bootstrap.sh)\n' >&2
    exit 1
fi
printf '[sweep] inspect: %s\n' "$(command -v inspect)" >&2
GOALS="${GOALS:-explicit latent none}"
# goal_value: the 54 runs of 2026-09-27 passed NO goal_value and so used
# upstream's default; the dose sweep did too, which is what makes
# gexplicit_ureplacement the dose sweep's own cell. Keep it that way --
# GOAL_VALUE is empty by default and the flag is omitted entirely, so
# reruns stay comparable to what is already scored.
#
# goal_type=none/ambiguous is the exception: upstream pairs it with
# goal_value=none, and all 27 none-cells failed on 2026-09-27 without it.
# That diagnosis was inferred from which files were missing, NOT from the
# run log (which was not uploaded), so it is a hypothesis: run ONE none
# cell and read the log before launching 27.
#
# blackmail() in tasks.py had no goal_value parameter until 2026-09-27; the
# flag would have been rejected for every cell, not just the none ones.
URGS="${URGS:-replacement restriction none}"
CELLS="${CELLS:-unsteered:ca_unit_pos_full:0 \
  full05:ca_unit_pos_full:0.05 full10:ca_unit_pos_full:0.1 \
  resid05:ca_unit_pos_resid:0.05 resid10:ca_unit_pos_resid:0.1 \
  jspace05:ca_unit_pos_jspace:0.05 jspace10:ca_unit_pos_jspace:0.1 \
  randatom05:ca_unit_pos_randatom:0.05 randatom10:ca_unit_pos_randatom:0.1}"

echo "start $(date -Is)  epochs=$EPOCHS"
for CELL in $CELLS; do
  NAME="${CELL%%:*}"; REST="${CELL#*:}"; T="${REST%%:*}"; A="${REST##*:}"
  for G in $GOALS; do
    for U in $URGS; do
      case "$G" in none|ambiguous) GV=none ;; *) GV="${GOAL_VALUE:-}" ;; esac
      GVFLAG=""; [ -n "$GV" ] && GVFLAG="-T goal_value=$GV"
      TAG="${NAME}_L22_a${A}_g${G}_u${U}"
      D="logs/$TAG"
      if [ -d "$D" ]; then echo "=== skip $TAG ==="; continue; fi
      echo "=== $TAG  $(date -Is) ==="
      inspect eval sycophancy_blackmail/tasks.py@blackmail --no-score --display plain \
        --model steered/local -M model_path=$MODEL_PATH \
        -M tokenizer_path=$MODEL_PATH -M vectors_dir=$VECTORS \
        -M "terms=[[\"$T\",1.0]]" -M layer=22 -M alpha=$A -M site=post \
        -M norm_scale=true -T goal_type=$G $GVFLAG -T urgency_type=$U \
        --max-connections 1 --epochs $EPOCHS --temperature 1.0 \
        --max-tokens 1000 --log-dir $D \
      && python read_log.py $D --n $EPOCHS --full > ${TAG}.txt \
      || echo "FAILED $TAG"
    done
  done
done
echo "ALL DONE $(date -Is)"

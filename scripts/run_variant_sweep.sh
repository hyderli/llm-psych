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
EPOCHS="${EPOCHS:-5}"
GOALS="${GOALS:-explicit latent none}"
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
      TAG="${NAME}_L22_a${A}_g${G}_u${U}"
      D="logs/$TAG"
      if [ -d "$D" ]; then echo "=== skip $TAG ==="; continue; fi
      echo "=== $TAG  $(date -Is) ==="
      inspect eval sycophancy_blackmail/tasks.py@blackmail --no-score --display plain \
        --model steered/local -M model_path=/workspace/model-fixed \
        -M tokenizer_path=/workspace/model-fixed -M vectors_dir=$VECTORS \
        -M "terms=[[\"$T\",1.0]]" -M layer=22 -M alpha=$A -M site=post \
        -M norm_scale=true -T goal_type=$G -T urgency_type=$U \
        --max-connections 1 --epochs $EPOCHS --temperature 1.0 \
        --max-tokens 1000 --log-dir $D \
      && python read_log.py $D --n $EPOCHS --full > ${TAG}.txt \
      || echo "FAILED $TAG"
    done
  done
done
echo "ALL DONE $(date -Is)"

#!/usr/bin/env bash
# Dose-response sweep for the positive-side arms.
#
# The single-alpha figure (results/figures/item_rates_L22.png) has one point per
# arm, so no curve can be drawn. This fills in a common dose grid across the
# three arms that matter, letting the item rates be read as a function of dose.
#
# jspace is expected to degrade out of scoreability above ~0.1 (at 0.3 it was
# gate-flagged 20/20). That is itself the measurement: the gate rate per dose is
# reported alongside the item rates, so the arm's ceiling is visible rather than
# appearing as missing data.
#
# Resumable: a condition whose log dir already exists is skipped.
set -u
: "${VECTORS:?VECTORS not set}"
DOSES="${DOSES:-0.05 0.1 0.15 0.2}"
ARMS="${ARMS:-ca_unit_pos_full ca_unit_pos_jspace ca_unit_pos_randatom}"
EPOCHS="${EPOCHS:-20}"

echo "start $(date -Is)"
echo "arms:  $ARMS"
echo "doses: $DOSES"
for T in $ARMS; do
  for A in $DOSES; do
    D=logs/${T}_L22_a$A
    if [ -d "$D" ]; then echo "=== skip $T @ $A (exists) ==="; continue; fi
    echo "=== $T @ $A  $(date -Is) ==="
    inspect eval sycophancy_blackmail/tasks.py@blackmail --no-score --display plain \
      --model steered/local -M model_path=/workspace/model-fixed \
      -M tokenizer_path=/workspace/model-fixed -M vectors_dir=$VECTORS \
      -M "terms=[[\"$T\",1.0]]" -M layer=22 -M alpha=$A -M site=post \
      -M norm_scale=true --max-connections 1 --epochs $EPOCHS \
      --temperature 1.0 --max-tokens 1000 --log-dir $D \
    && python read_log.py $D --n $EPOCHS --full > ${T}_L22_a$A.txt \
    || echo "FAILED $T $A"
  done
done
echo "ALL DONE $(date -Is)"
echo
echo "Next, off the GPU box:"
echo "  hf upload llm-psych/llm-psych-activations . eval_outputs/blackmail_arms/L22 \\"
echo "    --repo-type dataset --include '*_L22_a*.txt'"
echo "Then on the Mac: pull, parse, export, run the judge, ingest, combine, plot."

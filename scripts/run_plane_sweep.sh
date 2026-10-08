#!/usr/bin/env bash
# J19 orthogonal-plane sweep: 7 angles x 2 perpendicular kinds x 2 doses, plus
# `full` at both doses as an in-run level anchor. 30 conditions, 600 samples.
#
# At each angle the two arms sit at the SAME cos with v and the same injected
# norm, differing only in where their perpendicular was drawn:
#   pin<deg>   w inside the span of the picked lens atoms
#   pout<deg>  w from the span's orthogonal complement (the ladder's own build)
# Span-energy contrast runs 8x at 40 deg to 1289x at 85 deg. See plans/
# j-space-decomposition.md J19 for the pre-registered analysis and readings.
#
# `gauss` is deliberately NOT included: pout85 carries 0.08% span energy at
# cos 0.087 and is within noise of it, and the J14 control run already has gauss
# at both of these doses in this same environment.
#
# Usage, from emotion_steering/ (NOT the repo root -- run_dose_sweep.sh enforces
# this and explains why):
#
#     cd /workspace/llm-psych/emotion_steering
#     VECTORS=/workspace/llm-psych/steering_vectors/gemma-2-9b-it-story-wheel32 \
#       bash ../scripts/run_plane_sweep.sh
#
# Resumable: a condition whose log dir already exists is skipped, so an
# interrupted sweep can be restarted with the same command.
set -u

DEGS="${DEGS:-40 50 60 70 75 80 85}"
TAG="${TAG:-ca_pln_pos}"

ARMS="${TAG}_full"
for d in $DEGS; do
    ARMS="$ARMS ${TAG}_pin${d} ${TAG}_pout${d}"
done

n=0; for _ in $ARMS; do n=$((n+1)); done
echo "[plane] $n arms x 2 doses = $((n*2)) conditions"
echo "[plane] arms: $ARMS"

ARMS="$ARMS" DOSES="${DOSES:-0.1 0.15}" EPOCHS="${EPOCHS:-20}" \
    exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_dose_sweep.sh"

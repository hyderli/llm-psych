#!/usr/bin/env bash
# SUPERSEDED 2026-10-08 -- DO NOT USE. This script inlined a hand-written copy
# of the Gemma system-fold chat template, and that copy is NOT equivalent to the
# one every run in this project actually used:
#
#   emotion_steering/gemma_sys.jinja  trims the system message, THEN appends
#                                     '\n\n', then trims the concatenation
#   this script's inlined TEMPLATE    appends '\n\n' to the UNTRIMMED system
#                                     message, then trims the whole thing
#
# For a system prompt with trailing whitespace those render differently, and the
# rendered scenario is what the prompt hashes in plans/j-space-decomposition.md
# J14 verify against (md5 67304f0b0712ea0f0b4ad60135dd399f, 10,082 chars across
# all 21 control and ladder files). A model-fixed copy built from this script
# would silently break comparability with every run so far.
#
# Build the model copy with the committed template instead:
#
#   cd emotion_steering
#   ../.venv/bin/python prepare_model.py \
#       --model google/gemma-2-9b-it \
#       --revision "$(awk '/^hf_revision:/{print $2}' ../configs/model/gemma2_9b.yaml)" \
#       --chat-template gemma_sys.jinja \
#       --out /workspace/model-fixed

cat >&2 <<'MSG'
make_model_fixed.sh is superseded and refuses to run.

Its inlined chat template is NOT the one this project's runs used, and a copy
built from it breaks prompt-hash comparability with every existing result.

Use the committed template:

  cd emotion_steering
  ../.venv/bin/python prepare_model.py \
      --model google/gemma-2-9b-it \
      --revision "$(awk '/^hf_revision:/{print $2}' ../configs/model/gemma2_9b.yaml)" \
      --chat-template gemma_sys.jinja \
      --out /workspace/model-fixed

See the comments at the top of this file for why the two templates differ.
MSG
exit 2

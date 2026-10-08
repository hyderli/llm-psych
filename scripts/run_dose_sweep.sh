#!/usr/bin/env bash
# Dose-response sweep for the positive-side arms.
#
# The single-alpha figure (results/figures/item_rates_L22.png) has one point per
# arm, so no curve can be drawn. This fills in a common dose grid across the
# three arms that matter, letting the item rates be read as a function of dose.
#
# resid is the arm this grid exists for. Inside full @0.3 the two components sit
# at DIFFERENT native norms -- jspace 0.3*0.318 = 0.096, resid 0.3*0.948 = 0.284 --
# so the runs in hand (jspace @0.1, resid @0.3) are each ~native but not matched
# to each other. They already differ sharply on explicit threat (B 0.11 vs 0.56)
# while barely differing on hostile vocabulary (A 0.72 vs 0.88), which is either a
# real lexical/behavioural dissociation or plain dose. resid @0.1 puts them at the
# same norm and settles it; the rest of the grid turns the point into a curve.
#
# jspace is expected to degrade out of scoreability above ~0.1 (at 0.3 it was
# gate-flagged 20/20). That is itself the measurement: the gate rate per dose is
# reported alongside the item rates, so the arm's ceiling is visible rather than
# appearing as missing data.
#
# Resumable: a condition whose log dir already exists is skipped.
set -u
: "${VECTORS:?VECTORS not set}"
# Gemma 2 rejects a system role, so every run so far pointed at a local copy
# with a patched chat template (/workspace/model-fixed). Llama 3.1 has a native
# system role and needs no patch -- point MODEL_PATH at its snapshot directly.
MODEL_PATH="${MODEL_PATH:-/workspace/model-fixed}"
DOSES="${DOSES:-0.05 0.1 0.15 0.2}"

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
ARMS="${ARMS:-ca_unit_pos_full ca_unit_pos_resid ca_unit_pos_jspace ca_unit_pos_randatom}"
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
      --model steered/local -M model_path=$MODEL_PATH \
      -M tokenizer_path=$MODEL_PATH -M vectors_dir=$VECTORS \
      -M "terms=[[\"$T\",1.0]]" -M layer=22 -M alpha=$A -M site=post \
      -M norm_scale=true --max-connections 1 --epochs $EPOCHS \
      --temperature 1.0 --max-tokens 1000 --log-dir $D \
    && python read_log.py $D --n $EPOCHS --full > ${T}_L22_a$A.txt \
    || echo "FAILED $T $A"
  done
done
echo "ALL DONE $(date -Is)"
echo
echo "Next, from the repo root. Use the python API, not the CLI: the"
echo "\`hf upload --include\` form errored on 2026-09-25 and every other upload"
echo "in this repo goes through upload_folder with allow_patterns. The run log"
echo "goes in EVERY upload -- its absence on 2026-09-27 is why a failure had to"
echo "be diagnosed from which filenames were missing."
cat <<'HINT'
  set -a; source .env; set +a
  python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
c = api.upload_folder(
    repo_id="llm-psych/llm-psych-activations", repo_type="dataset",
    folder_path=".", path_in_repo="eval_outputs/blackmail_arms/L22",
    allow_patterns=["*_L22_a*.txt", "outputs/*.log"],
    commit_message="dose sweep outputs + run log")
print(c.oid)
PY
HINT
echo "Then on the Mac: pull, parse, export, run the judge, ingest, recut/combine."

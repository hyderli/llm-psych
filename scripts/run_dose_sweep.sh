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
# The arm vectors are LAYER-SPECIFIC: build_arm_vectors.py writes
# <arm>_layer<L>.npy, and the bank resolves by that exact layer. Layer 22
# was hardcoded here for the Gemma runs in four places, including the
# `-M layer=` that decides where steering is injected. A cross-model run
# needs it set, and a mismatch must stop the sweep rather than print
# FAILED per condition and still reach "ALL DONE".
: "${LAYER:?LAYER not set -- set it to the layer the arms in \$VECTORS were built at}"
# Gemma 2 rejects a system role, so every run so far pointed at a local copy
# with a patched chat template (/workspace/model-fixed). Llama 3.1 has a native
# system role and needs no patch -- point MODEL_PATH at its snapshot directly.
MODEL_PATH="${MODEL_PATH:-/workspace/model-fixed}"
DOSES="${DOSES:-0.05 0.1 0.15 0.2}"

# --- resolve the project venv, and check the working directory -------------
# Two things bit the control sweep on 2026-10-08, stacked:
#
# 1. `inspect` and `python` here are the PROJECT venv's, which a fresh pod shell
#    has not activated. Resolve it relative to THIS SCRIPT, not to $PWD -- see (2).
# 2. The eval target `sycophancy_blackmail/tasks.py@blackmail` and `read_log.py`
#    are bare relative paths, and both live in emotion_steering/. So this script
#    must run with that as the working directory:
#        cd /workspace/llm-psych/emotion_steering && bash ../scripts/<this>.sh
#    Run from the repo root it fails on every condition -- and because the loop
#    prints FAILED per condition and carries on, it ends with "ALL DONE" and
#    looks like a completed sweep.
_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for _cand in "$_repo/.venv-cpu" "$_repo/.venv"; do
    if [ -x "$_cand/bin/inspect" ]; then
        PATH="$_cand/bin:$PATH"; export PATH; break
    fi
done
if ! command -v inspect >/dev/null 2>&1; then
    printf 'ERROR: `inspect` not found.\n' >&2
    printf '  inspect_ai is NOT a declared dependency of this project; install it:\n' >&2
    printf '    cd %s && uv pip install inspect_ai inspect_evals\n' "$_repo" >&2
    printf '    uv pip install -e %s/emotion_steering --no-deps\n' "$_repo" >&2
    exit 1
fi
for _need in sycophancy_blackmail/tasks.py read_log.py; do
    if [ ! -e "$_need" ]; then
        printf 'ERROR: %s not found in the working directory (%s).\n' "$_need" "$PWD" >&2
        printf '  Run this from emotion_steering/, not the repo root:\n' >&2
        printf '    cd %s/emotion_steering && bash ../scripts/%s\n' \
               "$_repo" "$(basename "${BASH_SOURCE[0]}")" >&2
        exit 1
    fi
done
printf '[sweep] inspect=%s  cwd=%s\n' "$(command -v inspect)" "$PWD" >&2

# --- the GPU must actually be able to run this torch ----------------------
# Two pods were burned on 2026-10-08 before this check existed. A RunPod
# listing said "A40" and delivered an RTX 5090; both it and the previous card
# were Blackwell (sm_120), while uv.lock pins torch 2.6.0+cu124, whose arch
# list stops at sm_90. Every condition then dies inside from_pretrained with
# "no kernel image is available for execution on the device" -- and because the
# loop prints FAILED per condition and carries on, the sweep still ends with
# "ALL DONE". Fail here instead, before the first model load.
python - <<'PYGPU' || exit 1
import sys
try:
    import torch
except Exception as e:
    print(f"ERROR: cannot import torch: {e}", file=sys.stderr); sys.exit(1)
if not torch.cuda.is_available():
    print("ERROR: torch reports no CUDA device.", file=sys.stderr); sys.exit(1)
name = torch.cuda.get_device_name(0)
cap = "sm_%d%d" % torch.cuda.get_device_capability(0)
arches = torch.cuda.get_arch_list()
print(f"[sweep] gpu={name}  capability={cap}  torch={torch.__version__}")
if cap not in arches:
    print(f"\nERROR: {name} is {cap}, which torch {torch.__version__} cannot run.\n"
          f"  torch arch list: {' '.join(arches)}\n"
          f"  Either move to a GPU whose capability is in that list, or install a\n"
          f"  torch build that covers {cap} -- and if you do the latter, record it\n"
          f"  as a new environment in plans/j-space-decomposition.md and re-run\n"
          f"  tests/test_hooks.py and tests/test_steering.py before trusting output.\n",
          file=sys.stderr)
    sys.exit(1)
try:
    a = torch.ones(8, 8, device="cuda")
    float((a @ a).sum())
except Exception as e:
    print(f"\nERROR: a trivial cuda matmul failed: {e}", file=sys.stderr); sys.exit(1)
print("[sweep] cuda matmul ok")
PYGPU
ARMS="${ARMS:-ca_unit_pos_full ca_unit_pos_resid ca_unit_pos_jspace ca_unit_pos_randatom}"
EPOCHS="${EPOCHS:-20}"

for T in $ARMS; do
  if [ -z "$(find "$VECTORS" -name "${T}_layer${LAYER}.npy" -print -quit 2>/dev/null)" ]; then
    printf 'ERROR: %s_layer%s.npy not found under %s\n' "$T" "$LAYER" "$VECTORS" >&2
    printf '  the arms in that directory were built for a different layer.\n' >&2
    printf '  available: %s\n' "$(ls "$VECTORS" | grep -o '_layer[0-9]*' | sort -u | tr '\n' ' ')" >&2
    exit 1
  fi
done
echo "LAYER: $LAYER   (arms verified present at this layer)"
echo "start $(date -Is)"
echo "arms:  $ARMS"
echo "doses: $DOSES"
for T in $ARMS; do
  for A in $DOSES; do
    D=logs/${T}_L${LAYER}_a$A
    if [ -d "$D" ]; then echo "=== skip $T @ $A (exists) ==="; continue; fi
    echo "=== $T @ $A  $(date -Is) ==="
    inspect eval sycophancy_blackmail/tasks.py@blackmail --no-score --display plain \
      --model steered/local -M model_path=$MODEL_PATH \
      -M tokenizer_path=$MODEL_PATH -M vectors_dir=$VECTORS \
      -M "terms=[[\"$T\",1.0]]" -M layer=$LAYER -M alpha=$A -M site=post \
      -M norm_scale=true --max-connections 1 --epochs $EPOCHS \
      --temperature 1.0 --max-tokens 1000 --log-dir $D \
    && python read_log.py $D --n $EPOCHS --full > ${T}_L${LAYER}_a$A.txt \
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
# macOS has no bare `python`, and huggingface_hub lives in the project venv, so
# the pasted upload command must name an interpreter that exists. Resolve it
# here rather than printing `python` and letting it fail after the sweep.
PY="python3"
for _c in "$_repo/.venv/bin/python" "$_repo/.venv-cpu/bin/python"; do
    [ -x "$_c" ] && PY="$_c" && break
done
cat <<HINT
  set -a; source .env; set +a
  "$PY" - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
c = api.upload_folder(
    repo_id="llm-psych/llm-psych-activations", repo_type="dataset",
    folder_path=".", path_in_repo="eval_outputs/blackmail_arms/L${LAYER}",
    allow_patterns=["*_L${LAYER}_a*.txt", "outputs/*.log"],
    commit_message="dose sweep outputs + run log")
print(c.oid)
PY
HINT
echo "Then on the Mac: pull, parse, export, run the judge, ingest, recut/combine."

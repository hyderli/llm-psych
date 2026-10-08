#!/usr/bin/env bash
# Materialise /workspace/model-fixed: a local copy of a model whose chat
# template folds a system role into the first user turn.
#
# WHY THIS EXISTS AS A SCRIPT. Gemma 2's shipped chat template raises on a
# system message, and `agentic_misalignment` puts the whole scenario in the
# system prompt. Every Gemma steering run from 2026-09 onward therefore pointed
# at a hand-patched copy at /workspace/model-fixed, created ad hoc in a pod
# shell and never committed. The pod was torn down on 2026-09-27 and the
# template went with it.
#
# COMPARABILITY WARNING. The template below is a reconstruction. If it differs
# in any whitespace from the lost original, prompts rendered now are not
# byte-identical to the ones behind results/scoring_out, and new runs are not
# strictly comparable to them. Do not assume it matches. Re-run at least one
# condition that already has a result (resid or full at alpha 0.15, n=20) in the
# same session and check the rate against the recorded one before treating any
# new number as comparable. Print one rendered prompt and eyeball it too.
#
# Llama 3.1, Qwen 2.5 and Mistral all accept a system role natively and need
# none of this -- point MODEL_PATH straight at the snapshot for those.
#
# Usage:
#     export HF_TOKEN=...            # gated repo access
#     bash scripts/make_model_fixed.sh                      # gemma2_9b -> /workspace/model-fixed
#     bash scripts/make_model_fixed.sh --model gemma2_2b --out /workspace/mf-2b
set -euo pipefail

MODEL="${MODEL:-gemma2_9b}"
OUT="${OUT:-/workspace/model-fixed}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --out)   OUT="$2";   shift 2 ;;
        *) printf 'Unknown arg: %s\n' "$1" >&2; exit 1 ;;
    esac
done

CFG="configs/model/${MODEL}.yaml"
[[ -f "$CFG" ]] || { printf 'ERROR: no such model config: %s\n' "$CFG" >&2; exit 2; }

PY=".venv/bin/python"
[[ -x "$PY" ]] || PY="uv run --locked python"

$PY - "$CFG" "$OUT" <<'PYEOF'
import sys, shutil
from pathlib import Path
import yaml
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

cfg_path, out = sys.argv[1:]
cfg = yaml.safe_load(Path(cfg_path).read_text())
hf_id, rev = cfg["hf_model_id"], cfg.get("hf_revision")

# Fold a leading system message into the first user turn. Everything else is
# Gemma 2's own format: <start_of_turn>{user|model}\n ... <end_of_turn>\n
TEMPLATE = (
    "{{ bos_token }}"
    "{% if messages[0]['role'] == 'system' %}"
    "{% set loop_messages = messages[1:] %}"
    "{% set system_message = messages[0]['content'] %}"
    "{% else %}{% set loop_messages = messages %}{% endif %}"
    "{% for message in loop_messages %}"
    "{% if loop.index0 == 0 and system_message is defined %}"
    "{% set content = system_message + '\n\n' + message['content'] %}"
    "{% else %}{% set content = message['content'] %}{% endif %}"
    "<start_of_turn>{{ 'model' if message['role'] == 'assistant' else 'user' }}\n"
    "{{ content | trim }}<end_of_turn>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)

src = snapshot_download(hf_id, revision=rev)
dst = Path(out)
if dst.exists():
    print(f"{dst} exists; leaving it alone. Remove it first to rebuild.")
    raise SystemExit(0)
print(f"copying {src} -> {dst}")
shutil.copytree(src, dst, symlinks=False, ignore=shutil.ignore_patterns(".*"))

tok = AutoTokenizer.from_pretrained(str(dst))
tok.chat_template = TEMPLATE
tok.save_pretrained(str(dst))

rendered = tok.apply_chat_template(
    [{"role": "system", "content": "SYS-MARKER"},
     {"role": "user", "content": "USER-MARKER"}],
    tokenize=False, add_generation_prompt=True)
print("\n--- rendered probe prompt (check the fold by eye) ---")
print(rendered)
print("--- end probe ---")
assert "SYS-MARKER" in rendered and "USER-MARKER" in rendered, "fold lost content"
assert rendered.count("<start_of_turn>user") == 1, "system did not fold into user"
print(f"\nOK: {dst} ready. Pass it as MODEL_PATH to the sweep scripts.")
PYEOF

"""Prepare a local model copy — optionally with a custom chat template.

Why this exists: some models (e.g. Gemma-2/3) reject a `system` role, which breaks
evals that use system prompts. The fix is to save a local copy whose tokenizer has a
template that folds the system message into the first user turn, then point the eval
at that local copy. Models that already support system role need NO preparation — just
use their HF id directly.

Usage:
    # model that needs a template fix (e.g. Gemma-2):
    python prepare_model.py --model google/gemma-2-9b-it \\
        --chat-template templates/gemma_sys.jinja --out /workspace/gemma-2-9b-it-fixed

    # model that needs no fix — you usually don't need this script at all, but you can
    # still make a local copy if you want one:
    python prepare_model.py --model meta-llama/Llama-3.1-8B-Instruct --out /workspace/llama-local

After running, use it as:  --model steered/local -M model_path=<out> -M tokenizer_path=<out>

Verify the template took (for a system-fold template):
    grep -c "System role not supported" <out>/tokenizer_config.json   # want 0
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local path to load")
    ap.add_argument("--out", required=True, help="directory to save the prepared model")
    ap.add_argument("--chat-template", default=None,
                    help="path to a .jinja chat template to bake into the tokenizer")
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    if a.chat_template:
        tok.chat_template = open(a.chat_template).read()
        print(f"applied chat template from {a.chat_template}")

    dtype = getattr(torch, a.dtype)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype)
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"saved prepared model to {a.out}")

    # quick self-check: does a system role render without error?
    try:
        msgs = [{"role": "system", "content": "X"}, {"role": "user", "content": "hi"}]
        rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        print(">>> system role renders OK; first 120 chars:\n" + rendered[:120])
    except Exception as e:
        print(f">>> NOTE: system role still rejected by this tokenizer: {e}\n"
              f"    supply --chat-template with a system-folding template.")


if __name__ == "__main__":
    main()

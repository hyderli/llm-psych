"""Shared helpers for the J27 scripts (mu, manipulation check, runner).

The canonical scenario prompt is the same (system, user) the J24/J25 sweeps
feed the model. Preferred source: inspect_evals' generate_prompts via
scripts/build_j26a.py's ``canonical_messages``. Fallback: the SYSTEM/USER
sections recorded in ``eval_outputs/j24/L16/j24_full_L16_a0.txt`` (the .txt
writer strips a trailing newline, so the fallback is exact up to rstrip).
"""
from __future__ import annotations

import hashlib
import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_TXT = ROOT / "eval_outputs" / "j24" / "L16" / "j24_full_L16_a0.txt"
RESULTS = ROOT / "results" / "j27"
VEC_DIR = ROOT / "steering_vectors" / "Qwen2.5-7B-Instruct-story-wheel32"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
LAYER = 16
ALPHA_FULL = 0.25
ALPHA_POOL1 = 0.15
ABLATION_LAYERS = list(range(16, 27))          # L16..L26, the J27 ablation band
RANKS = (32, 64, 121)
N_TOKENS = 121
SEED = 20261110
CHUNK = 8192


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def messages_from_txt(ref: Path = REFERENCE_TXT) -> tuple[str, str]:
    """(system, user) extracted from the recorded J24 transcript."""
    seg = ref.read_text().split("########## SAMPLE 0 ##########")[1]
    m = re.search(
        r"----- SYSTEM -----\n(.*?)\n----- USER -----\n(.*?)(?=\n----- |\Z)",
        seg, re.S)
    return m.group(1).rstrip(), m.group(2).rstrip()


def canonical_messages() -> tuple[str, str, str]:
    """(system, user, source). Refuses if the two sources disagree."""
    sys_txt, user_txt = messages_from_txt()
    try:
        b = _load("build_j26a", "scripts/build_j26a.py")
        sys_ie, user_ie = b.canonical_messages()
    except SystemExit:
        # inspect_evals unavailable in this env: use the recorded transcript.
        return sys_txt, user_txt, "eval_outputs_txt"
    if sys_ie.rstrip() != sys_txt or user_ie.rstrip() != user_txt:
        raise SystemExit(
            "REFUSING: inspect_evals canonical messages disagree with "
            f"{REFERENCE_TXT} (the fallback source).")
    return sys_ie, user_ie, "inspect_evals+txt"


def rendered_prompt(tokenizer) -> tuple[str, str]:
    """Chat-templated scenario prompt and its sha256."""
    sys_msg, user_msg, source = canonical_messages()
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": sys_msg},
         {"role": "user", "content": user_msg}],
        add_generation_prompt=True, tokenize=False)
    return text, hashlib.sha256(text.encode()).hexdigest() + f" ({source})"

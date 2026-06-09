#!/usr/bin/env bash
# One-shot bootstrap for a fresh RunPod (or any Linux + CUDA) pod.
#
# Brings a clean Ubuntu image to a state where ``scripts/cloud_run.sh``
# can be executed. Idempotent — safe to re-run on a pod that has
# already been bootstrapped (it just no-ops at each step).
#
# Usage (paste into a fresh pod's terminal)::
#
#     export HF_TOKEN=hf_xxx            # required: write access to the
#                                       #           private HF dataset
#     curl -fsSL https://raw.githubusercontent.com/hyderli/llm-psych/main/scripts/cloud_bootstrap.sh | bash
#
# Or, equivalently, after cloning::
#
#     bash scripts/cloud_bootstrap.sh
#
# What this does
# --------------
# 1. Verifies ``nvidia-smi`` is present and reports a GPU.
# 2. Installs ``uv`` (Astral) if missing.
# 3. Clones the repo to ``$REPO_DIR`` (default ``/workspace/llm-psych``)
#    if it is not already present; otherwise ``git pull``s.
# 4. Runs ``uv sync`` to install CUDA torch + bitsandbytes + all deps.
# 5. Writes ``.env`` from the ambient ``HF_TOKEN`` (and other tokens
#    if present), so ``scripts/sync_hf.py`` can authenticate.
# 6. Runs a torch + CUDA preflight that imports torch, prints the
#    GPU name + VRAM, and imports bitsandbytes.
# 7. Verifies the private HF dataset is reachable with the token.
#
# Exit codes
# ----------
# 0   bootstrap complete; pod is ready
# 1   missing required env var (e.g. HF_TOKEN)
# 2   no CUDA GPU detected
# 3   uv sync failed
# 4   preflight failed (torch / bitsandbytes / HF auth)

set -euo pipefail

# --------------------------------------------------------------------------
# Config (overridable via env)
# --------------------------------------------------------------------------

REPO_URL="${REPO_URL:-https://github.com/hyderli/llm-psych.git}"
REPO_DIR="${REPO_DIR:-/workspace/llm-psych}"
GIT_REF="${GIT_REF:-main}"

# RunPod containers ship with a small root disk (~20 GB) and a large persistent
# /workspace volume. Default HF cache lives under $HOME, which fills the root
# disk on the first 7-8B model download. Pin the cache to the volume so models
# survive across pod restarts and the root disk does not run out.
HF_HOME_DEFAULT="/workspace/.cache/huggingface"
export HF_HOME="${HF_HOME:-$HF_HOME_DEFAULT}"

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

log()  { printf '\033[1;34m[bootstrap]\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33m[bootstrap WARN]\033[0m %s\n' "$*" >&2; }
fail() { printf '\033[1;31m[bootstrap FAIL]\033[0m %s\n' "$*" >&2; exit "${2:-1}"; }

# --------------------------------------------------------------------------
# 1. Required env vars
# --------------------------------------------------------------------------

if [[ -z "${HF_TOKEN:-}" ]]; then
    fail "HF_TOKEN is not set. Export it before bootstrap:
    export HF_TOKEN=hf_xxx
The token must have write access to the private dataset
llm-psych/llm-psych-activations (org: EmotionConceptsResearch)." 1
fi

# --------------------------------------------------------------------------
# 2. GPU check
# --------------------------------------------------------------------------

log "Checking for CUDA GPU…"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi not found. Is this actually a GPU pod?" 2
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || \
    fail "nvidia-smi failed; the GPU may be unavailable." 2

# --------------------------------------------------------------------------
# 3. Install uv
# --------------------------------------------------------------------------

if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # The installer puts uv in ~/.local/bin; surface it for this shell.
    export PATH="$HOME/.local/bin:$PATH"
fi
log "uv version: $(uv --version)"

# --------------------------------------------------------------------------
# 4. Clone or update repo
# --------------------------------------------------------------------------

if [[ ! -d "$REPO_DIR/.git" ]]; then
    log "Cloning $REPO_URL into $REPO_DIR…"
    mkdir -p "$(dirname "$REPO_DIR")"
    git clone "$REPO_URL" "$REPO_DIR"
else
    log "Repo exists; fetching latest $GIT_REF…"
    git -C "$REPO_DIR" fetch origin "$GIT_REF"
fi
git -C "$REPO_DIR" checkout "$GIT_REF"
git -C "$REPO_DIR" pull --ff-only origin "$GIT_REF" || \
    warn "git pull failed; continuing with current checkout."

cd "$REPO_DIR"
log "Repo at $(git rev-parse --short HEAD) on branch $(git rev-parse --abbrev-ref HEAD)"

# --------------------------------------------------------------------------
# 5. Write .env (gitignored)
# --------------------------------------------------------------------------

log "Writing .env from ambient secrets…"
{
    printf 'HF_TOKEN=%s\n' "$HF_TOKEN"
    [[ -n "${OPENAI_API_KEY:-}" ]]    && printf 'OPENAI_API_KEY=%s\n'    "$OPENAI_API_KEY"
    [[ -n "${ANTHROPIC_API_KEY:-}" ]] && printf 'ANTHROPIC_API_KEY=%s\n' "$ANTHROPIC_API_KEY"
} > .env
chmod 600 .env

# Persist HF_HOME to ~/.bashrc so future shells (e.g. a fresh tmux pane that
# runs cloud_run.sh) inherit the volume-backed cache without manual export.
mkdir -p "$HF_HOME"
if ! grep -q '^export HF_HOME=' "${HOME}/.bashrc" 2>/dev/null; then
    log "Persisting HF_HOME=$HF_HOME to ~/.bashrc"
    printf '\n# llm-psych: pin HF cache to /workspace volume\nexport HF_HOME=%s\n' \
        "$HF_HOME" >> "${HOME}/.bashrc"
fi
log "HF_HOME=$HF_HOME ($(df -h "$HF_HOME" | awk 'NR==2 {print $4 " free on " $6}'))"

# --------------------------------------------------------------------------
# 6. Dependencies
# --------------------------------------------------------------------------

log "Running uv sync (this may take a few minutes on first run)…"
if ! uv sync; then
    fail "uv sync failed. Check the output above." 3
fi

# --------------------------------------------------------------------------
# 7. Torch + CUDA + bitsandbytes preflight
# --------------------------------------------------------------------------

log "Running torch / CUDA / bitsandbytes preflight…"
uv run python - <<'PY' || fail "Python preflight failed." 4
import sys

import torch

print(f"torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("ERROR: torch reports no CUDA device.", file=sys.stderr)
    sys.exit(1)

dev = torch.cuda.current_device()
props = torch.cuda.get_device_properties(dev)
print(f"GPU: {props.name}")
print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
print(f"Compute capability: {props.major}.{props.minor}")

# bitsandbytes is required for 4-bit quantization paths.
try:
    import bitsandbytes  # noqa: F401
    print(f"bitsandbytes: {bitsandbytes.__version__}")
except Exception as exc:  # noqa: BLE001
    print(f"WARNING: bitsandbytes import failed: {exc}", file=sys.stderr)
    # Not fatal — bf16 paths still work without it.
PY

# --------------------------------------------------------------------------
# 8. HF auth + dataset accessibility
# --------------------------------------------------------------------------

log "Checking HF dataset accessibility…"
uv run python - <<'PY' || fail "HF dataset check failed." 4
import os
from pathlib import Path
from dotenv import load_dotenv

# Explicit path: find_dotenv() walks the call stack via frame inspection,
# which fails under `python - <<EOF` (no caller frame). Point at .env directly.
env_path = Path(".env")
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

from llm_psych.hf_sync import DEFAULT_DATASET_REPO_ID, list_remote

files = list_remote(repo_id=DEFAULT_DATASET_REPO_ID)
print(f"HF dataset {DEFAULT_DATASET_REPO_ID}: {len(files)} files visible")
PY

log "Bootstrap complete. Pod is ready."
log "Next: bash scripts/cloud_run.sh --model llama31_8b --emotions 'anger joy fear sadness'"

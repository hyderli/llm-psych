"""Pin HF revision SHAs in configs/model/*.yaml.

Requires HF_TOKEN in .env or environment for gated models (Llama, Gemma).
Usage:
    uv run python scripts/pin_hf_revisions.py
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from huggingface_hub import HfApi, list_repo_refs
from dotenv import load_dotenv


def get_main_sha(model_id: str) -> str | None:
    """Return the latest commit SHA on the main branch."""
    try:
        refs = list_repo_refs(model_id)
    except Exception as e:
        print(f"  ERROR querying {model_id}: {e}")
        return None

    for branch in refs.branches:
        if branch.name == "main":
            return branch.target_commit

    # Fallback: some repos use 'master' or the first branch
    if refs.branches:
        return refs.branches[0].target_commit

    return None


def main() -> None:
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if token:
        print("HF_TOKEN found in environment — using authenticated API.")
    else:
        print("WARNING: HF_TOKEN not found. Gated models (Llama, Gemma) will fail.")
        print("Set HF_TOKEN in .env or export it, then re-run.")

    config_dir = Path("configs/model")
    yaml_paths = sorted(config_dir.glob("*.yaml"))

    print(f"\nScanning {len(yaml_paths)} model configs...\n")

    updated_any = False
    for path in yaml_paths:
        with path.open() as f:
            cfg = yaml.safe_load(f)

        model_id = cfg.get("hf_model_id")
        current_rev = cfg.get("hf_revision")

        if not model_id:
            print(f"{path.name}: no hf_model_id — skipping")
            continue

        if current_rev not in (None, "", "null"):
            print(f"{path.name}: already pinned to {current_rev} — skipping")
            continue

        sha = get_main_sha(model_id)
        if sha is None:
            print(f"{path.name}: could not resolve SHA — skipping")
            continue

        # Update the file in-place
        raw = path.read_text()
        # Replace the hf_revision line while preserving comments
        new_raw = raw.replace("hf_revision: null", f"hf_revision: {sha}")
        if new_raw == raw:
            # Try other common null patterns
            new_raw = raw.replace("hf_revision: \"\"", f"hf_revision: {sha}")

        if new_raw != raw:
            path.write_text(new_raw)
            print(f"{path.name}: pinned to {sha}")
            updated_any = True
        else:
            print(f"{path.name}: could not find null hf_revision — skipping")

    if updated_any:
        print("\nDone. Review changes with `git diff` before committing.")
    else:
        print("\nNo configs updated.")


if __name__ == "__main__":
    main()

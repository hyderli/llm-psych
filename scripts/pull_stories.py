"""Pull generated story corpora from the HF dataset.

The inverse of ``scripts/push_stories.py``. ``sync_hf.py`` handles
activations / probes / steering_vectors, but story *texts* live at
``stories/<model_key>/`` on the dataset and have no pull path — so a
corpus pushed off a pod could not be fetched back without hand-rolling a
``snapshot_download``. This closes that.

Files land in the canonical local layout,
``data/derived/stories/<model_key>/``, which is where
``scripts/extract_story_activations.py`` and
``scripts/screen_story_corpora.py`` already look. No ``--stories-dir``
override needed downstream.

Usage
-----

.. code-block:: bash

    # list what corpora exist on the dataset
    uv run python scripts/pull_stories.py --list

    # fetch the three primaries
    uv run python scripts/pull_stories.py \\
        --model Llama-3.1-8B-Instruct Qwen2.5-7B-Instruct gemma-2-9b-it

    # fetch everything
    uv run python scripts/pull_stories.py

The dataset is private; the token is read from ``.env`` (``HF_TOKEN``)
exactly as the other sync scripts do. Never pass a token on the command
line — it lands in your shell history.

``snapshot_download`` overwrites a local file only when its content hash
differs. Pulling on top of a locally generated corpus of the same
``model_key`` will therefore replace it if the remote copy differs; the
script warns before doing so.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root / "src"))

from dotenv import load_dotenv  # noqa: E402
from huggingface_hub import HfApi, snapshot_download  # noqa: E402

from llm_psych.hf_sync import DEFAULT_DATASET_REPO_ID  # noqa: E402

_STORIES_PREFIX = "stories/"


def list_remote_models(repo_id: str) -> list[str]:
    """Return the ``model_key`` subfolders present under ``stories/``."""
    files = HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset")
    keys = {
        f[len(_STORIES_PREFIX):].split("/", 1)[0]
        for f in files
        if f.startswith(_STORIES_PREFIX) and "/" in f[len(_STORIES_PREFIX):]
    }
    return sorted(keys)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--model", dest="model_keys", nargs="+", default=None,
        help="base model keys (no -story suffix); default: all on the dataset",
    )
    ap.add_argument("--repo-id", default=DEFAULT_DATASET_REPO_ID)
    ap.add_argument("--revision", default=None, help="branch, tag or commit SHA")
    ap.add_argument(
        "--list", action="store_true",
        help="list remote corpora and exit without downloading",
    )
    args = ap.parse_args()

    load_dotenv(_repo_root / ".env")

    available = list_remote_models(args.repo_id)
    if args.list:
        if not available:
            print(f"No story corpora found on {args.repo_id} under stories/.")
        else:
            print(f"Story corpora on {args.repo_id}:")
            for k in available:
                print(f"  {k}")
        return 0

    if not available:
        print(
            f"ERROR: no story corpora under stories/ on {args.repo_id}. "
            "Nothing has been pushed with scripts/push_stories.py.",
            file=sys.stderr,
        )
        return 1

    wanted = args.model_keys or available
    missing = [k for k in wanted if k not in available]
    if missing:
        print(
            f"ERROR: not on the dataset: {', '.join(missing)}\n"
            f"Available: {', '.join(available)}",
            file=sys.stderr,
        )
        return 1

    dest_root = _repo_root / "data" / "derived"
    for key in wanted:
        local = dest_root / "stories" / key
        if local.exists():
            print(f"NOTE: {local} exists; differing files will be overwritten.")

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        allow_patterns=[f"{_STORIES_PREFIX}{k}/**" for k in wanted],
        local_dir=str(dest_root),
    )

    for key in wanted:
        local = dest_root / "stories" / key
        n = len(list(local.glob("*.parquet"))) if local.exists() else 0
        print(f"pulled {args.repo_id}:stories/{key} -> {local} ({n} parquets)")

    print(
        "\nNext: uv run python scripts/screen_story_corpora.py "
        f"--models {' '.join(wanted)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Push generated story corpora to the HF dataset.

``sync_hf.py`` syncs activations / probes / steering_vectors. The story
*texts* in ``data/derived/stories/<model_key>/`` are the **provenance** of the
``<model_key>-story`` activations (the activation meta keeps only a
``story_id``), and they are not bit-reproducible across hardware/torch because
generation uses seeded *sampling*. So they must be saved, not regenerated.

This uploads ``data/derived/stories/<model_key>/`` to ``stories/<model_key>/``
on the same private dataset. Dual purpose:

* called by ``scripts/run_story_pipeline.sh --push`` after each run, so future
  runs never strand their corpora;
* run by hand to rescue corpora off a pod before terminating it::

      uv run python scripts/push_stories.py --model Qwen2.5-0.5B-Instruct

Note ``--model`` is the **base** key (no ``-story`` suffix), matching the
``data/derived/stories/<base>/`` layout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root / "src"))

from dotenv import load_dotenv  # noqa: E402
from huggingface_hub import upload_folder  # noqa: E402

from llm_psych.hf_sync import DEFAULT_DATASET_REPO_ID  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        dest="model_key",
        required=True,
        help="Base model key, e.g. Qwen2.5-0.5B-Instruct (NOT the -story suffix).",
    )
    ap.add_argument("--repo-id", default=DEFAULT_DATASET_REPO_ID)
    args = ap.parse_args()

    load_dotenv(_repo_root / ".env")
    local = _repo_root / "data" / "derived" / "stories" / args.model_key
    if not local.exists():
        print(f"ERROR: no story corpus at {local}", file=sys.stderr)
        return 1

    upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(local),
        path_in_repo=f"stories/{args.model_key}",
        commit_message=f"stories: {args.model_key}",
    )
    print(f"pushed {local} -> {args.repo_id}:stories/{args.model_key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

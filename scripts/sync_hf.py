"""Push/pull activations, probes, and steering vectors to/from HF.

This script is the team's single entry point for syncing large
artefacts between machines via the private ``llm-psych/llm-psych-activations``
dataset (under the ``EmotionConceptsResearch`` HF org). See
``docs/methods.md`` §Activation storage.

Typical use
-----------

Cloud pod, after extracting activations for one model:

    uv run python scripts/sync_hf.py push activations --model Llama-3.1-8B-Instruct

Teammate's laptop, before training probes:

    uv run python scripts/sync_hf.py pull activations --model Llama-3.1-8B-Instruct

Tagging a milestone snapshot (after a complete H1 pilot for example):

    uv run python scripts/sync_hf.py push activations --tag h1-pilot-2026-05

List what's on the remote without downloading:

    uv run python scripts/sync_hf.py ls activations

Auth
----
Set ``HF_TOKEN`` in ``.env`` (gitignored) with read+write access to the
private dataset. The script loads ``.env`` automatically.

Exit codes
----------
0   success
1   user error (bad args, missing files, no token)
2   HF API error (network, auth, repo not found)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# --- make src/ importable when running without `pip install -e .`
_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root / "src"))

from dotenv import load_dotenv  # noqa: E402

from llm_psych.hf_sync import (  # noqa: E402
    ARTIFACT_KINDS,
    DEFAULT_DATASET_REPO_ID,
    download,
    list_remote,
    upload,
)

log = logging.getLogger("sync_hf")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sync_hf",
        description="Push/pull project artefacts to the private HF dataset.",
    )
    p.add_argument(
        "--repo-id",
        default=DEFAULT_DATASET_REPO_ID,
        help=f"HF dataset id (default: {DEFAULT_DATASET_REPO_ID}).",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging."
    )

    sub = p.add_subparsers(dest="command", required=True)

    # ---- push ----
    push = sub.add_parser("push", help="Upload local artefacts to HF.")
    push.add_argument("kind", choices=ARTIFACT_KINDS)
    push.add_argument(
        "--model",
        dest="model_key",
        default=None,
        help="Only push the <kind>/<model_key>/ subtree.",
    )
    push.add_argument(
        "--tag",
        default=None,
        help="Optional snapshot tag (e.g. h1-pilot-2026-05).",
    )
    push.add_argument(
        "--message",
        "-m",
        dest="commit_message",
        default=None,
        help="Commit message on the dataset repo.",
    )

    # ---- pull ----
    pull = sub.add_parser("pull", help="Download artefacts from HF.")
    pull.add_argument("kind", choices=ARTIFACT_KINDS)
    pull.add_argument("--model", dest="model_key", default=None)
    pull.add_argument(
        "--revision",
        default=None,
        help="Branch, tag, or commit SHA on the dataset (default: main).",
    )

    # ---- ls ----
    ls = sub.add_parser("ls", help="List remote files.")
    ls.add_argument(
        "kind",
        nargs="?",
        choices=ARTIFACT_KINDS,
        default=None,
        help="Filter by kind. Omit to list everything.",
    )
    ls.add_argument("--revision", default=None)

    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv(_repo_root / ".env")
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        if args.command == "push":
            result = upload(
                args.kind,
                repo_root=_repo_root,
                repo_id=args.repo_id,
                model_key=args.model_key,
                tag=args.tag,
                commit_message=args.commit_message,
            )
            log.info("OK %s", result.describe())
        elif args.command == "pull":
            result = download(
                args.kind,
                repo_root=_repo_root,
                repo_id=args.repo_id,
                model_key=args.model_key,
                revision=args.revision,
            )
            log.info("OK %s", result.describe())
        elif args.command == "ls":
            files = list_remote(
                args.kind, repo_id=args.repo_id, revision=args.revision
            )
            if not files:
                log.info("(empty)")
            else:
                for f in files:
                    print(f)
        else:  # pragma: no cover — argparse already validates
            raise AssertionError(f"unknown command: {args.command}")
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1
    except RuntimeError as exc:
        # No token / config error
        log.error("%s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        log.error("HF API failure: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

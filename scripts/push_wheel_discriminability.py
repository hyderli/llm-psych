"""Push wheel-discriminability results to the HF dataset.

``results/`` is gitignored, so the per-layer pairwise accuracy matrices produced
by ``scripts/wheel_discriminability.py`` live nowhere but the machine that ran
them until they are pushed. They are small (about 25 KB per model) and they are
the evidence behind the layer, intensity and cell-distinctness findings, so they
belong on the dataset beside ``vector_validation/``.

Uploads ``results/wheel_discriminability/<slug>/`` to
``wheel_discriminability/<slug>/`` on the private dataset, mirroring the
``vector_validation/<slug>/`` convention that ``scripts/h8_prep.sh`` pulls from.

Smoke tracks are skipped unless ``--include-smoke`` is passed: a 3-cell smoke
run produces a report that looks superficially like the real thing, and it
should not sit next to the real ones on the dataset.

Usage::

    uv run python scripts/push_wheel_discriminability.py            # all real tracks
    uv run python scripts/push_wheel_discriminability.py --list     # show, do not upload
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

_LOCAL_ROOT = _repo_root / "results" / "wheel_discriminability"
_PREFIX = "wheel_discriminability"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--slug", dest="slugs", nargs="+", default=None,
                    help="specific <model>-<track> slugs; default: every real track found")
    ap.add_argument("--include-smoke", action="store_true",
                    help="also push slugs containing 'smoke' (off by default)")
    ap.add_argument("--repo-id", default=DEFAULT_DATASET_REPO_ID)
    ap.add_argument("--list", action="store_true", help="list what would be pushed and exit")
    args = ap.parse_args()

    if not _LOCAL_ROOT.is_dir():
        print(f"ERROR: nothing at {_LOCAL_ROOT}", file=sys.stderr)
        return 1

    found = sorted(d.name for d in _LOCAL_ROOT.iterdir() if d.is_dir())
    if args.slugs:
        missing = [s for s in args.slugs if s not in found]
        if missing:
            print(f"ERROR: not found locally: {', '.join(missing)}\n"
                  f"Available: {', '.join(found)}", file=sys.stderr)
            return 1
        slugs = args.slugs
    else:
        slugs = [s for s in found if args.include_smoke or "smoke" not in s]

    if not slugs:
        print("Nothing to push (only smoke tracks found; use --include-smoke).")
        return 0

    for slug in slugs:
        local = _LOCAL_ROOT / slug
        files = sorted(p.name for p in local.iterdir() if p.is_file())
        print(f"{slug}: {', '.join(files)}")
    if args.list:
        return 0

    load_dotenv(_repo_root / ".env")
    failed = []
    for slug in slugs:
        local = _LOCAL_ROOT / slug
        try:
            upload_folder(
                repo_id=args.repo_id, repo_type="dataset",
                folder_path=str(local), path_in_repo=f"{_PREFIX}/{slug}",
                commit_message=f"wheel_discriminability: {slug}",
            )
        except Exception as exc:
            print(f"FAILED {slug}: {exc}", file=sys.stderr)
            failed.append(slug)
            continue
        print(f"pushed {local} -> {args.repo_id}:{_PREFIX}/{slug}")

    if failed:
        print(f"PUSH INCOMPLETE: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

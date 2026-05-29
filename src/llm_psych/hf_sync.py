"""HuggingFace Dataset sync for activations, probes, and steering vectors.

The project stores all large binary artefacts (residual-stream
activations, fitted probes, steering vectors) in a **private** HF
dataset that mirrors the on-disk layout documented in
``docs/methods.md``::

    <repo>/activations/<model_key>/<source>_<split>.npz
    <repo>/activations/<model_key>/<source>_<split>.meta.parquet
    <repo>/probes/<model_key>/<emotion>_layer<L>.joblib
    <repo>/probes/<model_key>/<emotion>_layer<L>.yaml
    <repo>/steering_vectors/<model_key>/<emotion>_layer<L>.pt

The HF dataset uses the same paths under repo root. This module wraps
``huggingface_hub`` so the team can run a single command from any
machine to push from a cloud pod or pull onto a local laptop.

Design constraints
------------------
* Dataset is **private**. Auth via ``HF_TOKEN`` (loaded from ``.env``
  if present, else from the environment).
* Uploads/downloads are **idempotent**: ``upload_folder`` skips
  unchanged files, ``snapshot_download`` resumes interrupted transfers.
* Operations are **scoped to a kind** (activations / probes /
  steering_vectors) and optionally **filtered by model_key** to keep
  bandwidth proportional to what the caller actually needs.
* The dataset repo is **created on demand** the first time anyone
  pushes to it. After that, ``ensure_dataset()`` is a no-op.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.utils import build_hf_headers

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

DEFAULT_DATASET_REPO_ID = "llm-psych/llm-psych-activations"
"""Private HF dataset repo (see docs/methods.md §Activation storage).

Namespace ``llm-psych`` is the URL slug of the HF organization
``EmotionConceptsResearch``; team members are added as org members
and use their own write tokens.
"""

ArtifactKind = Literal["activations", "probes", "steering_vectors"]
ARTIFACT_KINDS: tuple[ArtifactKind, ...] = (
    "activations",
    "probes",
    "steering_vectors",
)


@dataclass(frozen=True)
class SyncResult:
    """Outcome of an upload or download operation."""

    kind: ArtifactKind
    model_key: str | None
    repo_id: str
    revision: str | None
    n_files: int
    local_dir: Path

    def describe(self) -> str:
        scope = self.model_key or "all models"
        rev = f" @ {self.revision}" if self.revision else ""
        return (
            f"{self.kind}/{scope}: {self.n_files} files "
            f"<-> {self.repo_id}{rev}"
        )


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------

def _resolve_token(token: str | None = None) -> str:
    """Resolve an HF token from arg, environment, or ``.env``.

    Order: explicit arg > ``HF_TOKEN`` env var > ``HUGGING_FACE_HUB_TOKEN``
    env var. Loading from ``.env`` is handled at process start by the
    calling script (we deliberately do not import dotenv here to keep
    this module testable without filesystem side effects).
    """
    if token:
        return token
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(var)
        if val:
            return val
    raise RuntimeError(
        "No HuggingFace token found. Set HF_TOKEN in .env or the "
        "environment, or pass token=... explicitly. The project's HF "
        "dataset is private and requires authentication."
    )


# --------------------------------------------------------------------------
# Repo lifecycle
# --------------------------------------------------------------------------

def ensure_dataset(
    repo_id: str = DEFAULT_DATASET_REPO_ID,
    *,
    private: bool = True,
    token: str | None = None,
) -> str:
    """Create the dataset repo if it does not exist; return its id.

    Safe to call repeatedly. If the repo exists, this is a no-op apart
    from one API round-trip.
    """
    tok = _resolve_token(token)
    api = HfApi(token=tok)
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        return repo_id
    except RepositoryNotFoundError:
        pass
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    return repo_id


# --------------------------------------------------------------------------
# Path helpers
# --------------------------------------------------------------------------

def _local_dir_for(kind: ArtifactKind, repo_root: Path) -> Path:
    return repo_root / kind


def _hf_subpath(kind: ArtifactKind, model_key: str | None) -> str:
    """Path *inside* the HF dataset for a (kind, model) scope."""
    if model_key:
        return f"{kind}/{model_key}"
    return kind


def _validate_kind(kind: str) -> ArtifactKind:
    if kind not in ARTIFACT_KINDS:
        raise ValueError(
            f"Unknown artifact kind: {kind!r}. Expected one of "
            f"{ARTIFACT_KINDS}."
        )
    return kind  # type: ignore[return-value]


# --------------------------------------------------------------------------
# Upload
# --------------------------------------------------------------------------

def upload(
    kind: ArtifactKind,
    *,
    repo_root: Path,
    repo_id: str = DEFAULT_DATASET_REPO_ID,
    model_key: str | None = None,
    tag: str | None = None,
    commit_message: str | None = None,
    token: str | None = None,
    create_if_missing: bool = True,
) -> SyncResult:
    """Upload local artefacts of one ``kind`` to the HF dataset.

    Parameters
    ----------
    kind
        Which artefact tree to upload (``"activations"`` / ``"probes"``
        / ``"steering_vectors"``).
    repo_root
        Absolute path to the project root (the directory containing
        ``activations/``, ``probes/``, etc.).
    repo_id
        HF dataset id, default ``DEFAULT_DATASET_REPO_ID``.
    model_key
        If given, only the subtree ``<kind>/<model_key>/`` is uploaded.
        Otherwise the entire ``<kind>/`` tree is uploaded.
    tag
        Optional snapshot tag to create after upload (e.g.
        ``"h1-pilot-2026-05"``). The tag points to the resulting commit.
    commit_message
        Commit message on the dataset repo. Auto-generated if omitted.
    token
        HF token. Falls back to environment.
    create_if_missing
        If True (default), create the dataset repo before uploading.

    Returns
    -------
    SyncResult
    """
    kind = _validate_kind(kind)
    tok = _resolve_token(token)
    if create_if_missing:
        ensure_dataset(repo_id=repo_id, token=tok)

    local_dir = _local_dir_for(kind, repo_root)
    if model_key:
        local_dir = local_dir / model_key
    if not local_dir.exists():
        raise FileNotFoundError(
            f"Nothing to upload — local path does not exist: {local_dir}"
        )

    path_in_repo = _hf_subpath(kind, model_key)
    msg = commit_message or f"sync: upload {path_in_repo}"

    api = HfApi(token=tok)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(local_dir),
        path_in_repo=path_in_repo,
        commit_message=msg,
        ignore_patterns=[".DS_Store", "*.tmp", "__pycache__/*"],
    )

    if tag:
        api.create_tag(
            repo_id=repo_id,
            repo_type="dataset",
            tag=tag,
            tag_message=msg,
            exist_ok=True,
        )

    n_files = sum(1 for p in local_dir.rglob("*") if p.is_file())
    return SyncResult(
        kind=kind,
        model_key=model_key,
        repo_id=repo_id,
        revision=tag,
        n_files=n_files,
        local_dir=local_dir,
    )


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------

def download(
    kind: ArtifactKind,
    *,
    repo_root: Path,
    repo_id: str = DEFAULT_DATASET_REPO_ID,
    model_key: str | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> SyncResult:
    """Download artefacts of one ``kind`` from the HF dataset.

    Files land in ``<repo_root>/<kind>/...`` mirroring the dataset
    layout. Existing files are overwritten only if their content hash
    differs (``snapshot_download`` semantics).

    Parameters
    ----------
    kind
        Which artefact tree to fetch.
    repo_root
        Project root; downloads write to ``repo_root/<kind>/...``.
    repo_id
        HF dataset id.
    model_key
        If given, only the subtree ``<kind>/<model_key>/`` is fetched.
    revision
        Branch, tag, or commit SHA on the dataset. Defaults to ``main``.
    token
        HF token; required because the dataset is private.

    Returns
    -------
    SyncResult
    """
    kind = _validate_kind(kind)
    tok = _resolve_token(token)
    api = HfApi(token=tok)

    path_in_repo = _hf_subpath(kind, model_key)
    allow_patterns = [f"{path_in_repo}/**"]
    local_root = repo_root  # we want files under <repo_root>/<kind>/...

    api.snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(local_root),
        allow_patterns=allow_patterns,
    )

    local_dir = _local_dir_for(kind, repo_root)
    if model_key:
        local_dir = local_dir / model_key
    n_files = (
        sum(1 for p in local_dir.rglob("*") if p.is_file())
        if local_dir.exists()
        else 0
    )
    return SyncResult(
        kind=kind,
        model_key=model_key,
        repo_id=repo_id,
        revision=revision,
        n_files=n_files,
        local_dir=local_dir,
    )


# --------------------------------------------------------------------------
# Introspection
# --------------------------------------------------------------------------

def list_remote(
    kind: ArtifactKind | None = None,
    *,
    repo_id: str = DEFAULT_DATASET_REPO_ID,
    revision: str | None = None,
    token: str | None = None,
) -> list[str]:
    """List file paths in the HF dataset, optionally filtered by kind."""
    tok = _resolve_token(token)
    api = HfApi(token=tok)
    files = api.list_repo_files(
        repo_id=repo_id, repo_type="dataset", revision=revision
    )
    if kind is None:
        return list(files)
    kind = _validate_kind(kind)
    prefix = f"{kind}/"
    return [f for f in files if f.startswith(prefix)]


# Silence "unused import" linter warning while keeping the symbol
# available for downstream debugging utilities.
_ = build_hf_headers

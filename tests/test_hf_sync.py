"""Unit tests for ``llm_psych.hf_sync``.

These tests cover the pure-Python logic — token resolution, path
construction, argument validation — without making any network calls.
The actual upload/download paths are exercised by monkeypatching
``huggingface_hub.HfApi``; an integration smoke-test against the real
private dataset belongs in a separate ``@pytest.mark.slow`` suite.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_psych import hf_sync
from llm_psych.hf_sync import (
    ARTIFACT_KINDS,
    DEFAULT_DATASET_REPO_ID,
    SyncResult,
    _hf_subpath,
    _local_dir_for,
    _resolve_token,
    _validate_kind,
    download,
    upload,
)

# --------------------------------------------------------------------------
# Token resolution
# --------------------------------------------------------------------------

class TestResolveToken:
    def test_explicit_arg_wins(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "from_env")
        assert _resolve_token("explicit") == "explicit"

    def test_env_hf_token(self, monkeypatch):
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        monkeypatch.setenv("HF_TOKEN", "tok_a")
        assert _resolve_token() == "tok_a"

    def test_env_legacy_var_fallback(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "tok_b")
        assert _resolve_token() == "tok_b"

    def test_missing_raises(self, monkeypatch):
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
        with pytest.raises(RuntimeError, match="No HuggingFace token"):
            _resolve_token()


# --------------------------------------------------------------------------
# Kind validation and path helpers
# --------------------------------------------------------------------------

class TestKindValidation:
    @pytest.mark.parametrize("kind", ARTIFACT_KINDS)
    def test_known_kinds_pass(self, kind):
        assert _validate_kind(kind) == kind

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown artifact kind"):
            _validate_kind("models")  # close but not allowed


class TestPathHelpers:
    @pytest.mark.parametrize(
        "kind,model_key,expected",
        [
            ("activations", None, "activations"),
            ("activations", "Llama-3.1-8B-Instruct", "activations/Llama-3.1-8B-Instruct"),
            ("probes", "Qwen2.5-7B-Instruct", "probes/Qwen2.5-7B-Instruct"),
            ("steering_vectors", None, "steering_vectors"),
        ],
    )
    def test_hf_subpath(self, kind, model_key, expected):
        assert _hf_subpath(kind, model_key) == expected

    def test_local_dir_for(self, tmp_path):
        assert _local_dir_for("activations", tmp_path) == tmp_path / "activations"
        assert _local_dir_for("probes", tmp_path) == tmp_path / "probes"


# --------------------------------------------------------------------------
# SyncResult
# --------------------------------------------------------------------------

class TestSyncResult:
    def test_describe_with_model_and_tag(self):
        r = SyncResult(
            kind="activations",
            model_key="Llama-3.1-8B-Instruct",
            repo_id="org/repo",
            revision="h1-pilot",
            n_files=42,
            local_dir=Path("/tmp/x"),
        )
        s = r.describe()
        assert "activations/Llama-3.1-8B-Instruct" in s
        assert "42 files" in s
        assert "org/repo" in s
        assert "h1-pilot" in s

    def test_describe_without_model_or_tag(self):
        r = SyncResult(
            kind="probes",
            model_key=None,
            repo_id="org/repo",
            revision=None,
            n_files=0,
            local_dir=Path("/tmp/x"),
        )
        s = r.describe()
        assert "all models" in s
        assert "@" not in s  # no revision suffix


# --------------------------------------------------------------------------
# Upload — mocked API
# --------------------------------------------------------------------------

class TestUpload:
    @pytest.fixture
    def fake_api(self, monkeypatch):
        instance = MagicMock()
        # Default: dataset already exists (repo_info returns successfully)
        # so ensure_dataset() short-circuits without touching create_repo.
        instance.repo_info.return_value = MagicMock()
        cls = MagicMock(return_value=instance)
        monkeypatch.setattr(hf_sync, "HfApi", cls)
        monkeypatch.setenv("HF_TOKEN", "test_token")
        return instance

    def test_upload_full_kind(self, fake_api, tmp_path):
        # Create a fake activations tree.
        d = tmp_path / "activations" / "Llama-3.1-8B-Instruct"
        d.mkdir(parents=True)
        (d / "anger_train.npz").write_bytes(b"\x00" * 16)
        (d / "anger_train.meta.parquet").write_bytes(b"\x00" * 16)

        result = upload(
            "activations",
            repo_root=tmp_path,
            model_key="Llama-3.1-8B-Instruct",
        )
        assert result.kind == "activations"
        assert result.model_key == "Llama-3.1-8B-Instruct"
        assert result.n_files == 2
        assert result.repo_id == DEFAULT_DATASET_REPO_ID
        # Existing dataset path: repo_info succeeds, create_repo not called.
        fake_api.repo_info.assert_called_once()
        fake_api.create_repo.assert_not_called()
        fake_api.upload_folder.assert_called_once()
        call_kwargs = fake_api.upload_folder.call_args.kwargs
        assert call_kwargs["path_in_repo"] == "activations/Llama-3.1-8B-Instruct"
        assert call_kwargs["repo_type"] == "dataset"

    def test_upload_creates_repo_when_missing(self, monkeypatch, tmp_path):
        """ensure_dataset must create the repo when repo_info raises."""
        instance = MagicMock()
        # Simulate repo_info raising any exception that's NOT
        # RepositoryNotFoundError; ensure_dataset only catches the
        # specific subclass, so this confirms the catch is real.
        from huggingface_hub.errors import RepositoryNotFoundError

        # Build a real RepositoryNotFoundError by going through HfApi's
        # actual error path via a MagicMock response object.
        fake_response = MagicMock()
        fake_response.status_code = 404
        try:
            err = RepositoryNotFoundError("not found", response=fake_response)
        except TypeError:
            # Older signature: just message.
            err = RepositoryNotFoundError("not found")
        instance.repo_info.side_effect = err
        cls = MagicMock(return_value=instance)
        monkeypatch.setattr(hf_sync, "HfApi", cls)
        monkeypatch.setenv("HF_TOKEN", "test_token")

        d = tmp_path / "activations" / "Foo"
        d.mkdir(parents=True)
        (d / "x.npz").write_bytes(b"\x00")

        upload("activations", repo_root=tmp_path, model_key="Foo")
        instance.create_repo.assert_called_once()
        instance.upload_folder.assert_called_once()

    def test_upload_with_tag(self, fake_api, tmp_path):
        d = tmp_path / "probes" / "Llama-3.1-8B-Instruct"
        d.mkdir(parents=True)
        (d / "anger_layer18.joblib").write_bytes(b"\x00")

        result = upload(
            "probes",
            repo_root=tmp_path,
            model_key="Llama-3.1-8B-Instruct",
            tag="h1-pilot-2026-05",
        )
        assert result.revision == "h1-pilot-2026-05"
        fake_api.create_tag.assert_called_once()
        tag_kwargs = fake_api.create_tag.call_args.kwargs
        assert tag_kwargs["tag"] == "h1-pilot-2026-05"
        assert tag_kwargs["exist_ok"] is True

    def test_upload_missing_dir_raises(self, fake_api, tmp_path):
        with pytest.raises(FileNotFoundError):
            upload("activations", repo_root=tmp_path, model_key="ghost")

    def test_upload_rejects_unknown_kind(self, fake_api, tmp_path):
        with pytest.raises(ValueError):
            upload("models", repo_root=tmp_path)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Download — mocked API
# --------------------------------------------------------------------------

class TestDownload:
    @pytest.fixture
    def fake_api(self, monkeypatch):
        instance = MagicMock()
        cls = MagicMock(return_value=instance)
        monkeypatch.setattr(hf_sync, "HfApi", cls)
        monkeypatch.setenv("HF_TOKEN", "test_token")
        return instance

    def test_download_scopes_to_model(self, fake_api, tmp_path):
        result = download(
            "activations",
            repo_root=tmp_path,
            model_key="Qwen2.5-7B-Instruct",
        )
        fake_api.snapshot_download.assert_called_once()
        kwargs = fake_api.snapshot_download.call_args.kwargs
        assert kwargs["allow_patterns"] == [
            "activations/Qwen2.5-7B-Instruct/**"
        ]
        assert kwargs["repo_type"] == "dataset"
        assert result.n_files == 0  # nothing actually downloaded in mock

    def test_download_full_kind_when_no_model_key(self, fake_api, tmp_path):
        download("steering_vectors", repo_root=tmp_path)
        kwargs = fake_api.snapshot_download.call_args.kwargs
        assert kwargs["allow_patterns"] == ["steering_vectors/**"]

    def test_download_passes_revision(self, fake_api, tmp_path):
        download(
            "activations",
            repo_root=tmp_path,
            revision="h1-pilot-2026-05",
        )
        kwargs = fake_api.snapshot_download.call_args.kwargs
        assert kwargs["revision"] == "h1-pilot-2026-05"

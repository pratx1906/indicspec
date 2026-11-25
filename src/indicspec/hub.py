"""Hugging Face hub helpers shared by the harness and analyses."""

from __future__ import annotations

from pathlib import Path

# Files fetched for models/drafters (weights, configs, custom code, tokenizer files).
# Must match the download step so local-only resolution sees a "complete" snapshot.
MODEL_PATTERNS = ["*.json", "*.safetensors", "*.py", "*.txt", "*.md", "*.model"]


def local_snapshot(repo: str, revision: str | None) -> Path:
    """Path of a cached snapshot (no network); raises if the pinned files are not cached."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo, revision=revision, allow_patterns=MODEL_PATTERNS, local_files_only=True
        )
    )


def download_snapshot(repo: str, revision: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, revision=revision, allow_patterns=MODEL_PATTERNS))

"""Hugging Face model-cache helpers."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path


_DEFAULT_HF_NAMESPACE = "rphammonds"
_MODEL_REPOS = {
    "volume-v5": ("scalesurfer-v5", "transunet3d.safetensors"),
    "volume-v6": ("scalesurfer-v6", "transunet3d.safetensors"),
    "volume-v7": ("scalesurfer-v7", "transunet3d.safetensors"),
    "volume-v8": ("scalesurfer-v8", "transunet3d.safetensors"),
    "stats-base": ("scalesurfer-stats-base", "stats_model.safetensors"),
    "stats-v6": ("scalesurfer-stats-v6", "stats_model.safetensors"),
    "stats-v7": ("scalesurfer-stats-v7", "stats_model.safetensors"),
    "stats-v8": ("scalesurfer-stats-v8", "stats_model.safetensors"),
}


def _selected_models(fs_versions: int | str | Iterable[int | str] | None) -> dict[str, tuple[str, str]]:
    if fs_versions is None:
        return _MODEL_REPOS
    versions = [fs_versions] if isinstance(fs_versions, (int, str)) else list(fs_versions)
    selected: dict[str, tuple[str, str]] = {}
    for value in versions:
        version = str(value).strip().lower()
        if version in {"base", "all", "base_all", "stats_base_all"}:
            selected["stats-base"] = _MODEL_REPOS["stats-base"]
            continue
        for prefix in ("fsv", "fs", "v"):
            if version.startswith(prefix):
                version = version[len(prefix) :]
                break
        try:
            number = int(version)
        except ValueError as exc:
            raise ValueError(f"Unsupported FreeSurfer version: {value!r}") from exc
        matches = {name: spec for name, spec in _MODEL_REPOS.items() if name.endswith(f"-v{number}")}
        if not matches:
            raise ValueError(f"Unsupported FreeSurfer version: {value!r}")
        selected.update(matches)
    return selected


def fetch_all_models(
    *,
    fs_versions: int | str | Iterable[int | str] | None = None,
    force_download: bool = False,
    cache_dir: str | Path | None = None,
    token: str | None = None,
) -> dict[str, Path]:
    """Ensure every released ScaleSurfer model is present and current.

    By default, Hugging Face checks the latest revision and reuses matching
    content-addressed cache entries, downloading only missing or updated files.
    Set ``force_download=True`` only to redownload every file unconditionally.
    Pass one or more ``fs_versions`` to restrict downloads, for example
    ``fs_versions=7`` or ``fs_versions=(6, 7)``. ``"base"`` selects the
    version-independent base stats model.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("Install huggingface-hub to fetch ScaleSurfer models.") from exc

    namespace = os.environ.get("SCALESURFER_HF_NAMESPACE", _DEFAULT_HF_NAMESPACE).strip().strip("/")
    paths: dict[str, Path] = {}
    for name, (repo_name, weights_filename) in _selected_models(fs_versions).items():
        repo_id = f"{namespace}/{repo_name}" if namespace else repo_name
        paths[name] = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=weights_filename,
                repo_type="model",
                cache_dir=cache_dir,
                token=token,
                force_download=force_download,
            )
        )
        hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            repo_type="model",
            cache_dir=cache_dir,
            token=token,
            force_download=force_download,
        )
    return paths

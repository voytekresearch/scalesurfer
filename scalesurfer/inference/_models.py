"""Checkpoint resolution and model configuration for volume inference."""

import json
import os
from pathlib import Path

import torch

from ._settings import (
    DEFAULT_VOLUME_HF_NAMESPACE,
    VOLUME_DTYPE_ALIASES,
    VOLUME_MODEL_CONFIG,
    VOLUME_MODEL_FILENAME,
    VOLUME_MODEL_SPECS,
)


def normalize_fs_version(fs_version) -> int:
    version = str(fs_version).strip().lower()
    for prefix in ("fsv", "fs", "v"):
        if version.startswith(prefix):
            version = version[len(prefix) :]
            break
    version = version.split(".", 1)[0]
    try:
        normalized = int(version)
    except ValueError as exc:
        raise ValueError(f"Unsupported fs_version {fs_version!r}; expected one of {sorted(VOLUME_MODEL_SPECS)}") from exc
    if normalized not in VOLUME_MODEL_SPECS:
        raise ValueError(f"Unsupported fs_version {fs_version!r}; expected one of {sorted(VOLUME_MODEL_SPECS)}")
    return normalized


def volume_hf_repo_id(repo_name: str) -> str:
    namespace = os.environ.get("SCALESURFER_HF_NAMESPACE", DEFAULT_VOLUME_HF_NAMESPACE).strip().strip("/")
    return f"{namespace}/{repo_name}" if namespace else repo_name


def candidate_volume_checkpoint_paths(spec: dict) -> list[Path]:
    model_root = os.environ.get("SCALESURFER_VOLUME_MODEL_DIR")
    return [Path(model_root).expanduser() / spec["repo_name"] / VOLUME_MODEL_FILENAME] if model_root else []


def download_volume_checkpoint(spec: dict, *, local_files_only: bool = False) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Install huggingface-hub to download ScaleSurfer volume checkpoints, "
            "or set SCALESURFER_VOLUME_MODEL_DIR to a local model directory."
        ) from exc
    return Path(hf_hub_download(
        repo_id=volume_hf_repo_id(spec["repo_name"]),
        filename=VOLUME_MODEL_FILENAME,
        repo_type="model",
        local_files_only=local_files_only,
    ))


def resolve_volume_checkpoint_path(fs_version: int, *, local_files_only: bool = False) -> Path:
    spec = VOLUME_MODEL_SPECS[fs_version]
    for path in candidate_volume_checkpoint_paths(spec):
        if path.exists():
            return path
    return download_volume_checkpoint(spec, local_files_only=local_files_only)


def load_volume_state_dict(path: Path) -> dict:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("Install safetensors to load ScaleSurfer volume checkpoints.") from exc
        return load_file(str(path), device="cpu")
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError(f"Checkpoint missing model_state: {path}")
    return ckpt["model_state"]


def shape_tuple(value, fallback: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(fallback) if value is None else tuple(int(v) for v in value)


def volume_config_from_payload(payload: dict | None) -> dict:
    cfg = dict(VOLUME_MODEL_CONFIG)
    if not isinstance(payload, dict):
        return cfg
    model_cfg = payload.get("model_config")
    if isinstance(model_cfg, str):
        try:
            model_cfg = json.loads(model_cfg)
        except json.JSONDecodeError:
            model_cfg = {}
    if isinstance(model_cfg, dict) and ("model_config" in model_cfg or "base_volume_shape" in model_cfg):
        payload = model_cfg
        model_cfg = payload.get("model_config")
    if isinstance(model_cfg, dict):
        cfg.update(model_cfg)
    for key in ("n_classes", "in_channels", "transformer_depth", "n_heads"):
        if payload.get(key) is not None:
            cfg[key] = int(payload[key])
    if payload.get("dropout") is not None:
        cfg["dropout"] = float(payload["dropout"])
    if payload.get("base_shape") is not None:
        cfg["base_shape"] = shape_tuple(payload["base_shape"], cfg["base_shape"])
    if payload.get("base_volume_shape") is not None:
        cfg["base_shape"] = shape_tuple(payload["base_volume_shape"], cfg["base_shape"])
    if payload.get("patch_size") is not None:
        cfg["patch_size"] = shape_tuple(payload["patch_size"], cfg["patch_size"])
    if cfg.get("channels") is not None:
        cfg["channels"] = shape_tuple(cfg["channels"], VOLUME_MODEL_CONFIG["channels"])
    return cfg


def load_volume_model_config(path: Path) -> dict:
    config_path = path.with_name("config.json")
    if config_path.exists():
        return volume_config_from_payload(json.loads(config_path.read_text(encoding="utf-8")))
    if path.suffix == ".safetensors":
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ImportError("Install safetensors to load ScaleSurfer volume checkpoint metadata.") from exc
        with safe_open(str(path), framework="pt", device="cpu") as f:
            return volume_config_from_payload(f.metadata())
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        return dict(VOLUME_MODEL_CONFIG)
    return volume_config_from_payload({
        "n_classes": ckpt.get("n_classes"),
        "base_volume_shape": ckpt.get("base_volume_shape"),
        "patch_size": ckpt.get("patch_size"),
        "model_config": ckpt.get("model_cfg"),
    })


def normalize_volume_dtype(dtype) -> torch.dtype:
    if dtype is None:
        return torch.float32
    if isinstance(dtype, torch.dtype):
        if dtype not in {torch.float32, torch.float16, torch.bfloat16}:
            raise ValueError("volume_dtype must be torch.float32, torch.float16, or torch.bfloat16")
        return dtype
    key = str(dtype).strip().lower().removeprefix("torch.").replace("-", "").replace("_", "")
    try:
        return VOLUME_DTYPE_ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(sorted(VOLUME_DTYPE_ALIASES))
        raise ValueError(f"Unsupported volume_dtype {dtype!r}; expected one of: {supported}") from exc


def dense_label_save_dtype(n_classes: int) -> torch.dtype:
    if int(n_classes) <= 256:
        return torch.uint8
    if int(n_classes) <= torch.iinfo(torch.int16).max + 1:
        return torch.int16
    if int(n_classes) <= torch.iinfo(torch.int32).max + 1:
        return torch.int32
    return torch.int64


from __future__ import annotations

import json
from pathlib import Path
from time import strftime

import torch

from .config import MODULE_PATH, SEED
from .data import resolve_paths, split_pairs_by_group


DEFAULT_TRAINING_CHECKPOINT = MODULE_PATH.parent / "checkpoints" / "transunet3d_best.pt"
DEFAULT_SPLIT_MANIFEST_PATH = MODULE_PATH.parent / "notebooks" / "splits" / "transunet3d_best_split_manifest.json"


def _required_split_keys() -> tuple[str, ...]:
    return ("x_train", "y_train", "x_val", "y_val", "x_test", "y_test")


def _resolve_list(paths):
    return [str(Path(p).expanduser().resolve()) for p in paths]


def _normalize_split(split: dict[str, list[str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key in _required_split_keys():
        if key not in split:
            raise KeyError(f"Missing split key: {key}")
        out[key] = _resolve_list(split[key])
    return out


def _split_overlap_counts(split: dict[str, list[str]]) -> dict[str, int]:
    x_train = set(split["x_train"])
    x_val = set(split["x_val"])
    x_test = set(split["x_test"])
    return {
        "train_val": int(len(x_train & x_val)),
        "train_test": int(len(x_train & x_test)),
        "val_test": int(len(x_val & x_test)),
    }


def assert_disjoint_split(split: dict[str, list[str]]) -> None:
    overlaps = _split_overlap_counts(split)
    bad = {k: v for k, v in overlaps.items() if int(v) > 0}
    if bad:
        raise ValueError(f"Split leakage detected: {bad}")


def _random_train_val_only_split(
    x_all: list[str],
    y_all: list[str],
    *,
    val_fraction: float,
    seed: int,
) -> dict[str, list[str]]:
    n_total = len(x_all)
    if n_total < 2:
        raise ValueError("Need at least 2 samples for train/val split")

    n_val = max(1, int(round(n_total * float(val_fraction))))
    n_val = min(n_val, n_total - 1)
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n_total, generator=g).tolist()
    val_idx = set(perm[:n_val])

    x_train, y_train, x_val, y_val = [], [], [], []
    for i, (x, y) in enumerate(zip(x_all, y_all)):
        if i in val_idx:
            x_val.append(x)
            y_val.append(y)
        else:
            x_train.append(x)
            y_train.append(y)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": [],
        "y_test": [],
    }


def reconstruct_split_from_data_cfg(data_cfg: dict, *, default_seed: int = SEED) -> dict[str, list[str]]:
    x_all = resolve_paths(data_cfg.get("x_train_files", []))
    y_all = resolve_paths(data_cfg.get("y_train_files", []))
    if not x_all or not y_all:
        raise ValueError("data_cfg must contain non-empty x_train_files and y_train_files")
    if len(x_all) != len(y_all):
        raise ValueError("x_train_files and y_train_files must have same length")

    x_val_exp = resolve_paths(data_cfg.get("x_val_files", [])) if data_cfg.get("x_val_files") else []
    y_val_exp = resolve_paths(data_cfg.get("y_val_files", [])) if data_cfg.get("y_val_files") else []
    x_test_exp = resolve_paths(data_cfg.get("x_test_files", [])) if data_cfg.get("x_test_files") else []
    y_test_exp = resolve_paths(data_cfg.get("y_test_files", [])) if data_cfg.get("y_test_files") else []

    # Explicit split lists always win.
    if x_val_exp or y_val_exp or x_test_exp or y_test_exp:
        if len(x_val_exp) != len(y_val_exp):
            raise ValueError("Explicit val split has mismatched x/y lengths")
        if len(x_test_exp) != len(y_test_exp):
            raise ValueError("Explicit test split has mismatched x/y lengths")
        split = {
            "x_train": list(x_all),
            "y_train": list(y_all),
            "x_val": list(x_val_exp),
            "y_val": list(y_val_exp),
            "x_test": list(x_test_exp),
            "y_test": list(y_test_exp),
        }
        return _normalize_split(split)

    seed = int(data_cfg.get("split_seed", default_seed))

    if bool(data_cfg.get("group_split_enabled", True)):
        ratios = tuple(float(v) for v in data_cfg.get("split_ratios", (0.8, 0.1, 0.1)))
        x_tr, y_tr, x_va, y_va, x_te, y_te = split_pairs_by_group(
            x_all,
            y_all,
            root=data_cfg.get("group_split_root", "tensors"),
            ratios=ratios,
            seed=seed,
        )
        split = {
            "x_train": x_tr,
            "y_train": y_tr,
            "x_val": x_va,
            "y_val": y_va,
            "x_test": x_te,
            "y_test": y_te,
        }
        return _normalize_split(split)

    split = _random_train_val_only_split(
        x_all,
        y_all,
        val_fraction=float(data_cfg.get("val_fraction", 0.1)),
        seed=seed,
    )
    return _normalize_split(split)


def load_training_data_cfg_from_checkpoint(checkpoint_path: str | Path = DEFAULT_TRAINING_CHECKPOINT) -> dict:
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    data_cfg = ckpt.get("data_cfg")
    if not isinstance(data_cfg, dict):
        raise ValueError(f"Checkpoint missing data_cfg dict: {checkpoint_path}")
    return data_cfg


def build_training_split_manifest(
    *,
    checkpoint_path: str | Path = DEFAULT_TRAINING_CHECKPOINT,
    manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH,
) -> dict:
    checkpoint_path = Path(checkpoint_path)
    data_cfg = load_training_data_cfg_from_checkpoint(checkpoint_path)
    split = reconstruct_split_from_data_cfg(data_cfg, default_seed=int(SEED))
    assert_disjoint_split(split)

    payload = {
        "schema_version": 1,
        "created_utc": strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {
            "type": "training_checkpoint",
            "checkpoint_path": str(checkpoint_path.resolve()),
        },
        "data_cfg_snapshot": data_cfg,
        "splits": {
            "train_notebook_split": split,
        },
        "counts": {
            "train_notebook_split": {
                "n_train": len(split["x_train"]),
                "n_val": len(split["x_val"]),
                "n_test": len(split["x_test"]),
            },
            "overlaps": _split_overlap_counts(split),
        },
    }

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2))
    return payload


def load_split_manifest(path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH) -> dict:
    payload = json.loads(Path(path).read_text())
    if "splits" not in payload or "train_notebook_split" not in payload["splits"]:
        raise ValueError("Invalid split manifest: missing splits.train_notebook_split")
    split = _normalize_split(payload["splits"]["train_notebook_split"])
    assert_disjoint_split(split)
    payload["splits"]["train_notebook_split"] = split
    return payload


def build_or_load_training_split_manifest(
    *,
    checkpoint_path: str | Path = DEFAULT_TRAINING_CHECKPOINT,
    manifest_path: str | Path = DEFAULT_SPLIT_MANIFEST_PATH,
    force_rebuild: bool = False,
) -> dict:
    manifest_path = Path(manifest_path)
    if manifest_path.exists() and not bool(force_rebuild):
        return load_split_manifest(manifest_path)
    return build_training_split_manifest(
        checkpoint_path=checkpoint_path,
        manifest_path=manifest_path,
    )


def split_from_manifest(manifest: dict, split_key: str = "train_notebook_split") -> dict[str, list[str]]:
    if "splits" not in manifest or split_key not in manifest["splits"]:
        raise KeyError(f"Missing split key in manifest: {split_key}")
    split = _normalize_split(manifest["splits"][split_key])
    assert_disjoint_split(split)
    return split


__all__ = [
    "DEFAULT_TRAINING_CHECKPOINT",
    "DEFAULT_SPLIT_MANIFEST_PATH",
    "assert_disjoint_split",
    "reconstruct_split_from_data_cfg",
    "load_training_data_cfg_from_checkpoint",
    "build_training_split_manifest",
    "load_split_manifest",
    "build_or_load_training_split_manifest",
    "split_from_manifest",
]

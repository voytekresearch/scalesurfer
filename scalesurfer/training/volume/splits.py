from __future__ import annotations

import json
import random
import re
from pathlib import Path
from time import strftime

import torch

from .config import MODULE_PATH, SEED
from .data import resolve_paths, split_pairs_by_group


DEFAULT_TRAINING_CHECKPOINT = MODULE_PATH.parent / "checkpoints" / "transunet3d_best.pt"
DEFAULT_SPLIT_MANIFEST_PATH = MODULE_PATH.parent / "notebooks" / "splits" / "transunet3d_best_split_manifest.json"
DEFAULT_NOTEBOOK_SPLIT_MANIFEST_PATH = (
    MODULE_PATH.parent / "docs" / "notebooks" / "01_volume" / "04_train_split_manifest.json"
)
_STUDY_RE = re.compile(r"(ds\d{6,})", re.IGNORECASE)


def _required_split_keys() -> tuple[str, ...]:
    return ("x_train", "y_train", "x_val", "y_val", "x_test", "y_test")


def extract_study_id(path: str | Path) -> str:
    """Extract the study/dataset group used by the training notebook split."""
    text = str(path)
    match = _STUDY_RE.search(text)
    if match is not None:
        return match.group(1).lower()

    parts = {part.lower() for part in Path(path).parts}
    if "hcp_filt" in parts:
        return "hcp_filt"

    raise ValueError(f"Could not extract study id from path: {path}")


def _resolve_tensor_root(data_root: str | Path) -> Path:
    root = Path(data_root).expanduser()
    return root if root.name == "tensors" else root / "tensors"


def discover_training_pairs(
    root: str | Path,
    *,
    x_filename: str = "orig.pt",
    y_filename: str = "aparc+aseg.pt",
) -> dict[str, dict[str, str]]:
    """Discover paired tensor files under a tensor root.

    Returns a mapping keyed by the relative parent directory. Each value has
    ``orig`` and ``aparc+aseg`` entries for compatibility with the training
    notebooks.
    """
    root = Path(root).expanduser()
    if not root.exists():
        return {}

    file_map: dict[str, dict[str, str]] = {}
    for x_path in sorted(root.rglob(x_filename)):
        y_path = x_path.with_name(y_filename)
        if not y_path.exists():
            continue
        try:
            key = x_path.parent.relative_to(root).as_posix()
        except ValueError:
            key = x_path.parent.as_posix()
        file_map[key] = {
            "orig": str(x_path.resolve()),
            "aparc+aseg": str(y_path.resolve()),
        }
    return file_map


def _study_split_counts(n: int, ratios=(0.8, 0.1, 0.1)) -> tuple[int, int, int]:
    _, val_ratio, test_ratio = (float(ratios[0]), float(ratios[1]), float(ratios[2]))
    if n <= 0:
        return 0, 0, 0

    n_val = int(round(n * val_ratio))
    n_test = int(round(n * test_ratio))
    if n >= 10:
        n_val = max(1, n_val)
        n_test = max(1, n_test)

    if n_val + n_test >= n:
        overflow = n_val + n_test - (n - 1)
        while overflow > 0 and (n_val > 0 or n_test > 0):
            if n_val >= n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
            overflow -= 1

    n_train = n - n_val - n_test
    return n_train, n_val, n_test


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


def build_study_split_manifest(
    *,
    data_root: str | Path,
    manifest_path: str | Path = DEFAULT_NOTEBOOK_SPLIT_MANIFEST_PATH,
    x_filename: str = "orig.pt",
    y_filename: str = "aparc+aseg.pt",
    seed: int = SEED,
    ratios=(0.8, 0.1, 0.1),
) -> dict:
    """Build the per-study split manifest used by the volume notebooks."""
    data_root = Path(data_root).expanduser()
    tensors_root = _resolve_tensor_root(data_root)
    file_map = discover_training_pairs(
        tensors_root,
        x_filename=x_filename,
        y_filename=y_filename,
    )
    if not file_map:
        raise ValueError(f"No training pairs found under {tensors_root}")

    grouped: dict[str, list[tuple[str, str]]] = {}
    for paths in file_map.values():
        study = extract_study_id(paths["orig"])
        grouped.setdefault(study, []).append((paths["orig"], paths["aparc+aseg"]))

    split = {key: [] for key in _required_split_keys()}
    studies: dict[str, dict[str, int]] = {}
    for study in sorted(grouped):
        pairs = sorted(grouped[study])
        rng = random.Random(int(seed) + sum(ord(ch) for ch in study))
        rng.shuffle(pairs)
        n_train, n_val, n_test = _study_split_counts(len(pairs), ratios=ratios)
        studies[study] = {
            "n_total": len(pairs),
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
        }

        train_pairs = pairs[:n_train]
        val_pairs = pairs[n_train : n_train + n_val]
        test_pairs = pairs[n_train + n_val : n_train + n_val + n_test]
        for x, y in train_pairs:
            split["x_train"].append(x)
            split["y_train"].append(y)
        for x, y in val_pairs:
            split["x_val"].append(x)
            split["y_val"].append(y)
        for x, y in test_pairs:
            split["x_test"].append(x)
            split["y_test"].append(y)

    split = _normalize_split(split)
    assert_disjoint_split(split)
    manifest_path = Path(manifest_path).expanduser()
    payload = {
        "schema_version": 2,
        "created_utc": strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_path": str(manifest_path.resolve()),
        "source": {
            "type": "filesystem_discovery",
            "data_root": str(data_root.resolve()),
            "tensors_root": str(tensors_root.resolve()),
            "x_filename": str(x_filename),
            "y_filename": str(y_filename),
        },
        "split_strategy": {
            "type": "per_study",
            "study_pattern": "/ds*/",
            "seed": int(seed),
            "ratios": [float(r) for r in ratios],
        },
        "discovery": {
            "n_pairs": len(file_map),
            "n_studies": len(studies),
            "study_ids": sorted(studies),
        },
        "studies": studies,
        "splits": {
            "train_notebook_split": split,
        },
        "counts": {
            "train_notebook_split": {
                "n_train": len(split["x_train"]),
                "n_val": len(split["x_val"]),
                "n_test": len(split["x_test"]),
                "n_total": len(split["x_train"]) + len(split["x_val"]) + len(split["x_test"]),
            },
            "overlaps": _split_overlap_counts(split),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2))
    return payload


def build_or_load_study_split_manifest(
    *,
    data_root: str | Path,
    manifest_path: str | Path = DEFAULT_NOTEBOOK_SPLIT_MANIFEST_PATH,
    x_filename: str = "orig.pt",
    y_filename: str = "aparc+aseg.pt",
    seed: int = SEED,
    ratios=(0.8, 0.1, 0.1),
    force_rebuild: bool = False,
) -> dict:
    manifest_path = Path(manifest_path).expanduser()
    if manifest_path.exists() and not bool(force_rebuild):
        return load_split_manifest(manifest_path)
    return build_study_split_manifest(
        data_root=data_root,
        manifest_path=manifest_path,
        x_filename=x_filename,
        y_filename=y_filename,
        seed=seed,
        ratios=ratios,
    )


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
    "DEFAULT_NOTEBOOK_SPLIT_MANIFEST_PATH",
    "DEFAULT_TRAINING_CHECKPOINT",
    "DEFAULT_SPLIT_MANIFEST_PATH",
    "assert_disjoint_split",
    "extract_study_id",
    "discover_training_pairs",
    "reconstruct_split_from_data_cfg",
    "load_training_data_cfg_from_checkpoint",
    "build_training_split_manifest",
    "build_study_split_manifest",
    "build_or_load_study_split_manifest",
    "load_split_manifest",
    "build_or_load_training_split_manifest",
    "split_from_manifest",
]

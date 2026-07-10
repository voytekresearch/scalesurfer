from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ..training.volume.config import SEED
from ..experiments import build_v8_split_from_root
from ..training.volume.splits import assert_disjoint_split, split_from_manifest
from ..volume.eval import parse_freesurfer_stats


REQUIRED_STATS = (
    "stats/aseg.stats",
    "stats/lh.aparc.stats",
    "stats/rh.aparc.stats",
)

OPTIONAL_STATS = (
    "stats/brainvol.stats",
    "stats/wmparc.stats",
    "stats/aparc+aseg.stats",
    "stats/lh.aparc.pial.stats",
    "stats/rh.aparc.pial.stats",
    "stats/lh.curv.stats",
    "stats/rh.curv.stats",
)

DEFAULT_STATS_RELPATHS = REQUIRED_STATS + OPTIONAL_STATS
VOLUME_TABLE_COLUMNS = frozenset({"Volume_mm3"})
APARC_TABLE_COLUMNS = frozenset(
    {
        "NumVert",
        "SurfArea",
        "GrayVol",
        "ThickAvg",
        "ThickStd",
        "MeanCurv",
        "GausCurv",
        "FoldInd",
        "CurvInd",
    }
)
APARC_STATS_NAMES = frozenset({"lh.aparc", "rh.aparc"})
GLOBAL_STATS_NAMES = frozenset({"aseg", "wmparc", "aparc+aseg", "brainvol", "lh.aparc", "rh.aparc", "lh.curv", "rh.curv"})
APARC_REGIONS = frozenset(
    {
        "bankssts",
        "caudalanteriorcingulate",
        "caudalmiddlefrontal",
        "cuneus",
        "entorhinal",
        "fusiform",
        "inferiorparietal",
        "inferiortemporal",
        "isthmuscingulate",
        "lateraloccipital",
        "lateralorbitofrontal",
        "lingual",
        "medialorbitofrontal",
        "middletemporal",
        "parahippocampal",
        "paracentral",
        "parsopercularis",
        "parsorbitalis",
        "parstriangularis",
        "pericalcarine",
        "postcentral",
        "posteriorcingulate",
        "precentral",
        "precuneus",
        "rostralanteriorcingulate",
        "rostralmiddlefrontal",
        "superiorfrontal",
        "superiorparietal",
        "superiortemporal",
        "supramarginal",
        "frontalpole",
        "temporalpole",
        "transversetemporal",
        "insula",
    }
)

TARGET_VALUE_RANGES = {
    "Volume_mm3": (0.0, math.inf),
    "NumVert": (0.0, 200_000.0),
    "SurfArea": (0.0, 50_000.0),
    "GrayVol": (0.0, 200_000.0),
    "ThickAvg": (0.0, 8.0),
    "ThickStd": (0.0, 2.0),
    "MeanCurv": (-1.0, 1.0),
    "GausCurv": (-1.0, 1.0),
    "FoldInd": (0.0, 5_000.0),
    "CurvInd": (0.0, 200.0),
}

_DS_RE = re.compile(r"(ds\d{6,})", re.IGNORECASE)


def strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {str(k).removeprefix("_orig_mod."): v for k, v in state_dict.items()}


def extract_dataset_id(path: str | Path) -> str | None:
    match = _DS_RE.search(str(path))
    return match.group(1).lower() if match is not None else None


def derive_fs_major_by_dataset_from_checkpoints(checkpoints: dict[int, str | Path]) -> dict[str, int]:
    out: dict[str, int] = {}
    for major, ckpt_path in sorted(checkpoints.items()):
        ckpt = torch.load(Path(ckpt_path), map_location="cpu")
        data_cfg = ckpt.get("data_cfg", {})
        if not isinstance(data_cfg, dict):
            continue
        for key in ("x_train_files", "x_val_files", "x_test_files"):
            for x_path in data_cfg.get(key, []) or []:
                ds = extract_dataset_id(x_path)
                if ds is None:
                    continue
                previous = out.get(ds)
                if previous is not None and int(previous) != int(major):
                    raise ValueError(f"Conflicting FS major mapping for {ds}: {previous} vs {major}")
                out[ds] = int(major)
    return dict(sorted(out.items()))


def save_json(path: str | Path, payload: object) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _tensor_subject_root(x_path: str | Path, tensor_root: str | Path) -> str | None:
    x_path = Path(x_path).expanduser().resolve()
    tensor_root = Path(tensor_root).expanduser().resolve()
    if not _is_relative_to(x_path, tensor_root):
        return None
    rel = x_path.relative_to(tensor_root)
    if len(rel.parts) < 3 or rel.parts[-2] != "mri" or not rel.parts[0].startswith("ds"):
        return None
    return "/".join(rel.parts[:-2])


def _has_required_stats(stats_dir: str | Path, required_stats: Iterable[str] = REQUIRED_STATS) -> bool:
    stats_dir = Path(stats_dir)
    return all((stats_dir / Path(rel).name).exists() for rel in required_stats)


def build_openneuro_stats_rows(
    *,
    split: dict[str, list[str]],
    tensor_root: str | Path,
    stats_cache_root: str | Path,
    fs_major_by_dataset: dict[str, int],
    required_stats: Iterable[str] = REQUIRED_STATS,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    stats_cache_root = Path(stats_cache_root).expanduser().resolve()
    tensor_root = Path(tensor_root).expanduser().resolve()

    for split_name in ("train", "val", "test"):
        xs = split.get(f"x_{split_name}", [])
        ys = split.get(f"y_{split_name}", [])
        for x_path, seg_path in zip(xs, ys, strict=False):
            subject_root = _tensor_subject_root(x_path, tensor_root)
            if subject_root is None:
                continue
            dataset = subject_root.split("/", 1)[0].lower()
            fs_major = fs_major_by_dataset.get(dataset)
            if fs_major is None:
                continue
            stats_dir = stats_cache_root / subject_root / "stats"
            if not _has_required_stats(stats_dir, required_stats):
                continue
            rows.append(
                {
                    "sample_id": f"openneuro:{subject_root}",
                    "subject_root": subject_root,
                    "subject_id": subject_root.rsplit("/", 1)[-1],
                    "dataset": dataset,
                    "source": "openneuro",
                    "fs_major": int(fs_major),
                    "split": split_name,
                    "x_path": str(Path(x_path).expanduser().resolve()),
                    "seg_path": str(Path(seg_path).expanduser().resolve()),
                    "stats_dir": str(stats_dir),
                }
            )

    return pd.DataFrame(rows)


def _subject_id_from_v8_tensor_path(path: str | Path) -> str:
    path = Path(path).expanduser().resolve()
    if path.parent.name != "mri":
        raise ValueError(f"Expected .../<subject>/mri/orig.pt, got {path}")
    return path.parent.parent.name


def build_v8_stats_rows(
    *,
    tensors_root: str | Path,
    stats_root: str | Path,
    seed: int = SEED,
    ratios=(0.8, 0.1, 0.1),
    required_stats: Iterable[str] = REQUIRED_STATS,
) -> pd.DataFrame:
    split = build_v8_split_from_root(tensors_root=tensors_root, seed=seed, ratios=ratios)
    assert_disjoint_split(split)
    stats_root = Path(stats_root).expanduser().resolve()

    rows: list[dict[str, object]] = []
    for split_name in ("train", "val", "test"):
        xs = split.get(f"x_{split_name}", [])
        ys = split.get(f"y_{split_name}", [])
        for x_path, seg_path in zip(xs, ys, strict=False):
            subject_id = _subject_id_from_v8_tensor_path(x_path)
            stats_dir = stats_root / subject_id / "stats"
            if not _has_required_stats(stats_dir, required_stats):
                continue
            rows.append(
                {
                    "sample_id": f"fs8:{subject_id}",
                    "subject_root": subject_id,
                    "subject_id": subject_id,
                    "dataset": extract_dataset_id(subject_id) or "fs8",
                    "source": "fs8",
                    "fs_major": 8,
                    "split": split_name,
                    "x_path": str(Path(x_path).expanduser().resolve()),
                    "seg_path": str(Path(seg_path).expanduser().resolve()),
                    "stats_dir": str(stats_dir),
                }
            )

    return pd.DataFrame(rows)


def load_split(path: str | Path) -> dict[str, list[str]]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    return split_from_manifest(manifest, split_key="train_notebook_split")


def combine_sample_rows(openneuro_rows: pd.DataFrame, v8_rows: pd.DataFrame) -> pd.DataFrame:
    samples = pd.concat([openneuro_rows, v8_rows], ignore_index=True)
    if samples.empty:
        raise ValueError("No supervised stats samples were found.")
    if samples["sample_id"].duplicated().any():
        dupes = samples.loc[samples["sample_id"].duplicated(), "sample_id"].head(10).tolist()
        raise ValueError(f"Duplicate sample_id values: {dupes}")
    samples = samples.sort_values(["fs_major", "source", "dataset", "sample_id"]).reset_index(drop=True)
    return samples


def safe_name(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_.+\-]+", "_", text)
    return text.strip("_") or "value"


def row_key_from_stats_table(row: pd.Series) -> str:
    if "StructName" in row and pd.notna(row["StructName"]):
        return safe_name(row["StructName"])
    if "SegId" in row and pd.notna(row["SegId"]):
        return f"seg{int(row['SegId'])}"
    if "Index" in row and pd.notna(row["Index"]):
        return f"row{int(row['Index'])}"
    return f"row{int(row.name)}"


def group_for_target(target_name: str) -> str:
    stats_name = target_name.split(":", 1)[0]
    if ":global:" in target_name or stats_name == "brainvol":
        return "global"
    if stats_name.startswith("lh.aparc"):
        return "lh_aparc"
    if stats_name.startswith("rh.aparc"):
        return "rh_aparc"
    return "aseg"


def target_parts(target_name: str) -> tuple[str, str | None, str | None]:
    parts = str(target_name).split(":")
    stats_name = parts[0] if parts else ""
    if len(parts) >= 3 and parts[1] == "global":
        return stats_name, "global", parts[2]
    if len(parts) >= 3:
        return stats_name, parts[1], parts[2]
    return stats_name, None, None


def measure_for_target(target_name: str) -> str:
    _, _, measure = target_parts(target_name)
    return str(measure or "")


def target_is_supported(target_name: str) -> bool:
    stats_name, region, measure = target_parts(target_name)
    if region == "global":
        return stats_name in GLOBAL_STATS_NAMES
    if stats_name in {"aseg", "wmparc", "aparc+aseg"}:
        return measure in VOLUME_TABLE_COLUMNS
    if stats_name in APARC_STATS_NAMES:
        return region in APARC_REGIONS and measure in APARC_TABLE_COLUMNS
    return False


def target_value_range(target_name: str) -> tuple[float, float] | None:
    measure = measure_for_target(target_name)
    return TARGET_VALUE_RANGES.get(measure)


def sanitize_target_matrix(target_matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = target_matrix.copy()
    rows: list[dict[str, object]] = []
    for target in matrix.columns:
        stats_name, region, measure = target_parts(str(target))
        supported = target_is_supported(str(target))
        valid_before = int(matrix[target].notna().sum())
        invalid_n = 0
        if not supported:
            invalid_n = valid_before
            matrix[target] = np.nan
        else:
            value_range = target_value_range(str(target))
            if value_range is not None:
                lo, hi = value_range
                values = matrix[target]
                valid = values.isna() | ((values >= lo) & (values <= hi))
                invalid_n = int((~valid).sum())
                if invalid_n:
                    matrix.loc[~valid, target] = np.nan
        valid_after = int(matrix[target].notna().sum())
        if invalid_n or not supported:
            rows.append(
                {
                    "target": target,
                    "stats_name": stats_name,
                    "region": region,
                    "measure": measure,
                    "supported": bool(supported),
                    "n_before": valid_before,
                    "n_removed": invalid_n,
                    "n_after": valid_after,
                }
            )
    report = pd.DataFrame(rows)
    return matrix, report


def allowed_table_columns(stats_name: str) -> set[str]:
    if stats_name in {"aseg", "wmparc", "aparc+aseg"}:
        return set(VOLUME_TABLE_COLUMNS)
    if stats_name in APARC_STATS_NAMES:
        return set(APARC_TABLE_COLUMNS)
    return set()


def flatten_stats_file(sample_id: str, relpath: str, path: str | Path) -> list[dict[str, object]]:
    parsed = parse_freesurfer_stats(path)
    rows: list[dict[str, object]] = []
    stats_name = Path(relpath).name.removesuffix(".stats")

    measures = parsed.get("measures", {})
    if isinstance(measures, dict):
        for field, payload in measures.items():
            value = payload.get("value") if isinstance(payload, dict) else payload
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
                target_name = f"{stats_name}:global:{safe_name(field)}"
                rows.append(
                    {
                        "sample_id": sample_id,
                        "stats_relpath": relpath,
                        "kind": "measure",
                        "target_name": target_name,
                        "group": group_for_target(target_name),
                        "value": float(value),
                    }
                )

    table = parsed.get("table")
    allowed_cols = allowed_table_columns(stats_name)
    if isinstance(table, pd.DataFrame) and len(table) and "StructName" in table.columns and allowed_cols:
        skip_cols = {"Index", "SegId"}
        for _, table_row in table.iterrows():
            region = row_key_from_stats_table(table_row)
            if stats_name in APARC_STATS_NAMES and region not in APARC_REGIONS:
                continue
            for col, value in table_row.items():
                if col in skip_cols or col == "StructName" or col not in allowed_cols:
                    continue
                if pd.notna(value) and isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
                    target_name = f"{stats_name}:{region}:{safe_name(col)}"
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "stats_relpath": relpath,
                            "kind": "table",
                            "target_name": target_name,
                            "group": group_for_target(target_name),
                            "value": float(value),
                        }
                    )
    return rows


def build_target_long_table(
    samples: pd.DataFrame,
    *,
    stats_relpaths: Iterable[str] = DEFAULT_STATS_RELPATHS,
    desc: str = "parse stats",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    records = samples[["sample_id", "stats_dir"]].drop_duplicates().to_dict("records")
    for record in tqdm(records, desc=desc):
        stats_dir = Path(record["stats_dir"])
        sample_id = str(record["sample_id"])
        for relpath in stats_relpaths:
            path = stats_dir / Path(relpath).name
            if path.exists():
                rows.extend(flatten_stats_file(sample_id, relpath, path))
    if not rows:
        raise ValueError("No numeric stats targets were parsed.")
    return pd.DataFrame(rows)


def build_target_matrix(long_targets: pd.DataFrame) -> pd.DataFrame:
    matrix = (
        long_targets.pivot_table(index="sample_id", columns="target_name", values="value", aggfunc="first")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    return matrix


def select_target_columns(
    target_matrix: pd.DataFrame,
    train_sample_ids: Iterable[str],
    *,
    min_train_nonmissing: int = 10,
    min_train_std: float = 1e-6,
    require_supported: bool = True,
) -> list[str]:
    train_ids = [sid for sid in train_sample_ids if sid in target_matrix.index]
    if not train_ids:
        raise ValueError("No training sample IDs are present in the target matrix.")
    train = target_matrix.loc[train_ids]
    counts = train.notna().sum(axis=0)
    std = train.std(axis=0, skipna=True).fillna(0.0)
    keep = (counts >= int(min_train_nonmissing)) & (std > float(min_train_std))
    if require_supported:
        supported = pd.Series([target_is_supported(col) for col in train.columns], index=train.columns)
        keep &= supported
    selected = keep[keep].index.tolist()
    if not selected:
        raise ValueError("No targets passed min_train_nonmissing.")
    return selected


def target_normalization(target_matrix: pd.DataFrame, train_sample_ids: Iterable[str], columns: list[str]) -> pd.DataFrame:
    train_ids = [sid for sid in train_sample_ids if sid in target_matrix.index]
    train = target_matrix.loc[train_ids, columns]
    mean = train.mean(axis=0, skipna=True)
    std = train.std(axis=0, skipna=True).replace(0.0, 1.0).fillna(1.0)
    n = train.notna().sum(axis=0)
    groups = [group_for_target(col) for col in columns]
    return pd.DataFrame({"target": columns, "group": groups, "mean": mean.values, "std": std.values, "n_train": n.values})


def columns_by_group(columns: Iterable[str]) -> dict[str, list[str]]:
    out = {"aseg": [], "lh_aparc": [], "rh_aparc": [], "global": []}
    for col in columns:
        out.setdefault(group_for_target(col), []).append(col)
    return {k: v for k, v in out.items() if v}


def load_volume_tensor(path: str | Path, *, is_seg: bool = False) -> torch.Tensor:
    tensor = torch.load(Path(path), map_location="cpu")
    tensor = torch.as_tensor(tensor)

    if tensor.ndim == 3:
        pass
    elif tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    else:
        raise ValueError(f"Expected 3D tensor or [1,D,H,W], got {tuple(tensor.shape)} from {path}")

    if is_seg:
        return tensor.to(torch.int64)

    tensor = tensor.to(torch.float32)

    if not torch.isfinite(tensor).all():
        n_bad = int((~torch.isfinite(tensor)).sum().item())
        raise FloatingPointError(f"Non-finite values in image tensor: {path} ({n_bad} bad voxels)")

    std = tensor.std()
    if torch.isfinite(std) and float(std) > 1e-6:
        tensor = (tensor - tensor.mean()) / std.clamp_min(1e-6)
    else:
        tensor = tensor - tensor.mean()

    return tensor


class StatsTensorDataset(Dataset):
    def __init__(
        self,
        samples: pd.DataFrame,
        target_matrix: pd.DataFrame,
        target_stats: pd.DataFrame,
        columns_by_group: dict[str, list[str]],
    ):
        self.samples = samples.reset_index(drop=True).copy()
        self.target_matrix = target_matrix
        self.target_stats = target_stats.set_index("target")
        self.columns_by_group = {k: list(v) for k, v in columns_by_group.items()}
        self.sample_ids = self.samples["sample_id"].astype(str).tolist()
        self._y_by_group: dict[str, torch.Tensor] = {}
        self._mask_by_group: dict[str, torch.Tensor] = {}
        self._prepare_targets()

    def _prepare_targets(self) -> None:
        for group, cols in self.columns_by_group.items():
            values = self.target_matrix.reindex(self.sample_ids)[cols].to_numpy(dtype=np.float32)
            finite = np.isfinite(values)
            mean = self.target_stats.loc[cols, "mean"].to_numpy(dtype=np.float32)
            std = self.target_stats.loc[cols, "std"].to_numpy(dtype=np.float32)
            norm = (values - mean[None, :]) / np.maximum(std[None, :], 1e-6)
            norm = np.where(finite, norm, 0.0).astype(np.float32, copy=False)
            self._y_by_group[group] = torch.from_numpy(norm)
            self._mask_by_group[group] = torch.from_numpy(finite.astype(np.bool_))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.samples.iloc[idx]
        sample_id = self.sample_ids[idx]
        x = load_volume_tensor(row["x_path"], is_seg=False).unsqueeze(0)
        seg = load_volume_tensor(row["seg_path"], is_seg=True)
        y = {group: tensor[idx] for group, tensor in self._y_by_group.items()}
        mask = {group: tensor[idx] for group, tensor in self._mask_by_group.items()}
        return {"x": x, "seg": seg, "y": y, "mask": mask, "sample_id": sample_id, "row": row.to_dict()}


def collate_stats_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    groups = batch[0]["y"].keys()
    out = {
        "y": {g: torch.stack([item["y"][g] for item in batch], dim=0) for g in groups},
        "mask": {g: torch.stack([item["mask"][g] for item in batch], dim=0) for g in groups},
        "sample_id": [item["sample_id"] for item in batch],
        "rows": [item["row"] for item in batch],
    }
    if "h" in batch[0]:
        out["h"] = torch.stack([item["h"] for item in batch], dim=0)
    else:
        out["x"] = torch.stack([item["x"] for item in batch], dim=0)
        out["seg"] = torch.stack([item["seg"] for item in batch], dim=0)
    return out


def make_stats_loader(
    samples: pd.DataFrame,
    target_matrix: pd.DataFrame,
    target_stats: pd.DataFrame,
    cols_by_group: dict[str, list[str]],
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
) -> DataLoader:
    dataset = StatsTensorDataset(samples, target_matrix, target_stats, cols_by_group)
    kwargs = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "collate_fn": collate_stats_batch,
        "pin_memory": torch.cuda.is_available(),
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = bool(True if persistent_workers is None else persistent_workers)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **kwargs)


class StatsFeatureDataset(StatsTensorDataset):
    def __init__(
        self,
        samples: pd.DataFrame,
        target_matrix: pd.DataFrame,
        target_stats: pd.DataFrame,
        columns_by_group: dict[str, list[str]],
        *,
        feature_payload: dict[str, object],
    ):
        super().__init__(samples, target_matrix, target_stats, columns_by_group)
        ids = [str(v) for v in feature_payload["sample_id"]]
        features = torch.as_tensor(feature_payload["features"], dtype=torch.float32)
        if not torch.isfinite(features).all():
            raise FloatingPointError("Cached pooled features contain NaN/inf. Delete the feature cache and rebuild.")

        feature_map = {sample_id: i for i, sample_id in enumerate(ids)}
        missing = [sample_id for sample_id in self.sample_ids if sample_id not in feature_map]
        if missing:
            raise ValueError(f"Feature cache is missing {len(missing)} sample(s), first: {missing[:3]}")
        self.features = features[[feature_map[sample_id] for sample_id in self.sample_ids]].contiguous()

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.samples.iloc[idx]
        y = {group: tensor[idx] for group, tensor in self._y_by_group.items()}
        mask = {group: tensor[idx] for group, tensor in self._mask_by_group.items()}
        return {
            "h": self.features[idx],
            "y": y,
            "mask": mask,
            "sample_id": self.sample_ids[idx],
            "row": row.to_dict(),
        }


def make_stats_feature_loader(
    samples: pd.DataFrame,
    target_matrix: pd.DataFrame,
    target_stats: pd.DataFrame,
    cols_by_group: dict[str, list[str]],
    *,
    feature_payload: dict[str, object],
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
) -> DataLoader:
    dataset = StatsFeatureDataset(
        samples,
        target_matrix,
        target_stats,
        cols_by_group,
        feature_payload=feature_payload,
    )
    kwargs = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "collate_fn": collate_stats_batch,
        "pin_memory": torch.cuda.is_available(),
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = bool(True if persistent_workers is None else persistent_workers)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **kwargs)


def _fingerprint_payload(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def pooled_feature_cache_path(
    *,
    cache_dir: str | Path,
    stage: str,
    segmentation_checkpoint: str | Path,
    sample_ids: Iterable[str],
    pool_features: Iterable[str],
    label_ids: Iterable[int],
    feature_schema: str,
) -> Path:
    ckpt_path = Path(segmentation_checkpoint).expanduser().resolve()
    stat = ckpt_path.stat()
    key = _fingerprint_payload(
        {
            "stage": str(stage),
            "checkpoint": str(ckpt_path),
            "checkpoint_size": int(stat.st_size),
            "checkpoint_mtime_ns": int(stat.st_mtime_ns),
            "sample_ids": list(map(str, sample_ids)),
            "pool_features": list(map(str, pool_features)),
            "label_ids": list(map(int, label_ids)),
            "feature_schema": str(feature_schema),
        }
    )
    return Path(cache_dir) / f"{stage}_{key}.pt"


def load_or_build_pooled_feature_cache(
    model: StatsPredictionModel,
    samples: pd.DataFrame,
    *,
    cache_path: str | Path,
    target_matrix: pd.DataFrame,
    target_stats: pd.DataFrame,
    cols_by_group: dict[str, list[str]],
    batch_size: int = 1,
    num_workers: int = 0,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    amp: bool = False,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    cache_path = Path(cache_path)
    sample_ids = samples["sample_id"].astype(str).tolist()
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        cached_ids = [str(v) for v in payload.get("sample_id", [])]
        features = payload.get("features")
        if (
            cached_ids == sample_ids
            and torch.as_tensor(features).shape[1] == int(model.input_dim)
            and str(payload.get("feature_schema", "")) == str(model.feature_schema)
        ):
            print("loaded pooled feature cache:", cache_path)
            return payload
        print("pooled feature cache exists but does not match current samples/model; rebuilding:", cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    loader = make_stats_loader(
        samples,
        target_matrix,
        target_stats,
        cols_by_group,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    use_amp = bool(amp) and torch.device(device).type == "cuda"
    model.eval()
    features = []
    ids = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"cache features {cache_path.stem}", leave=False):
            x = batch["x"].to(device=device, non_blocking=True)
            seg = batch["seg"].to(device=device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                h = model.extract_features(x, seg)
            features.append(h.detach().float().cpu())
            ids.extend(batch["sample_id"])
    payload = {
        "sample_id": ids,
        "features": torch.cat(features, dim=0).contiguous(),
        "input_dim": int(model.input_dim),
        "pool_features": list(model.pool_features),
        "label_ids": list(model.label_ids),
        "feature_schema": str(model.feature_schema),
        "pool_stat_names": list(model.pool_stat_names),
        "label_size_feature_names": list(model.label_size_feature_names),
    }
    torch.save(payload, cache_path)
    print("wrote pooled feature cache:", cache_path)
    return payload

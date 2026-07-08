from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from ..data import build_label_lut, default_aparc_aseg_label_values
from ..metrics import dense_labels_to_fs_ids
from ..volume.model import TransUNet3D
from .models import StatsPredictionModel, TransUNetEncoderAdapter
from .utils import (
    APARC_STATS_NAMES,
    APARC_TABLE_COLUMNS,
    DEFAULT_STATS_RELPATHS,
    VOLUME_TABLE_COLUMNS,
    flatten_stats_file,
    safe_name,
    strip_compile_prefix,
    target_parts,
)


STATS_MODEL_FILENAME = "stats_model.safetensors"
STATS_CONFIG_FILENAME = "config.json"
STATS_METADATA_FILENAME = "metadata.json"
DEFAULT_STATS_HF_NAMESPACE = "rphammonds"
PREDICTED_STATS_LONG_FILENAME = "scalesurfer_stats_long.csv"
PREDICTED_STATS_WIDE_FILENAME = "scalesurfer_stats_wide.csv"
PREDICTED_STATS_PROVENANCE_FILENAME = "scalesurfer_stats.json"
PREDICTED_STATS_LOG_FILENAME = "scalesurfer_stats.log"

APARC_TABLE_MEASURE_ORDER = (
    "NumVert",
    "SurfArea",
    "GrayVol",
    "ThickAvg",
    "ThickStd",
    "MeanCurv",
    "GausCurv",
    "FoldInd",
    "CurvInd",
)
VOLUME_TABLE_MEASURE_ORDER = ("Volume_mm3",)


def stats_repo_name(fs_version: int | str) -> str:
    version = str(fs_version).strip().lower()
    if version in {"base", "all", "base_all", "stats_base_all"}:
        return "scalesurfer-stats-base"
    for prefix in ("stats_fsv", "fsv", "fs", "v"):
        if version.startswith(prefix):
            version = version[len(prefix) :]
            break
    return f"scalesurfer-stats-v{int(version)}"


def _stats_hf_repo_id(repo_name: str) -> str:
    namespace = os.environ.get("SCALESURFER_HF_NAMESPACE", DEFAULT_STATS_HF_NAMESPACE).strip().strip("/")
    return f"{namespace}/{repo_name}" if namespace else repo_name


def _candidate_stats_repo_dirs(repo_name: str) -> list[Path]:
    paths: list[Path] = []
    model_root = os.environ.get("SCALESURFER_STATS_MODEL_DIR")
    if model_root:
        paths.append(Path(model_root).expanduser() / repo_name)
    return paths


def _download_stats_file(repo_name: str, filename: str, *, local_files_only: bool = False) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Install huggingface-hub to download ScaleSurfer stats checkpoints, "
            "or set SCALESURFER_STATS_MODEL_DIR to a local model directory."
        ) from exc

    try:
        return Path(
            hf_hub_download(
                repo_id=_stats_hf_repo_id(repo_name),
                filename=filename,
                repo_type="model",
                local_files_only=local_files_only,
            )
        )
    except Exception as exc:
        if isinstance(exc, FileNotFoundError) or "EntryNotFound" in type(exc).__name__:
            raise FileNotFoundError(f"{repo_name}/{filename}") from exc
        raise


def resolve_stats_checkpoint_path(
    fs_version: int | str = 7,
    *,
    checkpoint_path: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    repo_name = stats_repo_name(fs_version)
    if checkpoint_path is not None:
        path = Path(checkpoint_path).expanduser()
        if path.suffix == ".safetensors" and path.exists():
            return path
        return _download_stats_file(repo_name, STATS_MODEL_FILENAME, local_files_only=local_files_only)

    for repo_dir in _candidate_stats_repo_dirs(repo_name):
        path = repo_dir / STATS_MODEL_FILENAME
        if path.exists():
            return path

    try:
        return _download_stats_file(repo_name, STATS_MODEL_FILENAME, local_files_only=local_files_only)
    except FileNotFoundError as exc:
        searched = ", ".join(
            str(path / STATS_MODEL_FILENAME) for path in _candidate_stats_repo_dirs(repo_name)
        )
        suffix = f" Searched override dirs: {searched}." if searched else ""
        raise FileNotFoundError(
            f"Could not find stats model for {fs_version!r} in the Hugging Face cache."
            f"{suffix}"
        ) from exc


def resolve_stats_config_path(
    fs_version: int | str = 7,
    *,
    config_path: str | Path | None = None,
    local_files_only: bool = False,
) -> Path | None:
    if config_path is not None:
        path = Path(config_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    repo_name = stats_repo_name(fs_version)
    for repo_dir in _candidate_stats_repo_dirs(repo_name):
        path = repo_dir / STATS_CONFIG_FILENAME
        if path.exists():
            return path

    try:
        return _download_stats_file(repo_name, STATS_CONFIG_FILENAME, local_files_only=local_files_only)
    except (FileNotFoundError, ImportError):
        return None


def _resolve_stats_config_path(checkpoint_path: Path) -> Path | None:
    path = checkpoint_path.with_name(STATS_CONFIG_FILENAME)
    return path if path.exists() else None


def _load_stats_bundle_config(checkpoint_path: Path) -> dict[str, object]:
    config_path = _resolve_stats_config_path(checkpoint_path)
    if config_path is None:
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def _encoder_config_from_stats_config(config: dict[str, object]) -> dict[str, object]:
    encoder = config.get("encoder")
    if isinstance(encoder, dict):
        model_cfg = dict(encoder.get("model_config") or {})
        return {
            "n_classes": int(encoder.get("n_classes", 118)),
            "in_channels": int(model_cfg.get("in_channels", 1)),
            "base_shape": tuple(int(v) for v in encoder.get("base_volume_shape", (256, 256, 256))),
            "patch_size": tuple(int(v) for v in encoder.get("patch_size", (16, 16, 16))),
            "channels": tuple(int(v) for v in model_cfg.get("channels", (12, 20, 32, 48, 64, 96))),
            "transformer_depth": int(model_cfg.get("transformer_depth", 2)),
            "n_heads": int(model_cfg.get("n_heads", 4)),
            "dropout": float(model_cfg.get("dropout", 0.0)),
            "positional_encoding": str(model_cfg.get("positional_encoding", "sincos")),
            "task_type": str(model_cfg.get("task_type", "classification")),
        }

    return {
        "n_classes": 118,
        "in_channels": 1,
        "base_shape": (256, 256, 256),
        "patch_size": (16, 16, 16),
        "channels": (12, 20, 32, 48, 64, 96),
        "transformer_depth": 2,
        "n_heads": 4,
        "dropout": 0.0,
        "positional_encoding": "sincos",
        "task_type": "classification",
    }


def _load_stats_safetensors(path: Path) -> dict[str, object]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError("Install safetensors to load ScaleSurfer stats checkpoints.") from exc

    with safe_open(str(path), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        model_state = {name: f.get_tensor(name) for name in f.keys()}
    serialized = metadata.get("checkpoint_metadata")
    if serialized is None:
        raise ValueError(f"Safetensors stats checkpoint is missing checkpoint_metadata: {path}")
    checkpoint = json.loads(serialized)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected checkpoint_metadata object: {path}")
    checkpoint["model_state"] = model_state
    return checkpoint


def _target_stats_frame(checkpoint: dict[str, object]) -> pd.DataFrame:
    rows = checkpoint.get("target_stats")
    if rows is None:
        raise ValueError("Stats checkpoint is missing target_stats.")
    frame = pd.DataFrame(rows)
    if "target" not in frame.columns:
        raise ValueError("Stats checkpoint target_stats is missing a target column.")
    return frame


def _feature_column_name(target: str) -> str:
    stats_name, region, measure = target_parts(target)
    pieces = [stats_name]
    if region:
        pieces.append(region)
    if measure:
        pieces.append(measure)
    return "__".join(str(piece).replace(":", "_") for piece in pieces)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _format_stats_value(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "nan"
    return f"{numeric:.10g}"


def stats_long_to_wide(long_df: pd.DataFrame, *, fill_value: float | None = None) -> pd.DataFrame:
    """Pivot a long ScaleSurfer stats table to one row per subject."""
    required = {"subject", "feature", "value"}
    missing = required.difference(long_df.columns)
    if missing:
        raise ValueError(f"Stats long table is missing required column(s): {sorted(missing)}")
    wide = long_df.pivot_table(index="subject", columns="feature", values="value", aggfunc="first")
    wide = wide.reindex(sorted(wide.columns), axis=1).reset_index()
    if fill_value is not None:
        wide = wide.fillna(fill_value)
    return wide


def _long_rows_from_flattened_stats(subject: str, rows: list[dict[str, object]]) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    for row in rows:
        target = str(row["target_name"])
        stats_name, region, measure = target_parts(target)
        out_rows.append(
            {
                "subject": subject,
                "target": target,
                "feature": _feature_column_name(target),
                "stats_name": stats_name,
                "region": region,
                "measure": measure,
                "group": row.get("group"),
                "value": float(row["value"]),
            }
        )
    return pd.DataFrame(out_rows)


def _subject_long_sidecar(subjects_dir: str | Path, subject: str) -> Path:
    return Path(subjects_dir) / str(subject) / "stats" / PREDICTED_STATS_LONG_FILENAME


def _subject_wide_sidecar(subjects_dir: str | Path, subject: str) -> Path:
    return Path(subjects_dir) / str(subject) / "stats" / PREDICTED_STATS_WIDE_FILENAME


def _subject_stats_provenance_path(subjects_dir: str | Path, subject: str) -> Path:
    return Path(subjects_dir) / str(subject) / "scripts" / PREDICTED_STATS_PROVENANCE_FILENAME


def _subject_stats_log_path(subjects_dir: str | Path, subject: str) -> Path:
    return Path(subjects_dir) / str(subject) / "scripts" / PREDICTED_STATS_LOG_FILENAME


def _stats_filename(stats_name: str) -> str:
    return f"{stats_name}.stats"


def _table_measure_order(stats_name: str, available_measures: Iterable[str]) -> list[str]:
    available = {str(v) for v in available_measures}
    if stats_name in APARC_STATS_NAMES:
        ordered = [name for name in APARC_TABLE_MEASURE_ORDER if name in available]
        extras = sorted(available.difference(APARC_TABLE_COLUMNS).difference({"global"}))
        return ordered + extras
    if stats_name in {"aseg", "wmparc", "aparc+aseg"}:
        ordered = [name for name in VOLUME_TABLE_MEASURE_ORDER if name in available]
        extras = sorted(available.difference(VOLUME_TABLE_COLUMNS).difference({"global"}))
        return ordered + extras
    return sorted(available)


def _write_predicted_stats_text(
    path: str | Path,
    frame: pd.DataFrame,
    *,
    subject: str,
    stats_name: str,
    fs_version: int | str,
    checkpoint_path: str | Path | None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Title ScaleSurfer predicted {stats_name} stats",
        f"# generating_program ScaleSurfer",
        f"# subjectname {subject}",
        f"# scalesurfer_stats_model_freesurfer_version {fs_version}",
        f"# scalesurfer_checkpoint {checkpoint_path or ''}",
        f"# creation_time {_utc_now_iso()}",
    ]

    global_rows = frame.loc[frame["region"].eq("global")].copy()
    if len(global_rows):
        for _, row in global_rows.sort_values("measure").iterrows():
            measure = safe_name(row["measure"])
            value = _format_stats_value(row["value"])
            lines.append(f"# Measure {measure}, {measure}, ScaleSurfer predicted {measure}, {value}, unknown")

    table_rows = frame.loc[~frame["region"].eq("global")].copy()
    if len(table_rows):
        table = table_rows.pivot_table(index="region", columns="measure", values="value", aggfunc="first")
        measures = _table_measure_order(stats_name, table.columns)
        if stats_name in APARC_STATS_NAMES:
            headers = ["StructName"] + measures
            lines.append("# ColHeaders " + " ".join(headers))
            for region, values in table.sort_index().iterrows():
                pieces = [str(region)]
                pieces.extend(_format_stats_value(values.get(measure, np.nan)) for measure in measures)
                lines.append(" ".join(pieces))
        elif stats_name in {"aseg", "wmparc", "aparc+aseg"}:
            headers = ["Index", "SegId", "NVoxels", "Volume_mm3", "StructName", "normMean", "normStdDev", "normMin", "normMax", "normRange"]
            lines.append("# ColHeaders " + " ".join(headers))
            for index, (region, values) in enumerate(table.sort_index().iterrows(), start=1):
                volume = float(values.get("Volume_mm3", np.nan))
                nvoxels = int(round(volume)) if np.isfinite(volume) and volume >= 0 else 0
                pieces = [
                    str(index),
                    "0",
                    str(nvoxels),
                    _format_stats_value(volume),
                    str(region),
                    "nan",
                    "nan",
                    "nan",
                    "nan",
                    "nan",
                ]
                lines.append(" ".join(pieces))
        else:
            headers = ["StructName"] + measures
            lines.append("# ColHeaders " + " ".join(headers))
            for region, values in table.sort_index().iterrows():
                pieces = [str(region)]
                pieces.extend(_format_stats_value(values.get(measure, np.nan)) for measure in measures)
                lines.append(" ".join(pieces))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_stats_outputs(
    long_df: pd.DataFrame,
    subjects_dir: str | Path,
    *,
    fs_version: int | str,
    checkpoint_path: str | Path | None = None,
    overwrite: bool = False,
    progress: bool = False,
    desc: str = "Writing stats",
) -> list[Path]:
    """Write predicted stats into each subject's FreeSurfer-style stats directory.

    The text `.stats` files are intentionally FreeSurfer-shaped: they contain
    `# Measure` lines for global values and `# ColHeaders` table sections for
    regional values. Exact target names are also preserved in CSV sidecars.
    """
    required = {"subject", "target", "feature", "stats_name", "region", "measure", "group", "value"}
    missing = required.difference(long_df.columns)
    if missing:
        raise ValueError(f"Stats long table is missing required column(s): {sorted(missing)}")

    subjects_dir = Path(subjects_dir)
    written: list[Path] = []
    grouped = long_df.groupby("subject", sort=False)
    iterator = tqdm(grouped, total=int(long_df["subject"].nunique()), desc=desc) if progress else grouped
    for subject, subject_frame in iterator:
        subject = str(subject)
        subject_dir = subjects_dir / subject
        stats_dir = subject_dir / "stats"
        scripts_dir = subject_dir / "scripts"
        stats_dir.mkdir(parents=True, exist_ok=True)
        scripts_dir.mkdir(parents=True, exist_ok=True)

        long_path = stats_dir / PREDICTED_STATS_LONG_FILENAME
        wide_path = stats_dir / PREDICTED_STATS_WIDE_FILENAME
        if not overwrite and (long_path.exists() or wide_path.exists()):
            # The caller is responsible for deciding whether to load or rewrite.
            # This guard avoids partial accidental overwrites if the function is
            # used directly.
            raise FileExistsError(f"Predicted stats sidecar already exists for {subject}; pass overwrite=True to replace it.")

        subject_frame = subject_frame.sort_values(["stats_name", "region", "measure", "target"]).reset_index(drop=True)
        subject_frame.to_csv(long_path, index=False)
        stats_long_to_wide(subject_frame, fill_value=0.0).to_csv(wide_path, index=False)
        written.extend([long_path, wide_path])

        stats_files = []
        for stats_name, stats_frame in subject_frame.groupby("stats_name", sort=True):
            stats_path = stats_dir / _stats_filename(str(stats_name))
            _write_predicted_stats_text(
                stats_path,
                stats_frame,
                subject=subject,
                stats_name=str(stats_name),
                fs_version=fs_version,
                checkpoint_path=checkpoint_path,
            )
            stats_files.append(str(stats_path.relative_to(subject_dir)))
            written.append(stats_path)

        provenance = {
            "kind": "stats",
            "generator": "ScaleSurferStatsPredictor",
            "created_at": _utc_now_iso(),
            "subject": subject,
            "fs_version": int(fs_version) if str(fs_version).strip().isdigit() else str(fs_version),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
            "n_features": int(len(subject_frame)),
            "outputs": sorted(stats_files + [f"stats/{PREDICTED_STATS_LONG_FILENAME}", f"stats/{PREDICTED_STATS_WIDE_FILENAME}"]),
        }
        provenance_path = scripts_dir / PREDICTED_STATS_PROVENANCE_FILENAME
        provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        log_path = scripts_dir / PREDICTED_STATS_LOG_FILENAME
        log_path.write_text(
            "\n".join(
                [
                    "ScaleSurfer predicted stats",
                    f"created_at={provenance['created_at']}",
                    f"subject={subject}",
                    f"fs_version={provenance['fs_version']}",
                    f"checkpoint_path={provenance['checkpoint_path']}",
                    f"n_features={provenance['n_features']}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        written.extend([provenance_path, log_path])
    return written


def _parse_subject_stats_files(
    subjects_dir: str | Path,
    subject: str,
    *,
    stats_relpaths: Iterable[str] = DEFAULT_STATS_RELPATHS,
) -> pd.DataFrame:
    subject_dir = Path(subjects_dir) / str(subject)
    rows: list[dict[str, object]] = []
    for relpath in stats_relpaths:
        path = subject_dir / relpath
        if path.exists():
            rows.extend(flatten_stats_file(str(subject), relpath, path))
    if not rows:
        raise FileNotFoundError(f"No stats sidecar or parseable .stats files found for subject {subject!r} in {subject_dir / 'stats'}")
    return _long_rows_from_flattened_stats(str(subject), rows)


def load_stats_features(
    subjects_dir: str | Path,
    subjects: Iterable[str] | None = None,
    *,
    return_format: Literal["long", "wide"] = "wide",
    fill_value: float | None = 0.0,
    prefer_sidecar: bool = True,
    stats_relpaths: Iterable[str] = DEFAULT_STATS_RELPATHS,
) -> pd.DataFrame:
    """Load predicted stats features from a ScaleSurfer subjects directory."""
    subjects_dir = Path(subjects_dir)
    if subjects is None:
        subject_list = sorted(
            path.name
            for path in subjects_dir.iterdir()
            if path.is_dir() and (path / "stats").exists()
        )
    else:
        subject_list = [str(subject) for subject in subjects]

    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for subject in subject_list:
        sidecar = _subject_long_sidecar(subjects_dir, subject)
        if prefer_sidecar and sidecar.exists():
            frame = pd.read_csv(sidecar)
        else:
            try:
                frame = _parse_subject_stats_files(subjects_dir, subject, stats_relpaths=stats_relpaths)
            except FileNotFoundError:
                missing.append(subject)
                continue
        frames.append(frame)

    if missing:
        raise FileNotFoundError(
            f"Missing predicted stats for {len(missing):,} subject(s). First few: {missing[:5]}"
        )
    long_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if return_format == "wide":
        return stats_long_to_wide(long_df, fill_value=fill_value)
    return long_df


def load_stats_feature_matrix(
    subjects_dir: str | Path,
    subjects: Iterable[str] | None = None,
    *,
    fill_value: float | None = 0.0,
    prefer_sidecar: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Return `(features, feature_cols)` with one row per subject."""
    features = load_stats_features(
        subjects_dir,
        subjects=subjects,
        return_format="wide",
        fill_value=fill_value,
        prefer_sidecar=prefer_sidecar,
    )
    feature_cols = [col for col in features.columns if col != "subject"]
    return features, feature_cols


def _normalize_t1_batch(t1: torch.Tensor) -> torch.Tensor:
    if t1.ndim == 3:
        t1 = t1[None, None]
    elif t1.ndim == 4:
        if t1.shape[0] == 1:
            t1 = t1[None]
        else:
            t1 = t1[:, None]
    elif t1.ndim != 5:
        raise ValueError(f"Expected T1 tensor [D,H,W], [B,D,H,W], or [B,1,D,H,W], got {tuple(t1.shape)}")
    if t1.shape[1] != 1:
        raise ValueError(f"Expected one T1 channel, got {tuple(t1.shape)}")
    t1 = t1.to(torch.float32)
    dims = tuple(range(2, t1.ndim))
    mean = t1.mean(dim=dims, keepdim=True)
    std = t1.std(dim=dims, keepdim=True).clamp_min(1e-6)
    return (t1 - mean) / std


def _normalize_seg_batch(seg: torch.Tensor) -> torch.Tensor:
    if seg.ndim == 5 and seg.shape[1] == 1:
        seg = seg[:, 0]
    elif seg.ndim == 3:
        seg = seg[None]
    elif seg.ndim != 4:
        raise ValueError(f"Expected segmentation [D,H,W], [B,D,H,W], or [B,1,D,H,W], got {tuple(seg.shape)}")
    return seg.to(torch.int64)


class ScaleSurferStatsPredictor:
    def __init__(
        self,
        model: StatsPredictionModel,
        *,
        target_stats: pd.DataFrame,
        columns_by_group: dict[str, list[str]],
        checkpoint_path: Path,
        config: dict[str, object] | None = None,
        device: str | torch.device = "cpu",
    ):
        self.model = model
        self.target_stats = target_stats.copy()
        self.columns_by_group = {str(k): list(v) for k, v in columns_by_group.items()}
        self.checkpoint_path = Path(checkpoint_path)
        self.config = dict(config or {})
        self.device = torch.device(device)
        class_values, _ = build_label_lut(default_aparc_aseg_label_values())
        self.class_values = class_values.cpu()

    @classmethod
    def from_pretrained(
        cls,
        fs_version: int | str = 7,
        *,
        checkpoint_path: str | Path | None = None,
        device: str | torch.device = "cpu",
        local_files_only: bool = False,
    ) -> "ScaleSurferStatsPredictor":
        resolved = resolve_stats_checkpoint_path(
            fs_version,
            checkpoint_path=checkpoint_path,
            local_files_only=local_files_only,
        )
        checkpoint = _load_stats_safetensors(resolved)
        bundle_config = _load_stats_bundle_config(resolved)
        if not bundle_config and checkpoint_path is None:
            resolved_config = resolve_stats_config_path(fs_version, local_files_only=local_files_only)
            if resolved_config is not None and resolved_config.exists():
                bundle_config = json.loads(resolved_config.read_text(encoding="utf-8"))
        encoder_cfg = _encoder_config_from_stats_config(bundle_config)
        encoder_model = TransUNet3D(**encoder_cfg)
        encoder = TransUNetEncoderAdapter(encoder_model)

        target_stats = _target_stats_frame(checkpoint)
        cols_by_group = checkpoint.get("columns_by_group")
        if not isinstance(cols_by_group, dict):
            raise ValueError("Stats checkpoint is missing columns_by_group.")

        config = checkpoint.get("config")
        config = dict(config) if isinstance(config, dict) else {}
        model = StatsPredictionModel(
            encoder,
            label_ids=checkpoint.get("label_ids", default_aparc_aseg_label_values()),
            out_dims={group: len(cols) for group, cols in cols_by_group.items()},
            pool_features=tuple(checkpoint.get("pool_features") or config.get("pool_features") or ("enc2", "enc3", "enc4", "z")),
            hidden=int(config.get("hidden", 256)),
            dropout=float(config.get("dropout", 0.1)),
            device="cpu",
        )
        state = strip_compile_prefix(checkpoint["model_state"])
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[ScaleSurferStatsPredictor] missing keys: {len(missing)}")
        if unexpected:
            print(f"[ScaleSurferStatsPredictor] unexpected keys: {len(unexpected)}")
        model.freeze_encoder(True)
        model.to(device=device)
        model.eval()

        return cls(
            model,
            target_stats=target_stats,
            columns_by_group={str(k): list(v) for k, v in cols_by_group.items()},
            checkpoint_path=resolved,
            config=bundle_config or config,
            device=device,
        )

    def _seg_to_fs_ids(self, seg: torch.Tensor, seg_is_dense: bool | Literal["auto"] = "auto") -> torch.Tensor:
        seg = _normalize_seg_batch(seg)
        convert = bool(seg_is_dense)
        if seg_is_dense == "auto":
            max_label = int(seg.max().item()) if seg.numel() else 0
            convert = max_label <= int(self.class_values.numel()) - 1
        if not convert:
            return seg
        fs = [
            torch.as_tensor(dense_labels_to_fs_ids(item.cpu(), class_values=self.class_values), dtype=torch.int64)
            for item in seg
        ]
        return torch.stack(fs, dim=0)

    def predict_tensors(
        self,
        t1: torch.Tensor,
        seg: torch.Tensor,
        *,
        subjects: Iterable[str] | None = None,
        seg_is_dense: bool | Literal["auto"] = "auto",
        return_format: Literal["long", "wide"] = "long",
    ) -> pd.DataFrame:
        x = _normalize_t1_batch(torch.as_tensor(t1)).to(self.device)
        seg_fs = self._seg_to_fs_ids(torch.as_tensor(seg), seg_is_dense=seg_is_dense).to(self.device)
        if x.shape[0] != seg_fs.shape[0]:
            raise ValueError(f"T1 batch size {x.shape[0]} does not match segmentation batch size {seg_fs.shape[0]}")
        subject_list = list(subjects) if subjects is not None else [f"subject_{i}" for i in range(int(x.shape[0]))]
        if len(subject_list) != int(x.shape[0]):
            raise ValueError(f"Expected {x.shape[0]} subject IDs, got {len(subject_list)}")

        stats_idx = self.target_stats.set_index("target")
        rows: list[dict[str, object]] = []
        with torch.no_grad():
            predictions = self.model(x, seg_fs)
        for group, cols in self.columns_by_group.items():
            pred_norm = predictions[group].detach().float().cpu().numpy()
            mean = stats_idx.loc[cols, "mean"].to_numpy(dtype=np.float32)
            std = stats_idx.loc[cols, "std"].to_numpy(dtype=np.float32)
            values = pred_norm * std[None, :] + mean[None, :]
            for i, subject in enumerate(subject_list):
                for j, target in enumerate(cols):
                    stats_name, region, measure = target_parts(str(target))
                    rows.append(
                        {
                            "subject": subject,
                            "target": target,
                            "feature": _feature_column_name(str(target)),
                            "stats_name": stats_name,
                            "region": region,
                            "measure": measure,
                            "group": group,
                            "value": float(values[i, j]),
                        }
                    )
        out = pd.DataFrame(rows)
        if return_format == "wide":
            return self.to_wide(out)
        return out

    @staticmethod
    def to_wide(long_df: pd.DataFrame, *, fill_value: float | None = None) -> pd.DataFrame:
        return stats_long_to_wide(long_df, fill_value=fill_value)

    def predict_subjects(
        self,
        subjects_dir: str | Path,
        subjects: Iterable[str],
        *,
        batch_size: int = 1,
        seg_is_dense: bool | Literal["auto"] = "auto",
        return_format: Literal["long", "wide"] = "long",
        progress: bool = False,
        desc: str = "Predicting stats",
    ) -> pd.DataFrame:
        subjects_dir = Path(subjects_dir)
        subject_list = [str(subject) for subject in subjects]
        frames: list[pd.DataFrame] = []
        starts = range(0, len(subject_list), int(batch_size))
        iterator = tqdm(
            starts,
            total=int(np.ceil(len(subject_list) / max(1, int(batch_size)))),
            desc=desc,
        ) if progress else starts
        for start in iterator:
            batch_subjects = subject_list[start : start + int(batch_size)]
            t1_batch = []
            seg_batch = []
            for subject in batch_subjects:
                mri_dir = subjects_dir / subject / "mri"
                t1_path = mri_dir / "orig.pt"
                seg_path = mri_dir / "aparc+aseg.pt"
                if not t1_path.exists() or not seg_path.exists():
                    missing = [str(path) for path in (t1_path, seg_path) if not path.exists()]
                    raise FileNotFoundError("Missing required ScaleSurfer tensor(s): " + ", ".join(missing))
                t1_batch.append(torch.load(t1_path, map_location="cpu", weights_only=True))
                seg_batch.append(torch.load(seg_path, map_location="cpu", weights_only=True))
            frames.append(
                self.predict_tensors(
                    torch.stack([torch.as_tensor(x).squeeze() for x in t1_batch], dim=0),
                    torch.stack([torch.as_tensor(y).squeeze() for y in seg_batch], dim=0),
                    subjects=batch_subjects,
                    seg_is_dense=seg_is_dense,
                    return_format="long",
                )
            )
        long_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if return_format == "wide":
            return self.to_wide(long_df)
        return long_df

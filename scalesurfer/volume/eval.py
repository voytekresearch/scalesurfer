from __future__ import annotations

import json
import math
import shutil
from functools import partial
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from nibabel.freesurfer.io import read_geometry, read_morph_data
from nibabel.processing import resample_from_to
from scipy.spatial import cKDTree

from ..convert import load_mgz
from .norm import build_deterministic_norm
from .targets import dequantize_norm_volume, load_tensor
from .fs import build_config, run


VOLUME_TARGETS = (
    "aparc+aseg",
    "aseg_presurf",
    "brainmask",
)
DETERMINISTIC_TARGETS = ("norm",)
DIRECT_SURFACE_RELPATHS = (
    "surf/lh.white",
    "surf/rh.white",
    "surf/lh.pial",
    "surf/rh.pial",
)
FINAL_VOLUME_OUTPUT_RELPATHS = (
    "mri/ribbon.mgz",
    "mri/aseg.mgz",
    "mri/aparc+aseg.mgz",
    "mri/wmparc.mgz",
)
FINAL_SURFACE_RELPATHS = (
    "surf/lh.white",
    "surf/rh.white",
    "surf/lh.pial",
    "surf/rh.pial",
)
FINAL_MORPH_RELPATHS = (
    "surf/lh.sulc",
    "surf/rh.sulc",
    "surf/lh.curv",
    "surf/rh.curv",
    "surf/lh.area",
    "surf/rh.area",
    "surf/lh.curv.pial",
    "surf/rh.curv.pial",
    "surf/lh.area.pial",
    "surf/rh.area.pial",
    "surf/lh.thickness",
    "surf/rh.thickness",
    "surf/lh.area.mid",
    "surf/rh.area.mid",
    "surf/lh.volume",
    "surf/rh.volume",
)
FINAL_STATS_RELPATHS = (
    "stats/aseg.stats",
    "stats/wmparc.stats",
    "stats/aparc+aseg.stats",
    "stats/lh.curv.stats",
    "stats/rh.curv.stats",
    "stats/lh.aparc.stats",
    "stats/rh.aparc.stats",
    "stats/lh.aparc.pial.stats",
    "stats/rh.aparc.pial.stats",
)


def exact_flow_targets() -> dict[str, tuple[str, ...]]:
    return {
        "tensor_models_required_now": VOLUME_TARGETS,
        "deterministic_tensor_targets": DETERMINISTIC_TARGETS,
        "direct_surface_files_required_by_fs": DIRECT_SURFACE_RELPATHS,
    }


def label_values_for_target(target_kind: str, *, norm_bins: int = 256) -> list[int]:
    if target_kind == "surface_bands":
        return [0, 1, 2, 3, 4]
    if target_kind in {"white_surfaces", "pial_surfaces"}:
        return [0, 1, 2]
    if target_kind in {"brainmask", "lh.white", "rh.white", "lh.pial", "rh.pial"}:
        return [0, 1]
    if target_kind == "norm_quantized":
        return list(range(int(norm_bins)))
    return []


def _load_metadata(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _parallel_map(
    worker,
    items: Iterable[object],
    *,
    n_jobs: int = 1,
    prefer: str = "processes",
) -> list[object]:
    items = list(items)
    if not items:
        return []
    if int(n_jobs) == 1 or len(items) == 1:
        return [worker(item) for item in items]
    return Parallel(n_jobs=int(n_jobs), prefer=prefer)(
        delayed(worker)(item) for item in items
    )


def _link_or_copy(src: str | Path, dst: str | Path, *, link_mode: str = "symlink", force: bool = True) -> None:
    src = Path(src).expanduser().resolve()
    dst = Path(dst).expanduser()
    if dst.exists() or dst.is_symlink():
        if not force:
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if link_mode == "symlink":
        dst.symlink_to(src)
    elif link_mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported link_mode: {link_mode}")


def _resample_prediction_to_native(
    prediction: np.ndarray,
    *,
    metadata: dict,
    native_ref_path: str | Path,
    is_labels: bool,
) -> nib.spatialimages.SpatialImage:
    tensor_img = nib.Nifti1Image(
        np.asarray(np.rint(prediction), dtype=np.int32) if is_labels else np.asarray(prediction, dtype=np.float32),
        np.asarray(metadata["tensor_affine"], dtype=np.float64),
    )
    native_ref = load_mgz(native_ref_path)
    order = 0 if is_labels else 1
    out = resample_from_to(tensor_img, native_ref, order=order)
    if is_labels:
        data = np.asarray(np.rint(out.get_fdata()), dtype=native_ref.header.get_data_dtype())
    else:
        data = np.asarray(out.get_fdata(dtype=np.float32), dtype=native_ref.header.get_data_dtype())
    return nib.MGHImage(data, native_ref.affine, header=native_ref.header)


def export_target_prediction_to_native_mgz(
    prediction: np.ndarray | torch.Tensor,
    *,
    metadata_path: str | Path,
    target_kind: str,
    out_path: str | Path,
) -> Path:
    metadata = _load_metadata(metadata_path)
    subject_dir = Path(metadata["subject_dir"])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pred = prediction.detach().cpu().numpy() if torch.is_tensor(prediction) else np.asarray(prediction)

    if target_kind == "aparc+aseg":
        native_ref = subject_dir / "mri" / "aparc+aseg.mgz"
        is_labels = True
    elif target_kind == "aseg_presurf":
        native_ref = subject_dir / "mri" / "aseg.presurf.mgz"
        is_labels = True
    elif target_kind == "brainmask":
        native_ref = subject_dir / "mri" / "brainmask.mgz"
        pred = (np.asarray(pred) > 0).astype(np.int16)
        is_labels = True
    elif target_kind == "norm":
        native_ref = subject_dir / "mri" / "norm.mgz"
        is_labels = False
    elif target_kind == "norm_quantized":
        native_ref = subject_dir / "mri" / "norm.mgz"
        q = metadata["norm_quantization"]
        pred = dequantize_norm_volume(pred, lo=float(q["lo"]), hi=float(q["hi"]), bins=int(q["bins"]))
        is_labels = False
    else:
        raise KeyError(f"Unsupported export target_kind: {target_kind}")

    out_img = _resample_prediction_to_native(
        pred,
        metadata=metadata,
        native_ref_path=native_ref,
        is_labels=is_labels,
    )
    nib.save(out_img, str(out_path))
    return out_path


def _export_prediction_job(job: tuple[np.ndarray | torch.Tensor, str | Path, str, str | Path]) -> Path:
    prediction, metadata_path, target_kind, out_path = job
    return export_target_prediction_to_native_mgz(
        prediction,
        metadata_path=metadata_path,
        target_kind=target_kind,
        out_path=out_path,
    )


def prepare_subject_scaffold(
    reference_subject_dir: str | Path,
    out_subject_dir: str | Path,
    *,
    link_mode: str = "symlink",
    force: bool = False,
) -> Path:
    reference_subject_dir = Path(reference_subject_dir).expanduser().resolve()
    out_subject_dir = Path(out_subject_dir).expanduser().resolve()
    for relpath in (
        "mri/orig.mgz",
        "mri/rawavg.mgz",
    ):
        src = reference_subject_dir / relpath
        if src.exists():
            _link_or_copy(src, out_subject_dir / relpath, link_mode=link_mode, force=force)
    for relpath in ("mri", "surf", "label", "stats", "scripts", "mri/transforms"):
        (out_subject_dir / relpath).mkdir(parents=True, exist_ok=True)
    return out_subject_dir


def create_reference_bundle_from_subject(
    subject_dir: str | Path,
    out_root: str | Path,
    *,
    link_mode: str = "symlink",
    force: bool = True,
) -> Path:
    subject_dir = Path(subject_dir).expanduser().resolve()
    bundle_root = Path(out_root).expanduser().resolve()
    for relpath in (
        "mri/aparc+aseg.mgz",
        "mri/aseg.presurf.mgz",
        "mri/brainmask.mgz",
        "mri/norm.mgz",
        "mri/transforms/talairach.xfm",
        "surf/lh.white",
        "surf/rh.white",
        "surf/lh.pial",
        "surf/rh.pial",
    ):
        src = subject_dir / relpath
        if src.exists():
            _link_or_copy(src, bundle_root / relpath, link_mode=link_mode, force=force)
    return bundle_root


def create_prediction_bundle_from_tensors(
    *,
    subject_dir: str | Path,
    metadata_path: str | Path,
    out_root: str | Path,
    predictions: dict[str, np.ndarray | torch.Tensor],
    rawavg_tensor: np.ndarray | torch.Tensor | None = None,
    surface_source: str = "ground_truth",
    surface_bundle_dir: str | Path | None = None,
    link_mode: str = "symlink",
    force: bool = True,
    include_talairach: bool = True,
    n_jobs: int = -1,
) -> Path:
    subject_dir = Path(subject_dir).expanduser().resolve()
    bundle_root = Path(out_root).expanduser().resolve()

    required_volume_targets = {"aparc+aseg", "aseg_presurf", "brainmask"}
    missing = sorted(required_volume_targets - set(predictions))
    if missing:
        raise KeyError(f"Missing required predicted volume targets: {missing}")

    predictions_local = dict(predictions)
    norm_meta: dict[str, object] | None = None
    if "norm" not in predictions_local and "norm_quantized" not in predictions_local:
        norm_tensor, norm_stats = build_deterministic_norm_prediction(
            predictions=predictions_local,
            metadata_path=metadata_path,
            rawavg_tensor=rawavg_tensor,
        )
        predictions_local["norm"] = norm_tensor
        norm_meta = {"source": "deterministic", "stats": norm_stats}

    norm_kind = "norm" if "norm" in predictions_local else "norm_quantized"
    export_jobs = [
        (predictions_local["aparc+aseg"], metadata_path, "aparc+aseg", bundle_root / "mri" / "aparc+aseg.mgz"),
        (predictions_local["aseg_presurf"], metadata_path, "aseg_presurf", bundle_root / "mri" / "aseg.presurf.mgz"),
        (predictions_local["brainmask"], metadata_path, "brainmask", bundle_root / "mri" / "brainmask.mgz"),
        (predictions_local[norm_kind], metadata_path, norm_kind, bundle_root / "mri" / "norm.mgz"),
    ]
    _parallel_map(
        _export_prediction_job,
        export_jobs,
        n_jobs=n_jobs,
        prefer="threads",
    )

    if include_talairach:
        xfm = subject_dir / "mri" / "transforms" / "talairach.xfm"
        if xfm.exists():
            _link_or_copy(xfm, bundle_root / "mri" / "transforms" / "talairach.xfm", link_mode=link_mode, force=force)

    if surface_source == "ground_truth":
        for relpath in DIRECT_SURFACE_RELPATHS:
            _link_or_copy(subject_dir / relpath, bundle_root / relpath, link_mode=link_mode, force=force)
    elif surface_source == "prediction_bundle":
        if surface_bundle_dir is None:
            raise ValueError("surface_bundle_dir is required when surface_source='prediction_bundle'")
        surface_bundle_dir = Path(surface_bundle_dir).expanduser().resolve()
        for relpath in DIRECT_SURFACE_RELPATHS:
            _link_or_copy(surface_bundle_dir / relpath, bundle_root / relpath, link_mode=link_mode, force=force)
    else:
        raise ValueError("surface_source must be 'ground_truth' or 'prediction_bundle'")

    if norm_meta is not None:
        norm_meta_path = bundle_root / "mri" / "norm.meta.json"
        norm_meta_path.parent.mkdir(parents=True, exist_ok=True)
        norm_meta_path.write_text(json.dumps(norm_meta, indent=2))

    return bundle_root


def build_deterministic_norm_prediction(
    *,
    predictions: dict[str, np.ndarray | torch.Tensor],
    metadata_path: str | Path,
    rawavg_tensor: np.ndarray | torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    metadata = _load_metadata(metadata_path)
    if rawavg_tensor is None:
        rawavg_path = metadata.get("pt_files", {}).get("rawavg")
        if rawavg_path is None:
            raise KeyError(f"metadata does not contain pt_files.rawavg: {metadata_path}")
        rawavg_tensor = load_tensor(rawavg_path)

    brainmask = predictions.get("brainmask")
    if brainmask is None:
        raise KeyError("Deterministic norm generation requires a predicted brainmask")

    result = build_deterministic_norm(
        rawavg_tensor,
        brainmask=brainmask,
        aseg_presurf=predictions.get("aseg_presurf"),
        aparc_aseg=predictions.get("aparc+aseg"),
    )
    return result.tensor, result.stats


def run_from_bundle(
    *,
    reference_subject_dir: str | Path,
    bundle_dir: str | Path,
    out_subject_dir: str | Path,
    freesurfer_home: str | Path,
    fs_license: str | Path | None = None,
    threads: int = 1,
    link_mode: str = "symlink",
    force: bool = False,
    dry_run: bool = False,
    brainvol_stats: bool = False,
    aparc_aseg_stats: bool = True,
) -> Path:
    out_subject_dir = prepare_subject_scaffold(
        reference_subject_dir,
        out_subject_dir,
        link_mode=link_mode,
        force=force,
    )
    config = build_config(
        subject_dir=out_subject_dir,
        predictions_dir=bundle_dir,
        freesurfer_home=freesurfer_home,
        fs_license=fs_license,
        threads=threads,
        link_mode=link_mode,
        dry_run=dry_run,
        force=force,
        run_autorecon1=False,
        stages=("all",),
        brainvol_stats=brainvol_stats,
        aparc_aseg_stats=aparc_aseg_stats,
    )
    run(config)
    return out_subject_dir


def _dice_for_label(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> float:
    true_mask = np.asarray(y_true == int(label))
    pred_mask = np.asarray(y_pred == int(label))
    denom = int(true_mask.sum()) + int(pred_mask.sum())
    if denom == 0:
        return 1.0
    inter = int((true_mask & pred_mask).sum())
    return float(2.0 * inter / max(1, denom))


def summarize_label_volume_overlap(
    reference_path: str | Path,
    predicted_path: str | Path,
) -> dict[str, float | int | str]:
    ref = np.asarray(load_mgz(reference_path).dataobj)
    pred = np.asarray(load_mgz(predicted_path).dataobj)
    labels = sorted(int(v) for v in np.union1d(np.unique(ref), np.unique(pred)))
    if labels and labels[0] == 0:
        labels_nonzero = labels[1:]
    else:
        labels_nonzero = labels

    if labels_nonzero:
        dices = [_dice_for_label(ref, pred, label) for label in labels_nonzero]
        macro_dice = float(np.mean(dices))
    else:
        macro_dice = 1.0
    return {
        "reference_path": str(reference_path),
        "predicted_path": str(predicted_path),
        "voxel_accuracy": float(np.mean(ref == pred)),
        "macro_dice_nonzero": macro_dice,
        "n_labels_union": int(len(labels_nonzero)),
    }


def _compare_final_volume_relpath(
    relpath: str,
    *,
    reference_subject_dir: str | Path,
    predicted_subject_dir: str | Path,
) -> dict[str, float | int | str] | None:
    reference_subject_dir = Path(reference_subject_dir)
    predicted_subject_dir = Path(predicted_subject_dir)
    ref_path = reference_subject_dir / relpath
    pred_path = predicted_subject_dir / relpath
    if not ref_path.exists() or not pred_path.exists():
        return None
    row = summarize_label_volume_overlap(ref_path, pred_path)
    row["relpath"] = relpath
    return row


def compare_final_volume_outputs(
    reference_subject_dir: str | Path,
    predicted_subject_dir: str | Path,
    *,
    relpaths: Iterable[str] = FINAL_VOLUME_OUTPUT_RELPATHS,
    n_jobs: int = -1,
) -> pd.DataFrame:
    reference_subject_dir = Path(reference_subject_dir)
    predicted_subject_dir = Path(predicted_subject_dir)
    rows = _parallel_map(
        partial(
            _compare_final_volume_relpath,
            reference_subject_dir=reference_subject_dir,
            predicted_subject_dir=predicted_subject_dir,
        ),
        relpaths,
        n_jobs=n_jobs,
        prefer="processes",
    )
    rows = [row for row in rows if row is not None]
    return pd.DataFrame(rows)


def _surface_distance_rows(reference_surface: str | Path, predicted_surface: str | Path) -> dict[str, float | int | str]:
    ref_vertices, ref_faces = read_geometry(str(reference_surface))
    pred_vertices, pred_faces = read_geometry(str(predicted_surface))

    paired_metrics: dict[str, float | int | str] = {
        "reference_surface": str(reference_surface),
        "predicted_surface": str(predicted_surface),
        "reference_vertices": int(ref_vertices.shape[0]),
        "predicted_vertices": int(pred_vertices.shape[0]),
        "reference_faces": int(ref_faces.shape[0]),
        "predicted_faces": int(pred_faces.shape[0]),
    }
    if ref_vertices.shape == pred_vertices.shape:
        paired = np.linalg.norm(ref_vertices - pred_vertices, axis=1)
        paired_metrics.update(
            {
                "paired_mean_mm": float(np.mean(paired)),
                "paired_rms_mm": float(math.sqrt(np.mean(np.square(paired)))),
                "paired_p95_mm": float(np.percentile(paired, 95)),
                "paired_max_mm": float(np.max(paired)),
            }
        )
    else:
        paired_metrics.update(
            {
                "paired_mean_mm": math.nan,
                "paired_rms_mm": math.nan,
                "paired_p95_mm": math.nan,
                "paired_max_mm": math.nan,
            }
        )

    ref_tree = cKDTree(ref_vertices)
    pred_tree = cKDTree(pred_vertices)
    ref_to_pred = pred_tree.query(ref_vertices, k=1)[0]
    pred_to_ref = ref_tree.query(pred_vertices, k=1)[0]
    paired_metrics.update(
        {
            "ref_to_pred_mean_mm": float(np.mean(ref_to_pred)),
            "pred_to_ref_mean_mm": float(np.mean(pred_to_ref)),
            "symmetric_mean_mm": float(0.5 * (np.mean(ref_to_pred) + np.mean(pred_to_ref))),
            "symmetric_hausdorff_mm": float(max(np.max(ref_to_pred), np.max(pred_to_ref))),
        }
    )
    return paired_metrics


def _compare_surface_relpath(
    relpath: str,
    *,
    reference_subject_dir: str | Path,
    predicted_subject_dir: str | Path,
) -> dict[str, float | int | str] | None:
    reference_subject_dir = Path(reference_subject_dir)
    predicted_subject_dir = Path(predicted_subject_dir)
    ref_path = reference_subject_dir / relpath
    pred_path = predicted_subject_dir / relpath
    if not ref_path.exists() or not pred_path.exists():
        return None
    row = _surface_distance_rows(ref_path, pred_path)
    row["relpath"] = relpath
    return row


def compare_final_surfaces(
    reference_subject_dir: str | Path,
    predicted_subject_dir: str | Path,
    *,
    relpaths: Iterable[str] = FINAL_SURFACE_RELPATHS,
    n_jobs: int = -1,
) -> pd.DataFrame:
    reference_subject_dir = Path(reference_subject_dir)
    predicted_subject_dir = Path(predicted_subject_dir)
    rows = _parallel_map(
        partial(
            _compare_surface_relpath,
            reference_subject_dir=reference_subject_dir,
            predicted_subject_dir=predicted_subject_dir,
        ),
        relpaths,
        n_jobs=n_jobs,
        prefer="processes",
    )
    rows = [row for row in rows if row is not None]
    return pd.DataFrame(rows)


def _corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return math.nan
    if np.allclose(a, a[:1]) or np.allclose(b, b[:1]):
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def compare_surface_morphometry(
    reference_subject_dir: str | Path,
    predicted_subject_dir: str | Path,
    *,
    relpaths: Iterable[str] = FINAL_MORPH_RELPATHS,
    n_jobs: int = -1,
) -> pd.DataFrame:
    rows = _parallel_map(
        partial(
            _compare_morphometry_relpath,
            reference_subject_dir=reference_subject_dir,
            predicted_subject_dir=predicted_subject_dir,
        ),
        relpaths,
        n_jobs=n_jobs,
        prefer="processes",
    )
    rows = [row for row in rows if row is not None]
    return pd.DataFrame(rows)


def parse_freesurfer_stats(path: str | Path) -> dict[str, object]:
    path = Path(path)
    measures: dict[str, dict[str, object]] = {}
    headers: list[str] = []
    rows: list[list[str]] = []

    for line in path.read_text().splitlines():
        if line.startswith("# Measure "):
            payload = line[len("# Measure ") :]
            parts = [part.strip() for part in payload.split(",")]
            if len(parts) >= 4:
                key = parts[1]
                value = parts[3]
                try:
                    value_obj: object = float(value)
                except ValueError:
                    value_obj = value
                measures[key] = {
                    "measure": parts[0],
                    "field": parts[1],
                    "description": parts[2],
                    "value": value_obj,
                    "units": parts[4] if len(parts) > 4 else "",
                }
        elif line.startswith("# ColHeaders"):
            headers = line[len("# ColHeaders") :].strip().split()
        elif line and not line.startswith("#"):
            rows.append(line.split())

    table = pd.DataFrame(rows, columns=headers if headers else None)
    for column in table.columns:
        try:
            table[column] = pd.to_numeric(table[column])
        except (ValueError, TypeError):
            continue
    return {"measures": measures, "table": table}


def _compare_morphometry_relpath(
    relpath: str,
    *,
    reference_subject_dir: str | Path,
    predicted_subject_dir: str | Path,
) -> dict[str, float | int | str] | None:
    reference_subject_dir = Path(reference_subject_dir)
    predicted_subject_dir = Path(predicted_subject_dir)
    ref_path = reference_subject_dir / relpath
    pred_path = predicted_subject_dir / relpath
    if not ref_path.exists() or not pred_path.exists():
        return None
    ref = np.asarray(read_morph_data(str(ref_path)), dtype=np.float64)
    pred = np.asarray(read_morph_data(str(pred_path)), dtype=np.float64)
    if ref.shape != pred.shape:
        return {
            "relpath": relpath,
            "n_ref": int(ref.shape[0]),
            "n_pred": int(pred.shape[0]),
            "mae": math.nan,
            "rmse": math.nan,
            "corr": math.nan,
        }
    delta = pred - ref
    return {
        "relpath": relpath,
        "n_ref": int(ref.shape[0]),
        "n_pred": int(pred.shape[0]),
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(math.sqrt(np.mean(np.square(delta)))),
        "corr": _corrcoef_safe(ref, pred),
    }


def compare_stats_files(reference_path: str | Path, predicted_path: str | Path) -> dict[str, pd.DataFrame]:
    ref = parse_freesurfer_stats(reference_path)
    pred = parse_freesurfer_stats(predicted_path)

    measure_rows = []
    shared_measure_keys = sorted(set(ref["measures"]) & set(pred["measures"]))
    for key in shared_measure_keys:
        ref_val = ref["measures"][key]["value"]
        pred_val = pred["measures"][key]["value"]
        if isinstance(ref_val, (int, float)) and isinstance(pred_val, (int, float)):
            abs_diff = float(abs(float(pred_val) - float(ref_val)))
            denom = max(abs(float(ref_val)), 1e-6)
            ape = float(100.0 * abs_diff / denom)
        else:
            abs_diff = math.nan
            ape = math.nan
        measure_rows.append(
            {
                "field": key,
                "reference_value": ref_val,
                "predicted_value": pred_val,
                "abs_diff": abs_diff,
                "abs_percent_error": ape,
            }
        )
    measures_df = pd.DataFrame(measure_rows)

    ref_table = ref["table"]
    pred_table = pred["table"]
    if "StructName" in ref_table.columns and "StructName" in pred_table.columns:
        key_cols = ["StructName"]
    elif "SegId" in ref_table.columns and "SegId" in pred_table.columns:
        key_cols = ["SegId"]
        if "StructName" in ref_table.columns and "StructName" in pred_table.columns:
            key_cols.append("StructName")
    else:
        key_cols = []

    if key_cols:
        merged = ref_table.merge(pred_table, on=key_cols, suffixes=("_ref", "_pred"))
        numeric_roots = []
        for column in ref_table.columns:
            if column in key_cols:
                continue
            if f"{column}_pred" in merged.columns and pd.api.types.is_numeric_dtype(merged[f"{column}_ref"]):
                numeric_roots.append(column)
        table_rows = []
        for column in numeric_roots:
            delta = merged[f"{column}_pred"] - merged[f"{column}_ref"]
            table_rows.append(
                {
                    "column": column,
                    "mae": float(np.mean(np.abs(delta))),
                    "rmse": float(math.sqrt(np.mean(np.square(delta)))),
                    "corr": _corrcoef_safe(
                        np.asarray(merged[f"{column}_ref"], dtype=np.float64),
                        np.asarray(merged[f"{column}_pred"], dtype=np.float64),
                    ),
                    "n_rows": int(len(merged)),
                }
            )
        table_df = pd.DataFrame(table_rows)
    else:
        table_df = pd.DataFrame()

    return {"measures": measures_df, "table": table_df}


def compare_final_stats_outputs(
    reference_subject_dir: str | Path,
    predicted_subject_dir: str | Path,
    *,
    relpaths: Iterable[str] = FINAL_STATS_RELPATHS,
    n_jobs: int = -1,
) -> dict[str, dict[str, pd.DataFrame]]:
    output_rows = _parallel_map(
        partial(
            _compare_stats_relpath,
            reference_subject_dir=reference_subject_dir,
            predicted_subject_dir=predicted_subject_dir,
        ),
        relpaths,
        n_jobs=n_jobs,
        prefer="processes",
    )
    return {relpath: stats for relpath, stats in output_rows if relpath is not None}


def _compare_stats_relpath(
    relpath: str,
    *,
    reference_subject_dir: str | Path,
    predicted_subject_dir: str | Path,
) -> tuple[str | None, dict[str, pd.DataFrame] | None]:
    reference_subject_dir = Path(reference_subject_dir)
    predicted_subject_dir = Path(predicted_subject_dir)
    ref_path = reference_subject_dir / relpath
    pred_path = predicted_subject_dir / relpath
    if not ref_path.exists() or not pred_path.exists():
        return (None, None)
    return (relpath, compare_stats_files(ref_path, pred_path))


def evaluate_run(
    reference_subject_dir: str | Path,
    predicted_subject_dir: str | Path,
    *,
    n_jobs: int = -1,
) -> dict[str, object]:
    return {
        "volumes": compare_final_volume_outputs(reference_subject_dir, predicted_subject_dir, n_jobs=n_jobs),
        "surfaces": compare_final_surfaces(reference_subject_dir, predicted_subject_dir, n_jobs=n_jobs),
        "morphometry": compare_surface_morphometry(reference_subject_dir, predicted_subject_dir, n_jobs=n_jobs),
        "stats": compare_final_stats_outputs(reference_subject_dir, predicted_subject_dir, n_jobs=n_jobs),
    }


__all__ = [
    "DETERMINISTIC_TARGETS",
    "DIRECT_SURFACE_RELPATHS",
    "VOLUME_TARGETS",
    "build_deterministic_norm_prediction",
    "FINAL_MORPH_RELPATHS",
    "FINAL_STATS_RELPATHS",
    "FINAL_SURFACE_RELPATHS",
    "FINAL_VOLUME_OUTPUT_RELPATHS",
    "compare_final_stats_outputs",
    "compare_final_surfaces",
    "compare_final_volume_outputs",
    "compare_stats_files",
    "compare_surface_morphometry",
    "create_prediction_bundle_from_tensors",
    "create_reference_bundle_from_subject",
    "evaluate_run",
    "exact_flow_targets",
    "export_target_prediction_to_native_mgz",
    "label_values_for_target",
    "parse_freesurfer_stats",
    "prepare_subject_scaffold",
    "run_from_bundle",
    "summarize_label_volume_overlap",
]

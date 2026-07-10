from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import torch
from joblib import Parallel, delayed
from nibabel.freesurfer.io import read_geometry
from nibabel.processing import resample_from_to

from ...convert import TARGET_SHAPE, TARGET_VOXEL_SIZE_MM, load_mgz, prepare_arrays_if_needed, prepare_images_if_needed


SURFACE_CLASS_MAP = {
    "lh.white": 1,
    "rh.white": 2,
    "lh.pial": 3,
    "rh.pial": 4,
}
SURFACE_PT_FILE_MAP = {
    "lh.white": "lh.white.pt",
    "rh.white": "rh.white.pt",
    "lh.pial": "lh.pial.pt",
    "rh.pial": "rh.pial.pt",
}
TARGET_PT_FILE_MAP = {
    "orig": "orig.pt",
    "aparc+aseg": "aparc+aseg.pt",
    "aseg_presurf": "aseg.presurf.pt",
    "brainmask": "brainmask.pt",
    "norm": "norm.pt",
    "norm_quantized": "norm.quantized.pt",
    "white_surfaces": "white_surfaces.pt",
    "pial_surfaces": "pial_surfaces.pt",
    "surface_bands": "surface_bands.pt",
    "lh.white": "lh.white.pt",
    "rh.white": "rh.white.pt",
    "lh.pial": "lh.pial.pt",
    "rh.pial": "rh.pial.pt",
}
TARGET_KIND_ORDER = (
    "aseg_presurf",
    "brainmask",
    "norm",
    "norm_quantized",
    "white_surfaces",
    "pial_surfaces",
    "lh.white",
    "rh.white",
    "lh.pial",
    "rh.pial",
    "surface_bands",
)
SURFACE_LABEL_NAMES = {
    0: "background",
    1: "lh.white",
    2: "rh.white",
    3: "lh.pial",
    4: "rh.pial",
}
CACHE_VERSION = 2


@dataclass(frozen=True)
class SubjectTensorCache:
    subject_id: str
    subject_dir: Path
    cache_dir: Path
    metadata_path: Path
    rawavg_pt: Path
    aparc_aseg_pt: Path
    aseg_presurf_pt: Path
    brainmask_pt: Path
    norm_pt: Path
    norm_quantized_pt: Path
    white_surfaces_pt: Path
    pial_surfaces_pt: Path
    surface_bands_pt: Path
    lh_white_pt: Path
    rh_white_pt: Path
    lh_pial_pt: Path
    rh_pial_pt: Path


def discover_subject_dirs(fs_root: str | Path) -> list[Path]:
    root = Path(fs_root)
    return sorted(path for path in root.iterdir() if path.is_dir())


def subject_required_relpaths() -> list[str]:
    return [
        "mri/rawavg.mgz",
        "mri/aparc+aseg.mgz",
        "mri/aseg.presurf.mgz",
        "mri/brainmask.mgz",
        "mri/norm.mgz",
        "mri/orig.mgz",
        "surf/lh.white",
        "surf/rh.white",
        "surf/lh.pial",
        "surf/rh.pial",
    ]


def subject_is_ready(subject_dir: str | Path) -> bool:
    subject_dir = Path(subject_dir)
    return all((subject_dir / relpath).exists() for relpath in subject_required_relpaths())


def ready_subject_dirs(fs_root: str | Path) -> list[Path]:
    return [subject_dir for subject_dir in discover_subject_dirs(fs_root) if subject_is_ready(subject_dir)]


def build_subject_cache_paths(subject_dir: str | Path, out_root: str | Path) -> SubjectTensorCache:
    subject_dir = Path(subject_dir)
    cache_dir = Path(out_root) / subject_dir.name
    return SubjectTensorCache(
        subject_id=subject_dir.name,
        subject_dir=subject_dir,
        cache_dir=cache_dir,
        metadata_path=cache_dir / "metadata.json",
        rawavg_pt=cache_dir / TARGET_PT_FILE_MAP["rawavg"],
        aparc_aseg_pt=cache_dir / TARGET_PT_FILE_MAP["aparc+aseg"],
        aseg_presurf_pt=cache_dir / TARGET_PT_FILE_MAP["aseg_presurf"],
        brainmask_pt=cache_dir / TARGET_PT_FILE_MAP["brainmask"],
        norm_pt=cache_dir / TARGET_PT_FILE_MAP["norm"],
        norm_quantized_pt=cache_dir / TARGET_PT_FILE_MAP["norm_quantized"],
        white_surfaces_pt=cache_dir / TARGET_PT_FILE_MAP["white_surfaces"],
        pial_surfaces_pt=cache_dir / TARGET_PT_FILE_MAP["pial_surfaces"],
        surface_bands_pt=cache_dir / TARGET_PT_FILE_MAP["surface_bands"],
        lh_white_pt=cache_dir / TARGET_PT_FILE_MAP["lh.white"],
        rh_white_pt=cache_dir / TARGET_PT_FILE_MAP["rh.white"],
        lh_pial_pt=cache_dir / TARGET_PT_FILE_MAP["lh.pial"],
        rh_pial_pt=cache_dir / TARGET_PT_FILE_MAP["rh.pial"],
    )


def _save_tensor_atomic(tensor: torch.Tensor, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(tensor, tmp_path)
    os.replace(tmp_path, out_path)


def _write_json_atomic(payload: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    os.replace(tmp_path, out_path)


def _load_metadata(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _cache_pt_paths(cache: SubjectTensorCache) -> tuple[Path, ...]:
    return (
        cache.rawavg_pt,
        cache.aparc_aseg_pt,
        cache.aseg_presurf_pt,
        cache.brainmask_pt,
        cache.norm_pt,
        cache.norm_quantized_pt,
        cache.white_surfaces_pt,
        cache.pial_surfaces_pt,
        cache.surface_bands_pt,
        cache.lh_white_pt,
        cache.rh_white_pt,
        cache.lh_pial_pt,
        cache.rh_pial_pt,
    )


def _cache_is_complete(
    cache: SubjectTensorCache,
    *,
    surface_dilation_iters: int,
    norm_bins: int,
) -> bool:
    del surface_dilation_iters
    del norm_bins
    if not cache.metadata_path.exists():
        return False
    if any(not path.exists() for path in _cache_pt_paths(cache)):
        return False
    try:
        _load_metadata(cache.metadata_path)
    except (OSError, json.JSONDecodeError, FileNotFoundError):
        return False
    return True


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


def _to_ras(img: nib.spatialimages.SpatialImage) -> nib.spatialimages.SpatialImage:
    return nib.as_closest_canonical(img)


def _resample_to_reference(
    moving: nib.spatialimages.SpatialImage,
    reference: nib.spatialimages.SpatialImage,
    *,
    is_labels: bool,
) -> nib.spatialimages.SpatialImage:
    order = 0 if is_labels else 1
    out = resample_from_to(moving, reference, order=order)
    data = out.get_fdata()
    if is_labels:
        data = np.asarray(np.rint(data), dtype=np.int32)
    else:
        data = np.asarray(data, dtype=np.float32)
    return nib.Nifti1Image(data, out.affine)


def _tensor_reference_image(subject_dir: str | Path) -> nib.spatialimages.SpatialImage:
    subject_dir = Path(subject_dir)
    _, aparc_out = prepare_images_if_needed(
        subject_dir / "mri" / "orig.mgz",
        subject_dir / "mri" / "aparc+aseg.mgz",
    )
    return aparc_out


def _load_native_label_in_tensor_space(subject_dir: str | Path, relpath: str) -> np.ndarray:
    subject_dir = Path(subject_dir)
    reference = _tensor_reference_image(subject_dir)
    native = _to_ras(load_mgz(subject_dir / relpath))
    resampled = _resample_to_reference(native, reference, is_labels=True)
    return np.asarray(resampled.dataobj, dtype=np.int32)


def _load_native_float_in_tensor_space(subject_dir: str | Path, relpath: str) -> np.ndarray:
    subject_dir = Path(subject_dir)
    reference = _tensor_reference_image(subject_dir)
    native = _to_ras(load_mgz(subject_dir / relpath))
    resampled = _resample_to_reference(native, reference, is_labels=False)
    return np.asarray(resampled.get_fdata(dtype=np.float32), dtype=np.float32)


def _densify_surface_points(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    centroids = tri.mean(axis=1)
    edge01 = 0.5 * (tri[:, 0] + tri[:, 1])
    edge12 = 0.5 * (tri[:, 1] + tri[:, 2])
    edge20 = 0.5 * (tri[:, 2] + tri[:, 0])
    return np.concatenate([vertices, centroids, edge01, edge12, edge20], axis=0)


def _surface_mask_in_orig_space(subject_dir: str | Path, surface_name: str, dilation_iters: int = 1) -> nib.spatialimages.SpatialImage:
    subject_dir = Path(subject_dir)
    orig_img = load_mgz(subject_dir / "mri" / "orig.mgz")
    vertices, faces = read_geometry(str(subject_dir / "surf" / surface_name))
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    points = _densify_surface_points(vertices, faces)

    vox2ras_tkr = np.asarray(orig_img.header.get_vox2ras_tkr(), dtype=np.float64)
    ras2vox_tkr = np.linalg.inv(vox2ras_tkr)
    pts_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    vox = (ras2vox_tkr @ pts_h.T).T[:, :3]
    vox = np.rint(vox).astype(np.int32)

    shape = np.asarray(orig_img.shape[:3], dtype=np.int32)
    valid = np.all((vox >= 0) & (vox < shape[None, :]), axis=1)
    vox = vox[valid]

    mask = np.zeros(tuple(int(v) for v in shape), dtype=bool)
    if vox.size > 0:
        mask[vox[:, 0], vox[:, 1], vox[:, 2]] = True
    if dilation_iters > 0:
        mask = ndi.binary_dilation(mask, iterations=int(dilation_iters))

    data = np.asarray(mask, dtype=np.uint8)
    return nib.MGHImage(data, orig_img.affine, header=orig_img.header)


def _surface_mask_in_tensor_space(subject_dir: str | Path, surface_name: str, dilation_iters: int = 1) -> np.ndarray:
    subject_dir = Path(subject_dir)
    reference = _tensor_reference_image(subject_dir)
    mask_img = _surface_mask_in_orig_space(subject_dir, surface_name, dilation_iters=dilation_iters)
    mask_ras = _to_ras(mask_img)
    resampled = _resample_to_reference(mask_ras, reference, is_labels=True)
    data = np.asarray(resampled.dataobj, dtype=np.int32)
    return (data > 0).astype(np.uint8)


def _surface_bands_in_tensor_space(subject_dir: str | Path, dilation_iters: int = 1) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    masks: dict[str, np.ndarray] = {}
    for surface_name in SURFACE_CLASS_MAP:
        masks[surface_name] = _surface_mask_in_tensor_space(subject_dir, surface_name, dilation_iters=dilation_iters)

    surface_bands = np.zeros_like(next(iter(masks.values())), dtype=np.int16)
    for surface_name in ("lh.white", "rh.white", "lh.pial", "rh.pial"):
        class_id = int(SURFACE_CLASS_MAP[surface_name])
        mask = masks[surface_name] > 0
        surface_bands[(surface_bands == 0) & mask] = class_id
    return surface_bands, masks


def _paired_surface_targets(surface_masks: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    white_surfaces = np.zeros_like(surface_masks["lh.white"], dtype=np.int16)
    pial_surfaces = np.zeros_like(surface_masks["lh.pial"], dtype=np.int16)

    white_surfaces[surface_masks["lh.white"] > 0] = 1
    white_surfaces[(white_surfaces == 0) & (surface_masks["rh.white"] > 0)] = 2

    pial_surfaces[surface_masks["lh.pial"] > 0] = 1
    pial_surfaces[(pial_surfaces == 0) & (surface_masks["rh.pial"] > 0)] = 2
    return white_surfaces, pial_surfaces


def _maybe_backfill_combined_surface_targets(cache: SubjectTensorCache) -> bool:
    if cache.white_surfaces_pt.exists() and cache.pial_surfaces_pt.exists():
        return True
    required = [
        cache.metadata_path,
        cache.lh_white_pt,
        cache.rh_white_pt,
        cache.lh_pial_pt,
        cache.rh_pial_pt,
    ]
    if any(not path.exists() for path in required):
        return False

    lh_white = np.asarray(load_tensor(cache.lh_white_pt), dtype=np.int16)
    rh_white = np.asarray(load_tensor(cache.rh_white_pt), dtype=np.int16)
    lh_pial = np.asarray(load_tensor(cache.lh_pial_pt), dtype=np.int16)
    rh_pial = np.asarray(load_tensor(cache.rh_pial_pt), dtype=np.int16)
    white_surfaces, pial_surfaces = _paired_surface_targets(
        {
            "lh.white": lh_white,
            "rh.white": rh_white,
            "lh.pial": lh_pial,
            "rh.pial": rh_pial,
        }
    )
    if not cache.white_surfaces_pt.exists():
        _save_tensor_atomic(torch.as_tensor(white_surfaces, dtype=torch.int16), cache.white_surfaces_pt)
    if not cache.pial_surfaces_pt.exists():
        _save_tensor_atomic(torch.as_tensor(pial_surfaces, dtype=torch.int16), cache.pial_surfaces_pt)

    metadata = _load_metadata(cache.metadata_path)
    metadata.setdefault("pt_files", {})
    metadata["cache_version"] = int(CACHE_VERSION)
    metadata.setdefault("surface_target_type", "shell_mask")
    metadata["pt_files"]["white_surfaces"] = str(cache.white_surfaces_pt.resolve())
    metadata["pt_files"]["pial_surfaces"] = str(cache.pial_surfaces_pt.resolve())
    _write_json_atomic(metadata, cache.metadata_path)
    return True


def _compute_norm_quantization(norm_arr: np.ndarray, brainmask_arr: np.ndarray, bins: int) -> tuple[np.ndarray, dict[str, float]]:
    mask = brainmask_arr > 0
    fg = np.asarray(norm_arr[mask], dtype=np.float32)
    if fg.size == 0:
        raise ValueError("brainmask is empty; cannot quantize norm volume")
    lo = float(np.percentile(fg, 0.5))
    hi = float(np.percentile(fg, 99.5))
    clipped = np.clip(np.asarray(norm_arr, dtype=np.float32), lo, hi)
    scaled = (clipped - lo) / max(1e-6, hi - lo)
    quantized = np.rint(scaled * float(bins - 1)).astype(np.int16)
    quantized[~mask] = 0
    meta = {"lo": lo, "hi": hi, "bins": int(bins)}
    return quantized, meta


def dequantize_norm_volume(quantized: np.ndarray, *, lo: float, hi: float, bins: int) -> np.ndarray:
    scaled = np.asarray(quantized, dtype=np.float32) / max(1.0, float(bins - 1))
    return lo + scaled * float(hi - lo)


def _subject_metadata_payload(
    cache: SubjectTensorCache,
    tensor_reference: nib.spatialimages.SpatialImage,
    norm_quant_meta: dict[str, float],
    *,
    surface_dilation_iters: int,
    norm_bins: int,
) -> dict:
    return {
        "cache_version": int(CACHE_VERSION),
        "subject_id": cache.subject_id,
        "subject_dir": str(cache.subject_dir.resolve()),
        "tensor_shape": [int(v) for v in tensor_reference.shape[:3]],
        "tensor_affine": np.asarray(tensor_reference.affine, dtype=np.float64).tolist(),
        "target_shape": [int(v) for v in TARGET_SHAPE],
        "target_voxel_size_mm": float(TARGET_VOXEL_SIZE_MM),
        "surface_target_type": "shell_mask",
        "surface_dilation_iters": int(surface_dilation_iters),
        "norm_bins": int(norm_bins),
        "pt_files": {
            "rawavg": str(cache.rawavg_pt.resolve()),
            "aparc+aseg": str(cache.aparc_aseg_pt.resolve()),
            "aseg_presurf": str(cache.aseg_presurf_pt.resolve()),
            "brainmask": str(cache.brainmask_pt.resolve()),
            "norm": str(cache.norm_pt.resolve()),
            "norm_quantized": str(cache.norm_quantized_pt.resolve()),
            "white_surfaces": str(cache.white_surfaces_pt.resolve()),
            "pial_surfaces": str(cache.pial_surfaces_pt.resolve()),
            "surface_bands": str(cache.surface_bands_pt.resolve()),
            "lh.white": str(cache.lh_white_pt.resolve()),
            "rh.white": str(cache.rh_white_pt.resolve()),
            "lh.pial": str(cache.lh_pial_pt.resolve()),
            "rh.pial": str(cache.rh_pial_pt.resolve()),
        },
        "native_files": {
            "rawavg": str((cache.subject_dir / "mri" / "rawavg.mgz").resolve()),
            "aparc+aseg": str((cache.subject_dir / "mri" / "aparc+aseg.mgz").resolve()),
            "aseg_presurf": str((cache.subject_dir / "mri" / "aseg.presurf.mgz").resolve()),
            "brainmask": str((cache.subject_dir / "mri" / "brainmask.mgz").resolve()),
            "norm": str((cache.subject_dir / "mri" / "norm.mgz").resolve()),
            "orig": str((cache.subject_dir / "mri" / "orig.mgz").resolve()),
        },
        "norm_quantization": norm_quant_meta,
    }


def build_subject_tensor_cache(
    subject_dir: str | Path,
    out_root: str | Path,
    *,
    force: bool = False,
    surface_dilation_iters: int = 1,
    norm_bins: int = 256,
) -> SubjectTensorCache:
    subject_dir = Path(subject_dir)
    if not subject_is_ready(subject_dir):
        missing = [rel for rel in subject_required_relpaths() if not (subject_dir / rel).exists()]
        raise FileNotFoundError(f"{subject_dir} is missing required files: {missing}")

    cache = build_subject_cache_paths(subject_dir, out_root)
    cache.cache_dir.mkdir(parents=True, exist_ok=True)
    if not force:
        _maybe_backfill_combined_surface_targets(cache)
    if not force and _cache_is_complete(
        cache,
        surface_dilation_iters=surface_dilation_iters,
        norm_bins=norm_bins,
    ):
        return cache

    rawavg_path = subject_dir / "mri" / "rawavg.mgz"
    aparc_path = subject_dir / "mri" / "aparc+aseg.mgz"
    tensor_reference = _tensor_reference_image(subject_dir)

    if force or not cache.rawavg_pt.exists() or not cache.aparc_aseg_pt.exists():
        rawavg_arr, aparc_arr = prepare_arrays_if_needed(rawavg_path, aparc_path)
        _save_tensor_atomic(torch.from_numpy(np.asarray(rawavg_arr, dtype=np.float32)), cache.rawavg_pt)
        _save_tensor_atomic(torch.from_numpy(np.asarray(aparc_arr, dtype=np.int16)), cache.aparc_aseg_pt)

    aseg_presurf = _load_native_label_in_tensor_space(subject_dir, "mri/aseg.presurf.mgz")
    brainmask = _load_native_label_in_tensor_space(subject_dir, "mri/brainmask.mgz")
    brainmask = (brainmask > 0).astype(np.int16)
    norm = _load_native_float_in_tensor_space(subject_dir, "mri/norm.mgz")
    norm_quantized, norm_quant_meta = _compute_norm_quantization(norm, brainmask, bins=norm_bins)
    surface_bands, surface_masks = _surface_bands_in_tensor_space(
        subject_dir,
        dilation_iters=surface_dilation_iters,
    )
    white_surfaces, pial_surfaces = _paired_surface_targets(surface_masks)

    targets_to_save: list[tuple[np.ndarray, Path, torch.dtype]] = [
        (aseg_presurf.astype(np.int16), cache.aseg_presurf_pt, torch.int16),
        (brainmask.astype(np.int16), cache.brainmask_pt, torch.int16),
        (norm.astype(np.float32), cache.norm_pt, torch.float32),
        (norm_quantized.astype(np.int16), cache.norm_quantized_pt, torch.int16),
        (white_surfaces.astype(np.int16), cache.white_surfaces_pt, torch.int16),
        (pial_surfaces.astype(np.int16), cache.pial_surfaces_pt, torch.int16),
        (surface_bands.astype(np.int16), cache.surface_bands_pt, torch.int16),
        (surface_masks["lh.white"].astype(np.int16), cache.lh_white_pt, torch.int16),
        (surface_masks["rh.white"].astype(np.int16), cache.rh_white_pt, torch.int16),
        (surface_masks["lh.pial"].astype(np.int16), cache.lh_pial_pt, torch.int16),
        (surface_masks["rh.pial"].astype(np.int16), cache.rh_pial_pt, torch.int16),
    ]
    for array, out_path, dtype in targets_to_save:
        if force or not out_path.exists():
            _save_tensor_atomic(torch.as_tensor(array, dtype=dtype), out_path)

    _write_json_atomic(
        _subject_metadata_payload(
            cache=cache,
            tensor_reference=tensor_reference,
            norm_quant_meta=norm_quant_meta,
            surface_dilation_iters=surface_dilation_iters,
            norm_bins=norm_bins,
        ),
        cache.metadata_path,
    )
    return cache


def _build_manifest_row(
    subject_dir: str | Path,
    out_root: str | Path,
    *,
    force: bool,
    surface_dilation_iters: int,
    norm_bins: int,
) -> dict:
    cache = build_subject_tensor_cache(
        subject_dir,
        out_root,
        force=force,
        surface_dilation_iters=surface_dilation_iters,
        norm_bins=norm_bins,
    )
    metadata = _load_metadata(cache.metadata_path)
    return {
        "subject_id": cache.subject_id,
        "subject_dir": str(cache.subject_dir),
        "cache_dir": str(cache.cache_dir),
        "metadata_path": str(cache.metadata_path),
        "rawavg_pt": str(cache.rawavg_pt),
        "aparc_aseg_pt": str(cache.aparc_aseg_pt),
        "aseg_presurf_pt": str(cache.aseg_presurf_pt),
        "brainmask_pt": str(cache.brainmask_pt),
        "norm_pt": str(cache.norm_pt),
        "norm_quantized_pt": str(cache.norm_quantized_pt),
        "white_surfaces_pt": str(cache.white_surfaces_pt),
        "pial_surfaces_pt": str(cache.pial_surfaces_pt),
        "surface_bands_pt": str(cache.surface_bands_pt),
        "lh_white_pt": str(cache.lh_white_pt),
        "rh_white_pt": str(cache.rh_white_pt),
        "lh_pial_pt": str(cache.lh_pial_pt),
        "rh_pial_pt": str(cache.rh_pial_pt),
        "tensor_shape": metadata["tensor_shape"],
    }


def _manifest_row_from_cache(
    cache: SubjectTensorCache,
) -> dict:
    metadata = _load_metadata(cache.metadata_path)
    return {
        "subject_id": cache.subject_id,
        "subject_dir": str(cache.subject_dir),
        "cache_dir": str(cache.cache_dir),
        "metadata_path": str(cache.metadata_path),
        "rawavg_pt": str(cache.rawavg_pt),
        "aparc_aseg_pt": str(cache.aparc_aseg_pt),
        "aseg_presurf_pt": str(cache.aseg_presurf_pt),
        "brainmask_pt": str(cache.brainmask_pt),
        "norm_pt": str(cache.norm_pt),
        "norm_quantized_pt": str(cache.norm_quantized_pt),
        "white_surfaces_pt": str(cache.white_surfaces_pt),
        "pial_surfaces_pt": str(cache.pial_surfaces_pt),
        "surface_bands_pt": str(cache.surface_bands_pt),
        "lh_white_pt": str(cache.lh_white_pt),
        "rh_white_pt": str(cache.rh_white_pt),
        "lh_pial_pt": str(cache.lh_pial_pt),
        "rh_pial_pt": str(cache.rh_pial_pt),
        "tensor_shape": metadata["tensor_shape"],
    }


def build_tensor_cache_manifest(
    fs_root: str | Path,
    out_root: str | Path,
    *,
    force: bool = False,
    max_subjects: int | None = None,
    surface_dilation_iters: int = 1,
    norm_bins: int = 256,
    n_jobs: int = -1,
) -> pd.DataFrame:
    subject_dirs = ready_subject_dirs(fs_root)
    if max_subjects is not None:
        subject_dirs = subject_dirs[: int(max_subjects)]

    rows_by_subject_id: dict[str, dict] = {}
    subjects_to_build: list[Path] = []
    out_root = Path(out_root)

    for subject_dir in subject_dirs:
        cache = build_subject_cache_paths(subject_dir, out_root)
        if not force and _cache_is_complete(
            cache,
            surface_dilation_iters=surface_dilation_iters,
            norm_bins=norm_bins,
        ):
            rows_by_subject_id[cache.subject_id] = _manifest_row_from_cache(cache)
        else:
            subjects_to_build.append(Path(subject_dir))

    if subjects_to_build:
        built_rows = _parallel_map(
            partial(
                _build_manifest_row,
                out_root=out_root,
                force=force,
                surface_dilation_iters=surface_dilation_iters,
                norm_bins=norm_bins,
            ),
            subjects_to_build,
            n_jobs=n_jobs,
            prefer="processes",
        )
        for row in built_rows:
            rows_by_subject_id[str(row["subject_id"])] = row

    ordered_rows = [rows_by_subject_id[subject_dir.name] for subject_dir in subject_dirs]
    return pd.DataFrame(ordered_rows)


def target_tensor_path_from_row(row: pd.Series | dict, target_kind: str) -> str:
    key_map = {
        "aseg_presurf": "aseg_presurf_pt",
        "brainmask": "brainmask_pt",
        "norm": "norm_pt",
        "norm_quantized": "norm_quantized_pt",
        "white_surfaces": "white_surfaces_pt",
        "pial_surfaces": "pial_surfaces_pt",
        "surface_bands": "surface_bands_pt",
        "lh.white": "lh_white_pt",
        "rh.white": "rh_white_pt",
        "lh.pial": "lh_pial_pt",
        "rh.pial": "rh_pial_pt",
    }
    if target_kind not in key_map:
        raise KeyError(f"Unsupported target_kind: {target_kind}")
    return str(row[key_map[target_kind]])


def load_tensor(path: str | Path) -> torch.Tensor:
    return torch.load(Path(path))


def dice_for_label(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> float:
    true_mask = np.asarray(y_true == int(label))
    pred_mask = np.asarray(y_pred == int(label))
    denom = int(true_mask.sum()) + int(pred_mask.sum())
    if denom == 0:
        return 1.0
    inter = int((true_mask & pred_mask).sum())
    return float(2.0 * inter / max(1, denom))


def evaluate_surface_bands(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for label, name in SURFACE_LABEL_NAMES.items():
        rows.append(
            {
                "label": int(label),
                "name": name,
                "dice": dice_for_label(y_true, y_pred, int(label)),
                "true_voxels": int((np.asarray(y_true) == int(label)).sum()),
                "pred_voxels": int((np.asarray(y_pred) == int(label)).sum()),
            }
        )
    return pd.DataFrame(rows)


def _tensor_space_image_from_prediction(prediction: np.ndarray, metadata: dict, *, is_labels: bool) -> nib.spatialimages.SpatialImage:
    affine = np.asarray(metadata["tensor_affine"], dtype=np.float64)
    data = np.asarray(prediction)
    if is_labels:
        data = np.asarray(np.rint(data), dtype=np.int32)
    else:
        data = np.asarray(data, dtype=np.float32)
    return nib.Nifti1Image(data, affine)


def _native_reference_for_target(subject_dir: str | Path, target_kind: str) -> Path:
    subject_dir = Path(subject_dir)
    if target_kind == "aseg_presurf":
        return subject_dir / "mri" / "aseg.presurf.mgz"
    if target_kind == "brainmask":
        return subject_dir / "mri" / "brainmask.mgz"
    if target_kind in {"norm", "norm_quantized"}:
        return subject_dir / "mri" / "norm.mgz"
    if target_kind in {"surface_bands", "white_surfaces", "pial_surfaces", "lh.white", "rh.white", "lh.pial", "rh.pial"}:
        return subject_dir / "mri" / "orig.mgz"
    raise KeyError(f"Unsupported target_kind: {target_kind}")


def export_prediction_to_native_mgz(
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
    if target_kind == "norm_quantized":
        q = metadata["norm_quantization"]
        pred = dequantize_norm_volume(pred, lo=float(q["lo"]), hi=float(q["hi"]), bins=int(q["bins"]))
        is_labels = False
    else:
        is_labels = target_kind not in {"norm"}

    tensor_img = _tensor_space_image_from_prediction(pred, metadata, is_labels=is_labels)
    native_ref = load_mgz(_native_reference_for_target(subject_dir, target_kind))
    resampled = _resample_to_reference(tensor_img, native_ref, is_labels=is_labels)

    if target_kind == "brainmask":
        data = (np.asarray(resampled.dataobj, dtype=np.int32) > 0).astype(np.uint8)
    elif is_labels:
        native_dtype = native_ref.header.get_data_dtype()
        data = np.asarray(resampled.dataobj, dtype=native_dtype)
    else:
        native_dtype = native_ref.header.get_data_dtype()
        data = np.asarray(resampled.get_fdata(dtype=np.float32), dtype=native_dtype)

    out_img = nib.MGHImage(data, native_ref.affine, header=native_ref.header)
    nib.save(out_img, str(out_path))
    return out_path


def export_surface_bands_bundle(
    prediction: np.ndarray | torch.Tensor,
    *,
    metadata_path: str | Path,
    out_root: str | Path,
) -> dict[str, Path]:
    metadata = _load_metadata(metadata_path)
    subject_dir = Path(metadata["subject_dir"])
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    pred = prediction.detach().cpu().numpy() if torch.is_tensor(prediction) else np.asarray(prediction)
    outputs: dict[str, Path] = {}
    combined = out_root / "surface_bands.native.mgz"
    outputs["surface_bands"] = export_prediction_to_native_mgz(
        pred,
        metadata_path=metadata_path,
        target_kind="surface_bands",
        out_path=combined,
    )
    for label, name in SURFACE_LABEL_NAMES.items():
        if label == 0:
            continue
        mask = (np.asarray(pred) == int(label)).astype(np.int16)
        outputs[name] = export_prediction_to_native_mgz(
            mask,
            metadata_path=metadata_path,
            target_kind=name,
            out_path=out_root / f"{name}.band.native.mgz",
        )
    return outputs


def build_subject_target_summary(subject_dir: str | Path, out_root: str | Path) -> dict:
    cache = build_subject_cache_paths(subject_dir, out_root)
    metadata = _load_metadata(cache.metadata_path)
    summary = {
        "subject_id": cache.subject_id,
        "subject_dir": str(cache.subject_dir),
        "tensor_shape": metadata["tensor_shape"],
        "tensor_affine": metadata["tensor_affine"],
        "targets": metadata["pt_files"],
    }
    return summary


__all__ = [
    "SURFACE_CLASS_MAP",
    "SURFACE_LABEL_NAMES",
    "TARGET_KIND_ORDER",
    "TARGET_PT_FILE_MAP",
    "SubjectTensorCache",
    "build_subject_cache_paths",
    "build_subject_target_summary",
    "build_subject_tensor_cache",
    "build_tensor_cache_manifest",
    "dequantize_norm_volume",
    "dice_for_label",
    "discover_subject_dirs",
    "evaluate_surface_bands",
    "export_prediction_to_native_mgz",
    "export_surface_bands_bundle",
    "load_tensor",
    "ready_subject_dirs",
    "subject_is_ready",
    "subject_required_relpaths",
    "target_tensor_path_from_row",
]

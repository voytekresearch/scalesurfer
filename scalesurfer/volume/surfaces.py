from __future__ import annotations

import json
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import torch
from joblib import Parallel, delayed
from nibabel.freesurfer.io import read_geometry, write_geometry
from skimage.measure import marching_cubes
from scipy.spatial import cKDTree

from ..convert import load_mgz
from .targets import export_prediction_to_native_mgz


COMBINED_SURFACE_TARGET_KINDS = ("white_surfaces", "pial_surfaces")
SURFACE_MASK_TARGET_KINDS = ("lh.white", "rh.white", "lh.pial", "rh.pial")
SURFACE_TARGET_KINDS = COMBINED_SURFACE_TARGET_KINDS + SURFACE_MASK_TARGET_KINDS
HEMIS = ("lh", "rh")


@dataclass(frozen=True)
class SurfaceBundleResult:
    bundle_dir: Path
    mask_paths: dict[str, Path]
    surface_paths: dict[str, Path]
    timings_sec: dict[str, float]


def _surface_masks_from_predictions(
    predictions: dict[str, np.ndarray | torch.Tensor],
) -> dict[str, np.ndarray | torch.Tensor]:
    if "white_surfaces" in predictions and "pial_surfaces" in predictions:
        white = predictions["white_surfaces"]
        pial = predictions["pial_surfaces"]
        white_arr = white.detach().cpu().numpy() if torch.is_tensor(white) else np.asarray(white)
        pial_arr = pial.detach().cpu().numpy() if torch.is_tensor(pial) else np.asarray(pial)
        return {
            "lh.white": (white_arr == 1).astype(np.int16),
            "rh.white": (white_arr == 2).astype(np.int16),
            "lh.pial": (pial_arr == 1).astype(np.int16),
            "rh.pial": (pial_arr == 2).astype(np.int16),
        }
    missing = [key for key in SURFACE_MASK_TARGET_KINDS if key not in predictions]
    if missing:
        raise KeyError(
            "Surface predictions must include either combined targets "
            f"{COMBINED_SURFACE_TARGET_KINDS} or direct mask targets {SURFACE_MASK_TARGET_KINDS}; missing {missing}"
        )
    return {key: predictions[key] for key in SURFACE_MASK_TARGET_KINDS}


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


def _surface_volume_info(subject_dir: str | Path, hemi: str) -> dict[str, object]:
    subject_dir = Path(subject_dir)
    reference_surface = subject_dir / "surf" / f"{hemi}.white"
    if reference_surface.exists():
        _coords, _faces, info = read_geometry(str(reference_surface), read_metadata=True)
        return {key: value.copy() if hasattr(value, "copy") else value for key, value in dict(info).items()}

    orig = load_mgz(subject_dir / "mri" / "orig.mgz")
    header = orig.header
    mdc = np.asarray(header["Mdc"], dtype=np.float64)
    return {
        "head": np.asarray([2, 0, 20], dtype=np.int32),
        "valid": "1  # volume info valid",
        "filename": str((subject_dir / "mri" / "orig.mgz").resolve()),
        "volume": np.asarray(orig.shape[:3], dtype=np.int32),
        "voxelsize": np.asarray(header["delta"], dtype=np.float64),
        "xras": np.asarray(mdc[0], dtype=np.float64),
        "yras": np.asarray(mdc[1], dtype=np.float64),
        "zras": np.asarray(mdc[2], dtype=np.float64),
        "cras": np.asarray(header["Pxyz_c"], dtype=np.float64),
    }


def _load_binary_mask(mask_path: str | Path) -> tuple[np.ndarray, nib.spatialimages.SpatialImage]:
    img = load_mgz(mask_path)
    mask = np.asarray(img.dataobj) > 0
    return mask, img


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    labels, n_labels = ndi.label(mask)
    if n_labels <= 1:
        return mask
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    keep = int(np.argmax(counts))
    return labels == keep


def _vox_to_tkr_ras(vertices_ijk: np.ndarray, orig_img: nib.spatialimages.SpatialImage) -> np.ndarray:
    vox2ras_tkr = np.asarray(orig_img.header.get_vox2ras_tkr(), dtype=np.float64)
    verts_h = np.concatenate(
        [np.asarray(vertices_ijk, dtype=np.float64), np.ones((vertices_ijk.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    return (vox2ras_tkr @ verts_h.T).T[:, :3]


def _extract_surface_from_mask(
    mask: np.ndarray,
    orig_img: nib.spatialimages.SpatialImage,
    *,
    closing_iters: int = 0,
    dilation_iters: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    work = np.asarray(mask, dtype=bool)
    if dilation_iters > 0:
        work = ndi.binary_dilation(work, iterations=int(dilation_iters))
    if closing_iters > 0:
        work = ndi.binary_closing(work, iterations=int(closing_iters))
    work = _largest_connected_component(work)
    if int(work.sum()) == 0:
        raise ValueError("Surface mask is empty after connected-component filtering.")

    vertices_ijk, faces, _normals, _values = marching_cubes(
        work.astype(np.float32),
        level=0.5,
        allow_degenerate=False,
    )
    vertices_ras = _vox_to_tkr_ras(vertices_ijk, orig_img)
    return np.asarray(vertices_ras, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = np.zeros_like(vertices, dtype=np.float64)
    for col in range(3):
        np.add.at(normals, faces[:, col], face_normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-8)

    center = np.mean(vertices, axis=0, keepdims=True)
    orientation = np.mean(np.sum((vertices - center) * normals, axis=1))
    if orientation < 0:
        normals = -normals
    return np.asarray(normals, dtype=np.float32)


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return _vertex_normals(vertices, faces)


def _mesh_scalar_smooth(values: np.ndarray, faces: np.ndarray, *, iters: int = 2) -> np.ndarray:
    smoothed = np.asarray(values, dtype=np.float32).copy()
    if iters <= 0 or smoothed.size == 0:
        return smoothed

    edges = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
            faces[:, [1, 0]],
            faces[:, [2, 1]],
            faces[:, [0, 2]],
        ],
        axis=0,
    )
    src = edges[:, 0]
    dst = edges[:, 1]
    for _ in range(int(iters)):
        neighbor_sum = np.zeros_like(smoothed)
        neighbor_count = np.zeros_like(smoothed)
        np.add.at(neighbor_sum, dst, smoothed[src])
        np.add.at(neighbor_count, dst, 1.0)
        valid = neighbor_count > 0
        updated = smoothed.copy()
        updated[valid] = 0.5 * smoothed[valid] + 0.5 * (neighbor_sum[valid] / neighbor_count[valid])
        smoothed = updated
    return smoothed


def _project_pial_to_white_topology(
    white_vertices: np.ndarray,
    white_faces: np.ndarray,
    pial_vertices_independent: np.ndarray,
    *,
    min_thickness_mm: float = 0.2,
    max_thickness_mm: float = 8.0,
    query_k: int = 8,
    smooth_iters: int = 2,
) -> np.ndarray:
    normals = _vertex_normals(white_vertices, white_faces)
    tree = cKDTree(np.asarray(pial_vertices_independent, dtype=np.float64))
    k = max(1, min(int(query_k), int(pial_vertices_independent.shape[0])))
    _distances, indices = tree.query(white_vertices, k=k)
    if k == 1:
        indices = indices[:, None]

    candidates = pial_vertices_independent[np.asarray(indices, dtype=np.int64)]
    deltas = candidates - white_vertices[:, None, :]
    normal_delta = np.einsum("vkc,vc->vk", deltas, normals)
    tangential = np.linalg.norm(deltas - normal_delta[..., None] * normals[:, None, :], axis=2)

    positive_penalty = np.where(normal_delta >= float(min_thickness_mm), 0.0, 1e3)
    score = tangential + positive_penalty
    best_idx = np.argmin(score, axis=1)
    thickness = normal_delta[np.arange(normal_delta.shape[0]), best_idx]
    thickness = np.clip(thickness, float(min_thickness_mm), float(max_thickness_mm))
    thickness = _mesh_scalar_smooth(thickness.astype(np.float32), white_faces, iters=smooth_iters)
    thickness = np.clip(thickness, float(min_thickness_mm), float(max_thickness_mm))
    return np.asarray(white_vertices + normals * thickness[:, None], dtype=np.float32)


def project_pial_to_white_topology(
    white_vertices: np.ndarray,
    white_faces: np.ndarray,
    pial_vertices_independent: np.ndarray,
    *,
    min_thickness_mm: float = 0.2,
    max_thickness_mm: float = 8.0,
    query_k: int = 8,
    smooth_iters: int = 2,
) -> np.ndarray:
    return _project_pial_to_white_topology(
        white_vertices,
        white_faces,
        pial_vertices_independent,
        min_thickness_mm=min_thickness_mm,
        max_thickness_mm=max_thickness_mm,
        query_k=query_k,
        smooth_iters=smooth_iters,
    )


def _write_surface(
    out_path: str | Path,
    *,
    coords: np.ndarray,
    faces: np.ndarray,
    volume_info: dict[str, object],
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_geometry(
        str(out_path),
        np.asarray(coords, dtype=np.float32),
        np.asarray(faces, dtype=np.int32),
        volume_info=volume_info,
    )
    return out_path


def _decode_surface_pair(
    hemi: str,
    *,
    subject_dir: str | Path,
    white_mask_path: str | Path,
    pial_mask_path: str | Path,
    out_dir: str | Path,
    white_closing_iters: int,
    pial_closing_iters: int,
    mask_dilation_iters: int,
    min_thickness_mm: float,
    max_thickness_mm: float,
    smooth_iters: int,
) -> tuple[str, dict[str, Path]]:
    out_dir = Path(out_dir)
    white_mask, white_img = _load_binary_mask(white_mask_path)
    pial_mask, pial_img = _load_binary_mask(pial_mask_path)

    white_vertices, white_faces = _extract_surface_from_mask(
        white_mask,
        white_img,
        closing_iters=white_closing_iters,
        dilation_iters=mask_dilation_iters,
    )
    pial_vertices_ind, _pial_faces_ind = _extract_surface_from_mask(
        pial_mask,
        pial_img,
        closing_iters=pial_closing_iters,
        dilation_iters=mask_dilation_iters,
    )
    pial_vertices = _project_pial_to_white_topology(
        white_vertices,
        white_faces,
        pial_vertices_ind,
        min_thickness_mm=min_thickness_mm,
        max_thickness_mm=max_thickness_mm,
        smooth_iters=smooth_iters,
    )

    volume_info = _surface_volume_info(subject_dir, hemi)
    white_path = _write_surface(out_dir / "surf" / f"{hemi}.white", coords=white_vertices, faces=white_faces, volume_info=volume_info)
    pial_path = _write_surface(out_dir / "surf" / f"{hemi}.pial", coords=pial_vertices, faces=white_faces, volume_info=volume_info)
    return hemi, {
        "white": white_path,
        "pial": pial_path,
    }


def create_surface_prediction_bundle(
    *,
    subject_dir: str | Path,
    metadata_path: str | Path,
    predictions: dict[str, np.ndarray | torch.Tensor],
    out_root: str | Path,
    force: bool = True,
    white_closing_iters: int = 0,
    pial_closing_iters: int = 0,
    mask_dilation_iters: int = 0,
    min_thickness_mm: float = 0.2,
    max_thickness_mm: float = 8.0,
    smooth_iters: int = 2,
    n_jobs: int = 2,
) -> SurfaceBundleResult:
    subject_dir = Path(subject_dir).expanduser().resolve()
    out_root = Path(out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    surface_dir = out_root / "surf"
    surface_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = out_root / "mri" / "surface_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    timings_sec: dict[str, float] = {}
    export_start = time.perf_counter()
    mask_predictions = _surface_masks_from_predictions(predictions)
    mask_paths: dict[str, Path] = {}
    for target_kind in SURFACE_MASK_TARGET_KINDS:
        mask_paths[target_kind] = export_prediction_to_native_mgz(
            mask_predictions[target_kind],
            metadata_path=metadata_path,
            target_kind=target_kind,
            out_path=mask_dir / f"{target_kind}.mgz",
        )
    timings_sec["export_surface_masks"] = time.perf_counter() - export_start

    decode_start = time.perf_counter()
    jobs = _parallel_map(
        partial(
            _decode_surface_pair_job,
            subject_dir=subject_dir,
            out_dir=out_root,
            white_closing_iters=white_closing_iters,
            pial_closing_iters=pial_closing_iters,
            mask_dilation_iters=mask_dilation_iters,
            min_thickness_mm=min_thickness_mm,
            max_thickness_mm=max_thickness_mm,
            smooth_iters=smooth_iters,
        ),
        [
            ("lh", mask_paths["lh.white"], mask_paths["lh.pial"]),
            ("rh", mask_paths["rh.white"], mask_paths["rh.pial"]),
        ],
        n_jobs=n_jobs,
        prefer="threads",
    )
    surface_paths: dict[str, Path] = {}
    for hemi, outputs in jobs:
        surface_paths[f"{hemi}.white"] = outputs["white"]
        surface_paths[f"{hemi}.pial"] = outputs["pial"]
    timings_sec["decode_surface_meshes"] = time.perf_counter() - decode_start

    payload = {
        "subject_dir": str(subject_dir),
        "metadata_path": str(Path(metadata_path).resolve()),
        "mask_paths": {key: str(value) for key, value in mask_paths.items()},
        "surface_paths": {key: str(value) for key, value in surface_paths.items()},
        "timings_sec": timings_sec,
    }
    (out_root / "surface_bundle.json").write_text(json.dumps(payload, indent=2))
    return SurfaceBundleResult(
        bundle_dir=out_root,
        mask_paths=mask_paths,
        surface_paths=surface_paths,
        timings_sec=timings_sec,
    )


def _decode_surface_pair_job(args, **kwargs):
    hemi, white_mask_path, pial_mask_path = args
    return _decode_surface_pair(
        hemi,
        white_mask_path=white_mask_path,
        pial_mask_path=pial_mask_path,
        **kwargs,
    )


__all__ = [
    "COMBINED_SURFACE_TARGET_KINDS",
    "HEMIS",
    "SURFACE_MASK_TARGET_KINDS",
    "SURFACE_TARGET_KINDS",
    "SurfaceBundleResult",
    "create_surface_prediction_bundle",
]

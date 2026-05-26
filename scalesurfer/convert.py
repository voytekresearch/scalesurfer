"""Conversions."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import shlex
import subprocess
from typing import Any
from typing import Tuple

from nilearn.datasets import load_mni152_template
import numpy as np
import torch
from nibabel.processing import conform, resample_from_to
from tqdm.contrib.concurrent import process_map
from nilearn.image import resample_img

import nibabel as nib
from nibabel.filebasedimages import ImageFileError

TARGET_VOXEL_SIZE_MM = 1.0
TARGET_SHAPE = (256, 256, 256)
CONFORM_SHAPE = (256, 256, 256)  # FreeSurfer orig.mgz conformed grid
CONFORM_UCHAR_HIGH_FRACTION = 0.999
_GZIP_MAGIC = b"\x1f\x8b"

MNI_SHAPE = (197, 233, 189)
MNI_AFFINE = np. array([
    [   1.,    0.,    0.,  -98.],
    [   0.,    1.,    0., -134.],
    [   0.,    0.,    1.,  -72.],
    [   0.,    0.,    0.,    1.]])

def _is_gzip_file(path: str | Path) -> bool:
    with open(path, "rb") as f:
        return f.read(2) == _GZIP_MAGIC

def load_mgz(path: str | Path):
    path = Path(path)
    try:
        return nib.load(str(path))
    except ImageFileError:
        # misnamed .nii.gz that is actually plain .nii
        if path.suffixes[-2:] == [".nii", ".gz"] and not _is_gzip_file(path):
            with tempfile.TemporaryDirectory() as tmpdir:
                fixed = Path(tmpdir) / path.name[:-3]  # drop ".gz"
                os.symlink(path, fixed)
                img = nib.load(str(fixed))
                # force data into memory before tempdir disappears
                data = img.get_fdata(dtype=np.float32)
                return nib.Nifti1Image(data, img.affine, img.header)
        raise


def _as_ras(img: nib.spatialimages.SpatialImage) -> nib.spatialimages.SpatialImage:
    """Reorient to canonical RAS without registration."""
    return nib.as_closest_canonical(img)


def _same_grid_img(
    a: nib.spatialimages.SpatialImage,
    b: nib.spatialimages.SpatialImage,
    atol: float = 1e-5,
) -> bool:
    return a.shape == b.shape and np.allclose(a.affine, b.affine, atol=atol)


def align_mgz_to_reference(
    moving_path: str | Path,
    reference_path: str | Path,
    *,
    is_labels: bool = False,
):
    """
    Resample `moving_path` onto the voxel grid of `reference_path`.
    """
    moving = _as_ras(load_mgz(moving_path))
    reference = _as_ras(load_mgz(reference_path))

    order = 0 if is_labels else 1
    out = resample_from_to(moving, reference, order=order)

    data = out.get_fdata()
    if is_labels:
        data = np.asarray(np.rint(data), dtype=np.int32)
    else:
        data = np.asarray(data, dtype=np.float32)

    return nib.Nifti1Image(data, out.affine)


def prepare_orig_and_aparc_arrays(
    orig_path: str | Path,
    aparc_path: str | Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return orig and aparc+aseg as NumPy arrays on the SAME grid.

    orig is resampled onto the aparc+aseg grid.
    aparc+aseg is kept as-is.
    """
    orig_resampled = align_mgz_to_reference(
        moving_path=orig_path,
        reference_path=aparc_path,
        is_labels=False,
    )
    aparc_img = _as_ras(load_mgz(aparc_path))

    orig_arr = np.asarray(orig_resampled.get_fdata(), dtype=np.float32)
    aparc_arr = np.asarray(np.rint(aparc_img.get_fdata()), dtype=np.int32)

    return orig_arr, aparc_arr


def arrays_are_same_grid(
    path_a: str | Path,
    path_b: str | Path,
    atol: float = 1e-5,
) -> bool:
    """
    Quick check before resampling.
    """
    a = _as_ras(load_mgz(path_a))
    b = _as_ras(load_mgz(path_b))
    return a.shape == b.shape and np.allclose(a.affine, b.affine, atol=atol)


def _resample_image_to_reference(
    moving: nib.spatialimages.SpatialImage,
    reference: nib.spatialimages.SpatialImage,
    *,
    is_labels: bool = False,
) -> nib.Nifti1Image:
    order = 0 if is_labels else 1
    out = resample_from_to(moving, reference, order=order)
    data = out.get_fdata()
    if is_labels:
        data = np.asarray(np.rint(data), dtype=np.int32)
    else:
        data = np.asarray(data, dtype=np.float32)
    return nib.Nifti1Image(data, out.affine)


def _build_isotropic_reference(
    img: nib.spatialimages.SpatialImage,
    *,
    voxel_size_mm: float,
) -> nib.Nifti1Image:
    """
    Build a same-space reference image at isotropic voxel size.
    """
    shape = np.asarray(img.shape[:3], dtype=np.float64)
    affine = np.asarray(img.affine, dtype=np.float64)
    zooms = np.asarray(nib.affines.voxel_sizes(affine)[:3], dtype=np.float64)
    target_zooms = np.asarray([voxel_size_mm, voxel_size_mm, voxel_size_mm], dtype=np.float64)

    # Preserve physical extent approximately: n' = floor((n-1)*z/z') + 1
    extent = np.maximum(shape - 1.0, 0.0) * zooms
    new_shape = np.floor(extent / target_zooms + 1.0).astype(np.int64)
    new_shape = np.maximum(new_shape, 1)

    # Keep axis directions; only change scale and translation so center is preserved.
    dirs = affine[:3, :3] / zooms[np.newaxis, :]
    new_affine = np.eye(4, dtype=np.float64)
    new_affine[:3, :3] = dirs * target_zooms[np.newaxis, :]

    old_center_vox = (shape - 1.0) / 2.0
    new_center_vox = (new_shape.astype(np.float64) - 1.0) / 2.0
    old_center_world = nib.affines.apply_affine(affine, old_center_vox)
    new_affine[:3, 3] = old_center_world - new_affine[:3, :3] @ new_center_vox

    ref_data = np.zeros(tuple(int(x) for x in new_shape), dtype=np.float32)
    return nib.Nifti1Image(ref_data, new_affine)


def _center_crop_or_pad_3d(
    arr: np.ndarray,
    target_shape: tuple[int, int, int],
    *,
    pad_value: float | int,
    safe: bool = True,
    affine: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Center crop/pad a 3D array.

    Returns
    -------
    out : np.ndarray
        Output array with `target_shape`.
    delta : np.ndarray
        Voxel offset mapping output indices to input indices: in = out + delta.
    safe : bool
        If True, the image is resample to MNI space.
        The old default lead to cropping out valid brain regions for images that
        were not well centered or irregularly shapeed Future should use safe=True,
        however resample_to_img it very slow at scale.
    """
    if safe:
        # ignore shape, uses MNI 1mm shape, prevents cropping out valid regions
        out = resample_img(nib.Nifti1Image(arr, affine), MNI_AFFINE, MNI_SHAPE)
        return out, np.zeros(3, dtype=np.float64)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

    target_shape = tuple(int(x) for x in target_shape)
    out = np.full(target_shape, pad_value, dtype=arr.dtype)
    delta = np.zeros(3, dtype=np.float64)

    in_slices: list[slice] = []
    out_slices: list[slice] = []
    for axis, out_len in enumerate(target_shape):
        in_len = int(arr.shape[axis])
        if in_len >= out_len:
            in_start = (in_len - out_len) // 2
            in_end = in_start + out_len
            out_start = 0
            out_end = out_len
        else:
            in_start = 0
            in_end = in_len
            out_start = (out_len - in_len) // 2
            out_end = out_start + in_len

        delta[axis] = float(in_start - out_start)
        in_slices.append(slice(in_start, in_end))
        out_slices.append(slice(out_start, out_end))

    out[tuple(out_slices)] = arr[tuple(in_slices)]
    return out, delta


def _shift_affine_for_voxel_offset(affine: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """
    If output voxel j maps to input voxel i = j + delta, adjust affine accordingly.
    """
    out = np.asarray(affine, dtype=np.float64).copy()
    out[:3, 3] = out[:3, :3] @ np.asarray(delta, dtype=np.float64) + out[:3, 3]
    return out


def _resample_to_1mm_same_space(
    img: nib.spatialimages.SpatialImage,
    *,
    is_labels: bool,
) -> nib.Nifti1Image:
    ref_1mm = _build_isotropic_reference(img, voxel_size_mm=TARGET_VOXEL_SIZE_MM)
    if _same_grid_img(img, ref_1mm):
        if is_labels:
            data = np.asarray(np.rint(img.get_fdata()), dtype=np.int32)
        else:
            data = img.get_fdata(dtype=np.float32)
        return nib.Nifti1Image(data, img.affine)

    return _resample_image_to_reference(img, ref_1mm, is_labels=is_labels)


def prepare_images_if_needed(
    orig_path: str | Path,
    aparc_path: str | Path,
    safe: bool = False
) -> Tuple[nib.Nifti1Image, nib.Nifti1Image]:
    """
    Return orig and aparc+aseg as Nifti1Image with:
    - shared voxel grid between image and labels
    - 1mm isotropic spacing
    - fixed output shape (center crop/pad)

    Steps:
    1) Reorient both to RAS (no registration).
    2) Align orig to aparc native grid if needed.
    3) Resample both to 1mm isotropic in that same native space.
    4) Center crop/pad to TARGET_SHAPE for uniform tensor size.
    """
    orig_img = _as_ras(load_mgz(orig_path))
    aparc_img = _as_ras(load_mgz(aparc_path))

    if _same_grid_img(orig_img, aparc_img):
        orig_native = nib.Nifti1Image(orig_img.get_fdata(dtype=np.float32), orig_img.affine)
        aparc_native = nib.Nifti1Image(
            np.asarray(np.rint(aparc_img.get_fdata()), dtype=np.int32),
            aparc_img.affine,
        )
    else:
        orig_native = _resample_image_to_reference(orig_img, aparc_img, is_labels=False)
        aparc_native = nib.Nifti1Image(
            np.asarray(np.rint(aparc_img.get_fdata()), dtype=np.int32),
            aparc_img.affine,
        )

    orig_1mm = _resample_to_1mm_same_space(orig_native, is_labels=False)
    aparc_1mm = _resample_to_1mm_same_space(aparc_native, is_labels=True)

    orig_data = orig_1mm.get_fdata(dtype=np.float32)
    aparc_data = np.asarray(aparc_1mm.dataobj, dtype=np.int32)

    orig_fixed, delta = _center_crop_or_pad_3d(
        orig_data,
        TARGET_SHAPE,
        pad_value=0.0,
        affine=orig_1mm.affine,
        safe=safe
    )
    aparc_fixed, _ = _center_crop_or_pad_3d(
        aparc_data,
        TARGET_SHAPE,
        pad_value=0,
        affine=aparc_1mm.affine,
        safe=safe
    )
    out_affine = _shift_affine_for_voxel_offset(orig_1mm.affine, delta)

    return (
        nib.Nifti1Image(orig_fixed, out_affine),
        nib.Nifti1Image(aparc_fixed, out_affine),
    )


def debug_prepare_images_report(
    orig_path: str | Path,
    aparc_path: str | Path
) -> dict[str, Any]:
    """
    Lightweight geometry sanity report for debugging conversion quality.
    """
    orig_orig = _as_ras(load_mgz(orig_path))
    aparc_orig = _as_ras(load_mgz(aparc_path))
    orig_out, aparc_out = prepare_images_if_needed(orig_path, aparc_path)

    orig_arr = orig_out.get_fdata(dtype=np.float32)
    aparc_arr = np.asarray(aparc_out.dataobj, dtype=np.int32)
    label_mask = aparc_arr > 0
    image_mask = orig_arr != 0
    overlap_ratio = float((label_mask & image_mask).sum() / max(1, int(label_mask.sum())))

    out_zooms = tuple(float(x) for x in nib.affines.voxel_sizes(orig_out.affine)[:3])
    orig_orig_zooms = tuple(float(x) for x in nib.affines.voxel_sizes(orig_orig.affine)[:3])
    orig_aparc_zooms = tuple(float(x) for x in nib.affines.voxel_sizes(aparc_orig.affine)[:3])

    return {
        "orig_orig_shape": tuple(int(x) for x in orig_orig.shape),
        "orig_aparc_shape": tuple(int(x) for x in aparc_orig.shape),
        "orig_orig_axcodes": tuple(nib.aff2axcodes(orig_orig.affine)),
        "orig_aparc_axcodes": tuple(nib.aff2axcodes(aparc_orig.affine)),
        "orig_orig_voxel_sizes": orig_orig_zooms,
        "orig_aparc_voxel_sizes": orig_aparc_zooms,
        "target_voxel_size_mm": (TARGET_VOXEL_SIZE_MM, TARGET_VOXEL_SIZE_MM, TARGET_VOXEL_SIZE_MM),
        "target_shape": TARGET_SHAPE,
        "out_orig_shape": tuple(int(x) for x in orig_out.shape),
        "out_aparc_shape": tuple(int(x) for x in aparc_out.shape),
        "out_orig_axcodes": tuple(nib.aff2axcodes(orig_out.affine)),
        "out_aparc_axcodes": tuple(nib.aff2axcodes(aparc_out.affine)),
        "out_voxel_sizes": out_zooms,
        "out_same_grid": bool(
            orig_out.shape == aparc_out.shape
            and np.allclose(orig_out.affine, aparc_out.affine)
        ),
        "out_is_1mm": bool(
            np.allclose(
                np.asarray(out_zooms),
                np.asarray([TARGET_VOXEL_SIZE_MM, TARGET_VOXEL_SIZE_MM, TARGET_VOXEL_SIZE_MM]),
                atol=1e-3,
            )
        ),
        "label_on_nonzero_image_overlap": overlap_ratio,
    }


def prepare_arrays_if_needed(
    orig_path: str | Path,
    aparc_path: str | Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load both images, co-register to shared native grid, resample to 1mm isotropic,
    then center crop/pad to fixed TARGET_SHAPE and return arrays.
    """
    orig_out, aparc_out = prepare_images_if_needed(orig_path, aparc_path)
    orig = orig_out.get_fdata(dtype=np.float32)
    aparc = np.asarray(aparc_out.dataobj, dtype=np.int32)
    return orig, aparc


def _safe_output_subdir(base_path: str | Path, out_root: str | Path) -> Path:
    """
    Create a stable output directory under `out_root` from a FreeSurfer mri base path.
    """
    base_path = Path(base_path)
    out_root = Path(out_root)

    parts = list(base_path.parts)
    if "openneuro_cache" in parts and "files" in parts:
        i = parts.index("files")
        rel = Path(*parts[i + 1 :])
    else:
        rel = Path(*parts[-5:])

    out_dir = out_root / rel
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _pt_exists(path: Path) -> bool:
    """Treat empty files as non-existent (common after interrupted writes)."""
    return path.exists() and path.stat().st_size > 0


def _process_one_entry(
    base_path: str,
    paths: dict[str, str],
    out_root: str | Path,
    unsafe_int8: bool = False,
) -> dict[str, Any]:
    """
    Run prepare_arrays_if_needed(orig, aparc+aseg), then save:
      - orig.pt as float32
      - aparc+aseg.pt as int16 by default, or int8 if unsafe_int8=True

    Robustness behavior:
    - Skip writing files that already exist.
    - On conversion/save error, print the failing file/path and continue.
    """
    if "orig" not in paths or "aparc+aseg" not in paths:
        return {
            "base_path": base_path,
            "ok": False,
            "error": "missing required keys: 'orig' and/or 'aparc+aseg'",
        }

    out_dir = _safe_output_subdir(base_path, out_root)
    orig_out = out_dir / "orig.pt"
    aparc_out = out_dir / "aparc+aseg.pt"

    orig_exists = _pt_exists(orig_out)
    aparc_exists = _pt_exists(aparc_out)
    if orig_exists and aparc_exists:
        return {
            "base_path": base_path,
            "ok": True,
            "skipped_existing": True,
            "orig_out": str(orig_out),
            "aparc_out": str(aparc_out),
        }

    orig_path = paths["orig"]
    aparc_path = paths["aparc+aseg"]

    try:
        orig_arr, aparc_arr = prepare_arrays_if_needed(orig_path, aparc_path)
    except Exception as e:
        print(f"[convert-error] {orig_path}", flush=True)
        print(f"[convert-error] {aparc_path}", flush=True)
        return {
            "base_path": base_path,
            "ok": False,
            "error": str(e),
            "orig_out": str(orig_out),
            "aparc_out": str(aparc_out),
        }

    orig_tensor = torch.from_numpy(np.asarray(orig_arr, dtype=np.float32))
    if unsafe_int8:
        aparc_tensor = torch.from_numpy(np.asarray(aparc_arr, dtype=np.int8))
        aparc_dtype = "int8"
    else:
        aparc_tensor = torch.from_numpy(np.asarray(aparc_arr, dtype=np.int16))
        aparc_dtype = "int16"

    save_errors: list[str] = []

    if not orig_exists:
        try:
            torch.save(orig_tensor, orig_out)
        except Exception as e:
            print(f"[save-error] {orig_out}", flush=True)
            if orig_out.exists():
                try:
                    orig_out.unlink()
                except Exception:
                    pass
            save_errors.append(f"{orig_out}: {e}")

    if not aparc_exists:
        try:
            torch.save(aparc_tensor, aparc_out)
        except Exception as e:
            print(f"[save-error] {aparc_out}", flush=True)
            if aparc_out.exists():
                try:
                    aparc_out.unlink()
                except Exception:
                    pass
            save_errors.append(f"{aparc_out}: {e}")

    if save_errors:
        return {
            "base_path": base_path,
            "ok": False,
            "error": " | ".join(save_errors),
            "orig_out": str(orig_out),
            "aparc_out": str(aparc_out),
            "aparc_dtype": aparc_dtype,
        }

    return {
        "base_path": base_path,
        "ok": True,
        "orig_out": str(orig_out),
        "aparc_out": str(aparc_out),
        "orig_shape": tuple(orig_tensor.shape),
        "aparc_shape": tuple(aparc_tensor.shape),
        "aparc_dtype": aparc_dtype,
        "skipped_existing": False,
    }


def _process_one_entry_star(args):
    base_path, paths, out_root, unsafe_int8 = args
    return _process_one_entry(
        base_path=base_path,
        paths=paths,
        out_root=out_root,
        unsafe_int8=unsafe_int8,
    )


def convert_file_map_to_pt(
    file_map: dict[str, dict[str, str]],
    out_root: str | Path = "data",
    n_jobs: int = -1,
    unsafe_int8: bool = False,
) -> list[dict[str, Any]]:
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    items = [
        (base_path, paths, out_root, unsafe_int8)
        for base_path, paths in file_map.items()
    ]

    max_workers = None if n_jobs == -1 else n_jobs
    worker_count = os.cpu_count() if max_workers is None else max_workers
    chunksize = max(1, len(items) // (max(1, worker_count) * 4))

    results = process_map(
        _process_one_entry_star,
        items,
        max_workers=max_workers,
        desc="Converting",
        chunksize=chunksize,
    )
    return results


def mri_convert_available() -> bool:
    """Return True when FreeSurfer's mri_convert is on PATH."""
    return shutil.which("mri_convert") is not None


def _run_mri_convert_binary(img: str | Path, orig: str | Path, extra_args: str = "--conform") -> None:
    args = ["mri_convert", str(img), str(orig), *shlex.split(extra_args)]
    subprocess.run(
        args,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _scale_conformed_to_uint8(data: np.ndarray) -> np.ndarray:
    """Approximate mri_convert --conform's uchar intensity conversion."""
    data = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(data)
    if not bool(finite.any()):
        return np.zeros(data.shape, dtype=np.uint8)

    lower = float(data[finite].min())
    nonzero_finite = finite & (data != 0)
    if bool(nonzero_finite.any()):
        upper = float(np.quantile(data[nonzero_finite], CONFORM_UCHAR_HIGH_FRACTION))
    else:
        upper = float(data[finite].max())

    if upper <= lower:
        return np.zeros(data.shape, dtype=np.uint8)

    scaled = (data - lower) * (255.0 / (upper - lower))
    scaled = np.where(finite, scaled, 0.0)
    scaled = np.clip(np.rint(scaled), 0.0, 255.0)
    scaled[(data == 0) | (~finite)] = 0.0
    return scaled.astype(np.uint8)


def _zscore_volume_for_model(volume: torch.Tensor) -> torch.Tensor:
    """Match the per-volume z-score used by the training cache."""
    volume = volume.to(torch.float32)
    return (volume - volume.mean()) / volume.std().clamp_min(1e-6)


def run_nibabel_mri_convert_conform(img: str | Path, orig: str | Path) -> None:
    """NiBabel replacement for the mri_convert --conform use in inference."""
    in_img = load_mgz(img)
    if len(in_img.shape) > 3:
        data = np.asanyarray(in_img.dataobj)[..., 0]
        in_img = nib.Nifti1Image(data, in_img.affine)

    conformed = conform(
        in_img,
        out_shape=CONFORM_SHAPE,
        voxel_size=(TARGET_VOXEL_SIZE_MM, TARGET_VOXEL_SIZE_MM, TARGET_VOXEL_SIZE_MM),
        order=1,
        orientation="RAS",
    )
    data = _scale_conformed_to_uint8(conformed.get_fdata(dtype=np.float32))
    out_img = nib.MGHImage(data, conformed.affine)
    orig = Path(orig)
    orig.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(orig))


def run_mri_convert(
    img: str | Path,
    orig: str | Path,
    extra_args: str = "--conform",
    *,
    backend: str = "auto",
) -> str:
    """Conform an image, using FreeSurfer when available or a NiBabel fallback."""
    backend = str(backend).strip().lower()
    if backend not in {"auto", "freesurfer", "nibabel"}:
        raise ValueError("backend must be 'auto', 'freesurfer', or 'nibabel'")

    if backend == "auto":
        backend = "freesurfer" if mri_convert_available() else "nibabel"

    if backend == "freesurfer":
        _run_mri_convert_binary(img, orig, extra_args=extra_args)
        return backend

    if extra_args.strip() not in {"--conform", "-c"}:
        raise ValueError("NiBabel mri_convert fallback only supports --conform")
    run_nibabel_mri_convert_conform(img, orig)
    return backend


def prepare_image(
    img_path: str | Path,
    out_path: str | Path,
    *,
    conform_backend: str = "auto",
) -> torch.Tensor:
    """
    Return a conformed, z-scored float32 tensor ready for model inference.

    The saved MGZ stays in conformed intensity space; only the returned tensor
    is z-scored to match the training cache.
    """
    run_mri_convert(img_path, out_path, backend=conform_backend)
    img = _as_ras(load_mgz(out_path))
    data = img.get_fdata(dtype=np.float32)
    return _zscore_volume_for_model(torch.from_numpy(data)).contiguous()

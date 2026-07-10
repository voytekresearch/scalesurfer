from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.ndimage as ndi
import torch


NORM_CONTROL_LABELS = (
    2,   # Left-Cerebral-White-Matter
    41,  # Right-Cerebral-White-Matter
    7,   # Left-Cerebellum-White-Matter
    46,  # Right-Cerebellum-White-Matter
    16,  # Brain-Stem
    28,  # Left-VentralDC
    60,  # Right-VentralDC
)
DEFAULT_TARGET_WM_INTENSITY = 110.0


@dataclass(frozen=True)
class DeterministicNormResult:
    tensor: torch.Tensor
    stats: dict[str, float | int]


def _as_float_array(array: np.ndarray | torch.Tensor) -> np.ndarray:
    if torch.is_tensor(array):
        return np.asarray(array.detach().cpu(), dtype=np.float32)
    return np.asarray(array, dtype=np.float32)


def _as_int_array(array: np.ndarray | torch.Tensor) -> np.ndarray:
    if torch.is_tensor(array):
        return np.asarray(array.detach().cpu(), dtype=np.int32)
    return np.asarray(array, dtype=np.int32)


def _segmentation_control_mask(
    seg: np.ndarray,
    brainmask: np.ndarray,
    *,
    erosion_iters: int,
) -> np.ndarray:
    control = np.isin(seg, NORM_CONTROL_LABELS)
    control &= brainmask
    if erosion_iters > 0 and np.any(control):
        eroded = ndi.binary_erosion(control, iterations=int(erosion_iters))
        if np.count_nonzero(eroded) >= 64:
            control = eroded
    opened = ndi.binary_opening(control)
    if np.count_nonzero(opened) >= 64:
        control = opened
    return control


def build_deterministic_norm(
    rawavg: np.ndarray | torch.Tensor,
    *,
    brainmask: np.ndarray | torch.Tensor,
    aseg_presurf: np.ndarray | torch.Tensor | None = None,
    aparc_aseg: np.ndarray | torch.Tensor | None = None,
    target_wm_intensity: float = DEFAULT_TARGET_WM_INTENSITY,
    erosion_iters: int = 2,
    sigma_large: float = 8.0,
    sigma_small: float = 1.5,
    min_bias: float = 0.20,
    max_bias: float = 2.00,
    fallback_percentile: float = 85.0,
) -> DeterministicNormResult:
    raw = _as_float_array(rawavg)
    brain = _as_int_array(brainmask) > 0
    if not np.any(brain):
        raise ValueError("brainmask is empty; cannot build deterministic norm volume")

    seg = None
    if aseg_presurf is not None:
        seg = _as_int_array(aseg_presurf)
    elif aparc_aseg is not None:
        seg = _as_int_array(aparc_aseg)
    if seg is None:
        raise ValueError("Need either aseg_presurf or aparc_aseg to build deterministic norm volume")
    if raw.shape != brain.shape or raw.shape != seg.shape:
        raise ValueError(
            "rawavg, brainmask, and segmentation must share a shape; "
            f"got rawavg={tuple(raw.shape)} brainmask={tuple(brain.shape)} seg={tuple(seg.shape)}"
        )

    control = _segmentation_control_mask(seg, brain, erosion_iters=erosion_iters)
    valid = control & np.isfinite(raw) & (raw > 1e-3)

    if np.count_nonzero(valid) < 64:
        in_brain = raw[brain & np.isfinite(raw)]
        if in_brain.size == 0:
            raise ValueError("rawavg has no finite in-brain voxels; cannot build deterministic norm volume")
        cutoff = float(np.percentile(in_brain, float(fallback_percentile)))
        valid = brain & np.isfinite(raw) & (raw >= cutoff) & (raw > 1e-3)

    samples = raw[valid]
    if samples.size == 0:
        raise ValueError("No usable control voxels found for deterministic norm volume")

    sample_bias = np.clip(float(target_wm_intensity) / samples, float(min_bias), float(max_bias))
    sample_log_bias = np.log(sample_bias).astype(np.float32)

    support = np.zeros_like(raw, dtype=np.float32)
    support[valid] = 1.0
    log_bias = np.zeros_like(raw, dtype=np.float32)
    log_bias[valid] = sample_log_bias

    num = ndi.gaussian_filter(log_bias * support, sigma=float(sigma_large), mode="nearest")
    den = ndi.gaussian_filter(support, sigma=float(sigma_large), mode="nearest")
    global_log_bias = float(np.median(sample_log_bias))
    field_log = np.where(den > 1e-5, num / np.maximum(den, 1e-5), global_log_bias)
    field_log = ndi.gaussian_filter(field_log, sigma=float(sigma_small), mode="nearest")
    field = np.clip(np.exp(field_log), float(min_bias), float(max_bias)).astype(np.float32)

    norm = (raw * field).astype(np.float32)
    control_after = norm[valid]
    if control_after.size > 0:
        wm_scale = float(target_wm_intensity) / float(np.median(control_after))
        wm_scale = float(np.clip(wm_scale, 0.50, 1.50))
        norm *= wm_scale
        field *= wm_scale

    norm *= brain.astype(np.float32)

    stats = {
        "brain_voxels": int(np.count_nonzero(brain)),
        "control_voxels": int(np.count_nonzero(valid)),
        "control_raw_median": float(np.median(samples)),
        "control_norm_median": float(np.median(norm[valid])),
        "field_mean": float(np.mean(field[brain])),
        "field_std": float(np.std(field[brain])),
        "field_min": float(np.min(field[brain])),
        "field_max": float(np.max(field[brain])),
    }
    return DeterministicNormResult(
        tensor=torch.as_tensor(norm, dtype=torch.float32),
        stats=stats,
    )


__all__ = [
    "DEFAULT_TARGET_WM_INTENSITY",
    "DeterministicNormResult",
    "NORM_CONTROL_LABELS",
    "build_deterministic_norm",
]

import hashlib
import os
import pickle
import re
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt
from tqdm.auto import tqdm

from .config import DEVICE, MODULE_PATH
from .data import load_freesurfer_lut


_CLASS_PLACEHOLDER_RE = re.compile(r"^class_\d+$")
_REQUIRED_FS_LUT_PATH = MODULE_PATH.parent / "FreeSurferColorLUT.txt"


def _clean_region_name(value):
    if pd.isna(value):
        return None
    name = str(value).strip()
    return name if name else None

def _is_placeholder_region_name(value):
    name = _clean_region_name(value)
    return bool(name is not None and _CLASS_PLACEHOLDER_RE.fullmatch(name))

@lru_cache(maxsize=1)
def _load_required_freesurfer_region_df():
    lut_path = Path(_REQUIRED_FS_LUT_PATH)
    if not lut_path.exists():
        raise FileNotFoundError(
            f"Required FreeSurfer LUT not found at {lut_path}. "
            "Region names must be resolved from FreeSurferColorLUT.txt."
        )

    region_df = load_freesurfer_lut(lut_path)
    if region_df.empty:
        raise ValueError(f"FreeSurfer LUT at {lut_path} did not produce any rows")
    if "fs_id" not in region_df.columns or "fs_name" not in region_df.columns:
        raise ValueError(f"FreeSurfer LUT at {lut_path} must contain fs_id and fs_name columns")

    invalid = region_df["fs_name"].map(lambda v: _clean_region_name(v) is None or _is_placeholder_region_name(v))
    if bool(invalid.any()):
        bad_ids = region_df.loc[invalid, "fs_id"].astype(int).tolist()[:10]
        raise ValueError(
            "FreeSurfer LUT contains invalid region names for fs_id values: "
            + ", ".join(str(v) for v in bad_ids)
        )

    return region_df.copy()

def _required_region_name_map(class_values):
    classes = torch.as_tensor(class_values, dtype=torch.int64).flatten()
    region_df = _load_required_freesurfer_region_df()
    fs_name_by_id = {
        int(fs_id): str(fs_name).strip()
        for fs_id, fs_name in region_df[["fs_id", "fs_name"]].itertuples(index=False)
        if pd.notna(fs_id) and _clean_region_name(fs_name) is not None and not _is_placeholder_region_name(fs_name)
    }

    name_by_y = {}
    missing = []
    for y_val, fs_id in enumerate(classes.tolist()):
        if int(y_val) <= 0:
            continue
        name = fs_name_by_id.get(int(fs_id))
        if _clean_region_name(name) is None:
            missing.append((int(y_val), int(fs_id)))
            continue
        name_by_y[int(y_val)] = str(name)

    if missing:
        preview = ", ".join(f"class_idx={y}/fs_id={fs_id}" for y, fs_id in missing[:10])
        raise ValueError(
            "Could not resolve FreeSurfer region names for all classes from "
            f"{_REQUIRED_FS_LUT_PATH}. Missing: {preview}"
        )

    return name_by_y

def _region_metrics_df_has_required_names(region_metrics_df, class_values, require_sample_idx=False):
    if region_metrics_df is None or len(region_metrics_df) == 0:
        return True
    if "class_idx" not in region_metrics_df.columns or "region_name" not in region_metrics_df.columns:
        return False
    if bool(require_sample_idx):
        if "sample_idx" not in region_metrics_df.columns:
            return False
        if bool(region_metrics_df["sample_idx"].isna().any()):
            return False

    expected = _required_region_name_map(class_values)
    present = set()
    for class_idx, region_name in region_metrics_df[["class_idx", "region_name"]].itertuples(index=False):
        if pd.isna(class_idx):
            return False
        class_idx = int(class_idx)
        if class_idx <= 0:
            continue
        expected_name = expected.get(class_idx)
        actual_name = _clean_region_name(region_name)
        if expected_name is None or actual_name != expected_name or _is_placeholder_region_name(actual_name):
            return False
        present.add(class_idx)
    return present.issuperset(expected.keys())

def _assert_region_metrics_df_has_resolved_names(region_metrics_df, context):
    if region_metrics_df is None or len(region_metrics_df) == 0:
        return
    if "region_name" not in region_metrics_df.columns:
        raise ValueError(f"{context} is missing the region_name column")
    invalid = region_metrics_df["region_name"].map(lambda v: _clean_region_name(v) is None or _is_placeholder_region_name(v))
    if bool(invalid.any()):
        bad = region_metrics_df.loc[invalid, ["class_idx", "region_name"]].head(10).to_dict("records")
        raise ValueError(
            f"{context} contains unresolved region names. "
            "FreeSurferColorLUT.txt must resolve every region. "
            f"Examples: {bad}"
        )

def _region_metrics_rows_from_confusion(
    cm_total,
    method,
    region_name_by_class,
    class_to_fs=None,
    sample_idx=None,
    domain=None,
    x_path=None,
):
    cm = cm_total.to(torch.float64)
    true_vol = cm.sum(dim=1)
    pred_vol = cm.sum(dim=0)
    inter = torch.diag(cm)
    den = true_vol + pred_vol

    dice = torch.full((cm.shape[0],), float("nan"), dtype=torch.float64)
    valid = den > 0
    if valid.any():
        dice[valid] = (2.0 * inter[valid]) / den[valid]

    rows = []
    for c in range(1, int(cm.shape[0])):
        region_name = region_name_by_class.get(int(c))
        if _clean_region_name(region_name) is None or _is_placeholder_region_name(region_name):
            raise ValueError(
                f"Missing valid FreeSurfer region name for class_idx={int(c)}. "
                "Region names must come from FreeSurferColorLUT.txt."
            )

        row = {
            "method": str(method),
            "class_idx": int(c),
            "region_name": region_name,
            "dice": float(dice[c].item()) if torch.isfinite(dice[c]) else np.nan,
            "vol_true": int(true_vol[c].item()),
            "vol_pred": int(pred_vol[c].item()),
        }
        if class_to_fs is not None:
            row["fs_id"] = int(class_to_fs[int(c)])
        if sample_idx is not None:
            row["sample_idx"] = int(sample_idx)
        if domain is not None:
            row["domain"] = str(domain)
        if x_path is not None:
            row["x_path"] = str(x_path)
        rows.append(row)
    return rows


def _region_name_map_from_region_df(class_values, region_df):
    del region_df
    return _required_region_name_map(class_values)


def _region_name_cache_tag(class_values, region_df):
    name_by_y = _region_name_map_from_region_df(class_values=class_values, region_df=region_df)
    if not name_by_y:
        return "none"
    sig = "|".join(f"{int(y)}={name_by_y[y]}" for y in sorted(name_by_y))
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]


def _sync_cuda(device=DEVICE):
    if str(device) == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def predict_volume_forward_dense(model, x: torch.Tensor) -> torch.Tensor:
    """
    Dense forward-style head path (features -> LN -> 1x1 conv classifier -> argmax).
    Used as fallback when model.forward is not implemented for dense logits.
    """
    feat = model.forward_features(x)

    mean = feat.mean(dim=1, keepdim=True)
    var = feat.var(dim=1, keepdim=True, unbiased=False)
    feat = (feat - mean) * torch.rsqrt(var + model.token_norm.eps)

    if model.token_norm.elementwise_affine:
        w_ln = model.token_norm.weight.view(1, -1, 1, 1, 1)
        b_ln = model.token_norm.bias.view(1, -1, 1, 1, 1)
        feat = feat * w_ln + b_ln

    w = model.classifier.weight[:, :, None, None, None]
    b = model.classifier.bias
    logits = F.conv3d(feat, w, b)
    return logits.argmax(dim=1)


@torch.inference_mode()
def predict_volume_forward_default(model, x: torch.Tensor) -> torch.Tensor:
    """
    Try the model's default forward path first.
    If unavailable for dense segmentation output, fall back to dense head path.
    """
    if type(model).forward is not torch.nn.Module.forward:
        out = model(x)
        if torch.is_tensor(out):
            if out.ndim == 5:  # [B, C, D, H, W] logits
                return out.argmax(dim=1)
            if out.ndim == 4:  # [B, D, H, W] labels
                return out.long()
        raise ValueError(f"Unsupported default forward output shape/type: {type(out)}")
    return predict_volume_forward_dense(model, x)


@torch.inference_mode()
def benchmark_inference_speed(
    model,
    loader,
    max_batches=16,
    patch_chunk_size=96,
    warmup=1,
    device=DEVICE,
):
    if loader is None:
        raise ValueError("pipeline.test_loader is None")

    methods = {
        "forward_default": lambda m, xx: predict_volume_forward_default(m, xx),
        "predict_volume": lambda m, xx: m.predict_volume(xx, patch_chunk_size=patch_chunk_size),
    }

    model.eval()
    rows = []
    n_total = len(loader) if max_batches is None else min(len(loader), int(max_batches))
    pbar = tqdm(loader, total=n_total, desc="test inference timing", leave=False)

    for b_idx, (x_cpu, _y_cpu) in enumerate(pbar):
        if max_batches is not None and b_idx >= int(max_batches):
            break

        x = x_cpu.to(device, non_blocking=True).unsqueeze(1).float()
        voxels = int(np.prod(x.shape[2:])) * int(x.shape[0])

        for method_name, fn in methods.items():
            for _ in range(int(warmup)):
                _ = fn(model, x)
                _sync_cuda(device=device)

            t0 = perf_counter()
            _ = fn(model, x)
            _sync_cuda(device=device)
            dt = perf_counter() - t0

            rows.append(
                {
                    "batch_idx": int(b_idx),
                    "method": method_name,
                    "batch_size": int(x.shape[0]),
                    "voxels": int(voxels),
                    "sec": float(dt),
                    "ms_per_mvox": float(dt * 1000.0 / max(1e-6, voxels / 1_000_000.0)),
                }
            )

    return pd.DataFrame(rows)


def _as_3d_float_tensor(x):
    xt = torch.as_tensor(x)
    if xt.ndim != 3:
        raise ValueError(f"Expected 3D tensor/array [D,H,W], got shape={tuple(xt.shape)}")
    return xt.to(torch.float32).contiguous()


def _get_dataset_sample_pair(ds, sample_idx):
    idx = int(sample_idx)
    if idx < 0:
        raise ValueError("sample_idx must be non-negative")

    if hasattr(ds, "dataset") and hasattr(ds, "indices"):
        base = ds.dataset
        base_idx = int(ds.indices[idx])
        x_t, y_t = base[base_idx]
    else:
        x_t, y_t = ds[idx]

    return x_t, y_t


@torch.inference_mode()
def predict_volume_from_unpadded(
    model,
    x_3d,
    patch_size,
    patch_chunk_size=96,
    device=DEVICE,
):
    """
    Predict labels for a single unbatched volume [D,H,W].

    The volume is padded to a patch-divisible shape using the same collate logic
    as training/evaluation, then cropped back to original shape.
    """
    from .data import collate_pad_to_patch

    x = _as_3d_float_tensor(x_3d)
    d, h, w = (int(v) for v in x.shape)

    dummy_y = torch.zeros((d, h, w), dtype=torch.int64)
    x_pad_b, _ = collate_pad_to_patch(
        [(x, dummy_y)],
        patch_size=tuple(int(v) for v in patch_size),
    )

    x_pad = x_pad_b.to(device, non_blocking=True).unsqueeze(1).float()
    y_pad = model.predict_volume(x_pad, patch_chunk_size=int(patch_chunk_size))[0].detach().to("cpu", dtype=torch.int64)
    return y_pad[:d, :h, :w].contiguous()


def dense_labels_to_fs_ids(y_dense, class_values):
    y = torch.as_tensor(y_dense, dtype=torch.int64)
    classes = torch.as_tensor(class_values, dtype=torch.int64)
    if y.ndim != 3:
        raise ValueError(f"y_dense must be 3D [D,H,W], got {tuple(y.shape)}")
    return classes[y].cpu().numpy().astype(np.int32)


def csf_fs_ids_from_class_values(class_values, region_df=None):
    tissue = infer_tissue_class_indices(
        class_values=class_values,
        region_df=region_df,
        exclude_label_24_from_tissue=False,
    )
    classes = torch.as_tensor(class_values, dtype=torch.int64)
    return np.asarray([int(classes[int(i)].item()) for i in tissue.get("CSF", [])], dtype=np.int32)


def build_csf_binary_mask(fs_label_volume, csf_fs_ids):
    fsv = np.asarray(fs_label_volume, dtype=np.int32)
    ids = np.asarray(csf_fs_ids, dtype=np.int32)
    return np.isin(fsv, ids).astype(np.uint8)


def prepare_napari_sample_from_test_dataset(
    model,
    pipeline,
    class_values,
    sample_idx=0,
    patch_size=None,
    patch_chunk_size=96,
    region_df=None,
    device=DEVICE,
):
    """
    Build aligned raw image / GT / prediction volumes for napari from test_ds.

    Returns tissue-level binary masks for GM/WM/CSF/FG on both GT and prediction.
    """
    if pipeline.test_ds is None:
        raise ValueError("pipeline.test_ds is None")

    x_t, y_t = _get_dataset_sample_pair(pipeline.test_ds, sample_idx)
    x = _as_3d_float_tensor(x_t).cpu()
    y_true_dense = torch.as_tensor(y_t, dtype=torch.int64).cpu().contiguous()

    if patch_size is None:
        patch_size = tuple(int(v) for v in getattr(model, "patch_size", (16, 16, 16)))
    else:
        patch_size = tuple(int(v) for v in patch_size)

    y_pred_dense = predict_volume_from_unpadded(
        model=model,
        x_3d=x,
        patch_size=patch_size,
        patch_chunk_size=patch_chunk_size,
        device=device,
    )

    y_true_fs = dense_labels_to_fs_ids(y_true_dense, class_values=class_values)
    y_pred_fs = dense_labels_to_fs_ids(y_pred_dense, class_values=class_values)
    tissue_idx = infer_tissue_class_indices(
        class_values=class_values,
        region_df=region_df,
        exclude_label_24_from_tissue=False,
    )
    class_values_t = torch.as_tensor(class_values, dtype=torch.int64)

    def _tissue_fs_ids(tissue_name):
        return np.asarray(
            [int(class_values_t[int(i)].item()) for i in tissue_idx.get(str(tissue_name), [])],
            dtype=np.int32,
        )

    csf_fs_ids = _tissue_fs_ids("CSF")
    gm_fs_ids = _tissue_fs_ids("GM")
    wm_fs_ids = _tissue_fs_ids("WM")
    fg_fs_ids = _tissue_fs_ids("FG")

    gt_csf_bin = build_csf_binary_mask(y_true_fs, csf_fs_ids)
    pred_csf_bin = build_csf_binary_mask(y_pred_fs, csf_fs_ids)
    gt_gm_bin = np.isin(y_true_fs, gm_fs_ids).astype(np.uint8)
    pred_gm_bin = np.isin(y_pred_fs, gm_fs_ids).astype(np.uint8)
    gt_wm_bin = np.isin(y_true_fs, wm_fs_ids).astype(np.uint8)
    pred_wm_bin = np.isin(y_pred_fs, wm_fs_ids).astype(np.uint8)
    gt_fg_bin = np.isin(y_true_fs, fg_fs_ids).astype(np.uint8)
    pred_fg_bin = np.isin(y_pred_fs, fg_fs_ids).astype(np.uint8)

    return {
        "sample_idx": int(sample_idx),
        "x": x.numpy().astype(np.float32),
        "y_true_dense": y_true_dense.numpy().astype(np.int32),
        "y_pred_dense": y_pred_dense.numpy().astype(np.int32),
        "y_true_fs": y_true_fs,
        "y_pred_fs": y_pred_fs,
        "tissue_fs_ids": {
            "CSF": csf_fs_ids,
            "GM": gm_fs_ids,
            "WM": wm_fs_ids,
            "FG": fg_fs_ids,
        },
        "csf_fs_ids": csf_fs_ids,
        "gm_fs_ids": gm_fs_ids,
        "wm_fs_ids": wm_fs_ids,
        "fg_fs_ids": fg_fs_ids,
        "gt_csf_bin": gt_csf_bin,
        "pred_csf_bin": pred_csf_bin,
        "gt_gm_bin": gt_gm_bin,
        "pred_gm_bin": pred_gm_bin,
        "gt_wm_bin": gt_wm_bin,
        "pred_wm_bin": pred_wm_bin,
        "gt_fg_bin": gt_fg_bin,
        "pred_fg_bin": pred_fg_bin,
    }


def _test_x_files(ds):
    if ds is None:
        return []
    if hasattr(ds, "x_files"):
        return list(ds.x_files)
    if hasattr(ds, "dataset") and hasattr(ds, "indices") and hasattr(ds.dataset, "x_files"):
        return [ds.dataset.x_files[i] for i in ds.indices]
    return ["unknown"] * len(ds)


def _domain_from_path(path):
    p = Path(str(path))
    parts = p.parts
    if "tensors" in parts:
        i = parts.index("tensors")
        if i + 1 < len(parts):
            return parts[i + 1]
    return p.parent.name if p.parent.name else "unknown"


def _safe_pearson_np(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _class_confusion_and_stats(y_true, y_pred, n_classes):
    """
    Fast per-sample class stats using a single bincount-based confusion matrix.
    Inputs: y_true/y_pred as integer tensors with same shape [D,H,W].
    Returns: confusion [C,C], dice [C], true_vol [C], pred_vol [C].
    """
    yt = y_true.to(torch.int64).reshape(-1)
    yp = y_pred.to(torch.int64).reshape(-1)

    idx = yt * int(n_classes) + yp
    cm = torch.bincount(idx, minlength=int(n_classes) * int(n_classes)).reshape(int(n_classes), int(n_classes))

    true_vol = cm.sum(dim=1)
    pred_vol = cm.sum(dim=0)
    inter = torch.diag(cm)
    den = true_vol + pred_vol

    dice = torch.full((int(n_classes),), float("nan"), dtype=torch.float32)
    valid = den > 0
    if valid.any():
        dice[valid] = (2.0 * inter[valid].to(torch.float32)) / den[valid].to(torch.float32)

    return cm, dice, true_vol, pred_vol


def _surface_voxels(mask):
    if not mask.any():
        return mask
    eroded = binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    return np.logical_xor(mask, eroded)


def _hd95_assd_foreground(pred_fg, true_fg, spacing=(1.0, 1.0, 1.0), downsample=4):
    """
    Boundary metrics on foreground union only.
    Downsampling reduces runtime substantially for large 3D volumes.
    """
    pred_fg = np.asarray(pred_fg, dtype=bool)
    true_fg = np.asarray(true_fg, dtype=bool)

    if int(downsample) > 1:
        ds = int(downsample)
        pred_fg = pred_fg[::ds, ::ds, ::ds]
        true_fg = true_fg[::ds, ::ds, ::ds]
        spacing = tuple(float(s) * ds for s in spacing)

    if (not pred_fg.any()) and (not true_fg.any()):
        return np.nan, np.nan
    if (not pred_fg.any()) or (not true_fg.any()):
        return np.inf, np.inf

    pred_s = _surface_voxels(pred_fg)
    true_s = _surface_voxels(true_fg)
    if (not pred_s.any()) or (not true_s.any()):
        return np.inf, np.inf

    dt_true = distance_transform_edt(~true_s, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_s, sampling=spacing)

    d_pred_to_true = dt_true[pred_s]
    d_true_to_pred = dt_pred[true_s]

    all_d = np.concatenate([d_pred_to_true, d_true_to_pred])
    hd95 = float(np.percentile(all_d, 95))
    assd = float((d_pred_to_true.mean() + d_true_to_pred.mean()) / 2.0)
    return hd95, assd


def _aggregate_region_metrics_from_confusion(cm_total, method, region_name_by_class):
    return _region_metrics_rows_from_confusion(
        cm_total=cm_total,
        method=method,
        region_name_by_class=region_name_by_class,
    )


def _sum_confusions(confusion_by_sample, sample_ids, n_classes):
    total = torch.zeros((int(n_classes), int(n_classes)), dtype=torch.int64)
    for sid in sample_ids:
        cm = confusion_by_sample.get(int(sid))
        if cm is not None:
            total += cm
    return total


def _select_failed_sample_ids(
    sample_metrics_df,
    threshold,
    metric_col="mean_dice_fg",
    method="model",
    mode="lt",
):
    if threshold is None or len(sample_metrics_df) == 0:
        return set()

    mode = str(mode).strip().lower()
    if mode not in {"lt", "le", "gt", "ge"}:
        raise ValueError("failure mode must be one of: 'lt', 'le', 'gt', 'ge'")

    sub = sample_metrics_df[sample_metrics_df["method"] == str(method)].copy()
    if sub.empty:
        return set()

    vals = pd.to_numeric(sub[metric_col], errors="coerce")
    thr = float(threshold)

    if mode == "lt":
        mask = vals < thr
    elif mode == "le":
        mask = vals <= thr
    elif mode == "gt":
        mask = vals > thr
    else:
        mask = vals >= thr

    return set(int(v) for v in sub.loc[mask.fillna(False), "sample_idx"].tolist())


def filter_sparse_regions(
    region_metrics_df,
    method="model",
    min_true_voxels=None,
    min_true_quantile=0.05,
):
    """
    Split region metrics into kept and sparse subsets based on true voxel count.

    If `min_true_voxels` is None, a data-driven threshold is computed from the
    specified method's `vol_true` distribution using `min_true_quantile`.
    """
    if region_metrics_df is None or len(region_metrics_df) == 0:
        empty = pd.DataFrame(columns=["method", "class_idx", "region_name", "dice", "vol_true", "vol_pred"])
        return empty.copy(), empty.copy(), int(min_true_voxels or 0)

    df = region_metrics_df.copy()
    vol_true = pd.to_numeric(df["vol_true"], errors="coerce").fillna(0.0)
    df["vol_true"] = vol_true

    ref = df[df["method"] == str(method)]["vol_true"].to_numpy(dtype=np.float64)
    if ref.size == 0:
        ref = df["vol_true"].to_numpy(dtype=np.float64)

    if min_true_voxels is None:
        q = float(np.clip(float(min_true_quantile), 0.0, 1.0))
        min_true_voxels = int(np.nanquantile(ref, q)) if ref.size else 0

    thr = int(max(0, int(min_true_voxels)))
    keep_mask = df["vol_true"] >= thr

    kept = df.loc[keep_mask].reset_index(drop=True)
    sparse = df.loc[~keep_mask].reset_index(drop=True)
    return kept, sparse, thr


def build_tissue_assignment_df(
    class_values,
    region_df=None,
    ignore_fs_ids=None,
    ignore_name_tokens=None,
    exclude_label_24_from_tissue=False,
):
    """
    Build an explicit FS-label -> tissue assignment table for auditability.
    """
    vals = torch.as_tensor(class_values, dtype=torch.int64)
    if vals.ndim != 1:
        raise ValueError("class_values must be 1D")

    n_classes = int(vals.numel())
    y_to_fs = {int(y): int(fs_id) for y, fs_id in enumerate(vals.tolist())}

    name_by_y = _region_name_map_from_region_df(class_values=vals, region_df=region_df)

    # Canonical VBM-like tissue mapping for FreeSurfer IDs.
    csf_ids = {4, 5, 14, 15, 24, 31, 43, 44, 63, 72}
    wm_ids = {2, 7, 41, 46, 77, 251, 252, 253, 254, 255}

    # Common non-tissue or unstable labels to ignore for tissue Dice comparability.
    # Brain-Stem is treated as IGNORE for VBM-style GM/WM/CSF comparability.
    if ignore_fs_ids is None:
        ignore_fs_ids = {16, 30, 62, 80, 85}
    ignore_fs_ids = {int(v) for v in ignore_fs_ids}
    if bool(exclude_label_24_from_tissue):
        ignore_fs_ids.add(24)

    if ignore_name_tokens is None:
        ignore_name_tokens = (
            "vessel",
            "optic-chiasm",
            "non-wm-hypointensities",
            "unknown",
        )

    rows = []

    for y in range(1, n_classes):
        name = name_by_y.get(y, "")
        lname = name.lower()
        fs_id = int(y_to_fs.get(y))

        if fs_id in ignore_fs_ids or any(tok in lname for tok in ignore_name_tokens):
            tissue = "IGNORE"
            reason = "ignore_list"
        elif any(tok in lname for tok in ("ventricle", "csf", "choroid-plexus")):
            tissue = "CSF"
            reason = "name_rule"
        elif any(tok in lname for tok in ("white-matter", "wm-hypointensities", "corpuscallosum")):
            tissue = "WM"
            reason = "name_rule"
        elif fs_id in csf_ids:
            tissue = "CSF"
            reason = "fs_id_rule"
        elif fs_id in wm_ids:
            tissue = "WM"
            reason = "fs_id_rule"
        else:
            tissue = "GM"
            reason = "default_gm"

        rows.append(
            {
                "class_idx": int(y),
                "fs_id": fs_id,
                "region_name": name,
                "tissue": tissue,
                "reason": reason,
            }
        )

    return pd.DataFrame(rows)


def infer_tissue_class_indices(
    class_values,
    region_df=None,
    ignore_fs_ids=None,
    ignore_name_tokens=None,
    exclude_label_24_from_tissue=False,
):
    """
    Infer dense class indices for CSF/GM/WM and FG from FreeSurfer-style labels.
    Returns label groups plus an IGNORE group for non-tissue classes.
    """
    assign_df = build_tissue_assignment_df(
        class_values=class_values,
        region_df=region_df,
        ignore_fs_ids=ignore_fs_ids,
        ignore_name_tokens=ignore_name_tokens,
        exclude_label_24_from_tissue=exclude_label_24_from_tissue,
    )

    csf = sorted(assign_df.loc[assign_df["tissue"] == "CSF", "class_idx"].astype(int).tolist())
    gm = sorted(assign_df.loc[assign_df["tissue"] == "GM", "class_idx"].astype(int).tolist())
    wm = sorted(assign_df.loc[assign_df["tissue"] == "WM", "class_idx"].astype(int).tolist())
    ign = sorted(assign_df.loc[assign_df["tissue"] == "IGNORE", "class_idx"].astype(int).tolist())

    fg = sorted(set(csf) | set(gm) | set(wm))
    return {
        "CSF": csf,
        "GM": gm,
        "WM": wm,
        "FG": fg,
        "IGNORE": ign,
    }


def _dice_from_confusion_subset(cm, class_indices):
    if class_indices is None or len(class_indices) == 0:
        return np.nan

    idx = torch.as_tensor(class_indices, dtype=torch.int64)
    tp = cm.index_select(0, idx).index_select(1, idx).sum().to(torch.float64)
    true_count = cm.index_select(0, idx).sum().to(torch.float64)
    pred_count = cm.index_select(1, idx).sum().to(torch.float64)

    den = true_count + pred_count
    if den.item() <= 0:
        return np.nan
    return float((2.0 * tp / den).item())


def _compute_tissue_dice_from_confusion(cm, tissue_class_indices, drop_true_ignore=True):
    """
    Compute CSF/GM/WM Dice from a multiclass confusion matrix.

    If `drop_true_ignore` is True, rows corresponding to IGNORE classes are
    zeroed before Dice computation. This removes true-IGNORE voxels from the
    evaluation mask (so predictions there do not count as false positives),
    while still penalizing tissue voxels predicted as IGNORE.
    """
    cm_eval = cm
    if bool(drop_true_ignore):
        ign = tissue_class_indices.get("IGNORE", [])
        if ign:
            idx = torch.as_tensor(sorted(set(int(v) for v in ign)), dtype=torch.int64, device=cm.device)
            cm_eval = cm.clone()
            cm_eval.index_fill_(0, idx, 0)

    dice_csf = _dice_from_confusion_subset(cm_eval, tissue_class_indices.get("CSF", []))
    dice_gm = _dice_from_confusion_subset(cm_eval, tissue_class_indices.get("GM", []))
    dice_wm = _dice_from_confusion_subset(cm_eval, tissue_class_indices.get("WM", []))

    fg_mean = np.nanmean([dice_csf, dice_gm, dice_wm])
    gm_wm_mean = np.nanmean([dice_gm, dice_wm])

    return {
        "dice_csf": float(dice_csf) if np.isfinite(dice_csf) else np.nan,
        "dice_gm": float(dice_gm) if np.isfinite(dice_gm) else np.nan,
        "dice_wm": float(dice_wm) if np.isfinite(dice_wm) else np.nan,
        "dice_fg": float(fg_mean) if np.isfinite(fg_mean) else np.nan,
        "dice_gm_wm_mean": float(gm_wm_mean) if np.isfinite(gm_wm_mean) else np.nan,
    }


@torch.inference_mode()
def collect_test_metrics_fast(
    model,
    pipeline,
    class_values,
    max_batches=None,
    patch_chunk_size=96,
    compute_boundary=True,
    boundary_every_n=16,
    boundary_downsample=4,
    null_seed=1337,
    region_df=None,
    device=DEVICE,
    failure_threshold=None,
    failure_metric="mean_dice_fg",
    failure_mode="lt",
    failure_method="model",
    return_failure_data=False,
    tissue_ignore_fs_ids=None,
    tissue_ignore_name_tokens=None,
    tissue_drop_true_ignore=True,
    exclude_label_24_from_tissue=False,
):
    """
    Fast test metric collection suitable for export into segmentation/metrics.py.

    Speedups vs old cell:
    - Vectorized class stats via confusion matrix (no per-class boolean loops)
    - Region-wise metrics emitted once per sample from per-sample confusion matrices
    - Optional sparse/downsampled boundary metric computation

    Failure filtering:
    - If failure_threshold is set, samples matching the failure rule are excluded
      from returned primary metrics and returned separately.
    """
    if pipeline.test_loader is None:
        raise ValueError("pipeline.test_loader is None")

    n_classes = int(class_values.numel())
    class_values_i64 = torch.as_tensor(class_values, dtype=torch.int64)

    x_files = _test_x_files(pipeline.test_ds)
    tissue_class_indices = infer_tissue_class_indices(
        class_values=class_values,
        region_df=region_df,
        ignore_fs_ids=tissue_ignore_fs_ids,
        ignore_name_tokens=tissue_ignore_name_tokens,
        exclude_label_24_from_tissue=exclude_label_24_from_tissue,
    )
    drop_fg_idx = set()
    if bool(exclude_label_24_from_tissue):
        drop_fg_idx.update(int(i) for i in torch.nonzero(class_values_i64 == 24, as_tuple=False).flatten().tolist() if int(i) > 0)
    fg_eval_idx = [i for i in range(1, n_classes) if i not in drop_fg_idx]
    if not fg_eval_idx:
        fg_eval_idx = list(range(1, n_classes))
    fg_eval_idx_t = torch.as_tensor(fg_eval_idx, dtype=torch.int64)

    region_name_by_class = _region_name_map_from_region_df(class_values=class_values, region_df=region_df)
    class_to_fs = {int(i): int(fs_id) for i, fs_id in enumerate(class_values_i64.tolist())}

    sample_rows = []
    region_rows = []
    timing_rows = []
    sample_idx = 0

    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(null_seed))

    n_total = len(pipeline.test_loader) if max_batches is None else min(len(pipeline.test_loader), int(max_batches))
    pbar = tqdm(pipeline.test_loader, total=n_total, desc="test metrics (fast)", leave=False)

    for b_idx, (x_cpu, y_cpu) in enumerate(pbar):
        if max_batches is not None and b_idx >= int(max_batches):
            break

        x = x_cpu.to(device, non_blocking=True).unsqueeze(1).float()
        y_true_batch = y_cpu.to(torch.int64)

        _sync_cuda(device=device)
        t0 = perf_counter()
        y_pred_batch = model.predict_volume(x, patch_chunk_size=patch_chunk_size)
        _sync_cuda(device=device)
        infer_sec_batch = perf_counter() - t0

        if y_pred_batch.device.type != "cpu":
            y_pred_batch = y_pred_batch.cpu()
        y_pred_batch = y_pred_batch.to(torch.int64)

        y_null_batch = torch.randint(
            low=0,
            high=n_classes,
            size=y_true_batch.shape,
            generator=rng,
            dtype=torch.int64,
        )

        bs = int(y_true_batch.shape[0])
        infer_sec_per_sample = float(infer_sec_batch / max(1, bs))

        t_metrics_batch_start = perf_counter()

        for bi in range(bs):
            yt = y_true_batch[bi]
            yp = y_pred_batch[bi]
            yn = y_null_batch[bi]

            cm_m, dice_m, vol_true, vol_pred = _class_confusion_and_stats(yt, yp, n_classes)
            cm_n, dice_n, _vol_true_n, vol_null = _class_confusion_and_stats(yt, yn, n_classes)

            d_model_fg = dice_m.index_select(0, fg_eval_idx_t)
            d_null_fg = dice_n.index_select(0, fg_eval_idx_t)

            valid_model = torch.isfinite(d_model_fg)
            valid_null = torch.isfinite(d_null_fg)

            mean_dice_model = float(d_model_fg[valid_model].mean().item()) if valid_model.any() else np.nan
            mean_dice_null = float(d_null_fg[valid_null].mean().item()) if valid_null.any() else np.nan

            vol_true_fg = vol_true.index_select(0, fg_eval_idx_t).to(torch.float64).numpy()
            vol_pred_fg = vol_pred.index_select(0, fg_eval_idx_t).to(torch.float64).numpy()
            vol_null_fg = vol_null.index_select(0, fg_eval_idx_t).to(torch.float64).numpy()

            tissue_model = _compute_tissue_dice_from_confusion(
                cm_m,
                tissue_class_indices,
                drop_true_ignore=tissue_drop_true_ignore,
            )
            tissue_null = _compute_tissue_dice_from_confusion(
                cm_n,
                tissue_class_indices,
                drop_true_ignore=tissue_drop_true_ignore,
            )

            hd95, assd = np.nan, np.nan
            if bool(compute_boundary):
                every_n = max(1, int(boundary_every_n))
                if sample_idx % every_n == 0:
                    yp_fg = yp.numpy() > 0
                    yt_fg = yt.numpy() > 0
                    hd95, assd = _hd95_assd_foreground(
                        pred_fg=yp_fg,
                        true_fg=yt_fg,
                        downsample=int(boundary_downsample),
                    )

            path = x_files[sample_idx] if sample_idx < len(x_files) else "unknown"
            domain = _domain_from_path(path)
            sample_region_rows_model = _region_metrics_rows_from_confusion(
                cm_total=cm_m,
                method="model",
                region_name_by_class=region_name_by_class,
                class_to_fs=class_to_fs,
                sample_idx=sample_idx,
                domain=domain,
                x_path=path,
            )
            sample_region_rows_null = _region_metrics_rows_from_confusion(
                cm_total=cm_n,
                method="null_random",
                region_name_by_class=region_name_by_class,
                class_to_fs=class_to_fs,
                sample_idx=sample_idx,
                domain=domain,
                x_path=path,
            )
            region_rows.extend(sample_region_rows_model)
            region_rows.extend(sample_region_rows_null)

            sample_rows.append(
                {
                    "sample_idx": int(sample_idx),
                    "domain": domain,
                    "method": "model",
                    "mean_dice_fg": mean_dice_model,
                    "hd95_fg": float(hd95) if np.isfinite(hd95) else np.nan,
                    "assd_fg": float(assd) if np.isfinite(assd) else np.nan,
                    "vol_corr": _safe_pearson_np(vol_pred_fg, vol_true_fg),
                    "abs_vol_error_mean": float(np.mean(np.abs(vol_pred_fg - vol_true_fg))),
                    "inference_sec": infer_sec_per_sample,
                    **tissue_model,
                }
            )
            sample_rows.append(
                {
                    "sample_idx": int(sample_idx),
                    "domain": domain,
                    "method": "null_random",
                    "mean_dice_fg": mean_dice_null,
                    "hd95_fg": np.nan,
                    "assd_fg": np.nan,
                    "vol_corr": _safe_pearson_np(vol_null_fg, vol_true_fg),
                    "abs_vol_error_mean": float(np.mean(np.abs(vol_null_fg - vol_true_fg))),
                    "inference_sec": np.nan,
                    **tissue_null,
                }
            )

            sample_idx += 1

        metrics_sec_batch = perf_counter() - t_metrics_batch_start
        timing_rows.append(
            {
                "batch_idx": int(b_idx),
                "batch_size": int(bs),
                "infer_sec": float(infer_sec_batch),
                "metrics_sec": float(metrics_sec_batch),
                "total_sec": float(infer_sec_batch + metrics_sec_batch),
            }
        )

    sample_metrics_full_df = pd.DataFrame(sample_rows)

    failed_sample_ids = _select_failed_sample_ids(
        sample_metrics_df=sample_metrics_full_df,
        threshold=failure_threshold,
        metric_col=failure_metric,
        method=failure_method,
        mode=failure_mode,
    )

    if failed_sample_ids:
        sample_metrics_df = sample_metrics_full_df[~sample_metrics_full_df["sample_idx"].isin(failed_sample_ids)].reset_index(drop=True)
        failure_samples_df = sample_metrics_full_df[sample_metrics_full_df["sample_idx"].isin(failed_sample_ids)].reset_index(drop=True)
    else:
        sample_metrics_df = sample_metrics_full_df.reset_index(drop=True)
        failure_samples_df = sample_metrics_full_df.iloc[0:0].copy()

    region_metrics_full_df = pd.DataFrame(region_rows)
    if failed_sample_ids:
        region_metrics_df = region_metrics_full_df[~region_metrics_full_df["sample_idx"].isin(failed_sample_ids)].reset_index(drop=True)
        failure_region_metrics_df = region_metrics_full_df[region_metrics_full_df["sample_idx"].isin(failed_sample_ids)].reset_index(drop=True)
    else:
        region_metrics_df = region_metrics_full_df.reset_index(drop=True)
        failure_region_metrics_df = region_metrics_full_df.iloc[0:0].copy()

    if region_metrics_df.empty:
        region_metrics_df = pd.DataFrame(
            columns=["sample_idx", "domain", "x_path", "method", "class_idx", "fs_id", "region_name", "dice", "vol_true", "vol_pred"]
        )
    if failure_region_metrics_df.empty:
        failure_region_metrics_df = pd.DataFrame(
            columns=["sample_idx", "domain", "x_path", "method", "class_idx", "fs_id", "region_name", "dice", "vol_true", "vol_pred"]
        )

    metrics_timing_df = pd.DataFrame(timing_rows)

    if return_failure_data:
        return sample_metrics_df, region_metrics_df, metrics_timing_df, failure_samples_df, failure_region_metrics_df
    return sample_metrics_df, region_metrics_df, metrics_timing_df


def _save_torch_atomic(t, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(t, tmp)
    os.replace(tmp, out_path)


def _save_pickle_atomic(obj, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out_path)


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _paths_digest(paths):
    h = hashlib.sha1()
    for p in paths:
        h.update(str(p).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def _default_test_results_cache_name(
    pipeline,
    class_values,
    region_df,
    max_batches,
    patch_chunk_size,
    compute_boundary,
    boundary_every_n,
    boundary_downsample,
    failure_threshold,
    failure_metric,
    failure_mode,
    failure_method,
    tissue_ignore_fs_ids,
    tissue_ignore_name_tokens,
    tissue_drop_true_ignore,
    exclude_label_24_from_tissue,
):
    x_files = _test_x_files(pipeline.test_ds)
    data_sig = _paths_digest(x_files)
    region_sig = _region_name_cache_tag(class_values=class_values, region_df=region_df)

    if tissue_ignore_fs_ids is None:
        fs_ign = "default"
    else:
        fs_ign = ",".join(str(int(v)) for v in sorted(set(int(v) for v in tissue_ignore_fs_ids)))

    if tissue_ignore_name_tokens is None:
        name_ign = "default"
    else:
        name_ign = ",".join(str(v).strip().lower() for v in tissue_ignore_name_tokens)

    sig = (
        f"test|n={len(x_files)}|d={data_sig}|cls={int(torch.as_tensor(class_values).numel())}"
        f"|rgn={region_sig}"
        f"|mb={str(max_batches)}|pc={int(patch_chunk_size)}"
        f"|bd={int(bool(compute_boundary))}:{int(boundary_every_n)}:{int(boundary_downsample)}"
        f"|fail={str(failure_metric)}:{str(failure_mode)}:{str(failure_method)}:{str(failure_threshold)}"
        f"|tignfs={fs_ign}|tignname={name_ign}|tdrop={int(bool(tissue_drop_true_ignore))}"
        f"|x24={int(bool(exclude_label_24_from_tissue))}"
        f"|rrows=per_sample_v2"
    )
    digest = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]
    return f"test_eval_{digest}.pkl"


def collect_test_metrics_fast_cached(
    model,
    pipeline,
    class_values,
    max_batches=None,
    patch_chunk_size=96,
    compute_boundary=True,
    boundary_every_n=16,
    boundary_downsample=4,
    null_seed=1337,
    region_df=None,
    device=DEVICE,
    failure_threshold=None,
    failure_metric="mean_dice_fg",
    failure_mode="lt",
    failure_method="model",
    return_failure_data=False,
    tissue_ignore_fs_ids=None,
    tissue_ignore_name_tokens=None,
    tissue_drop_true_ignore=True,
    exclude_label_24_from_tissue=False,
    results_cache_dir=None,
    results_cache_name=None,
    reuse_results_cache=True,
):
    results_cache_path = None
    if results_cache_dir is not None:
        results_cache_dir = Path(results_cache_dir)
        results_cache_dir.mkdir(parents=True, exist_ok=True)
        if results_cache_name is None:
            results_cache_name = _default_test_results_cache_name(
                pipeline=pipeline,
                class_values=class_values,
                region_df=region_df,
                max_batches=max_batches,
                patch_chunk_size=patch_chunk_size,
                compute_boundary=compute_boundary,
                boundary_every_n=boundary_every_n,
                boundary_downsample=boundary_downsample,
                failure_threshold=failure_threshold,
                failure_metric=failure_metric,
                failure_mode=failure_mode,
                failure_method=failure_method,
                tissue_ignore_fs_ids=tissue_ignore_fs_ids,
                tissue_ignore_name_tokens=tissue_ignore_name_tokens,
                tissue_drop_true_ignore=tissue_drop_true_ignore,
                exclude_label_24_from_tissue=exclude_label_24_from_tissue,
            )
        results_cache_path = results_cache_dir / str(results_cache_name)

    _region_name_map_from_region_df(class_values=class_values, region_df=region_df)

    if (
        results_cache_path is not None
        and bool(reuse_results_cache)
        and results_cache_path.exists()
        and results_cache_path.stat().st_size > 0
    ):
        payload = _load_pickle(results_cache_path)
        sample_metrics_df = payload["sample_metrics_df"]
        region_metrics_df = payload["region_metrics_df"]
        metrics_timing_df = payload["metrics_timing_df"]
        failure_samples_df = payload.get("failure_samples_df")
        failure_region_metrics_df = payload.get("failure_region_metrics_df")
        if failure_samples_df is None:
            failure_samples_df = pd.DataFrame(columns=getattr(sample_metrics_df, "columns", []))
        if failure_region_metrics_df is None:
            failure_region_metrics_df = pd.DataFrame(columns=getattr(region_metrics_df, "columns", []))
        if (
            _region_metrics_df_has_required_names(region_metrics_df, class_values=class_values, require_sample_idx=True)
            and _region_metrics_df_has_required_names(failure_region_metrics_df, class_values=class_values, require_sample_idx=True)
        ):
            if return_failure_data:
                return sample_metrics_df, region_metrics_df, metrics_timing_df, failure_samples_df, failure_region_metrics_df
            return sample_metrics_df, region_metrics_df, metrics_timing_df

    out = collect_test_metrics_fast(
        model=model,
        pipeline=pipeline,
        class_values=class_values,
        max_batches=max_batches,
        patch_chunk_size=patch_chunk_size,
        compute_boundary=compute_boundary,
        boundary_every_n=boundary_every_n,
        boundary_downsample=boundary_downsample,
        null_seed=null_seed,
        region_df=region_df,
        device=device,
        failure_threshold=failure_threshold,
        failure_metric=failure_metric,
        failure_mode=failure_mode,
        failure_method=failure_method,
        return_failure_data=True,
        tissue_ignore_fs_ids=tissue_ignore_fs_ids,
        tissue_ignore_name_tokens=tissue_ignore_name_tokens,
        tissue_drop_true_ignore=tissue_drop_true_ignore,
        exclude_label_24_from_tissue=exclude_label_24_from_tissue,
    )

    sample_metrics_df, region_metrics_df, metrics_timing_df, failure_samples_df, failure_region_metrics_df = out
    payload = {
        "sample_metrics_df": sample_metrics_df,
        "region_metrics_df": region_metrics_df,
        "metrics_timing_df": metrics_timing_df,
        "failure_samples_df": failure_samples_df,
        "failure_region_metrics_df": failure_region_metrics_df,
    }
    if results_cache_path is not None:
        _save_pickle_atomic(payload, results_cache_path)

    if return_failure_data:
        return sample_metrics_df, region_metrics_df, metrics_timing_df, failure_samples_df, failure_region_metrics_df
    return sample_metrics_df, region_metrics_df, metrics_timing_df


def _preprocess_cache_tag(patch_size, apply_zscore, label_lut):
    lut_sig = int((torch.as_tensor(label_lut) >= 0).sum().item()) if label_lut is not None else 0
    ps = tuple(int(v) for v in patch_size)
    return f"pp_v1|ps={ps}|zs={int(bool(apply_zscore))}|lut={lut_sig}"


def _pair_digest(rawavg_path, aparc_path, cache_tag):
    key = f"{Path(rawavg_path).resolve()}|{Path(aparc_path).resolve()}|{cache_tag}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]


def discover_valid_mgz_pairs(mgz_root, expected_valid_pairs=None):
    root = Path(mgz_root).expanduser().resolve()
    raw_paths = sorted(root.rglob("rawavg.mgz"))
    aparc_paths = sorted(root.rglob("aparc+aseg.mgz"))

    valid_pairs = []
    missing_raw_for_aparc = []
    for aparc_path in aparc_paths:
        rawavg_path = aparc_path.with_name("rawavg.mgz")
        if rawavg_path.exists():
            valid_pairs.append((rawavg_path, aparc_path))
        else:
            missing_raw_for_aparc.append(aparc_path)

    if expected_valid_pairs is not None and len(valid_pairs) != int(expected_valid_pairs):
        raise RuntimeError(
            f"Expected {int(expected_valid_pairs)} valid pairs under {root}, found {len(valid_pairs)}"
        )

    inventory_df = pd.DataFrame(
        {
            "mgz_root": [str(root)],
            "rawavg_files": [int(len(raw_paths))],
            "aparc_files": [int(len(aparc_paths))],
            "valid_pairs": [int(len(valid_pairs))],
            "missing_raw_for_aparc": [int(len(missing_raw_for_aparc))],
        }
    )
    return valid_pairs, inventory_df, missing_raw_for_aparc


def preprocess_mgz_pair_to_padded_tensors(rawavg_path, aparc_path, label_lut, apply_zscore, patch_size):
    """
    Convert mgz pair -> padded tensors using the module conversion/preprocess path.

    Returns:
      x_pad: [D, H, W] float32
      y_pad: [D, H, W] int64 (dense class ids)
    """
    # Local imports keep baseline metrics paths lightweight.
    from convert import prepare_arrays_if_needed
    from .data import collate_pad_to_patch

    x_arr, y_arr = prepare_arrays_if_needed(rawavg_path, aparc_path)

    x = torch.from_numpy(np.asarray(x_arr, dtype=np.float32))
    y = torch.from_numpy(np.asarray(y_arr, dtype=np.int64))

    if label_lut is not None:
        lut = torch.as_tensor(label_lut, dtype=torch.int64)
        y_max = int(y.max().item())
        if y_max >= int(lut.numel()):
            raise ValueError(
                f"{aparc_path}: label id {y_max} exceeds LUT range {int(lut.numel()) - 1}."
            )
        y_mapped = lut[y]
        if bool((y_mapped < 0).any()):
            bad_values = torch.unique(y[y_mapped < 0]).cpu().tolist()
            raise ValueError(
                f"{aparc_path}: labels not present in LUT. bad labels (first 20): {bad_values[:20]}"
            )
        y = y_mapped

    if bool(apply_zscore):
        x = (x - x.mean()) / x.std().clamp_min(1e-6)

    x_pad_b, y_pad_b = collate_pad_to_patch([(x.contiguous(), y.contiguous())], patch_size=tuple(int(v) for v in patch_size))
    return x_pad_b[0].to(torch.float32).contiguous(), y_pad_b[0].to(torch.int64).contiguous()


def build_mgz_preprocess_cache(
    valid_pairs,
    cache_dir,
    label_lut,
    apply_zscore,
    patch_size,
    rebuild=False,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_tag = _preprocess_cache_tag(
        patch_size=patch_size,
        apply_zscore=apply_zscore,
        label_lut=label_lut,
    )

    rows = []
    built = 0
    reused = 0

    pbar = tqdm(valid_pairs, desc="mgz preprocess cache", leave=False)
    for rawavg_path, aparc_path in pbar:
        digest = _pair_digest(rawavg_path, aparc_path, cache_tag=cache_tag)
        x_cache = cache_dir / f"{digest}.x.pt"
        y_cache = cache_dir / f"{digest}.y.pt"

        if (not rebuild) and x_cache.exists() and y_cache.exists():
            reused += 1
        else:
            x_pad, y_pad = preprocess_mgz_pair_to_padded_tensors(
                rawavg_path=rawavg_path,
                aparc_path=aparc_path,
                label_lut=label_lut,
                apply_zscore=apply_zscore,
                patch_size=patch_size,
            )
            _save_torch_atomic(x_pad, x_cache)
            _save_torch_atomic(y_pad, y_cache)
            built += 1

        rows.append(
            {
                "rawavg_path": str(rawavg_path),
                "aparc_path": str(aparc_path),
                "x_cache_path": str(x_cache),
                "y_cache_path": str(y_cache),
            }
        )

    cache_index_df = pd.DataFrame(rows)
    cache_stats_df = pd.DataFrame(
        {
            "cache_dir": [str(cache_dir)],
            "cache_tag": [cache_tag],
            "n_pairs": [int(len(rows))],
            "cached_built": [int(built)],
            "cached_reused": [int(reused)],
        }
    )
    return cache_index_df, cache_stats_df


def _load_padded_pair_from_cache_row(row):
    x = torch.load(row["x_cache_path"]).to(torch.float32)
    y = torch.load(row["y_cache_path"]).to(torch.int64)
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError(
            f"Cached tensors must be [D,H,W]. Got x={tuple(x.shape)} y={tuple(y.shape)} for {row['x_cache_path']}"
        )
    return x.contiguous(), y.contiguous()


def _default_results_cache_name(
    mgz_root,
    expected_valid_pairs,
    patch_size,
    patch_chunk_size,
    apply_zscore,
    class_values,
    region_df,
    n_classes,
    compute_boundary,
    boundary_downsample,
    tissue_ignore_fs_ids,
    tissue_ignore_name_tokens,
    tissue_drop_true_ignore,
    exclude_label_24_from_tissue,
):
    region_sig = _region_name_cache_tag(class_values=class_values, region_df=region_df)
    if tissue_ignore_fs_ids is None:
        fs_ign = "default"
    else:
        fs_ign = ",".join(str(int(v)) for v in sorted(set(int(v) for v in tissue_ignore_fs_ids)))

    if tissue_ignore_name_tokens is None:
        name_ign = "default"
    else:
        name_ign = ",".join(str(v).strip().lower() for v in tissue_ignore_name_tokens)

    sig = (
        f"root={Path(mgz_root).resolve()}|pairs={expected_valid_pairs}|ps={tuple(int(v) for v in patch_size)}"
        f"|pc={int(patch_chunk_size)}|zs={int(bool(apply_zscore))}|cls={int(n_classes)}"
        f"|rgn={region_sig}"
        f"|bd={int(bool(compute_boundary))}:{int(boundary_downsample)}"
        f"|tignfs={fs_ign}|tignname={name_ign}|tdrop={int(bool(tissue_drop_true_ignore))}"
        f"|x24={int(bool(exclude_label_24_from_tissue))}"
    )
    digest = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]
    return f"throughput_eval_{digest}.pkl"


@torch.inference_mode()
def evaluate_mgz_throughput_cached(
    model,
    mgz_root,
    class_values,
    label_lut,
    patch_size,
    patch_chunk_size=96,
    apply_zscore=True,
    expected_valid_pairs=None,
    region_df=None,
    device=DEVICE,
    use_amp=False,
    amp_dtype=torch.float16,
    preprocess_cache_dir=None,
    preprocess_rebuild=False,
    results_cache_dir=None,
    results_cache_name=None,
    reuse_results_cache=True,
    compute_boundary=True,
    boundary_downsample=4,
    tissue_ignore_fs_ids=None,
    tissue_ignore_name_tokens=None,
    tissue_drop_true_ignore=True,
    exclude_label_24_from_tissue=False,
):
    """
    End-to-end throughput-set evaluation with disk caching.

    Cache layers:
    1) Preprocess cache: mgz -> padded tensors (avoids conversion/resampling repeats)
    2) Results cache: computed losses/metrics dataframes (avoids model re-run repeats)
    """
    valid_pairs, inventory_df, missing_raw_for_aparc = discover_valid_mgz_pairs(
        mgz_root=mgz_root,
        expected_valid_pairs=expected_valid_pairs,
    )
    if missing_raw_for_aparc:
        raise RuntimeError(
            f"Found aparc files without sibling rawavg.mgz: {len(missing_raw_for_aparc)}"
        )

    n_classes = int(torch.as_tensor(class_values).numel())
    class_values_i64 = torch.as_tensor(class_values, dtype=torch.int64)

    if results_cache_dir is not None:
        results_cache_dir = Path(results_cache_dir)
        results_cache_dir.mkdir(parents=True, exist_ok=True)
        if results_cache_name is None:
            results_cache_name = _default_results_cache_name(
                mgz_root=mgz_root,
                expected_valid_pairs=(len(valid_pairs) if expected_valid_pairs is None else int(expected_valid_pairs)),
                patch_size=patch_size,
                patch_chunk_size=patch_chunk_size,
                apply_zscore=apply_zscore,
                class_values=class_values,
                region_df=region_df,
                n_classes=n_classes,
                compute_boundary=compute_boundary,
                boundary_downsample=boundary_downsample,
                tissue_ignore_fs_ids=tissue_ignore_fs_ids,
                tissue_ignore_name_tokens=tissue_ignore_name_tokens,
                tissue_drop_true_ignore=tissue_drop_true_ignore,
                exclude_label_24_from_tissue=exclude_label_24_from_tissue,
            )
        results_cache_path = results_cache_dir / str(results_cache_name)
    else:
        results_cache_path = None

    _region_name_map_from_region_df(class_values=class_values, region_df=region_df)

    if (
        results_cache_path is not None
        and bool(reuse_results_cache)
        and results_cache_path.exists()
        and results_cache_path.stat().st_size > 0
    ):
        payload = _load_pickle(results_cache_path)
        region_metrics_df = payload.get("throughput_region_metrics_df")
        if _region_metrics_df_has_required_names(region_metrics_df, class_values=class_values):
            payload["throughput_inventory_df"] = inventory_df
            return payload

    if preprocess_cache_dir is not None:
        cache_index_df, cache_stats_df = build_mgz_preprocess_cache(
            valid_pairs=valid_pairs,
            cache_dir=preprocess_cache_dir,
            label_lut=label_lut,
            apply_zscore=apply_zscore,
            patch_size=patch_size,
            rebuild=preprocess_rebuild,
        )
    else:
        rows = []
        for rawavg_path, aparc_path in valid_pairs:
            rows.append(
                {
                    "rawavg_path": str(rawavg_path),
                    "aparc_path": str(aparc_path),
                    "x_cache_path": None,
                    "y_cache_path": None,
                }
            )
        cache_index_df = pd.DataFrame(rows)
        cache_stats_df = pd.DataFrame(
            {
                "cache_dir": [None],
                "cache_tag": ["none"],
                "n_pairs": [int(len(rows))],
                "cached_built": [0],
                "cached_reused": [0],
            }
        )

    region_name_by_class = _region_name_map_from_region_df(class_values=class_values, region_df=region_df)

    tissue_class_indices = infer_tissue_class_indices(
        class_values=class_values,
        region_df=region_df,
        ignore_fs_ids=tissue_ignore_fs_ids,
        ignore_name_tokens=tissue_ignore_name_tokens,
        exclude_label_24_from_tissue=exclude_label_24_from_tissue,
    )
    drop_fg_idx = set()
    if bool(exclude_label_24_from_tissue):
        drop_fg_idx.update(int(i) for i in torch.nonzero(class_values_i64 == 24, as_tuple=False).flatten().tolist() if int(i) > 0)
    fg_eval_idx = [i for i in range(1, n_classes) if i not in drop_fg_idx]
    if not fg_eval_idx:
        fg_eval_idx = list(range(1, n_classes))
    fg_eval_idx_t = torch.as_tensor(fg_eval_idx, dtype=torch.int64)

    cm_total = torch.zeros((n_classes, n_classes), dtype=torch.int64)
    loss_rows = []
    sample_rows = []

    model.eval()
    amp_enabled = bool(use_amp) and str(device) == "cuda"

    pbar = tqdm(cache_index_df.itertuples(index=False), total=len(cache_index_df), desc="throughput cached eval", leave=False)
    for sample_idx, row in enumerate(pbar):
        rawavg_path = Path(row.rawavg_path)
        aparc_path = Path(row.aparc_path)

        t0 = perf_counter()
        if row.x_cache_path and row.y_cache_path:
            x_pad, y_pad = _load_padded_pair_from_cache_row(
                {"x_cache_path": row.x_cache_path, "y_cache_path": row.y_cache_path}
            )
        else:
            x_pad, y_pad = preprocess_mgz_pair_to_padded_tensors(
                rawavg_path=rawavg_path,
                aparc_path=aparc_path,
                label_lut=label_lut,
                apply_zscore=apply_zscore,
                patch_size=patch_size,
            )
        preprocess_sec = perf_counter() - t0

        x_dev = x_pad.unsqueeze(0).unsqueeze(0).to(device, non_blocking=True).float()  # [1,1,D,H,W]
        y_dev = y_pad.unsqueeze(0).to(device, non_blocking=True).long()                # [1,D,H,W]

        _sync_cuda(device=device)
        t1 = perf_counter()
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            feat = model.forward_features(x_dev)
            loss = model.loss_from_features(feat, y_dev, patch_chunk_size=patch_chunk_size)
        _sync_cuda(device=device)
        loss_forward_sec = perf_counter() - t1

        _sync_cuda(device=device)
        t2 = perf_counter()
        y_pred = model.predict_volume(x_dev, patch_chunk_size=patch_chunk_size)
        _sync_cuda(device=device)
        predict_sec = perf_counter() - t2

        yt = y_dev[0].detach().to("cpu", dtype=torch.int64)
        yp = y_pred[0].detach().to("cpu", dtype=torch.int64)

        cm, dice_vec, vol_true_vec, vol_pred_vec = _class_confusion_and_stats(yt, yp, n_classes)
        cm_total += cm

        fg_dice = dice_vec.index_select(0, fg_eval_idx_t)
        valid_fg = torch.isfinite(fg_dice)
        mean_dice_fg = float(fg_dice[valid_fg].mean().item()) if valid_fg.any() else np.nan

        tissue_dice = _compute_tissue_dice_from_confusion(
            cm,
            tissue_class_indices,
            drop_true_ignore=tissue_drop_true_ignore,
        )

        vol_true_fg = vol_true_vec.index_select(0, fg_eval_idx_t).to(torch.float64).numpy()
        vol_pred_fg = vol_pred_vec.index_select(0, fg_eval_idx_t).to(torch.float64).numpy()

        if bool(compute_boundary):
            yp_fg = yp.numpy() > 0
            yt_fg = yt.numpy() > 0
            hd95_fg, assd_fg = _hd95_assd_foreground(
                pred_fg=yp_fg,
                true_fg=yt_fg,
                downsample=int(boundary_downsample),
            )
        else:
            hd95_fg, assd_fg = np.nan, np.nan

        n_vox = int(y_dev.numel())
        sample_rows.append(
            {
                "sample_idx": int(sample_idx),
                "method": "model",
                "study": rawavg_path.parts[-4] if len(rawavg_path.parts) >= 4 else "unknown",
                "subject": rawavg_path.parts[-3] if len(rawavg_path.parts) >= 3 else rawavg_path.parent.name,
                "rawavg_path": str(rawavg_path),
                "aparc_path": str(aparc_path),
                "mean_dice_fg": mean_dice_fg,
                "dice_csf": tissue_dice.get("dice_csf", np.nan),
                "dice_gm": tissue_dice.get("dice_gm", np.nan),
                "dice_wm": tissue_dice.get("dice_wm", np.nan),
                "dice_fg": tissue_dice.get("dice_fg", np.nan),
                "dice_gm_wm_mean": tissue_dice.get("dice_gm_wm_mean", np.nan),
                "hd95_fg": float(hd95_fg) if np.isfinite(hd95_fg) else np.nan,
                "assd_fg": float(assd_fg) if np.isfinite(assd_fg) else np.nan,
                "vol_corr_fg": _safe_pearson_np(vol_pred_fg, vol_true_fg),
                "abs_vol_error_mean_fg": float(np.mean(np.abs(vol_pred_fg - vol_true_fg))),
                "n_voxels": n_vox,
                "preprocess_sec": float(preprocess_sec),
                "loss_forward_sec": float(loss_forward_sec),
                "predict_sec": float(predict_sec),
                "total_sec": float(preprocess_sec + loss_forward_sec + predict_sec),
                "loss": float(loss.detach().item()),
            }
        )

        loss_rows.append(
            {
                "sample_idx": int(sample_idx),
                "study": rawavg_path.parts[-4] if len(rawavg_path.parts) >= 4 else "unknown",
                "subject": rawavg_path.parts[-3] if len(rawavg_path.parts) >= 3 else rawavg_path.parent.name,
                "rawavg_path": str(rawavg_path),
                "aparc_path": str(aparc_path),
                "loss": float(loss.detach().item()),
                "n_voxels": n_vox,
                "preprocess_sec": float(preprocess_sec),
                "loss_forward_sec": float(loss_forward_sec),
                "predict_sec": float(predict_sec),
                "total_sec": float(preprocess_sec + loss_forward_sec + predict_sec),
            }
        )

    throughput_loss_df = pd.DataFrame(loss_rows)
    throughput_sample_metrics_df = pd.DataFrame(sample_rows)

    throughput_region_metrics_df = pd.DataFrame(
        _aggregate_region_metrics_from_confusion(cm_total, "model", region_name_by_class)
    )

    class_to_fs = {int(i): int(fs_id) for i, fs_id in enumerate(torch.as_tensor(class_values).tolist())}
    if len(throughput_region_metrics_df):
        throughput_region_metrics_df["fs_id"] = throughput_region_metrics_df["class_idx"].map(class_to_fs).astype("Int64")

    if len(throughput_loss_df):
        throughput_loss_summary = pd.DataFrame(
            {
                "n_pairs": [int(len(throughput_loss_df))],
                "loss_mean": [float(throughput_loss_df["loss"].mean())],
                "loss_median": [float(throughput_loss_df["loss"].median())],
                "loss_std": [float(throughput_loss_df["loss"].std(ddof=1)) if len(throughput_loss_df) > 1 else np.nan],
                "loss_weighted_by_voxels": [
                    float(np.average(throughput_loss_df["loss"].to_numpy(), weights=throughput_loss_df["n_voxels"].to_numpy()))
                ],
                "preprocess_sec_mean": [float(throughput_loss_df["preprocess_sec"].mean())],
                "loss_forward_sec_mean": [float(throughput_loss_df["loss_forward_sec"].mean())],
                "predict_sec_mean": [float(throughput_loss_df["predict_sec"].mean())],
                "total_sec_mean": [float(throughput_loss_df["total_sec"].mean())],
            }
        )
    else:
        throughput_loss_summary = pd.DataFrame()

    if len(throughput_sample_metrics_df):
        throughput_metrics_summary = pd.DataFrame(
            {
                "n_samples": [int(len(throughput_sample_metrics_df))],
                "mean_dice_fg": [float(throughput_sample_metrics_df["mean_dice_fg"].mean())],
                "mean_dice_csf": [float(throughput_sample_metrics_df["dice_csf"].mean())],
                "mean_dice_gm": [float(throughput_sample_metrics_df["dice_gm"].mean())],
                "mean_dice_wm": [float(throughput_sample_metrics_df["dice_wm"].mean())],
                "mean_dice_tissue_fg": [float(throughput_sample_metrics_df["dice_fg"].mean())],
                "mean_dice_gm_wm_mean": [float(throughput_sample_metrics_df["dice_gm_wm_mean"].mean())],
                "mean_hd95_fg": [float(throughput_sample_metrics_df["hd95_fg"].mean())],
                "mean_assd_fg": [float(throughput_sample_metrics_df["assd_fg"].mean())],
                "mean_vol_corr_fg": [float(throughput_sample_metrics_df["vol_corr_fg"].mean())],
                "mean_abs_vol_error_fg": [float(throughput_sample_metrics_df["abs_vol_error_mean_fg"].mean())],
                "mean_preprocess_sec": [float(throughput_sample_metrics_df["preprocess_sec"].mean())],
                "mean_loss_forward_sec": [float(throughput_sample_metrics_df["loss_forward_sec"].mean())],
                "mean_predict_sec": [float(throughput_sample_metrics_df["predict_sec"].mean())],
                "mean_total_sec": [float(throughput_sample_metrics_df["total_sec"].mean())],
            }
        )

        throughput_tissue_summary = (
            throughput_sample_metrics_df[["dice_csf", "dice_gm", "dice_wm", "dice_fg", "dice_gm_wm_mean"]]
            .agg(["mean", "median", "std", "min", "max"])
            .T
            .reset_index()
            .rename(columns={"index": "tissue_metric"})
        )
    else:
        throughput_metrics_summary = pd.DataFrame()
        throughput_tissue_summary = pd.DataFrame()

    payload = {
        "throughput_inventory_df": inventory_df,
        "throughput_cache_stats_df": cache_stats_df,
        "throughput_cache_index_df": cache_index_df,
        "throughput_loss_df": throughput_loss_df,
        "throughput_loss_summary": throughput_loss_summary,
        "throughput_sample_metrics_df": throughput_sample_metrics_df,
        "throughput_region_metrics_df": throughput_region_metrics_df,
        "throughput_metrics_summary": throughput_metrics_summary,
        "throughput_tissue_summary": throughput_tissue_summary,
    }

    if results_cache_path is not None:
        _save_pickle_atomic(payload, results_cache_path)

    return payload


def build_tissue_long_df(sample_metrics_df, method="model", as_percent=True):
    required_cols = ["dice_csf", "dice_gm", "dice_wm", "dice_fg"]
    if sample_metrics_df is None or len(sample_metrics_df) == 0:
        raise ValueError("sample_metrics_df is empty")

    missing = [c for c in required_cols if c not in sample_metrics_df.columns]
    if missing:
        raise ValueError(f"Missing tissue Dice columns: {missing}")

    sub = sample_metrics_df[sample_metrics_df["method"] == str(method)].copy()
    if sub.empty:
        raise ValueError(f"No rows found for method={method}")

    tissue_map = {
        "dice_csf": "CSF",
        "dice_gm": "GM",
        "dice_wm": "WM",
        "dice_fg": "FG",
    }

    tissue_long = (
        sub[["sample_idx"] + list(tissue_map.keys())]
        .melt(id_vars=["sample_idx"], var_name="metric", value_name="dice")
    )
    tissue_long["tissue"] = tissue_long["metric"].map(tissue_map)
    tissue_long["dice"] = pd.to_numeric(tissue_long["dice"], errors="coerce")
    tissue_long = tissue_long.replace([np.inf, -np.inf], np.nan).dropna(subset=["dice"]).reset_index(drop=True)

    if bool(as_percent):
        tissue_long["dice_plot"] = tissue_long["dice"] * 100.0
        tissue_long["dice_unit"] = "%"
    else:
        tissue_long["dice_plot"] = tissue_long["dice"]
        tissue_long["dice_unit"] = "[0,1]"

    return tissue_long


def summarize_tissue_long_df(tissue_long_df):
    if tissue_long_df is None or len(tissue_long_df) == 0:
        return pd.DataFrame(columns=["tissue", "n", "mean", "median", "std", "min", "max", "unit"])

    out = (
        tissue_long_df.groupby("tissue", as_index=False)
        .agg(
            n=("dice_plot", "count"),
            mean=("dice_plot", "mean"),
            median=("dice_plot", "median"),
            std=("dice_plot", "std"),
            min=("dice_plot", "min"),
            max=("dice_plot", "max"),
        )
    )
    unit = tissue_long_df["dice_unit"].iloc[0] if "dice_unit" in tissue_long_df.columns and len(tissue_long_df) else ""
    out["unit"] = unit

    order = ["CSF", "GM", "WM", "FG"]
    out["tissue"] = pd.Categorical(out["tissue"], categories=order, ordered=True)
    out = out.sort_values("tissue").reset_index(drop=True)
    return out


def summarize_sample_metrics_by_method(sample_metrics_df):
    if sample_metrics_df is None or len(sample_metrics_df) == 0:
        return pd.DataFrame()

    col_alias = [
        ("mean_dice_fg", "mean_dice_fg"),
        ("dice_csf", "mean_dice_csf"),
        ("dice_gm", "mean_dice_gm"),
        ("dice_wm", "mean_dice_wm"),
        ("dice_fg", "mean_dice_tissue_fg"),
        ("dice_gm_wm_mean", "mean_dice_gm_wm_mean"),
        ("hd95_fg", "mean_hd95_fg"),
        ("assd_fg", "mean_assd_fg"),
        ("vol_corr", "mean_vol_corr"),
        ("vol_corr_fg", "mean_vol_corr_fg"),
        ("abs_vol_error_mean", "mean_abs_vol_error"),
        ("abs_vol_error_mean_fg", "mean_abs_vol_error_fg"),
        ("inference_sec", "mean_inference_sec"),
        ("predict_sec", "mean_predict_sec"),
        ("total_sec", "mean_total_sec"),
        ("loss", "mean_loss"),
    ]

    agg = {"n_samples": ("sample_idx", "nunique")}
    for src, dst in col_alias:
        if src in sample_metrics_df.columns:
            agg[dst] = (src, "mean")

    return sample_metrics_df.groupby("method", as_index=False).agg(**agg)


def summarize_domain_robustness(sample_metrics_df, method="model", domain_col="domain"):
    if sample_metrics_df is None or len(sample_metrics_df) == 0:
        return pd.DataFrame()
    if domain_col not in sample_metrics_df.columns:
        return pd.DataFrame()

    sub = sample_metrics_df[sample_metrics_df["method"] == str(method)].copy()
    if sub.empty:
        return pd.DataFrame()

    agg = {"n_samples": ("sample_idx", "nunique")}
    if "mean_dice_fg" in sub.columns:
        agg["mean_dice_fg"] = ("mean_dice_fg", "mean")
    if "vol_corr" in sub.columns:
        agg["mean_vol_corr"] = ("vol_corr", "mean")
    if "vol_corr_fg" in sub.columns:
        agg["mean_vol_corr_fg"] = ("vol_corr_fg", "mean")

    out = sub.groupby(domain_col, as_index=False).agg(**agg)
    sort_cols = [c for c in ["n_samples", "mean_dice_fg", "mean_vol_corr", "mean_vol_corr_fg"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    return out.reset_index(drop=True)


def summarize_timing_df(timing_df):
    if timing_df is None or len(timing_df) == 0:
        return pd.Series(dtype=float)

    cols = [c for c in ["infer_sec", "metrics_sec", "total_sec"] if c in timing_df.columns]
    if not cols:
        return pd.Series(dtype=float)

    mapping = {c: f"{c}_mean" for c in cols}
    return timing_df[cols].mean().rename(index=mapping)


def build_eval_dataset_report(
    sample_metrics_df,
    region_metrics_df,
    method="model",
    failure_samples_df=None,
    failure_metric="mean_dice_fg",
    exclude_region_names=None,
    min_true_voxels=None,
    min_true_quantile=0.05,
    tissue_as_percent=True,
    worst_n=10,
    bottom_regions_n=25,
):
    plot_bundle = build_eval_plot_bundle(
        sample_metrics_df=sample_metrics_df,
        region_metrics_df=region_metrics_df,
        method=method,
        min_true_voxels=min_true_voxels,
        min_true_quantile=min_true_quantile,
        exclude_region_names=exclude_region_names,
        tissue_as_percent=tissue_as_percent,
    )

    summary_df = summarize_sample_metrics_by_method(sample_metrics_df)
    domain_df = summarize_domain_robustness(sample_metrics_df, method=method, domain_col="domain")

    model_samples = sample_metrics_df[sample_metrics_df["method"] == str(method)].copy()
    sort_metric = str(failure_metric) if str(failure_metric) in model_samples.columns else "mean_dice_fg"
    worst_samples = model_samples.sort_values(sort_metric, ascending=True).head(int(worst_n))
    bottom_regions = plot_bundle["region_plot_df"].head(int(bottom_regions_n)).copy()

    failed_unique = 0
    if failure_samples_df is not None and len(failure_samples_df):
        failed_unique = int(failure_samples_df.loc[failure_samples_df["method"] == str(method), "sample_idx"].nunique())
    kept_unique = int(model_samples["sample_idx"].nunique()) if len(model_samples) else 0

    return {
        "plot_bundle": plot_bundle,
        "summary_df": summary_df,
        "domain_df": domain_df,
        "worst_samples_df": worst_samples,
        "bottom_regions_df": bottom_regions,
        "kept_n": kept_unique,
        "failed_n": failed_unique,
    }


def prepare_region_dice_plot_df(
    region_metrics_df,
    method="model",
    min_true_voxels=None,
    min_true_quantile=0.05,
    exclude_region_names=None,
):
    if region_metrics_df is None or len(region_metrics_df) == 0:
        raise ValueError("region_metrics_df is empty")

    model_region = region_metrics_df[region_metrics_df["method"] == str(method)].copy()
    if model_region.empty:
        raise ValueError(f"No rows found for method={method}")

    model_region = (
        model_region.groupby(["class_idx", "region_name"], dropna=False, as_index=False)
        .agg(
            dice=("dice", "mean"),
            vol_true=("vol_true", "sum"),
            vol_pred=("vol_pred", "sum"),
        )
    )

    model_region["dice"] = pd.to_numeric(model_region["dice"], errors="coerce")
    model_region["vol_true"] = pd.to_numeric(model_region["vol_true"], errors="coerce").fillna(0)
    model_region = model_region.replace([np.inf, -np.inf], np.nan).dropna(subset=["dice"]).reset_index(drop=True)

    kept, sparse, sparse_thr = filter_sparse_regions(
        region_metrics_df=model_region.assign(method=str(method)),
        method=str(method),
        min_true_voxels=min_true_voxels,
        min_true_quantile=min_true_quantile,
    )
    kept = kept.drop(columns=["method"], errors="ignore")

    if exclude_region_names:
        excl = set(str(x) for x in exclude_region_names)
        kept = kept[~kept["region_name"].isin(excl)].reset_index(drop=True)

    _assert_region_metrics_df_has_resolved_names(kept, context="region_metrics_df")
    kept["label"] = kept["region_name"].astype(str).str.strip()

    kept = kept.sort_values("dice", ascending=True).reset_index(drop=True)
    return kept, sparse, int(sparse_thr)


def build_eval_plot_bundle(
    sample_metrics_df,
    region_metrics_df,
    method="model",
    min_true_voxels=None,
    min_true_quantile=0.05,
    exclude_region_names=None,
    tissue_as_percent=True,
):
    region_plot_df, sparse_region_df, sparse_threshold = prepare_region_dice_plot_df(
        region_metrics_df=region_metrics_df,
        method=method,
        min_true_voxels=min_true_voxels,
        min_true_quantile=min_true_quantile,
        exclude_region_names=exclude_region_names,
    )

    tissue_long_df = build_tissue_long_df(
        sample_metrics_df=sample_metrics_df,
        method=method,
        as_percent=tissue_as_percent,
    )
    tissue_summary_df = summarize_tissue_long_df(tissue_long_df)

    return {
        "region_plot_df": region_plot_df,
        "sparse_region_df": sparse_region_df,
        "sparse_threshold": int(sparse_threshold),
        "tissue_long_df": tissue_long_df,
        "tissue_summary_df": tissue_summary_df,
    }


def plot_eval_bundle(bundle, dataset_label="Evaluation"):
    import matplotlib.pyplot as plt

    region_plot_df = bundle["region_plot_df"]
    tissue_long_df = bundle["tissue_long_df"]

    # Plot 1: per-region Dice
    fig_h = max(10, min(30, 0.22 * len(region_plot_df) + 2))
    fig1, ax1 = plt.subplots(figsize=(12, fig_h))
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    colors = plt.cm.viridis(norm(region_plot_df["dice"].values))
    ax1.barh(region_plot_df["label"], region_plot_df["dice"], color=colors, edgecolor="none")
    ax1.set_xlim(0.0, 1.0)
    ax1.set_xlabel("Dice")
    ax1.set_ylabel("Region")
    ax1.set_title(f"{dataset_label}: Per-Region Dice (Model)")
    ax1.grid(axis="x", alpha=0.2)
    mean_dice = float(region_plot_df["dice"].mean()) if len(region_plot_df) else np.nan
    median_dice = float(region_plot_df["dice"].median()) if len(region_plot_df) else np.nan
    if np.isfinite(mean_dice):
        ax1.axvline(mean_dice, color="black", linestyle="--", linewidth=1.2, label=f"mean={mean_dice:.3f}")
    if np.isfinite(median_dice):
        ax1.axvline(median_dice, color="gray", linestyle=":", linewidth=1.2, label=f"median={median_dice:.3f}")
    if ax1.get_legend_handles_labels()[0]:
        ax1.legend(loc="lower right")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, pad=0.01)
    cbar.set_label("Dice")
    plt.tight_layout()

    # Plot 2: tissue Dice distribution
    order = ["CSF", "GM", "WM", "FG"]
    data = [tissue_long_df.loc[tissue_long_df["tissue"] == t, "dice_plot"].values for t in order]
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    positions = np.arange(1, len(order) + 1)
    vp = ax2.violinplot(data, positions=positions, widths=0.8, showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(vp["bodies"], ["#4C78A8", "#59A14F", "#F28E2B", "#B07AA1"]):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.55)
    ax2.boxplot(
        data,
        positions=positions,
        widths=0.20,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "black", "linewidth": 1.2},
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.0},
        whiskerprops={"color": "black", "linewidth": 1.0},
        capprops={"color": "black", "linewidth": 1.0},
    )
    ax2.set_xticks(positions)
    ax2.set_xticklabels(order)
    is_percent = bool(len(tissue_long_df) and str(tissue_long_df["dice_unit"].iloc[0]) == "%")
    ax2.set_ylabel("Dice score (%)" if is_percent else "Dice score")
    if is_percent:
        ax2.set_ylim(0, 100)
    ax2.set_title(f"{dataset_label}: Tissue Dice Distribution")
    ax2.grid(axis="y", alpha=0.2)
    for i, arr in enumerate(data, start=1):
        if len(arr) > 0:
            med = float(np.nanmedian(arr))
            y_txt = min(99.0, med + 1.0) if is_percent else med + 0.01
            ax2.text(i, y_txt, f"{med:.1f}" if is_percent else f"{med:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()

    return (fig1, ax1), (fig2, ax2)


__all__ = [
    "_sync_cuda",
    "predict_volume_forward_dense",
    "predict_volume_forward_default",
    "benchmark_inference_speed",
    "predict_volume_from_unpadded",
    "prepare_napari_sample_from_test_dataset",
    "dense_labels_to_fs_ids",
    "csf_fs_ids_from_class_values",
    "build_csf_binary_mask",
    "_test_x_files",
    "_domain_from_path",
    "_safe_pearson_np",
    "_class_confusion_and_stats",
    "_surface_voxels",
    "_hd95_assd_foreground",
    "_aggregate_region_metrics_from_confusion",
    "_sum_confusions",
    "_select_failed_sample_ids",
    "filter_sparse_regions",
    "build_tissue_assignment_df",
    "infer_tissue_class_indices",
    "_dice_from_confusion_subset",
    "_compute_tissue_dice_from_confusion",
    "collect_test_metrics_fast",
    "collect_test_metrics_fast_cached",
    "evaluate_mgz_throughput_cached",
    "plot_eval_bundle",
    "build_eval_plot_bundle",
    "build_eval_dataset_report",
    "prepare_region_dice_plot_df",
    "summarize_sample_metrics_by_method",
    "summarize_domain_robustness",
    "summarize_timing_df",
    "summarize_tissue_long_df",
    "build_tissue_long_df",
    "build_mgz_preprocess_cache",
    "preprocess_mgz_pair_to_padded_tensors",
    "discover_valid_mgz_pairs",
]

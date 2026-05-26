import hashlib
import json
import os
import random
import re
from pathlib import Path
from tqdm.auto import tqdm

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from nibabel.freesurfer.io import write_geometry

from .config import DATA_CFG, DEVICE, PATCH_SIZE, SEED, TRAIN_CFG


def resolve_paths(items):
    if items is None:
        return []
    if isinstance(items, (str, Path)):
        items = [items]

    paths = []
    for item in items:
        p = Path(item).expanduser()
        if p.suffix.lower() in {".txt", ".lst"}:
            lines_ = [ln.strip() for ln in p.read_text().splitlines() if ln.strip() and not ln.strip().startswith("#")]
            paths.extend(lines_)
        elif p.suffix.lower() == ".json":
            payload = json.loads(p.read_text())
            if not isinstance(payload, list):
                raise ValueError(f"Expected list in {p}, got {type(payload)}")
            paths.extend([str(x) for x in payload])
        else:
            paths.append(str(p))

    return [str(Path(x).expanduser()) for x in paths]


def _to_batched_3d(t, name):
    # Normalize to [N, D, H, W]
    if t.ndim == 3:
        t = t.unsqueeze(0)
    elif t.ndim == 4:
        pass
    elif t.ndim == 5 and t.shape[1] == 1:
        t = t[:, 0]
    else:
        raise ValueError(f"{name}: expected [D,H,W], [N,D,H,W], or [N,1,D,H,W], got {tuple(t.shape)}")
    return t


def load_tensor_pt(path):
    # Requested simple loader path.
    t = torch.load(path)
    if not torch.is_tensor(t):
        raise TypeError(f"{path} did not load as a Tensor. Got: {type(t)}")
    return _to_batched_3d(t, path)


def _group_key_from_path(x_path, root):
    p = Path(x_path).resolve()
    root = Path(root).resolve()
    try:
        rel = p.relative_to(root)
        return rel.parts[0] if len(rel.parts) > 0 else p.parent.name
    except Exception:
        return p.parent.as_posix()


def _group_split_counts(n, ratios):
    tr, va, te = ratios
    if n <= 0:
        return 0, 0, 0

    n_val = int(round(n * va))
    n_test = int(round(n * te))

    # For reasonably sized groups, enforce non-empty val/test.
    if n >= 10:
        n_val = max(1, n_val)
        n_test = max(1, n_test)

    # Ensure at least 1 train sample.
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


def split_pairs_by_group(x_files, y_files, root, ratios=(0.8, 0.1, 0.1), seed=1337):
    if len(x_files) != len(y_files):
        raise ValueError("x_files and y_files must have same length")

    grouped = {}
    for x, y in zip(x_files, y_files):
        k = _group_key_from_path(x, root)
        grouped.setdefault(k, []).append((x, y))

    x_tr, y_tr, x_va, y_va, x_te, y_te = [], [], [], [], [], []

    for gk in sorted(grouped.keys()):
        pairs = grouped[gk][:]
        rng = random.Random(int(seed) + sum(ord(c) for c in gk))
        rng.shuffle(pairs)

        n_train, n_val, n_test = _group_split_counts(len(pairs), ratios)

        train_pairs = pairs[:n_train]
        val_pairs = pairs[n_train:n_train + n_val]
        test_pairs = pairs[n_train + n_val:n_train + n_val + n_test]

        for x, y in train_pairs:
            x_tr.append(x); y_tr.append(y)
        for x, y in val_pairs:
            x_va.append(x); y_va.append(y)
        for x, y in test_pairs:
            x_te.append(x); y_te.append(y)

    return x_tr, y_tr, x_va, y_va, x_te, y_te


def summarize_group_splits(x_train_files, x_val_files, x_test_files, root):
    train_counts = {}
    val_counts = {}
    test_counts = {}

    for p in x_train_files:
        k = _group_key_from_path(p, root)
        train_counts[k] = train_counts.get(k, 0) + 1
    for p in x_val_files:
        k = _group_key_from_path(p, root)
        val_counts[k] = val_counts.get(k, 0) + 1
    for p in x_test_files:
        k = _group_key_from_path(p, root)
        test_counts[k] = test_counts.get(k, 0) + 1

    keys = sorted(set(train_counts.keys()) | set(val_counts.keys()) | set(test_counts.keys()))
    if not keys:
        return

    print("per-group file split (train/val/test):")
    for k in keys:
        tr = train_counts.get(k, 0)
        va = val_counts.get(k, 0)
        te = test_counts.get(k, 0)
        n = tr + va + te
        print(f"  {k}: {tr}/{va}/{te} (n={n})")


def build_label_lut(label_values):
    values = torch.as_tensor(label_values, dtype=torch.int64)
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("label_values must be a non-empty 1D sequence")
    values = torch.unique(values).sort().values
    if int(values[0].item()) < 0:
        raise ValueError("label_values must be non-negative")
    lut = torch.full((int(values[-1].item()) + 1,), -1, dtype=torch.int64)
    lut[values] = torch.arange(values.numel(), dtype=torch.int64)
    return values, lut


def infer_label_values(y_files, max_files=64, seed=SEED):
    if not y_files:
        raise ValueError("Cannot infer labels from an empty list")

    files = list(y_files)
    if max_files is not None and int(max_files) > 0 and len(files) > int(max_files):
        rng = random.Random(int(seed))
        rng.shuffle(files)
        files = files[: int(max_files)]

    values = set()
    for yp in files:
        y = load_tensor_pt(yp).to(torch.int64)
        values.update(int(v) for v in torch.unique(y).tolist())

    return sorted(values), len(files)

def default_aparc_aseg_label_values():
    # Common FreeSurfer aparc+aseg IDs mapped to a compact contiguous class space.
    vals = [
        0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 30,
        31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62, 63, 72,
        77, 80, 85, 251, 252, 253, 254, 255,
    ]
    vals.extend(range(1000, 1036))
    vals.extend(range(2000, 2036))
    return vals


def dtype_from_name(name):
    n = str(name).lower()
    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "int64": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
    }
    if n not in table:
        raise ValueError(f"Unsupported dtype name: {name}")
    return table[n]


def _resolved_pair_id(x_path, y_path):
    return f"{Path(x_path).resolve()}|{Path(y_path).resolve()}"


def _cache_info_for_pair(x_path, y_path, cache_dir, cache_tag):
    pair_id = _resolved_pair_id(x_path, y_path)
    pair_key = hashlib.sha1(pair_id.encode("utf-8")).hexdigest()
    digest = hashlib.sha1(f"{pair_id}|{cache_tag}".encode("utf-8")).hexdigest()[:16]
    x_name = Path(x_path).stem
    y_name = Path(y_path).stem
    x_cache = Path(cache_dir) / f"{x_name}.{digest}.x.pt"
    y_cache = Path(cache_dir) / f"{y_name}.{digest}.y.pt"
    return pair_key, x_cache, y_cache


def _cache_paths_for_pair(x_path, y_path, cache_dir, cache_tag):
    _, x_cache, y_cache = _cache_info_for_pair(
        x_path,
        y_path,
        cache_dir=cache_dir,
        cache_tag=cache_tag,
    )
    return x_cache, y_cache


def _save_tensor_atomic(t, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    torch.save(t, tmp)
    os.replace(tmp, out_path)


def _write_json_atomic(payload, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, out_path)


def _load_json_dict(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _load_cache_request_manifest(path):
    payload = _load_json_dict(path)
    requests = payload.get("requests")
    if payload.get("version") != 1 or not isinstance(requests, dict):
        return {"version": 1, "requests": {}}
    return {"version": 1, "requests": requests}


def _cache_request_signature(x_files, y_files, cache_tag):
    h = hashlib.sha1()
    h.update(str(cache_tag).encode("utf-8"))
    h.update(b"\0")
    for xf, yf in zip(x_files, y_files):
        h.update(str(xf).encode("utf-8"))
        h.update(b"\0")
        h.update(str(yf).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def build_cache_for_pairs(x_files, y_files, cache_dir, label_lut, rebuild=False, recheck=False):
    if len(x_files) != len(y_files):
        raise ValueError("x_files and y_files must have same length")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    skipped_db_path = cache_dir / "_skipped.json"
    request_manifest_path = cache_dir / "_pair_manifest.json"
    skipped_db = _load_json_dict(skipped_db_path)
    request_manifest = _load_cache_request_manifest(request_manifest_path)

    cache_x_dtype = dtype_from_name(DATA_CFG.get("cache_x_dtype", "float16"))
    cache_y_dtype = dtype_from_name(DATA_CFG.get("cache_y_dtype", "int16"))
    cache_zscore_x = bool(DATA_CFG.get("cache_zscore_x", True))
    cache_apply_label_lut = bool(DATA_CFG.get("cache_apply_label_lut", True))
    cache_on_load_error = str(DATA_CFG.get("cache_on_load_error", "skip")).strip().lower()
    if cache_on_load_error not in {"skip", "raise"}:
        raise ValueError("DATA_CFG['cache_on_load_error'] must be 'skip' or 'raise'")
    cache_on_label_miss = str(DATA_CFG.get("cache_on_label_miss", "skip")).strip().lower()
    if cache_on_label_miss not in {"skip", "raise"}:
        raise ValueError("DATA_CFG['cache_on_label_miss'] must be 'skip' or 'raise'")
    cache_retry_skipped = bool(DATA_CFG.get("cache_retry_skipped", False))

    lut_sig = "nolut"
    if label_lut is not None:
        lut_sig = str(int((label_lut >= 0).sum().item()))

    cache_tag = (
        f"xd={DATA_CFG.get('cache_x_dtype','float16')}"
        f"|yd={DATA_CFG.get('cache_y_dtype','int16')}"
        f"|zs={int(cache_zscore_x)}"
        f"|map={int(cache_apply_label_lut)}"
        f"|lut={lut_sig}"
    )
    request_sig = _cache_request_signature(x_files, y_files, cache_tag=cache_tag)

    if (not rebuild) and (not recheck) and (not cache_retry_skipped):
        request_entry = request_manifest["requests"].get(request_sig)
        out_x_cached = request_entry.get("out_x") if isinstance(request_entry, dict) else None
        out_y_cached = request_entry.get("out_y") if isinstance(request_entry, dict) else None
        skipped_cached = request_entry.get("skipped", 0) if isinstance(request_entry, dict) else 0
        if isinstance(out_x_cached, list) and isinstance(out_y_cached, list) and len(out_x_cached) == len(out_y_cached):
            return [str(p) for p in out_x_cached], [str(p) for p in out_y_cached], 0, len(out_x_cached), int(skipped_cached)

    out_x, out_y = [], []
    built = 0
    reused = 0
    skipped = 0

    iterator = tqdm(zip(x_files, y_files), total=len(x_files), desc="cache build/check", leave=False)

    for xf, yf in iterator:
        pair_key, x_cache, y_cache = _cache_info_for_pair(
            xf,
            yf,
            cache_dir=cache_dir,
            cache_tag=cache_tag,
        )

        if (not rebuild) and x_cache.exists() and y_cache.exists():
            out_x.append(str(x_cache))
            out_y.append(str(y_cache))
            reused += 1
            if pair_key in skipped_db:
                skipped_db.pop(pair_key, None)
            continue

        if (not rebuild) and (not cache_retry_skipped) and pair_key in skipped_db:
            skipped += 1
            continue

        try:
            x = load_tensor_pt(xf)
            y = load_tensor_pt(yf)
        except Exception as e:
            msg = f"Failed to load source tensors: {type(e).__name__}: {e}"
            if cache_on_load_error == "skip":
                skipped_db[pair_key] = {"x": str(xf), "y": str(yf), "reason": msg}
                skipped += 1
                continue
            raise RuntimeError(f"{msg} | x={xf} | y={yf}") from e

        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Sample count mismatch: {xf} has {x.shape[0]}, {yf} has {y.shape[0]}")
        if tuple(x.shape[1:]) != tuple(y.shape[1:]):
            raise ValueError(f"Spatial mismatch: {xf} {tuple(x.shape[1:])} vs {yf} {tuple(y.shape[1:])}")
        if y.dtype.is_floating_point:
            raise TypeError(f"{yf}: labels must be integer class IDs, got float dtype {y.dtype}")
        if x.shape[0] != 1:
            raise ValueError(
                f"{xf}: expected one sample per file ([D,H,W] or [1,D,H,W]), got {tuple(x.shape)}. "
                "Split batched tensors into one sample per file to use this cache pipeline."
            )

        x3 = x[0].to(torch.float32)
        y3 = y[0].to(torch.int64)

        if cache_apply_label_lut and label_lut is not None:
            y_max = int(y3.max().item())
            if y_max >= label_lut.numel():
                msg = (
                    f"Label id {y_max} exceeds current LUT range {label_lut.numel()-1}. "
                    "Set DATA_CFG['label_values'] explicitly or set cache_on_label_miss='skip'."
                )
                if cache_on_label_miss == "skip":
                    skipped_db[pair_key] = {"x": str(xf), "y": str(yf), "reason": msg}
                    skipped += 1
                    continue
                raise ValueError(msg)
            y_map = label_lut[y3]
            if bool((y_map < 0).any()):
                msg = (
                    "Found labels not present in LUT during cache build. "
                    "Set DATA_CFG['label_values'] explicitly or set cache_on_label_miss='skip'."
                )
                if cache_on_label_miss == "skip":
                    skipped_db[pair_key] = {"x": str(xf), "y": str(yf), "reason": msg}
                    skipped += 1
                    continue
                raise ValueError(msg)
            y3 = y_map

        if cache_zscore_x:
            x3 = (x3 - x3.mean()) / x3.std().clamp_min(1e-6)

        x3 = x3.to(cache_x_dtype).contiguous()
        y3 = y3.to(cache_y_dtype).contiguous()

        _save_tensor_atomic(x3, x_cache)
        _save_tensor_atomic(y3, y_cache)
        out_x.append(str(x_cache))
        out_y.append(str(y_cache))
        skipped_db.pop(pair_key, None)
        built += 1

    request_manifest["requests"][request_sig] = {
        "out_x": out_x,
        "out_y": out_y,
        "skipped": int(skipped),
    }
    _write_json_atomic(skipped_db, skipped_db_path)
    _write_json_atomic(request_manifest, request_manifest_path)
    return out_x, out_y, built, reused, skipped


class LazyTensorPairDataset(Dataset):
    # Strictly on-the-fly: no full-dataset loads during __init__.

    def __init__(
        self,
        x_files,
        y_files,
        zscore_x=True,
        max_open_files=1,
        label_lut=None,
        target_mode="classification",
    ):
        if len(x_files) != len(y_files):
            raise ValueError("x_files and y_files must have same length")

        self.x_files = [str(Path(p)) for p in x_files]
        self.y_files = [str(Path(p)) for p in y_files]
        self.zscore_x = bool(zscore_x)
        self.max_open_files = max(1, int(max_open_files))
        self.length = len(self.x_files)
        if self.length == 0:
            raise ValueError("Dataset is empty")

        self.label_lut = label_lut
        self.target_mode = str(target_mode).strip().lower()
        if self.target_mode not in {"classification", "regression"}:
            raise ValueError("target_mode must be 'classification' or 'regression'")
        self._cache = {}
        self._cache_order = []

    def __len__(self):
        return self.length

    def _open_pair(self, file_idx):
        if file_idx in self._cache:
            return self._cache[file_idx]

        while len(self._cache_order) >= self.max_open_files:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)

        xf = self.x_files[file_idx]
        yf = self.y_files[file_idx]
        x = load_tensor_pt(xf)
        y = load_tensor_pt(yf)

        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Sample count mismatch: {xf} has {x.shape[0]}, {yf} has {y.shape[0]}")
        if tuple(x.shape[1:]) != tuple(y.shape[1:]):
            raise ValueError(f"Spatial mismatch: {xf} {tuple(x.shape[1:])} vs {yf} {tuple(y.shape[1:])}")
        if self.target_mode == "classification" and y.dtype.is_floating_point:
            raise TypeError(f"{yf}: labels must be integer class IDs, got float dtype {y.dtype}")

        if x.shape[0] != 1:
            raise ValueError(
                f"{xf}: expected one sample per file ([D,H,W] or [1,D,H,W]), got {tuple(x.shape)}. "
                "Split batched tensors into one sample per file to keep loading fully on-the-fly."
            )

        x3 = x[0]
        y3 = y[0]
        self._cache[file_idx] = (x3, y3)
        self._cache_order.append(file_idx)
        return x3, y3

    def __getitem__(self, idx):
        if idx < 0:
            idx += self.length

        x, y = self._open_pair(int(idx))
        x = x.to(torch.float32)
        if self.target_mode == "classification":
            y = y.to(torch.int64)
        else:
            y = y.to(torch.float32)

        if self.label_lut is not None:
            y_max = int(y.max().item())
            if y_max >= self.label_lut.numel():
                raise ValueError(
                    f"Label id {y_max} exceeds current LUT range {self.label_lut.numel()-1}. "
                    "Increase DATA_CFG['label_scan_max_files'] or set DATA_CFG['label_values'] explicitly."
                )
            y = self.label_lut[y]
            if bool((y < 0).any()):
                raise ValueError(
                    "Found labels not present in LUT. "
                    "Increase DATA_CFG['label_scan_max_files'] or set DATA_CFG['label_values'] explicitly."
                )

        if self.zscore_x:
            x = (x - x.mean()) / x.std().clamp_min(1e-6)

        return x.contiguous(), y.contiguous()


def collate_pad_to_patch(batch, patch_size=PATCH_SIZE):
    xs, ys = zip(*batch)
    max_d = max(int(x.shape[0]) for x in xs)
    max_h = max(int(x.shape[1]) for x in xs)
    max_w = max(int(x.shape[2]) for x in xs)

    tgt_d = ((max_d + patch_size[0] - 1) // patch_size[0]) * patch_size[0]
    tgt_h = ((max_h + patch_size[1] - 1) // patch_size[1]) * patch_size[1]
    tgt_w = ((max_w + patch_size[2] - 1) // patch_size[2]) * patch_size[2]

    x_out, y_out = [], []
    for x, y in zip(xs, ys):
        d_pad = tgt_d - int(x.shape[0])
        h_pad = tgt_h - int(x.shape[1])
        w_pad = tgt_w - int(x.shape[2])
        pads = (0, w_pad, 0, h_pad, 0, d_pad)
        x_out.append(F.pad(x, pads, mode="constant", value=0.0))
        y_out.append(F.pad(y, pads, mode="constant", value=0))

    return torch.stack(x_out, dim=0), torch.stack(y_out, dim=0)


def limit_dataset(ds, max_samples, seed=SEED):
    if max_samples is None or int(max_samples) >= len(ds):
        return ds
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[: int(max_samples)].tolist()
    return Subset(ds, idx)


def make_loader(ds, batch_size, shuffle, num_workers=TRAIN_CFG["num_workers"], pin_memory=TRAIN_CFG["pin_memory"], prefetch_factor=TRAIN_CFG["prefetch_factor"]):
    num_workers = int(num_workers)
    pin_memory = bool(pin_memory and DEVICE == "cuda")
    kwargs = {
        "dataset": ds,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "collate_fn": collate_pad_to_patch,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**kwargs)


def build_dataset(x_files, y_files, max_open_files, zscore_x, label_lut, target_mode="classification"):
    return LazyTensorPairDataset(
        x_files,
        y_files,
        zscore_x=zscore_x,
        max_open_files=max_open_files,
        label_lut=label_lut,
        target_mode=target_mode,
    )


def load_freesurfer_lut(fs_lut_path):
    pat = re.compile(r"^\s*(\d+)\s+(.+?)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*$")
    rows = []
    with open(fs_lut_path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            m = pat.match(ln)
            if not m:
                continue
            fs_id, fs_name, r, g, b, a = m.groups()
            rows.append({
                "fs_id": int(fs_id),
                "fs_name": fs_name,
                "R": int(r), "G": int(g), "B": int(b), "A": int(a),
            })
    return pd.DataFrame(rows)


def save_surfaces_to_subject_dir(
    surfaces: dict[str, dict[str, np.ndarray]],
    out_dir: str | Path,
    volume_info: dict | None = None,
) -> dict[str, Path]:
    """
    Write predicted surfaces as FreeSurfer binary surface files.
    surfaces: dict of surface_name → {"vertices_ras": [N,3], "faces": [F,3]}
    volume_info: FreeSurfer volume metadata (vox2ras, vox2ras_tkr, etc.) from orig.mgz.
    Returns dict: surface_name → written path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, surf in surfaces.items():
        verts = np.asarray(surf["vertices_ras"], dtype=np.float32)
        faces = np.asarray(surf["faces"], dtype=np.int32)
        if verts.shape[0] == 0:
            print(f"Skipping {name}: empty surface")
            continue
        # CortexODE produces inward normals (negative nibabel signed_vol).
        # FreeSurfer's mri_brainvol_stats needs outward normals (positive signed_vol)
        # for CortexVol = PialVol - WhiteVol to be positive. Flip face winding.
        faces = np.ascontiguousarray(faces[:, [0, 2, 1]])
        out_path = out_dir / name
        if volume_info is not None:
            write_geometry(str(out_path), verts, faces, volume_info=volume_info)
        else:
            write_geometry(str(out_path), verts, faces)
        paths[name] = out_path
    return paths

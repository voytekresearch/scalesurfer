from pathlib import Path

import torch

from .config import DEVICE, PATCH_SIZE

def auto_discover_pairs(root=Path("tensors")):
    root = Path(root)
    if not root.exists():
        return [], []

    x_files, y_files = [], []
    for sub in sorted([p for p in root.rglob("*") if p.is_dir()]):
        x = sub / "rawavg.pt"
        y = sub / "aparc+aseg.pt"
        if x.exists() and y.exists():
            x_files.append(str(x))
            y_files.append(str(y))
    return x_files, y_files

def gpu_mem_gb():
    if DEVICE != "cuda":
        return 0.0, 0.0
    return (
        torch.cuda.memory_allocated() / (1024 ** 3),
        torch.cuda.max_memory_allocated() / (1024 ** 3),
    )

def ensure_divisible(shape, patch_size=PATCH_SIZE):
    if any(s % p != 0 for s, p in zip(shape, patch_size)):
        raise ValueError(f"shape={shape} must be divisible by patch_size={patch_size}")

def patches_to_volume(patches, volume_shape, patch_size=PATCH_SIZE):
    # patches: [B, N, V] -> volume: [B, D, H, W]
    if patches.ndim != 3:
        raise ValueError(f"Expected [B, N, V], got {tuple(patches.shape)}")
    b, n, v = patches.shape
    pd, ph, pw = patch_size
    gd, gh, gw = (volume_shape[0] // pd, volume_shape[1] // ph, volume_shape[2] // pw)
    expected_n = gd * gh * gw
    expected_v = pd * ph * pw
    if (n, v) != (expected_n, expected_v):
        raise ValueError(f"Expected [B, {expected_n}, {expected_v}], got {tuple(patches.shape)}")
    x = patches.reshape(b, gd, gh, gw, pd, ph, pw)
    x = x.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
    return x.reshape(b, *volume_shape)

def volume_to_patches(volume, patch_size=PATCH_SIZE):
    # volume: [B, D, H, W] -> patches: [B, N, V]
    if volume.ndim != 4:
        raise ValueError(f"Expected [B, D, H, W], got {tuple(volume.shape)}")
    b, d, h, w = volume.shape
    ensure_divisible((d, h, w), patch_size)
    pd, ph, pw = patch_size
    gd, gh, gw = (d // pd, h // ph, w // pw)
    x = volume.reshape(b, gd, pd, gh, ph, gw, pw)
    x = x.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
    return x.reshape(b, gd * gh * gw, pd * ph * pw)

def _positions_1d(dim, win, stride):
    starts = list(range(0, dim - win + 1, stride))
    if starts[-1] != dim - win:
        starts.append(dim - win)
    return starts

def make_window_slices(spatial_shape, window_shape, stride_shape):
    d, h, w = spatial_shape
    wd, wh, ww = window_shape
    sd, sh, sw = stride_shape
    zs = _positions_1d(d, wd, sd)
    ys = _positions_1d(h, wh, sh)
    xs = _positions_1d(w, ww, sw)
    return [
        (slice(z0, z0 + wd), slice(y0, y0 + wh), slice(x0, x0 + ww))
        for z0 in zs for y0 in ys for x0 in xs
    ]

def fit_window_to_shape(spatial_shape, desired_window, patch_size=PATCH_SIZE):
    # Clamp requested window to fit current volume and keep divisibility by patch_size.
    if desired_window is None:
        return tuple(int(s) for s in spatial_shape)
    out = []
    for dim, req, p in zip(spatial_shape, desired_window, patch_size):
        w = min(int(req), int(dim))
        w = max(int(p), (w // int(p)) * int(p))
        out.append(w)
    return tuple(out)

def halve_window_shape(window_shape, patch_size=PATCH_SIZE):
    out = []
    for s, p in zip(window_shape, patch_size):
        half = max(int(p), int(s) // 2)
        half = max(int(p), (half // int(p)) * int(p))
        out.append(half)
    return tuple(out)

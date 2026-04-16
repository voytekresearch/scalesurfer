"""Surface metrics."""

from pathlib import Path
import numpy as np
import pandas as pd
from nibabel.freesurfer.io import read_geometry

SURFACE_NAMES = ("lh.white", "rh.white", "lh.pial", "rh.pial")


def symmetric_mean_distance(
    pred_verts: np.ndarray,
    ref_verts: np.ndarray,
    sample_n: int = 10_000,
    rng_seed: int = 0,
) -> float:
    """
    Symmetric mean surface distance (mm) between two point clouds.
    Samples up to sample_n points from each for efficiency.
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(int(rng_seed))
    if pred_verts.shape[0] > sample_n:
        pred_verts = pred_verts[rng.choice(pred_verts.shape[0], sample_n, replace=False)]
    if ref_verts.shape[0] > sample_n:
        ref_verts = ref_verts[rng.choice(ref_verts.shape[0], sample_n, replace=False)]
    if pred_verts.shape[0] == 0 or ref_verts.shape[0] == 0:
        return float("nan")

    tree_ref = cKDTree(ref_verts)
    tree_pred = cKDTree(pred_verts)
    d_pr, _ = tree_ref.query(pred_verts, k=1)
    d_rp, _ = tree_pred.query(ref_verts, k=1)
    return float(0.5 * (d_pr.mean() + d_rp.mean()))


def evaluate_surfaces(
    surfaces: dict[str, dict[str, np.ndarray]],
    subject_dir: str | Path
) -> pd.DataFrame:
    """
    Compare predicted surfaces to FreeSurfer ground truth.
    surfaces: output of extract_surfaces_from_sdf or infer_surfnet.
    Returns DataFrame with (surface, n_pred, n_ref, sym_mean_mm).
    """
    subject_dir = Path(subject_dir)
    rows = []
    for name in SURFACE_NAMES:
        pred_info = surfaces.get(name, {})
        pred_verts = pred_info.get("vertices_ras", np.zeros((0, 3), np.float32))

        ref_path = subject_dir / "surf" / name
        if not ref_path.exists():
            rows.append({"surface": name, "n_pred": pred_verts.shape[0], "n_ref": 0, "sym_mean_mm": float("nan")})
            continue

        ref_verts, _ = read_geometry(str(ref_path))
        ref_verts = np.asarray(ref_verts, dtype=np.float32)
        dist = symmetric_mean_distance(pred_verts, ref_verts)
        rows.append({
            "surface": name,
            "n_pred": int(pred_verts.shape[0]),
            "n_ref": int(ref_verts.shape[0]),
            "sym_mean_mm": dist,
        })
    return pd.DataFrame(rows)

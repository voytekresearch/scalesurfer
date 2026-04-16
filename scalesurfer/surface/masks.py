from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..config import MODULE_PATH
from ..data import load_freesurfer_lut


_FS_LUT_PATH = MODULE_PATH.parent / "FreeSurferColorLUT.txt"

LEFT_WHITE_LABELS = frozenset([2, 4, 7, 10, 11, 12, 13, 17, 26, 28, 254, 255])
RIGHT_WHITE_LABELS = frozenset([27, 41, 43, 49, 50, 51, 52, 53, 58, 60, 254, 255])
SHARED_PIAL_EXCLUDE_LABELS = frozenset([0, 15, 16, 24])
LEFT_PIAL_EXCLUDE_LABELS = frozenset([5, 7, 8, 17, 94])
RIGHT_PIAL_EXCLUDE_LABELS = frozenset([44, 46, 47, 53, 87])
SURFACE_MASK_KEYS = ("lh.white", "rh.white", "lh.pial", "rh.pial")


@lru_cache(maxsize=1)
def _hemi_label_sets() -> tuple[frozenset[int], frozenset[int]]:
    region_df = load_freesurfer_lut(_FS_LUT_PATH)
    left = region_df[
        region_df["fs_name"].str.contains("Left") | region_df["fs_name"].str.contains("ctx-lh")
    ]["fs_id"].astype(int)
    right = region_df[
        region_df["fs_name"].str.contains("Right") | region_df["fs_name"].str.contains("ctx-rh")
    ]["fs_id"].astype(int)
    return frozenset(left.tolist()), frozenset(right.tolist())


def aparc_masks(aparc: np.ndarray) -> dict[str, np.ndarray]:
    """Return accurate broad white/pial volumes derived directly from aparc+aseg."""
    aparc = np.asarray(aparc, dtype=np.int32)
    if aparc.ndim != 3:
        raise ValueError(f"Expected a 3D aparc+aseg volume, got shape={aparc.shape}")

    left_labels, right_labels = _hemi_label_sets()
    left_pial_exclude = LEFT_PIAL_EXCLUDE_LABELS | SHARED_PIAL_EXCLUDE_LABELS
    right_pial_exclude = RIGHT_PIAL_EXCLUDE_LABELS | SHARED_PIAL_EXCLUDE_LABELS

    lh_white = np.isin(aparc, sorted(LEFT_WHITE_LABELS))
    rh_white = np.isin(aparc, sorted(RIGHT_WHITE_LABELS))
    lh_pial = (~np.isin(aparc, sorted(left_pial_exclude))) & np.isin(aparc, sorted(left_labels))
    rh_pial = (~np.isin(aparc, sorted(right_pial_exclude))) & np.isin(aparc, sorted(right_labels))

    masks = {
        "lh.white": np.asarray(lh_white, dtype=bool),
        "rh.white": np.asarray(rh_white, dtype=bool),
        "lh.pial": np.asarray(lh_pial, dtype=bool),
        "rh.pial": np.asarray(rh_pial, dtype=bool),
    }

    # Backward-compatible aliases for tmp.py-style exploratory code.
    masks["wm_left_mask_broad"] = masks["lh.white"]
    masks["wm_right_mask_broad"] = masks["rh.white"]
    masks["pial_left_mask_broad"] = masks["lh.pial"]
    masks["pial_right_mask_broad"] = masks["rh.pial"]
    masks["lh.white.volume"] = masks["lh.white"]
    masks["rh.white.volume"] = masks["rh.white"]
    masks["lh.pial.volume"] = masks["lh.pial"]
    masks["rh.pial.volume"] = masks["rh.pial"]
    return masks


def combined_surface_volume(aparc: np.ndarray, *, surface_kind: str) -> np.ndarray:
    masks = aparc_masks(aparc)
    if surface_kind == "white":
        left = masks["lh.white"]
        right = masks["rh.white"]
    elif surface_kind == "pial":
        left = masks["lh.pial"]
        right = masks["rh.pial"]
    else:
        raise ValueError(f"surface_kind must be 'white' or 'pial', got {surface_kind!r}")

    combined = np.zeros_like(np.asarray(aparc, dtype=np.int16), dtype=np.int16)
    combined[left] = 1
    combined[right] = 2
    return combined


__all__ = [
    "LEFT_PIAL_EXCLUDE_LABELS",
    "LEFT_WHITE_LABELS",
    "RIGHT_PIAL_EXCLUDE_LABELS",
    "RIGHT_WHITE_LABELS",
    "SHARED_PIAL_EXCLUDE_LABELS",
    "SURFACE_MASK_KEYS",
    "aparc_masks",
    "combined_surface_volume",
]

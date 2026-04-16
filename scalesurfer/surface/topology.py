"""Topology."""

import gzip
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

from .CortexODE.util.tca import bit_map, tca_fill, tca_mask_fill


class TopologyCorrector:
    """Fast topology-preserving correction for implicit surfaces.

    The implicit field must use the standard CortexODE sign convention:

    - negative inside the object
    - positive outside the object

    ``threshold`` controls how much of the field is treated as object-side
    during initialization. For clipped neural level sets, values around
    ``0.5`` to ``2.0`` are usually more sensible than the much larger
    thresholds used on wide-range distance transforms.
    """

    def __init__(self, lut_path: str | Path | None = None) -> None:
        lut_path = Path(lut_path) if lut_path is not None else _default_topology_lut_path()
        with gzip.open(str(lut_path), "rb") as lut_file:
            self.lut = lut_file.read()
        self.bit = bit_map()
        self.lut_path = Path(lut_path)

        toy = np.ones((10, 10, 10), dtype=np.float64)
        toy[4:6, 4:6, 4:6] = -1.0
        mask, init_pts = tca_mask_fill(toy, threshold=0.5)
        tca_fill(toy, mask, init_pts, self.bit, self.lut)

    def apply(self, levelset: np.ndarray, threshold: float = 1.0) -> np.ndarray:
        levelset = np.asarray(levelset, dtype=np.float64)
        mask, init_pts = tca_mask_fill(levelset, threshold=float(threshold))
        if init_pts.size == 0:
            raise ValueError(
                "Topology correction initialization produced no frontier points. "
                "The threshold is likely too large for this field's numeric range."
            )
        corrected = tca_fill(levelset, mask, init_pts, self.bit, self.lut)
        return np.asarray(corrected, dtype=np.float32)



def extract_topology_corrected_mesh_from_mask(
    mask: np.ndarray,
    *,
    sigma: float = 0.5,
    topology_threshold: float = 1.0,
    level: float = 0.0,
    step_size: int = 1,
    keep_lcc: bool = True,
    n_smooth: int = 2,
    smooth_lambda: float = 1.0,
    corrector: TopologyCorrector | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Paper-style seed extraction from a binary mask.
    """
    mask = np.asarray(mask, dtype=bool)
    if keep_lcc:
        mask = _largest_component(mask)
    if not mask.any():
        empty_verts = np.zeros((0, 3), dtype=np.float32)
        empty_faces = np.zeros((0, 3), dtype=np.int32)
        empty_field = np.zeros(mask.shape, dtype=np.float32)
        return empty_verts, empty_faces, empty_field
    sdf = signed_distance_from_mask(mask, sigma=float(sigma))
    return extract_topology_corrected_mesh_from_levelset(
        sdf,
        sigma=0.0,
        topology_threshold=float(topology_threshold),
        level=float(level),
        step_size=int(step_size),
        n_smooth=int(n_smooth),
        smooth_lambda=float(smooth_lambda),
        corrector=corrector,
    )


def extract_topology_corrected_mesh_from_levelset(
    levelset: np.ndarray,
    *,
    sigma: float = 0.5,
    topology_threshold: float = 1.0,
    level: float = 0.0,
    step_size: int = 1,
    n_smooth: int = 2,
    smooth_lambda: float = 1.0,
    corrector: TopologyCorrector | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Seed extraction from an implicit field.

    The sequence is:

    1. optional Gaussian smoothing of the field
    2. topology correction
    3. marching cubes on the corrected field
    4. optional mesh smoothing

    Returns ``(verts, faces, corrected_levelset)``.
    """
    field = np.asarray(levelset, dtype=np.float32)
    if sigma > 0:
        field = ndi.gaussian_filter(field, sigma=float(sigma)).astype(np.float32)
    corrected = topology_correct_levelset(field, threshold=float(topology_threshold), corrector=corrector)
    verts, faces = extract_mesh_from_levelset(corrected, level=float(level), step_size=int(step_size))
    if verts.shape[0] > 0 and faces.shape[0] > 0 and n_smooth > 0:
        verts = laplacian_smooth_mesh(verts, faces, n_iters=int(n_smooth), lambd=float(smooth_lambda))
    return verts, faces, corrected


def signed_distance_from_mask(mask: np.ndarray, *, sigma: float = 0.5) -> np.ndarray:
    """
    Convert a binary mask into a smooth signed-distance-like field.

    Negative values are inside the mask and positive values are outside. This is
    the sign convention expected by the topology-correction stage and by
    zero-level marching cubes.
    """
    mask = np.asarray(mask, dtype=bool)
    sdf = -ndi.distance_transform_cdt(mask) + ndi.distance_transform_cdt(~mask)
    sdf = np.asarray(sdf, dtype=np.float32)
    if sigma > 0:
        sdf = ndi.gaussian_filter(sdf, sigma=float(sigma))
    return sdf.astype(np.float32)


def extract_mesh_from_levelset(
    levelset: np.ndarray,
    *,
    level: float = 0.0,
    step_size: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a seed mesh from an implicit surface / level-set volume.

    Intuition:
    the zero level set of the volume is treated as the current surface, and
    marching cubes turns that continuous boundary into an explicit triangle mesh.

    This is often a better seed source than a hard binary mask because the
    implicit field already contains sub-voxel boundary information.
    """
    vol = np.asarray(levelset, dtype=np.float32)
    if vol.min() >= level or vol.max() <= level:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)

    try:
        verts, faces, _, _ = marching_cubes(
            vol,
            level=float(level),
            step_size=max(1, int(step_size)),
            allow_degenerate=False,
        )
    except RuntimeError:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def laplacian_smooth_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    n_iters: int = 1,
    lambd: float = 1.0,
) -> np.ndarray:
    """
    Uniform Laplacian smoothing on a triangle mesh in voxel coordinates.
    """
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    if verts.size == 0 or faces.size == 0 or n_iters <= 0:
        return verts.copy()

    out = verts.astype(np.float32, copy=True)
    edges = _face_vertex_adjacency(faces)
    for _ in range(int(n_iters)):
        sums = np.zeros_like(out, dtype=np.float32)
        counts = np.zeros((out.shape[0], 1), dtype=np.float32)
        np.add.at(sums, edges[:, 0], out[edges[:, 1]])
        np.add.at(counts[:, 0], edges[:, 0], 1.0)
        neighbor_mean = sums / np.maximum(counts, 1.0)
        mask = counts[:, 0] > 0
        out[mask] = out[mask] + float(lambd) * (neighbor_mean[mask] - out[mask])
    return out.astype(np.float32)



def inflate_and_smooth_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    n_iters: int = 2,
    smooth_lambda: float = 1.0,
    normal_step: float = 0.25,
) -> np.ndarray:
    """
    Apply the white-to-pial handoff from the CortexODE pipeline.

    Each iteration smooths the mesh slightly and then nudges vertices outward
    along their normals.
    """
    out = np.asarray(verts, dtype=np.float32).copy()
    faces = np.asarray(faces, dtype=np.int32)
    if out.size == 0 or faces.size == 0 or n_iters <= 0:
        return out
    for _ in range(int(n_iters)):
        out = laplacian_smooth_mesh(out, faces, n_iters=1, lambd=float(smooth_lambda))
        out = out + float(normal_step) * compute_vertex_normals(out, faces)
    return out.astype(np.float32)


def compute_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute area-weighted vertex normals for a triangle mesh.
    """
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    if verts.size == 0 or faces.size == 0:
        return np.zeros_like(verts, dtype=np.float32)

    tri = verts[faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = np.zeros_like(verts, dtype=np.float32)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    denom = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(denom, 1e-6)
    return normals.astype(np.float32)


def _default_topology_lut_path() -> Path:
    candidates = [
        Path(__file__).resolve().parent / "CortexODE" / "util" / "critical186LUT.raw.gz",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not locate critical186LUT.raw.gz. Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def _largest_component(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    labels, n_labels = ndi.label(mask)
    if n_labels <= 1:
        return mask
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    keep = int(counts.argmax())
    return labels == keep


def topology_correct_levelset(
    levelset: np.ndarray,
    *,
    threshold: float = 1.0,
    corrector: TopologyCorrector | None = None,
) -> np.ndarray:
    """
    Apply CortexODE-style topology correction to an implicit field.
    """
    if corrector is None:
        corrector = _get_default_topology_corrector()
    return corrector.apply(levelset, threshold=float(threshold))



_DEFAULT_TOPOLOGY_CORRECTOR: TopologyCorrector | None = None
def _get_default_topology_corrector() -> TopologyCorrector:
    global _DEFAULT_TOPOLOGY_CORRECTOR
    if _DEFAULT_TOPOLOGY_CORRECTOR is None:
        _DEFAULT_TOPOLOGY_CORRECTOR = TopologyCorrector()
    return _DEFAULT_TOPOLOGY_CORRECTOR


def _face_vertex_adjacency(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
            faces[:, [1, 0]],
            faces[:, [2, 1]],
            faces[:, [0, 2]],
        ],
        axis=0,
    ).astype(np.int64)

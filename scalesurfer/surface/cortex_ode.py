"""Helpers for running the copied pretrained CortexODE models locally.

This module is intentionally narrow in scope: it keeps the author's released
deformation network architecture intact so the shipped checkpoints under
``CortexODE/ckpts/pretrained`` can be used directly, while swapping in our
local preprocessing and topology-corrected seed extraction.

The practical pipeline implemented here is:

1. load ``orig.mgz`` from a FreeSurfer subject
2. preprocess that MRI exactly with the reference CortexODE ``process_volume``
3. derive hemisphere-specific white-matter masks from a native-space
   ``aparc+aseg`` volume
4. run the local fast topology correction and marching cubes to get a white
   seed mesh
5. deform that seed with the pretrained white-surface model
6. inflate and smooth the predicted white surface
7. deform the inflated white surface with the pretrained pial model
8. map the predicted surfaces back into FreeSurfer surface coordinates

Why a separate module?
----------------------
The notebook that uses this path should stay focused on data loading, caching,
visualization, and evaluation. The geometry and model-compatibility details are
easy to get wrong, so it is better to keep them in one tested place.

Important constraint
--------------------
The released pretrained CortexODE checkpoints are hemisphere-specific. That
means pretrained inference still loads separate ``lh`` and ``rh`` white/pial
models. This module does not try to hide that; it simply makes using them
practical in our pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to

from .CortexODE.data.preprocess import process_surface_inverse, process_volume
from .CortexODE.model.net import CortexODE as ReferenceCortexODE

from scalesurfer.surface.masks import aparc_masks
from scalesurfer.surface.topology import (
    extract_topology_corrected_mesh_from_mask,
    inflate_and_smooth_mesh,
)


__all__ = [
    "PretrainedCortexODEConfig",
    "DEFAULT_PRETRAINED_ROOT",
    "load_pretrained_model_bundles",
    "predict_surfaces_from_native_aparc",
    "prepare_reference_volume",
    "resample_tensor_labels_to_native",
]


DEFAULT_PRETRAINED_ROOT = Path(__file__).resolve().parent / "CortexODE" / "ckpts" / "pretrained"
HEMIS = ("lh", "rh")


@dataclass(frozen=True)
class PretrainedCortexODEConfig:
    """Configuration for author-compatible pretrained CortexODE inference.

    Parameters are grouped by stage:

    - reference preprocessing: ``data_name``
    - seed extraction: ``seed_*``
    - white / pial ODE integration: ``*_step_size`` and ``total_time``
    - white-to-pial handoff: ``inflate_*``
    - execution: ``device``

    The defaults mirror the reference ADNI evaluation path closely:

    - ADNI-style MRI preprocessing
    - topology threshold ``16``
    - marching-cubes level ``0.8``
    - 10 Euler steps for white (``dt=0.1``)
    - 20 Euler steps for pial (``dt=0.05``)
    """

    data_name: str = "adni"
    seed_sigma: float = 0.5
    seed_topology_threshold: float = 16.0
    seed_level: float = 0.8
    seed_step_size: int = 2
    seed_smooth_iters: int = 2
    seed_smooth_lambda: float = 1.0
    total_time: float = 1.0
    wm_step_size: float = 0.1
    gm_step_size: float = 0.05
    inflate_iters: int = 2
    inflate_smooth_lambda: float = 1.0
    inflate_voxel_step: float | None = None
    device: str | torch.device = "cpu"


def _as_device(device: str | torch.device) -> torch.device:
    return torch.device(device)


def _integration_steps(step_size: float, total_time: float) -> int:
    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}")
    return max(1, int(round(float(total_time) / float(step_size))))


def _resolve_pretrained_root(
    *,
    pretrained_root: str | Path | None = None,
    data_name: str,
) -> Path:
    root = Path(pretrained_root) if pretrained_root is not None else DEFAULT_PRETRAINED_ROOT / data_name
    if not root.exists():
        raise FileNotFoundError(f"Missing pretrained CortexODE directory: {root}")
    return root


def _load_reference_model(
    *,
    hemi: str,
    surface_kind: str,
    config: PretrainedCortexODEConfig,
    pretrained_root: str | Path | None = None,
) -> ReferenceCortexODE:
    if hemi not in HEMIS:
        raise ValueError(f"hemi must be one of {HEMIS}, got {hemi!r}")
    if surface_kind not in {"wm", "gm"}:
        raise ValueError(f"surface_kind must be 'wm' or 'gm', got {surface_kind!r}")

    ckpt_root = _resolve_pretrained_root(pretrained_root=pretrained_root, data_name=config.data_name)
    ckpt_path = ckpt_root / f"model_{surface_kind}_{config.data_name}_{hemi}_pretrained.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing pretrained checkpoint: {ckpt_path}")

    device = _as_device(config.device)
    model = ReferenceCortexODE(dim_in=3, dim_h=128, kernel_size=5, n_scale=3)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_pretrained_model_bundles(
    *,
    config: PretrainedCortexODEConfig,
    pretrained_root: str | Path | None = None,
) -> dict[str, dict[str, ReferenceCortexODE]]:
    """Load the pretrained white/pial deformation nets for both hemispheres."""
    bundles: dict[str, dict[str, ReferenceCortexODE]] = {}
    for hemi in HEMIS:
        bundles[hemi] = {
            "wm": _load_reference_model(
                hemi=hemi,
                surface_kind="wm",
                config=config,
                pretrained_root=pretrained_root,
            ),
            "gm": _load_reference_model(
                hemi=hemi,
                surface_kind="gm",
                config=config,
                pretrained_root=pretrained_root,
            ),
        }
    return bundles


def resample_tensor_labels_to_native(
    prediction: np.ndarray | torch.Tensor,
    *,
    metadata_path: str | Path,
    native_ref_path: str | Path,
) -> nib.MGHImage:
    """Resample a tensor-space label volume back to a native FreeSurfer grid."""
    import json

    metadata = json.loads(Path(metadata_path).read_text())
    pred = prediction.detach().cpu().numpy() if torch.is_tensor(prediction) else np.asarray(prediction)
    tensor_img = nib.Nifti1Image(np.asarray(np.rint(pred), dtype=np.int16), np.asarray(metadata["tensor_affine"], dtype=np.float64))
    native_ref = nib.load(str(native_ref_path))
    out = resample_from_to(tensor_img, native_ref, order=0)
    data = np.asarray(np.rint(out.get_fdata()), dtype=native_ref.header.get_data_dtype())
    return nib.MGHImage(data, native_ref.affine, header=native_ref.header)


def _mni_tensor_to_conform(arr: np.ndarray) -> np.ndarray:
    """Resample a (197,233,189) MNI-space array to a synthetic 256³ FreeSurfer conform grid.

    ``process_volume`` and ``process_surface_inverse`` both assume a 256³ input:
    the crop indices ``[40:-40, 24:-24, 40:-40]`` and the hardcoded inverse
    arithmetic only give correct results when the input is 256×256×256.
    ``rawavg.pt`` / ``aparc+aseg.pt`` are stored in MNI space (197×233×189);
    this function pads them back to the conformed grid before CortexODE sees them.
    """
    from scalesurfer.convert import MNI_AFFINE, _build_fs_conform_reference
    mni_img = nib.Nifti1Image(np.asarray(arr, dtype=np.float32), MNI_AFFINE)
    conform_ref = _build_fs_conform_reference(mni_img)
    order = 0 if np.issubdtype(arr.dtype, np.integer) else 1
    out = resample_from_to(mni_img, conform_ref, order=order)
    return out.get_fdata(dtype=np.float32)


def prepare_reference_volume(
    subject_dir: str | Path,
    *,
    config: PretrainedCortexODEConfig,
) -> np.ndarray:
    """Load ``orig.mgz`` and preprocess it exactly like the reference code."""
    subject_dir = Path(subject_dir)

    try:
        orig_path = subject_dir / "mri" / "rawavg.pt"
        orig_arr = torch.load(orig_path).cpu().numpy()
        # rawavg.pt is MNI space (197×233×189); process_volume and
        # process_surface_inverse both assume 256³ — pad back before processing.
        orig_arr = _mni_tensor_to_conform(orig_arr)
    except:
        orig_path = subject_dir / "mri" / "orig.mgz"
        if not orig_path.exists():
            raise FileNotFoundError(f"Missing orig.mgz: {orig_path}")
        orig = nib.load(str(orig_path))
        orig_arr = np.asarray(orig.get_fdata(), dtype=np.float32)
    return process_volume((orig_arr / 255.0).astype(np.float32), config.data_name).astype(np.float32)



def _vox_zyx_to_norm_xyz(verts_zyx: np.ndarray, volume_shape: tuple[int, int, int]) -> np.ndarray:
    d, h, w = (int(v) for v in volume_shape)
    scale = float(max(volume_shape))
    verts_xyz = np.asarray(verts_zyx, dtype=np.float32)[:, [2, 1, 0]]
    return (2.0 * verts_xyz - np.array([w, h, d], dtype=np.float32)) / scale


def _norm_xyz_to_vox_zyx(verts_norm_xyz: np.ndarray, volume_shape: tuple[int, int, int]) -> np.ndarray:
    d, h, w = (int(v) for v in volume_shape)
    scale = float(max(volume_shape))
    verts_xyz = (np.asarray(verts_norm_xyz, dtype=np.float32) * scale + np.array([w, h, d], dtype=np.float32)) / 2.0
    return verts_xyz[:, [2, 1, 0]].astype(np.float32)


def _preprocess_white_mask(
    native_aparc: np.ndarray,
    *,
    hemi: str,
    config: PretrainedCortexODEConfig,
) -> np.ndarray:
    masks = aparc_masks(np.asarray(native_aparc, dtype=np.int32))
    white_native = masks[f"{hemi}.white"].astype(np.float32)
    return process_volume(white_native, config.data_name)[0] > 0.5


def _integrate_reference_euler(
    model: ReferenceCortexODE,
    verts_norm_xyz: torch.Tensor,
    *,
    step_size: float,
    total_time: float,
) -> torch.Tensor:
    current = verts_norm_xyz
    for _ in range(_integration_steps(step_size, total_time)):
        current = current + float(step_size) * model(None, current)
    return current


def _surface_entry_from_norm(
    verts_norm_xyz: np.ndarray,
    faces: np.ndarray,
    *,
    config: PretrainedCortexODEConfig,
    volume_shape: tuple[int, int, int],
) -> dict[str, np.ndarray]:
    verts_norm_xyz = np.asarray(verts_norm_xyz, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    verts_vox = _norm_xyz_to_vox_zyx(verts_norm_xyz, volume_shape)
    verts_ras, _ = process_surface_inverse(verts_norm_xyz.copy(), faces.copy(), config.data_name)
    return {
        "verts_norm_xyz": verts_norm_xyz,
        "verts_vox": verts_vox.astype(np.float32),
        "vertices_ras": np.asarray(verts_ras, dtype=np.float32),
        "faces": faces,
    }


def predict_surfaces_from_native_aparc(
    *,
    subject_dir: str | Path,
    native_aparc: np.ndarray,
    config: PretrainedCortexODEConfig,
    pretrained_root: str | Path | None = None,
    model_bundles: dict[str, dict[str, ReferenceCortexODE]] | None = None,
) -> dict[str, Any]:
    """Run pretrained CortexODE from a native-space ``aparc+aseg`` volume.

    Parameters
    ----------
    subject_dir:
        FreeSurfer subject directory containing ``mri/orig.mgz`` and GT surfaces.
    native_aparc:
        Native-grid ``aparc+aseg`` volume, usually either ground truth or a
        tensor-space prediction resampled back to ``orig.mgz``.
    config:
        Inference settings for the reference preprocessing, seed extraction, and
        Euler integration.
    pretrained_root:
        Optional override for the pretrained checkpoint directory. Defaults to
        ``CortexODE/ckpts/pretrained/{data_name}``.
    model_bundles:
        Optional preloaded models from ``load_pretrained_model_bundles``.

    Returns
    -------
    dict
        A bundle with keys:

        - ``surfaces``: ``surface_name -> {vertices_ras, faces, verts_vox, ...}``
        - ``seed_surfaces``: topology-corrected white seeds for ``lh`` and ``rh``
        - ``volume_shape``: processed MRI shape ``(D, H, W)``
    """
    subject_dir = Path(subject_dir)
    volume_proc = prepare_reference_volume(subject_dir, config=config)
    volume_shape = tuple(int(v) for v in volume_proc.shape[1:])
    device = _as_device(config.device)
    volume_tensor = torch.from_numpy(volume_proc[None]).to(device=device, dtype=torch.float32)

    if model_bundles is None:
        model_bundles = load_pretrained_model_bundles(config=config, pretrained_root=pretrained_root)

    surfaces: dict[str, dict[str, np.ndarray]] = {}
    seed_surfaces: dict[str, dict[str, np.ndarray]] = {}

    for hemi in HEMIS:
        white_mask = _preprocess_white_mask(native_aparc, hemi=hemi, config=config)
        seed_verts_vox, seed_faces, _ = extract_topology_corrected_mesh_from_mask(
            white_mask,
            sigma=float(config.seed_sigma),
            topology_threshold=float(config.seed_topology_threshold),
            level=float(config.seed_level),
            step_size=int(config.seed_step_size),
            n_smooth=int(config.seed_smooth_iters),
            smooth_lambda=float(config.seed_smooth_lambda),
        )
        if seed_verts_vox.shape[0] == 0 or seed_faces.shape[0] == 0:
            raise RuntimeError(f"Seed extraction produced an empty mesh for {hemi}")

        seed_norm_xyz = _vox_zyx_to_norm_xyz(seed_verts_vox, volume_shape)
        seed_surfaces[f"{hemi}.white"] = _surface_entry_from_norm(
            seed_norm_xyz,
            seed_faces,
            config=config,
            volume_shape=volume_shape,
        )

        wm_model = model_bundles[hemi]["wm"]
        gm_model = model_bundles[hemi]["gm"]

        with torch.no_grad():
            wm_in = torch.from_numpy(seed_norm_xyz).unsqueeze(0).to(device=device, dtype=torch.float32)
            wm_model.set_data(wm_in, volume_tensor)
            wm_norm_xyz = _integrate_reference_euler(
                wm_model,
                wm_in,
                step_size=float(config.wm_step_size),
                total_time=float(config.total_time),
            )[0].detach().cpu().numpy()

            wm_surface = _surface_entry_from_norm(
                wm_norm_xyz,
                seed_faces,
                config=config,
                volume_shape=volume_shape,
            )

            inflate_step = (
                float(config.inflate_voxel_step)
                if config.inflate_voxel_step is not None
                else 0.001 * float(max(volume_shape))
            )
            gm_seed_vox = inflate_and_smooth_mesh(
                wm_surface["verts_vox"],
                seed_faces,
                n_iters=int(config.inflate_iters),
                smooth_lambda=float(config.inflate_smooth_lambda),
                normal_step=float(inflate_step),
            )
            gm_seed_norm_xyz = _vox_zyx_to_norm_xyz(gm_seed_vox, volume_shape)
            gm_in = torch.from_numpy(gm_seed_norm_xyz).unsqueeze(0).to(device=device, dtype=torch.float32)
            gm_model.set_data(gm_in, volume_tensor)
            gm_norm_xyz = _integrate_reference_euler(
                gm_model,
                gm_in,
                step_size=float(config.gm_step_size),
                total_time=float(config.total_time),
            )[0].detach().cpu().numpy()

        surfaces[f"{hemi}.white"] = wm_surface
        surfaces[f"{hemi}.pial"] = _surface_entry_from_norm(
            gm_norm_xyz,
            seed_faces,
            config=config,
            volume_shape=volume_shape,
        )

    return {
        "surfaces": surfaces,
        "seed_surfaces": seed_surfaces,
        "volume_shape": volume_shape,
    }

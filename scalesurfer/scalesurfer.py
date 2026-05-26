"""High-level API for interacting with models."""
import os
from time import time
from pathlib import Path
from joblib import Parallel, delayed
from math import ceil
import numpy as np
import torch
from tqdm.auto import tqdm

from scalesurfer.config import DEVICE, MODULE_PATH
import nibabel as nib
from nibabel.processing import resample_from_to

from scalesurfer.convert import prepare_image
from scalesurfer.data import (
    build_label_lut,
    default_aparc_aseg_label_values,
    save_surfaces_to_subject_dir,
)
from scalesurfer.metrics import dense_labels_to_fs_ids
from scalesurfer.surface.cortex_ode import (
    PretrainedCortexODEConfig,
    load_pretrained_model_bundles,
    predict_surfaces_from_native_aparc,
)
from scalesurfer.volume import fs as _fs
from scalesurfer.volume.model import TransUNet3D


_VOLUME_MODEL_FILENAME = "transunet3d.safetensors"
_VOLUME_MODEL_ROOT = MODULE_PATH.parent / "docs" / "notebooks" / "huggingface"
_DEFAULT_VOLUME_HF_NAMESPACE = "rphammonds"
_VOLUME_MODEL_CONFIG = {
    "n_classes": 118,
    "in_channels": 1,
    "base_shape": (208, 240, 192),
    "patch_size": (16, 16, 16),
    "channels": (12, 20, 32, 48, 64, 96),
    "transformer_depth": 2,
    "n_heads": 4,
    "dropout": 0.0,
    "positional_encoding": "sincos",
    "task_type": "classification",
}
_VOLUME_MODEL_SPECS = {
    5: {
        "repo_name": "scalesurfer-v5",
        "checkpoint_dir": MODULE_PATH.parent
        / "docs"
        / "notebooks"
        / "checkpoints_fsv5"
        / "fsv5_20260402_015649",
    },
    6: {
        "repo_name": "scalesurfer-v6",
        "checkpoint_dir": MODULE_PATH.parent
        / "docs"
        / "notebooks"
        / "checkpoints_fsv6"
        / "fsv6_20260402_021927",
    },
    7: {
        "repo_name": "scalesurfer-v7",
        "checkpoint_dir": MODULE_PATH.parent
        / "docs"
        / "notebooks"
        / "checkpoints_fsv7"
        / "fsv7_20260402_031018",
    },
    8: {
        "repo_name": "scalesurfer-v8",
        "checkpoint_dir": MODULE_PATH.parent
        / "docs"
        / "notebooks"
        / "checkpoints_fsv8"
        / "fsv8_20260413_164217",
    },
}


def _normalize_fs_version(fs_version) -> int:
    version = str(fs_version).strip().lower()
    for prefix in ("fsv", "fs", "v"):
        if version.startswith(prefix):
            version = version[len(prefix) :]
            break
    version = version.split(".", 1)[0]
    try:
        normalized = int(version)
    except ValueError as exc:
        raise ValueError(f"Unsupported fs_version {fs_version!r}; expected one of {sorted(_VOLUME_MODEL_SPECS)}") from exc

    if normalized not in _VOLUME_MODEL_SPECS:
        raise ValueError(f"Unsupported fs_version {fs_version!r}; expected one of {sorted(_VOLUME_MODEL_SPECS)}")
    return normalized


def _volume_hf_repo_id(repo_name: str) -> str:
    namespace = os.environ.get("SCALESURFER_HF_NAMESPACE", _DEFAULT_VOLUME_HF_NAMESPACE).strip().strip("/")
    return f"{namespace}/{repo_name}" if namespace else repo_name


def _candidate_volume_checkpoint_paths(spec: dict) -> list[Path]:
    paths = []
    model_root = os.environ.get("SCALESURFER_VOLUME_MODEL_DIR")
    if model_root:
        paths.append(Path(model_root).expanduser() / spec["repo_name"] / _VOLUME_MODEL_FILENAME)

    paths.extend(
        [
            _VOLUME_MODEL_ROOT / spec["repo_name"] / _VOLUME_MODEL_FILENAME,
            spec["checkpoint_dir"] / _VOLUME_MODEL_FILENAME,
        ]
    )
    return paths


def _download_volume_checkpoint(spec: dict) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Install huggingface-hub to download ScaleSurfer volume checkpoints, "
            "or set SCALESURFER_VOLUME_MODEL_DIR to a local model directory."
        ) from exc

    return Path(
        hf_hub_download(
            repo_id=_volume_hf_repo_id(spec["repo_name"]),
            filename=_VOLUME_MODEL_FILENAME,
            repo_type="model",
        )
    )


def _resolve_volume_checkpoint_path(fs_version: int) -> Path:
    spec = _VOLUME_MODEL_SPECS[fs_version]
    for path in _candidate_volume_checkpoint_paths(spec):
        if path.exists():
            return path
    return _download_volume_checkpoint(spec)


def _load_volume_state_dict(path: Path) -> dict:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("Install safetensors to load ScaleSurfer volume checkpoints.") from exc
        return load_file(str(path), device="cpu")

    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        raise ValueError(f"Checkpoint missing model_state: {path}")
    return ckpt["model_state"]


class ScaleSurfer:

    def __init__(
        self,
        anat_files,
        subjects,
        subject_dir,
        *,
        batch_size=1,
        n_jobs_cpu=1,
        fs_version=8,
        device=None,
        progress=True,
        pretrained_data_name="adni",
        conform_backend="auto",
        overwrite=False,
        verbose=True,
    ):
        """
        Initilize object.

        Parameters
        ----------
        """

        self.anat_files = anat_files
        self.subjects = subjects
        self.subject_dir = Path(subject_dir)
        self.batch_size =  batch_size
        self.n_jobs = n_jobs_cpu
        self.fs_version = _normalize_fs_version(fs_version)
        self.pretrained_data_name = pretrained_data_name
        self.conform_backend = conform_backend
        self.overwrite = overwrite
        self.verbose = verbose

        self._model_volume = None
        self._model_surface_bundles = None
        self._surface_config = None
        self.device = DEVICE if device is None else device

        self.chkpt_path_volume = _resolve_volume_checkpoint_path(self.fs_version)

        assert len(anat_files) == len(subjects), "anat_files and subjects must have the same length"
        self.prepare_directories()
        self._tqdm = tqdm if progress else lambda i: i # todo fix this is progress is False


    def _log_timing(self, label: str, elapsed: float) -> None:
        n = len(self.subjects)
        print(f"[scalesurfer] {label}: {elapsed:.1f}s total, {n/elapsed:.2f} img/s ({elapsed/n:.1f}s/img) for {n} subject(s)")

    def _empty_cache(self) -> None:
        """Release unused memory for the active device backend."""
        d = str(self.device)
        if d.startswith("cuda"):
            torch.cuda.empty_cache()
        elif d.startswith("mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    # Model loaders
    @property
    def model_volume(self):
        """Lazy load volumetric model."""
        if self._model_volume is None:
            self._model_volume = TransUNet3D(**_VOLUME_MODEL_CONFIG)
            self._model_volume.load_state_dict(_load_volume_state_dict(self.chkpt_path_volume))
            self._model_volume.to(self.device)
            self._model_volume.eval()
        return self._model_volume


    @property
    def surface_config(self):
        """Lazy build CortexODE config."""
        if self._surface_config is None:
            self._surface_config = PretrainedCortexODEConfig(
                data_name=self.pretrained_data_name,
                device=torch.device(self.device),
            )
        return self._surface_config


    @property
    def model_surface(self):
        """Lazy load pretrained CortexODE model bundles (white + pial, lh + rh)."""
        if self._model_surface_bundles is None:
            self._model_surface_bundles = load_pretrained_model_bundles(config=self.surface_config)
        return self._model_surface_bundles

    # Preprocessing
    def prepare_directories(self):
        """Create FreeSurfer-style directory structure."""
        # base dir
        self.subject_dir.mkdir(parents=True, exist_ok=True)

        for subject in self.subjects:
            # subject-level dirs
            (self.subject_dir / subject).mkdir(exist_ok=True)

            for d in ["label", "mri", "scripts", "stats", "surf", "tmp", "touch", "trash"]:
                # canonical freesurfer dirs
                (self.subject_dir / subject / d).mkdir(exist_ok=True)


    def _prepare_image(self, subject, anat_file, subject_dir):
        out_file = subject_dir / subject / "mri" / "orig.pt"
        img_tensor = prepare_image(
            anat_file,
            subject_dir / subject / "mri" / "orig.mgz",
            conform_backend=self.conform_backend,
        ).to(DEVICE)
        torch.save(img_tensor, out_file)


    def prepare_images(self):
        t0 = time()
        Parallel(n_jobs=self.n_jobs)(
            delayed(self._prepare_image)(subject, anat_file, self.subject_dir)
            for subject, anat_file in tqdm(
                zip(self.subjects, self.anat_files),
                total=len(self.subjects),
                desc="Conforming images"
            )
        )
        if self.verbose:
            self._log_timing("prepare_images", time() - t0)

    # Torch models
    def _predict_volume(self, x):
        """Predict aparc+aseg volumes."""
        aparc_aseg_pred = self.model_volume.predict_volume(x.unsqueeze(1))
        return aparc_aseg_pred


    def predict_volumes(self, batch_size: int | None = None):
        """Predict aparc+aseg for all subject and save."""
        _ = self.model_volume  # trigger lazy load before timing starts
        t0 = time()
        bs = batch_size if batch_size is not None else self.batch_size

        for idx in self._tqdm(
            range(0, len(self.subjects), bs),
            total=ceil(len(self.subjects) / bs),
            desc="Predicting volumes"
        ):
            # Construct orig tensor
            n_subjects = min(bs, len(self.subjects)-idx)
            X = torch.zeros((n_subjects, 256, 256, 256), dtype=torch.float32, device=self.device)
            for i, isub in enumerate(range(idx, idx+n_subjects)):
                X[i] = torch.load(self.subject_dir / self.subjects[isub] / "mri"/ "orig.pt")

            # Predict
            aparc_aseg_pred = self._predict_volume(X)

            # Write
            for i, isub in enumerate(range(idx, idx+n_subjects)):
                torch.save(aparc_aseg_pred[i], self.subject_dir / self.subjects[isub] / "mri" / "aparc+aseg.pt")

        self._model_volume = None
        self._empty_cache()
        if self.verbose:
            self._log_timing("predict_volumes", time() - t0)

    def plot_volume(
        self,
        subject_id: str,
        *,
        draw_cross: bool = False,
        colorbar: bool = False,
        display_mode: str = "mosaic",
        **plot_kwargs,
    ):
        """Plot a predicted aparc+aseg volume over the prepared orig image."""
        from nilearn.plotting import plot_roi

        subject_id = str(subject_id)
        mri_dir = self.subject_dir / subject_id / "mri"
        if not mri_dir.exists():
            known = ", ".join(str(s) for s in self.subjects[:5])
            suffix = "..." if len(self.subjects) > 5 else ""
            raise ValueError(f"Unknown subject_id {subject_id!r}. Known subjects: {known}{suffix}")

        orig_path = mri_dir / "orig.mgz"
        x_path = mri_dir / "orig.pt"
        y_path = mri_dir / "aparc+aseg.pt"
        missing = [str(path) for path in (orig_path, x_path, y_path) if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing required volume file(s): " + ", ".join(missing))

        orig_img = nib.as_closest_canonical(nib.load(str(orig_path)))
        x = torch.load(x_path, map_location="cpu")
        y = torch.load(y_path, map_location="cpu")
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if x.ndim == 4 and x.shape[0] == 1:
            x = x[0]
        if y.ndim == 4 and y.shape[0] == 1:
            y = y[0]
        if x.ndim != 3:
            raise ValueError(f"{x_path} must contain a 3D volume, got shape {x.shape}")
        if y.ndim != 3:
            raise ValueError(f"{y_path} must contain a 3D volume, got shape {y.shape}")

        bg_img = nib.Nifti1Image(x, affine=orig_img.affine)
        roi_img = nib.Nifti1Image(y, affine=orig_img.affine)
        return plot_roi(
            roi_img,
            bg_img,
            draw_cross=draw_cross,
            colorbar=colorbar,
            display_mode=display_mode,
            **plot_kwargs,
        )

    def _predict_surface(self, subject, aparc_fs_np):
        """Predict lh/rh white and pial surfaces for a single subject."""
        subject_dir = self.subject_dir / subject
        result = predict_surfaces_from_native_aparc(
            subject_dir=subject_dir,
            native_aparc=aparc_fs_np,
            config=self.surface_config,
            model_bundles=self.model_surface,
        )
        return result

    @staticmethod
    def _canonical_labels_to_orig_grid(labels: np.ndarray, orig_img: nib.spatialimages.SpatialImage) -> np.ndarray:
        """Map model-label tensors from canonical RAS voxel order back to orig.mgz voxels."""
        canonical_ref = nib.as_closest_canonical(orig_img)
        labels = np.asarray(np.rint(labels), dtype=np.int32)
        if tuple(labels.shape) != tuple(canonical_ref.shape[:3]):
            raise ValueError(
                f"Predicted label shape {labels.shape} does not match canonical orig shape "
                f"{canonical_ref.shape[:3]}"
            )

        label_img = nib.Nifti1Image(labels, canonical_ref.affine)
        native = resample_from_to(label_img, orig_img, order=0)
        return np.asarray(np.rint(native.get_fdata()), dtype=np.int32)


    def predict_surfaces(self, batch_size: int | None = None):
        """Predict lh/rh white and pial surfaces for all subjects and save."""
        _ = self.surface_config  # trigger lazy load before timing starts
        _ = self.model_surface   # trigger lazy load before timing starts
        t0 = time()
        bs = batch_size if batch_size is not None else self.batch_size
        class_values, _ = build_label_lut(default_aparc_aseg_label_values())

        for idx in self._tqdm(
            range(0, len(self.subjects), bs),
            total=ceil(len(self.subjects) / bs),
            desc="Predicting surfaces",
        ):
            batch_subjs = self.subjects[idx : idx + bs]
            for subj in batch_subjs:
                mri_dir = self.subject_dir / subj / "mri"
                aparc_dense = torch.load(mri_dir / "aparc+aseg.pt")
                aparc_fs = dense_labels_to_fs_ids(aparc_dense, class_values=class_values)
                if torch.is_tensor(aparc_fs):
                    aparc_fs = aparc_fs.detach().cpu().numpy()
                aparc_fs_np = np.asarray(aparc_fs, dtype=np.int32)
                orig_img = nib.load(str(mri_dir / "orig.mgz"))
                aparc_native = self._canonical_labels_to_orig_grid(aparc_fs_np, orig_img)
                result = self._predict_surface(subj, aparc_native)
                volume_info = self._surface_volume_info(mri_dir / "orig.mgz")
                save_surfaces_to_subject_dir(
                    result["surfaces"],
                    self.subject_dir / subj / "surf",
                    volume_info=volume_info,
                )

        self._model_surface_bundles = None
        self._surface_config = None
        self._empty_cache()
        if self.verbose:
            self._log_timing("predict_surfaces", time() - t0)

    @staticmethod
    def _surface_volume_info(orig_mgz: Path) -> dict | None:
        """Build nibabel surface volume_info from a FreeSurfer orig.mgz."""
        try:
            img = nib.load(str(orig_mgz))
            affine = img.affine
            voxelsize = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
            return {
                "head": np.array([20], dtype=np.int32),
                "valid": "1  # volume info valid",
                "filename": str(orig_mgz),
                "volume": np.array(img.shape[:3], dtype=np.int32),
                "voxelsize": voxelsize,
                "xras": affine[:3, 0] / voxelsize[0],
                "yras": affine[:3, 1] / voxelsize[1],
                "zras": affine[:3, 2] / voxelsize[2],
                "cras": np.array(img.header["Pxyz_c"], dtype=np.float64),
            }
        except Exception:
            return None

    def run_freesurfer_stats(
        self,
        *,
        freesurfer_home: str | Path | None = None,
        fs_license: str | Path | None = None,
        n_jobs: int | None = None,
        aparc_aseg_stats: bool = True,
        brainvol_stats: bool = False,
    ) -> dict:
        """Run the FreeSurfer morphometry tail on all predicted subjects.

        Writes aseg.presurf.mgz + aparc+aseg.mgz from aparc+aseg.pt, then
        runs the FreeSurfer tail pipeline (annotation -> derived surfaces ->
        ribbon -> volumes -> stats). No symlinks are created anywhere.

        Parameters
        ----------
        freesurfer_home:
            Path to FreeSurfer install. Defaults to $FREESURFER_HOME.
        fs_license:
            Path to FreeSurfer license file. Defaults to $FS_LICENSE.
        n_jobs:
            Number of subjects to process in parallel. Defaults to self.n_jobs.
        aparc_aseg_stats:
            Compute per-ROI aparc.stats (thickness, area, etc.).
        brainvol_stats:
            Compute brainvol.stats. Requires mri/transforms/talairach.xfm;
            run autorecon1 first or set this to False.
        """
        t0 = time()
        class_values, _ = build_label_lut(default_aparc_aseg_label_values())
        fs_home = _fs.resolve_freesurfer_home(freesurfer_home)
        fs_license_path = _fs.resolve_fs_license_path(fs_license, freesurfer_home=fs_home, allow_missing=False)

        import os
        import warnings
        if fs_license_path is not None:
            os.environ["FS_LICENSE"] = str(fs_license_path)

        if self.overwrite:
            stale = [
                s for s in self.subjects
                if any((self.subject_dir / s / rel).exists() for rel in _fs.CORE_OUTPUT_RELPATHS)
            ]
            if stale:
                warnings.warn(
                    f"overwrite=True: pre-existing FreeSurfer output files found for "
                    f"{len(stale)} subject(s) and will be overwritten: {stale[:5]}"
                )

        def _run_one(subj: str) -> dict:
            subj_dir = self.subject_dir / subj
            mri_dir = subj_dir / "mri"

            # Convert aparc+aseg.pt -> aseg.presurf.mgz + aparc+aseg.mgz
            orig_img = nib.load(str(mri_dir / "orig.mgz"))
            aparc_dense = torch.load(mri_dir / "aparc+aseg.pt", weights_only=True)
            aparc_fs = dense_labels_to_fs_ids(aparc_dense, class_values=class_values)
            if torch.is_tensor(aparc_fs):
                aparc_fs = aparc_fs.detach().cpu().numpy()
            aparc_arr = self._canonical_labels_to_orig_grid(aparc_fs, orig_img)
            # Write as int32 MGHImage (NOT reusing orig_img.header which is uint8
            # and would clip aparc+aseg label IDs 1001-1035 to 255).
            seg_img = nib.MGHImage(aparc_arr, orig_img.affine)
            nib.save(seg_img, str(mri_dir / "aseg.presurf.mgz"))
            nib.save(seg_img, str(mri_dir / "aparc+aseg.mgz"))

            # norm.mgz and brainmask.mgz are required by the stats stage.
            if not (mri_dir / "norm.mgz").exists():
                nib.save(orig_img, str(mri_dir / "norm.mgz"))
            if not (mri_dir / "brainmask.mgz").exists():
                mask = nib.MGHImage(
                    (aparc_arr > 0).astype(np.uint8),
                    orig_img.affine,
                )
                nib.save(mask, str(mri_dir / "brainmask.mgz"))

            # mris_anatomical_stats requires talairach.xfm even for surface
            # stats. Write an identity transform so stats run; MNI-normalized
            # volume stats (eTIV) will be inaccurate but ROI morphometry is fine.
            xfm_path = mri_dir / "transforms" / "talairach.xfm"
            if not xfm_path.exists():
                xfm_path.parent.mkdir(parents=True, exist_ok=True)
                xfm_path.write_text(
                    "MNI Transform File\n\n"
                    "Transform_Type = Linear;\n"
                    "Linear_Transform =\n"
                    " 1 0 0 0\n"
                    " 0 1 0 0\n"
                    " 0 0 1 0;\n"
                )

            config = _fs.build_config(
                subject_dir=subj_dir,
                predictions_dir=subj_dir,
                freesurfer_home=fs_home,
                fs_license=fs_license_path,
                link_mode="copy",
                force=self.overwrite,
                verbose=self.verbose,
                run_autorecon1=False,
                brainvol_stats=brainvol_stats,
                aparc_aseg_stats=aparc_aseg_stats,
            )
            try:
                _fs.run(config)
                return {"subject": subj, "status": "ok"}
            except Exception as e:
                return {"subject": subj, "status": f"error: {e}"}

        n = n_jobs if n_jobs is not None else self.n_jobs
        results = Parallel(n_jobs=n)(
            delayed(_run_one)(subj)
            for subj in self._tqdm(self.subjects, desc="FreeSurfer stats")
        )
        statuses = {r["subject"]: r["status"] for r in results}
        failed = [s for s, v in statuses.items() if v.startswith("error")]
        if failed:
            import warnings
            warnings.warn(f"{len(failed)} subject(s) failed FreeSurfer stats: {failed[:5]}")
        if self.verbose:
            self._log_timing("run_freesurfer_stats", time() - t0)
        return statuses

"""High-level API for interacting with models."""
from pathlib import Path
from joblib import Parallel, delayed

import numpy as np
import torch
from tqdm.auto import tqdm

from scalesurfer.config import DEVICE, MODULE_PATH
import nibabel as nib
from nibabel.processing import resample_from_to

from scalesurfer.convert import MNI_AFFINE, prepare_image, _build_fs_conform_reference
from scalesurfer.surface.cortex_ode import _mni_tensor_to_conform
from scalesurfer.data import (
    build_label_lut,
    default_aparc_aseg_label_values,
    save_surfaces_to_subject_dir,
)
from scalesurfer.metrics import dense_labels_to_fs_ids, predict_volume_from_unpadded
from scalesurfer.surface.cortex_ode import (
    PretrainedCortexODEConfig,
    load_pretrained_model_bundles,
    predict_surfaces_from_native_aparc,
)
from scalesurfer.volume.model import TransUNet3D


class ScaleSurfer:

    def __init__(
            self,
            anat_files,
            subjects,
            subject_dir,
            *,
            in_memory=False,
            n_jobs=1,
            fs_version=8,
            device=None,
            progress=True,
            pretrained_data_name="adni",
        ):
        """
        Initilize object.

        Parameters
        ----------
        # todo: doc

        TODO: implement an efficient batch_size.
        """

        self.anat_files = anat_files
        self.subjects = subjects
        self.subject_dir = Path(subject_dir)
        self.n_jobs = n_jobs
        self.fs_version = fs_version
        self.pretrained_data_name = pretrained_data_name

        self._model_volume = None
        self._model_surface_bundles = None
        self._surface_config = None
        self.device = DEVICE if device is None else device

        # todo: load based on fs_version kwarg
        self.chkpt_path_volume = MODULE_PATH.parent / "docs" / "notebooks" / "checkpoints_fsv8" / "fsv8_20260413_164217" / "transunet3d_best.pt"

        assert len(anat_files) == len(subjects), "anat_files and subjects must have the same length"
        self.prepare_directories()
        self._tqdm = tqdm if progress else lambda i: i # todo fix this is progress is False
        self.in_memory = in_memory


    @property
    def model_volume(self):
        """Lazy load volumetric model."""
        if self._model_volume is None:
            self._model_volume = TransUNet3D(
                n_classes=118,
                in_channels=1,
                base_shape=(208, 240, 192),
                patch_size=(16, 16, 16),
                channels=(12, 20, 32, 48, 64, 96),
                transformer_depth=2,
                n_heads=4,
                dropout=0.0,
                positional_encoding="sincos",
            )
            ckpt = torch.load(self.chkpt_path_volume, map_location=self.device)
            self._model_volume.load_state_dict(ckpt["model_state"])
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
        out_file = subject_dir / subject / "mri" / "rawavg.pt"
        if not out_file.exists():
            img_tensor = prepare_image(anat_file).to(DEVICE)
            torch.save(img_tensor, out_file)
        elif self.in_memory:
            img_tensor = torch.load(out_file)

        if self.in_memory:
            return img_tensor
        else:
            return None


    def prepare_images(self):
        img_tensors = Parallel(n_jobs=self.n_jobs)(
            delayed(self._prepare_image)(subject, anat_file, self.subject_dir)
            for subject, anat_file in tqdm(
                zip(self.subjects, self.anat_files),
                total=len(self.subjects),
                desc="Converting niftis to tensors"
            )
        )
        if self.in_memory:
            self._img_tensors = torch.stack(img_tensors) # [B, 1, D, H, W]


    def _predict_volume(self, x):
        """Predict single aparc+aseg."""
        aparc_aseg_pred = predict_volume_from_unpadded(
            model=self.model_volume,
            x_3d=x,
            patch_size=(16, 16, 16),
            device=self.device,
        )
        return aparc_aseg_pred


    def predict_volumes(self):
        """Predict aparc+aseg for all subject and save."""
        for subj in self._tqdm(self.subjects, desc="Predicting volumes"):
            if not self.in_memory:
                x = torch.load(self.subject_dir / subj / "mri"/ "rawavg.pt")
            else:
                x = self._img_tensors[self.subjects.index(subj)]
            aparc_aseg_pred = self._predict_volume(x)
            torch.save(aparc_aseg_pred, self.subject_dir / subj / "mri" / "aparc+aseg.pt")


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


    def predict_surfaces(self):
        """Predict lh/rh white and pial surfaces for all subjects and save."""
        class_values, _ = build_label_lut(default_aparc_aseg_label_values())
        for subj in self._tqdm(self.subjects, desc="Predicting surfaces"):
            aparc_dense = torch.load(self.subject_dir / subj / "mri" / "aparc+aseg.pt")
            aparc_fs = dense_labels_to_fs_ids(aparc_dense, class_values=class_values)
            if torch.is_tensor(aparc_fs):
                aparc_fs = aparc_fs.detach().cpu().numpy()
            aparc_fs_np = np.asarray(aparc_fs, dtype=np.int32)
            # aparc_fs_np is MNI space (197×233×189); pad to 256³ so CortexODE
            # process_volume crop indices and process_surface_inverse math are correct.
            aparc_conform = np.asarray(np.rint(_mni_tensor_to_conform(aparc_fs_np)), dtype=np.int32)

            result = self._predict_surface(subj, aparc_conform)
            pred_surfaces = result["surfaces"]
            save_surfaces_to_subject_dir(pred_surfaces, self.subject_dir / subj / "surf")

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

import scalesurfer.scalesurfer as scalesurfer_module
import scalesurfer.inference._datasets as inference_datasets
import scalesurfer.inference._io as inference_io
from scalesurfer import ScaleSurfer


class _SimulatedVolumeModel(torch.nn.Module):
    n_classes = 118

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def predict_volume_fast(self, x, patch_chunk_size=64):
        # A deterministic stand-in for the downloaded network. It retains the
        # exact input/output contract while keeping this smoke test fast.
        return (x[:, 0] > x[:, 0].mean(dim=(1, 2, 3), keepdim=True)).to(torch.uint8)


class _SimulatedStatsPredictor:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = Path(checkpoint_path)

    def predict_tensors(self, t1, seg, *, subjects, **kwargs):
        return pd.DataFrame(
            {
                "subject": subjects,
                "stats_name": ["aseg"] * len(subjects),
                "region": ["BrainSeg"] * len(subjects),
                "measure": ["Volume_mm3"] * len(subjects),
                "target": ["aseg.BrainSeg.Volume_mm3"] * len(subjects),
                "feature": ["aseg.BrainSeg.Volume_mm3"] * len(subjects),
                "group": ["volume"] * len(subjects),
                "value": [float((item > 0).sum()) for item in seg],
            }
        )


def test_notebook_and_readme_inference_flow_with_simulated_data(tmp_path, monkeypatch):
    anat_path = tmp_path / "sub-sim_T1w.nii.gz"
    grid = np.indices((24, 24, 24), dtype=np.float32)
    data = np.exp(-sum((axis - 12) ** 2 for axis in grid) / 50).astype(np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), anat_path)

    checkpoint = tmp_path / "transunet3d.safetensors"
    checkpoint.touch()
    monkeypatch.setattr(scalesurfer_module, "resolve_volume_checkpoint_path", lambda version: checkpoint)
    monkeypatch.setattr(inference_datasets, "CONFORM_SHAPE", (24, 24, 24))
    monkeypatch.setattr(inference_io, "CONFORM_SHAPE", (24, 24, 24))

    def prepare_simulated(subject, anat_file, subject_dir, **kwargs):
        image = nib.load(anat_file)
        mri_dir = Path(subject_dir) / subject / "mri"
        nib.save(nib.MGHImage(np.asarray(image.dataobj), image.affine), mri_dir / "orig.mgz")
        torch.save(torch.from_numpy(np.asarray(image.dataobj).copy()), mri_dir / "orig.pt")
        return {"subject": subject, "skipped_existing": False}

    monkeypatch.setattr(ScaleSurfer, "_prepare_image", staticmethod(prepare_simulated))
    monkeypatch.setattr("nilearn.plotting.plot_roi", lambda *args, **kwargs: "simulated-display")

    surfer = ScaleSurfer(
        [anat_path],
        ["sub-sim"],
        tmp_path / "subjects",
        device="cpu",
        n_jobs_cpu=1,
        inference_num_workers=0,
        progress=False,
        verbose=False,
    )
    surfer._model_volume = _SimulatedVolumeModel()
    surfer._stats_predictors[(7, None)] = _SimulatedStatsPredictor(checkpoint)

    surfer.prepare_images()
    surfer.predict_volumes()
    stats = surfer.predict_stats()
    display = surfer.plot_volume("sub-sim")

    mri_dir = tmp_path / "subjects" / "sub-sim" / "mri"
    assert (mri_dir / "orig.mgz").is_file()
    assert tuple(torch.load(mri_dir / "orig.pt", weights_only=True).shape) == (24, 24, 24)
    assert tuple(torch.load(mri_dir / "aparc+aseg.pt", weights_only=True).shape) == (24, 24, 24)
    assert list(stats["subject"]) == ["sub-sim"]
    assert display == "simulated-display"

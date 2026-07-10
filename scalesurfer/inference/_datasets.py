"""Datasets used by the high-level inference pipeline."""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from scalesurfer.convert import CONFORM_SHAPE

from ._io import torch_load_weights


class VolumeOrigDataset(Dataset):
    def __init__(self, subject_dir: str | Path, subjects: list[str], indices: list[int]):
        self.subject_dir = Path(subject_dir)
        self.subjects = [str(subject) for subject in subjects]
        self.indices = [int(idx) for idx in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, object]:
        isub = self.indices[int(item)]
        subject = self.subjects[isub]
        orig_path = self.subject_dir / subject / "mri" / "orig.pt"
        orig = torch_load_weights(orig_path, mmap=True)
        if not isinstance(orig, torch.Tensor) or tuple(orig.shape) != tuple(CONFORM_SHAPE):
            shape = tuple(orig.shape) if isinstance(orig, torch.Tensor) else type(orig).__name__
            raise RuntimeError(
                f"{orig_path} has shape {shape}, expected {CONFORM_SHAPE}. "
                "Run surfer.prepare_images(overwrite=True) or delete this stale prepared tensor."
            )
        return {"isub": isub, "subject": subject, "orig": torch.as_tensor(orig).squeeze().contiguous()}


class StatsTensorDataset(Dataset):
    def __init__(self, subject_dir: str | Path, subjects: list[str]):
        self.subject_dir = Path(subject_dir)
        self.subjects = [str(subject) for subject in subjects]

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, item: int) -> dict[str, object]:
        subject = self.subjects[int(item)]
        mri_dir = self.subject_dir / subject / "mri"
        t1_path = mri_dir / "orig.pt"
        seg_path = mri_dir / "aparc+aseg.pt"
        if not t1_path.exists() or not seg_path.exists():
            missing = [str(path) for path in (t1_path, seg_path) if not path.exists()]
            raise FileNotFoundError("Missing required ScaleSurfer tensor(s): " + ", ".join(missing))
        return {
            "subject": subject,
            "t1": torch.as_tensor(torch_load_weights(t1_path, mmap=True)).squeeze().contiguous(),
            "seg": torch.as_tensor(torch_load_weights(seg_path, mmap=True)).squeeze().contiguous(),
        }


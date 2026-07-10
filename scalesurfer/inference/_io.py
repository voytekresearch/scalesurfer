"""Tensor I/O, validation, provenance, and background writer jobs."""

import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import pandas as pd
import torch

from scalesurfer.convert import CONFORM_SHAPE, image_is_loadable
from scalesurfer.stats import write_stats_outputs

from ._settings import VOLUME_LOG_FILENAME, VOLUME_PROVENANCE_FILENAME


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def save_torch_zip_deflated(obj, path: str | Path, *, compresslevel: int = 1) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_tmp = path.with_suffix(path.suffix + ".tmp")
    compressed_tmp = path.with_suffix(path.suffix + ".tmp.deflated")
    try:
        torch.save(obj, raw_tmp)
        with zipfile.ZipFile(raw_tmp, "r") as zin, zipfile.ZipFile(
            compressed_tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=int(compresslevel)
        ) as zout:
            for info in zin.infolist():
                zi = zipfile.ZipInfo(info.filename, date_time=info.date_time)
                zi.external_attr = info.external_attr
                zi.compress_type = zipfile.ZIP_DEFLATED
                zout.writestr(zi, zin.read(info.filename))
        os.replace(compressed_tmp, path)
    finally:
        raw_tmp.unlink(missing_ok=True)
        compressed_tmp.unlink(missing_ok=True)


def torch_load_weights(path: str | Path, *, mmap: bool = False):
    kwargs = {"map_location": "cpu", "weights_only": True}
    if mmap:
        try:
            return torch.load(path, mmap=True, **kwargs)
        except RuntimeError:
            return torch.load(path, **kwargs)
    return torch.load(path, **kwargs)


def prepared_orig_tensor_is_valid(path: str | Path) -> bool:
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        tensor = torch_load_weights(path, mmap=True)
    except Exception:
        return False
    return isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == tuple(CONFORM_SHAPE)


def prepared_orig_mgz_is_valid(path: str | Path) -> bool:
    path = Path(path)
    if not image_is_loadable(path):
        return False
    try:
        return tuple(nib.load(str(path)).shape[:3]) == tuple(CONFORM_SHAPE)
    except Exception:
        return False


def volume_provenance_payload(*, subject: str, fs_version: int, checkpoint_path: str | Path) -> dict:
    return {
        "kind": "aparc+aseg",
        "generator": "ScaleSurfer.predict_volumes",
        "created_at": utc_now_iso(),
        "subject": str(subject),
        "fs_version": int(fs_version),
        "checkpoint_path": str(checkpoint_path),
        "outputs": ["mri/aparc+aseg.pt"],
    }


def write_volume_provenance_files(
    *, subject_dir: str | Path, subject: str, fs_version: int, checkpoint_path: str | Path
) -> None:
    subject_dir = Path(subject_dir) / str(subject)
    scripts_dir = subject_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    payload = volume_provenance_payload(
        subject=str(subject), fs_version=int(fs_version), checkpoint_path=checkpoint_path
    )
    (scripts_dir / VOLUME_PROVENANCE_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (scripts_dir / VOLUME_LOG_FILENAME).write_text(
        "\n".join([
            "ScaleSurfer predicted aparc+aseg",
            f"created_at={payload['created_at']}",
            f"subject={payload['subject']}",
            f"fs_version={payload['fs_version']}",
            f"checkpoint_path={payload['checkpoint_path']}",
        ]) + "\n",
        encoding="utf-8",
    )


def write_volume_prediction_job(
    dense_label: torch.Tensor,
    out_path: str | Path,
    subject_dir: str | Path,
    subject: str,
    fs_version: int,
    checkpoint_path: str | Path,
    save_dtype: torch.dtype | None = None,
) -> str:
    if dense_label.device.type != "cpu" or (save_dtype is not None and dense_label.dtype != save_dtype):
        dense_label = dense_label.detach().to(device="cpu", dtype=save_dtype or dense_label.dtype)
    dense_label = dense_label.contiguous().clone()
    save_torch_zip_deflated(dense_label, out_path)
    write_volume_provenance_files(
        subject_dir=subject_dir,
        subject=subject,
        fs_version=fs_version,
        checkpoint_path=checkpoint_path,
    )
    return str(out_path)


def write_stats_prediction_job(
    subject_frame: pd.DataFrame,
    subjects_dir: str | Path,
    fs_version: int | str,
    checkpoint_path: str | Path | None,
) -> list[str]:
    written = write_stats_outputs(
        subject_frame,
        subjects_dir,
        fs_version=fs_version,
        checkpoint_path=checkpoint_path,
        overwrite=True,
        progress=False,
    )
    return [str(path) for path in written]


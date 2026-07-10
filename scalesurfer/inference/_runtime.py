"""Private runtime behavior mixed into :class:`ScaleSurfer`."""

import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from scalesurfer.convert import prepare_image

from ._io import (
    prepared_orig_mgz_is_valid,
    prepared_orig_tensor_is_valid,
    save_torch_zip_deflated,
    write_volume_provenance_files,
)
from ._models import normalize_fs_version


def normalize_writer_backend(value: str) -> str:
    backend = str(value).strip().lower()
    if backend not in {"thread", "process"}:
        raise ValueError("writer_backend must be 'thread' or 'process'")
    return backend


def default_inference_num_workers(n_jobs_cpu) -> int:
    if n_jobs_cpu is None or int(n_jobs_cpu) == 0:
        return 0
    if int(n_jobs_cpu) < 0:
        return min(4, max(1, os.cpu_count() or 1))
    return max(0, int(n_jobs_cpu))


class ScaleSurferRuntime:
    def _log_timing(self, label: str, elapsed: float, *, n: int | None = None) -> None:
        n = len(self.subjects) if n is None else int(n)
        print(
            f"[scalesurfer] {label}: {elapsed:.1f}s total, {n/elapsed:.2f} img/s "
            f"({elapsed/n:.1f}s/img) for {n} subject(s)"
        )

    def _empty_cache(self) -> None:
        d = str(self.device)
        if d.startswith("cuda"):
            torch.cuda.empty_cache()
        elif d.startswith("mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    def _sync_device(self) -> None:
        d = str(self.device)
        if d.startswith("cuda"):
            torch.cuda.synchronize()
        elif d.startswith("mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()

    def _profile_enabled(self) -> bool:
        value = os.environ.get("SCALESURFER_PROFILE", "")
        return value.strip().lower() not in {"", "0", "false", "no", "off"}

    def _make_inference_loader(self, dataset: Dataset, *, batch_size: int) -> DataLoader:
        num_workers = max(0, int(self.inference_num_workers))
        kwargs = {
            "dataset": dataset,
            "batch_size": max(1, int(batch_size)),
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": bool(self.pin_memory and str(self.device).startswith("cuda")),
        }
        if num_workers > 0:
            kwargs["persistent_workers"] = bool(self.persistent_workers)
            if self.prefetch_factor is not None:
                kwargs["prefetch_factor"] = int(self.prefetch_factor)
        return DataLoader(**kwargs)

    def _read_provenance(self, subject: str, filename: str) -> dict | None:
        path = self.subject_dir / str(subject) / "scripts" / filename
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse ScaleSurfer provenance file: {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"ScaleSurfer provenance file must contain a JSON object: {path}")
        return payload

    def _provenance_fs_version(self, payload: dict | None) -> int | None:
        if not payload:
            return None
        value = payload.get("fs_version", payload.get("freesurfer_version"))
        return None if value is None else normalize_fs_version(value)

    def _check_existing_output_version(
        self,
        subject: str,
        *,
        output_path: Path,
        provenance_filename: str,
        requested_fs_version: int,
        output_label: str,
    ) -> bool:
        if not output_path.exists():
            return False
        payload = self._read_provenance(subject, provenance_filename)
        existing_version = self._provenance_fs_version(payload)
        if existing_version is not None and int(existing_version) != int(requested_fs_version):
            raise ValueError(
                f"Refusing to reuse existing {output_label} for subject {subject!r}: "
                f"it was created with FreeSurfer/model version {existing_version}, "
                f"but the current call requested version {requested_fs_version}. "
                "Pass overwrite=True or use a separate subjects_dir."
            )
        return True

    def _write_volume_provenance(self, subject: str) -> None:
        write_volume_provenance_files(
            subject_dir=self.subject_dir,
            subject=str(subject),
            fs_version=int(self.fs_version),
            checkpoint_path=self.chkpt_path_volume,
        )

    def _prepare_image(
        self,
        subject,
        anat_file,
        subject_dir,
        *,
        compress_orig: bool = False,
        overwrite: bool = False,
    ):
        subject = str(subject)
        mri_dir = subject_dir / subject / "mri"
        out_file = mri_dir / "orig.pt"
        out_mgz = mri_dir / "orig.mgz"
        if not overwrite and prepared_orig_tensor_is_valid(out_file) and prepared_orig_mgz_is_valid(out_mgz):
            return {"subject": subject, "skipped_existing": True, "out_file": str(out_file)}
        img_tensor = prepare_image(
            anat_file, out_mgz, conform_backend=self.conform_backend, overwrite=overwrite
        ).to(dtype=self.volume_dtype).contiguous()
        if img_tensor.device.type != "cpu":
            img_tensor = img_tensor.cpu()
        if compress_orig:
            save_torch_zip_deflated(img_tensor, out_file)
        else:
            torch.save(img_tensor, out_file)
        return {"subject": subject, "skipped_existing": False, "out_file": str(out_file)}

    def _predict_volume(self, x, *, patch_chunk_size=64, volume_dtype=None):
        return self.model_volume.predict_volume_fast(x.unsqueeze(1), patch_chunk_size=patch_chunk_size)

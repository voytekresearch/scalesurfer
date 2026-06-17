"""High-level API for interacting with models."""
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
import json
import os
import zipfile
from datetime import datetime, timezone
from time import perf_counter, time
from pathlib import Path
from joblib import Parallel, delayed
from math import ceil
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from scalesurfer.config import DEVICE
import nibabel as nib
from nibabel.processing import resample_from_to
import pandas as pd

from scalesurfer.convert import CONFORM_SHAPE, image_is_loadable, prepare_image
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
from scalesurfer.stats import (
    PREDICTED_STATS_LONG_FILENAME,
    PREDICTED_STATS_PROVENANCE_FILENAME,
    ScaleSurferStatsPredictor,
    load_stats_features,
    stats_long_to_wide,
    write_stats_outputs,
)
from scalesurfer.volume import fs as _fs
from scalesurfer.volume.model import TransUNet3D


_VOLUME_MODEL_FILENAME = "transunet3d.safetensors"
_DEFAULT_VOLUME_HF_NAMESPACE = "rphammonds"
_VOLUME_MODEL_CONFIG = {
    "n_classes": 118,
    "in_channels": 1,
    "base_shape": (256, 256, 256),
    "patch_size": (16, 16, 16),
    "channels": (12, 20, 32, 48, 64, 96),
    "transformer_depth": 2,
    "n_heads": 4,
    "dropout": 0.0,
    "positional_encoding": "sincos",
    "task_type": "classification",
}
_VOLUME_DTYPE_ALIASES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "f32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "f16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}
_VOLUME_MODEL_SPECS = {
    5: {"repo_name": "scalesurfer-v5"},
    6: {"repo_name": "scalesurfer-v6"},
    7: {"repo_name": "scalesurfer-v7"},
    8: {"repo_name": "scalesurfer-v8"},
}
_VOLUME_PROVENANCE_FILENAME = "scalesurfer_aparc_aseg.json"
_VOLUME_LOG_FILENAME = "scalesurfer_aparc_aseg.log"


def _normalize_writer_backend(value: str) -> str:
    backend = str(value).strip().lower()
    if backend not in {"thread", "process"}:
        raise ValueError("writer_backend must be 'thread' or 'process'")
    return backend


def _default_inference_num_workers(n_jobs_cpu) -> int:
    if n_jobs_cpu is None:
        return 0
    n_jobs = int(n_jobs_cpu)
    if n_jobs == 0:
        return 0
    if n_jobs < 0:
        return min(4, max(1, os.cpu_count() or 1))
    return max(0, n_jobs)


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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _volume_hf_repo_id(repo_name: str) -> str:
    namespace = os.environ.get("SCALESURFER_HF_NAMESPACE", _DEFAULT_VOLUME_HF_NAMESPACE).strip().strip("/")
    return f"{namespace}/{repo_name}" if namespace else repo_name


def _candidate_volume_checkpoint_paths(spec: dict) -> list[Path]:
    paths = []
    model_root = os.environ.get("SCALESURFER_VOLUME_MODEL_DIR")
    if model_root:
        paths.append(Path(model_root).expanduser() / spec["repo_name"] / _VOLUME_MODEL_FILENAME)
    return paths


def _download_volume_checkpoint(spec: dict, *, local_files_only: bool = False) -> Path:
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
            local_files_only=local_files_only,
        )
    )


def _resolve_volume_checkpoint_path(fs_version: int, *, local_files_only: bool = False) -> Path:
    spec = _VOLUME_MODEL_SPECS[fs_version]
    for path in _candidate_volume_checkpoint_paths(spec):
        if path.exists():
            return path
    return _download_volume_checkpoint(spec, local_files_only=local_files_only)


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


def _shape_tuple(value, fallback: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return tuple(fallback)
    return tuple(int(v) for v in value)


def _volume_config_from_payload(payload: dict | None) -> dict:
    cfg = dict(_VOLUME_MODEL_CONFIG)
    if not isinstance(payload, dict):
        return cfg

    # HF config.json stores architecture fields at the top level and the
    # TransUNet constructor kwargs under "model_config". Safetensors metadata
    # stores the same HF config as a JSON string under "model_config".
    model_cfg = payload.get("model_config")
    if isinstance(model_cfg, str):
        try:
            model_cfg = json.loads(model_cfg)
        except json.JSONDecodeError:
            model_cfg = {}
    if isinstance(model_cfg, dict) and ("model_config" in model_cfg or "base_volume_shape" in model_cfg):
        payload = model_cfg
        model_cfg = payload.get("model_config")
    if isinstance(model_cfg, dict):
        cfg.update(model_cfg)

    for key in ("n_classes", "in_channels", "transformer_depth", "n_heads"):
        if payload.get(key) is not None:
            cfg[key] = int(payload[key])
    if payload.get("dropout") is not None:
        cfg["dropout"] = float(payload["dropout"])
    if payload.get("base_shape") is not None:
        cfg["base_shape"] = _shape_tuple(payload["base_shape"], cfg["base_shape"])
    if payload.get("base_volume_shape") is not None:
        cfg["base_shape"] = _shape_tuple(payload["base_volume_shape"], cfg["base_shape"])
    if payload.get("patch_size") is not None:
        cfg["patch_size"] = _shape_tuple(payload["patch_size"], cfg["patch_size"])
    if cfg.get("channels") is not None:
        cfg["channels"] = _shape_tuple(cfg["channels"], _VOLUME_MODEL_CONFIG["channels"])
    return cfg


def _load_volume_model_config(path: Path) -> dict:
    config_path = path.with_name("config.json")
    if config_path.exists():
        return _volume_config_from_payload(json.loads(config_path.read_text(encoding="utf-8")))

    if path.suffix == ".safetensors":
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ImportError("Install safetensors to load ScaleSurfer volume checkpoint metadata.") from exc
        with safe_open(str(path), framework="pt", device="cpu") as f:
            return _volume_config_from_payload(f.metadata())

    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        return dict(_VOLUME_MODEL_CONFIG)
    return _volume_config_from_payload(
        {
            "n_classes": ckpt.get("n_classes"),
            "base_volume_shape": ckpt.get("base_volume_shape"),
            "patch_size": ckpt.get("patch_size"),
            "model_config": ckpt.get("model_cfg"),
        }
    )


def _normalize_volume_dtype(dtype) -> torch.dtype:
    if dtype is None:
        return torch.float32

    if isinstance(dtype, torch.dtype):
        if dtype not in {torch.float32, torch.float16, torch.bfloat16}:
            raise ValueError("volume_dtype must be torch.float32, torch.float16, or torch.bfloat16")
        return dtype

    key = str(dtype).strip().lower().removeprefix("torch.").replace("-", "").replace("_", "")
    try:
        return _VOLUME_DTYPE_ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(sorted(_VOLUME_DTYPE_ALIASES))
        raise ValueError(f"Unsupported volume_dtype {dtype!r}; expected one of: {supported}") from exc


def _dense_label_save_dtype(n_classes: int) -> torch.dtype:
    if int(n_classes) <= 256:
        return torch.uint8
    if int(n_classes) <= torch.iinfo(torch.int16).max + 1:
        return torch.int16
    if int(n_classes) <= torch.iinfo(torch.int32).max + 1:
        return torch.int32
    return torch.int64


def _save_torch_zip_deflated(obj, path: str | Path, *, compresslevel: int = 1) -> None:
    """Save a torch object as a deflated zip archive readable by torch.load."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_tmp = path.with_suffix(path.suffix + ".tmp")
    compressed_tmp = path.with_suffix(path.suffix + ".tmp.deflated")
    try:
        torch.save(obj, raw_tmp)
        with zipfile.ZipFile(raw_tmp, "r") as zin, zipfile.ZipFile(
            compressed_tmp,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=int(compresslevel),
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


def _torch_load_weights(path: str | Path, *, mmap: bool = False):
    """Load a tensor/checkpoint, retrying without mmap for compressed torch zips."""
    kwargs = {"map_location": "cpu", "weights_only": True}
    if mmap:
        try:
            return torch.load(path, mmap=True, **kwargs)
        except RuntimeError:
            return torch.load(path, **kwargs)
    return torch.load(path, **kwargs)


def _prepared_orig_tensor_is_valid(path: str | Path) -> bool:
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        tensor = _torch_load_weights(path, mmap=True)
    except Exception:
        return False
    return isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == tuple(CONFORM_SHAPE)


def _prepared_orig_mgz_is_valid(path: str | Path) -> bool:
    path = Path(path)
    if not image_is_loadable(path):
        return False
    try:
        return tuple(nib.load(str(path)).shape[:3]) == tuple(CONFORM_SHAPE)
    except Exception:
        return False


def _volume_provenance_payload(
    *,
    subject: str,
    fs_version: int,
    checkpoint_path: str | Path,
) -> dict:
    return {
        "kind": "aparc+aseg",
        "generator": "ScaleSurfer.predict_volumes",
        "created_at": _utc_now_iso(),
        "subject": str(subject),
        "fs_version": int(fs_version),
        "checkpoint_path": str(checkpoint_path),
        "outputs": ["mri/aparc+aseg.pt"],
    }


def _write_volume_provenance_files(
    *,
    subject_dir: str | Path,
    subject: str,
    fs_version: int,
    checkpoint_path: str | Path,
) -> None:
    subject_dir = Path(subject_dir) / str(subject)
    scripts_dir = subject_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    payload = _volume_provenance_payload(
        subject=str(subject),
        fs_version=int(fs_version),
        checkpoint_path=checkpoint_path,
    )
    (scripts_dir / _VOLUME_PROVENANCE_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (scripts_dir / _VOLUME_LOG_FILENAME).write_text(
        "\n".join(
            [
                "ScaleSurfer predicted aparc+aseg",
                f"created_at={payload['created_at']}",
                f"subject={payload['subject']}",
                f"fs_version={payload['fs_version']}",
                f"checkpoint_path={payload['checkpoint_path']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_volume_prediction_job(
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
    _save_torch_zip_deflated(dense_label, out_path)
    _write_volume_provenance_files(
        subject_dir=subject_dir,
        subject=subject,
        fs_version=fs_version,
        checkpoint_path=checkpoint_path,
    )
    return str(out_path)


def _write_stats_prediction_job(
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


class _VolumeOrigDataset(Dataset):
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
        orig = _torch_load_weights(orig_path, mmap=True)
        if not isinstance(orig, torch.Tensor) or tuple(orig.shape) != tuple(CONFORM_SHAPE):
            shape = tuple(orig.shape) if isinstance(orig, torch.Tensor) else type(orig).__name__
            raise RuntimeError(
                f"{orig_path} has shape {shape}, expected {CONFORM_SHAPE}. "
                "Run surfer.prepare_images(overwrite=True) or delete this stale prepared tensor."
            )
        return {
            "isub": isub,
            "subject": subject,
            "orig": torch.as_tensor(orig).squeeze().contiguous(),
        }


class _StatsTensorDataset(Dataset):
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
        t1 = _torch_load_weights(t1_path, mmap=True)
        seg = _torch_load_weights(seg_path, mmap=True)
        return {
            "subject": subject,
            "t1": torch.as_tensor(t1).squeeze().contiguous(),
            "seg": torch.as_tensor(seg).squeeze().contiguous(),
        }


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
        volume_dtype="float32",
        progress=True,
        pretrained_data_name="adni",
        conform_backend="auto",
        compress_orig=False,
        async_writes=False,
        writer_backend="thread",
        writer_workers=1,
        writer_queue_size=2,
        inference_num_workers=None,
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True,
        overwrite=False,
        verbose=True,
    ):
        """
        Initilize object.

        Parameters
        ----------
        volume_dtype : str or torch.dtype, default="float32"
            Floating dtype for volume inference weights and input batches.
        compress_orig : bool, default=False
            If True, write mri/orig.pt as a deflated torch zip archive.
            This saves disk space but makes prepare_images CPU-bound.
        async_writes : bool, default=False
            If True, write predicted volume/stat outputs and provenance in a
            bounded background executor while later GPU batches run.
        writer_backend : {"thread", "process"}, default="thread"
            Background executor type used when async_writes=True.
            Threads avoid serializing large tensors between processes and are
            usually the right first choice for disk writes.
        writer_workers : int, default=1
            Number of background writer workers.
        writer_queue_size : int, default=2
            Maximum number of pending background write jobs before prediction
            waits for at least one write to complete.
        inference_num_workers : int, optional
            Number of PyTorch DataLoader workers for inference input loading.
            Defaults to a conservative value derived from n_jobs_cpu.
        prefetch_factor : int or None, default=2
            Number of batches prefetched per DataLoader worker when
            inference_num_workers > 0.
        pin_memory : bool, default=True
            Pin CPU input batches before CUDA transfer when running on CUDA.
        persistent_workers : bool, default=True
            Keep DataLoader workers alive for the duration of an inference call.
        """

        self.anat_files = anat_files
        self.subjects = subjects
        self.subject_dir = Path(subject_dir)
        self.batch_size =  batch_size
        self.n_jobs = n_jobs_cpu
        self.fs_version = _normalize_fs_version(fs_version)
        self.pretrained_data_name = pretrained_data_name
        self.conform_backend = conform_backend
        self.compress_orig = bool(compress_orig)
        self.async_writes = bool(async_writes)
        self.writer_backend = _normalize_writer_backend(writer_backend)
        self.writer_workers = max(1, int(writer_workers))
        self.writer_queue_size = max(1, int(writer_queue_size))
        self.inference_num_workers = (
            _default_inference_num_workers(self.n_jobs)
            if inference_num_workers is None
            else max(0, int(inference_num_workers))
        )
        self.prefetch_factor = None if prefetch_factor is None else max(1, int(prefetch_factor))
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.overwrite = overwrite
        self.verbose = verbose
        self.volume_dtype = _normalize_volume_dtype(volume_dtype)

        self._model_volume = None
        self._model_surface_bundles = None
        self._surface_config = None
        self._stats_predictors = {}
        self.predicted_volumes = None
        self.df_stats = None
        self.df_stats_long = None
        self.device = DEVICE if device is None else device

        self.chkpt_path_volume = _resolve_volume_checkpoint_path(self.fs_version)

        assert len(anat_files) == len(subjects), "anat_files and subjects must have the same length"
        self.prepare_directories()
        self.progress = bool(progress)
        self._tqdm = tqdm if self.progress else (lambda iterable, *args, **kwargs: iterable)


    def _log_timing(self, label: str, elapsed: float, *, n: int | None = None) -> None:
        n = len(self.subjects) if n is None else int(n)
        print(f"[scalesurfer] {label}: {elapsed:.1f}s total, {n/elapsed:.2f} img/s ({elapsed/n:.1f}s/img) for {n} subject(s)")

    def free(self) -> None:
        """Free memory."""
        self._empty_cache()
        self._model_surface_bundles = None
        self._surface_config = None
        self._model_volume = None
        self._stats_predictors = {}
        self.df_stats = None
        self.df_stats_long = None

    def _empty_cache(self) -> None:
        """Release unused memory for the active device backend."""
        d = str(self.device)
        if d.startswith("cuda"):
            torch.cuda.empty_cache()
        elif d.startswith("mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    def _sync_device(self) -> None:
        """Synchronize pending work before measuring device-side timing."""
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

    # Model loaders
    @property
    def model_volume(self):
        """Lazy load volumetric model."""
        if self._model_volume is None:
            self._model_volume = TransUNet3D(**_load_volume_model_config(self.chkpt_path_volume))
            self._model_volume.load_state_dict(_load_volume_state_dict(self.chkpt_path_volume))
            self._model_volume.to(device=self.device, dtype=self.volume_dtype)
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

    def stats_predictor(
        self,
        fs_version: int | str | None = None,
        *,
        checkpoint_path: str | Path | None = None,
        local_files_only: bool = False,
    ) -> ScaleSurferStatsPredictor:
        """Lazy load a stats prediction model."""
        key = (
            _normalize_fs_version(self.fs_version if fs_version is None else fs_version),
            None if checkpoint_path is None else str(Path(checkpoint_path).expanduser().resolve()),
        )
        if key not in self._stats_predictors:
            self._stats_predictors[key] = ScaleSurferStatsPredictor.from_pretrained(
                fs_version=key[0],
                checkpoint_path=checkpoint_path,
                device=self.device,
                local_files_only=local_files_only,
            )
        return self._stats_predictors[key]

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
        if value is None:
            return None
        return _normalize_fs_version(value)

    def _check_existing_output_version(
        self,
        subject: str,
        *,
        output_path: Path,
        provenance_filename: str,
        requested_fs_version: int,
        output_label: str,
    ) -> bool:
        """Return True when an existing output can be reused."""
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
        _write_volume_provenance_files(
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
        if (
            not overwrite
            and _prepared_orig_tensor_is_valid(out_file)
            and _prepared_orig_mgz_is_valid(out_mgz)
        ):
            return {"subject": subject, "skipped_existing": True, "out_file": str(out_file)}

        img_tensor = prepare_image(
            anat_file,
            out_mgz,
            conform_backend=self.conform_backend,
            overwrite=overwrite,
        ).to(dtype=torch.float32).contiguous()
        if img_tensor.device.type != "cpu":
            img_tensor = img_tensor.cpu()
        if compress_orig:
            _save_torch_zip_deflated(img_tensor, out_file)
        else:
            torch.save(img_tensor, out_file)
        return {"subject": subject, "skipped_existing": False, "out_file": str(out_file)}


    def prepare_images(
        self,
        *,
        compress_orig: bool | None = None,
        overwrite: bool | None = None,
    ):
        """Prepare conformed orig.mgz/orig.pt inputs for inference.

        Parameters
        ----------
        compress_orig:
            One-call override for `self.compress_orig`.
        overwrite:
            One-call override for `self.overwrite`. When False, subjects with
            existing non-empty mri/orig.pt and loadable mri/orig.mgz are reused.
        """
        compress_orig = self.compress_orig if compress_orig is None else bool(compress_orig)
        overwrite = self.overwrite if overwrite is None else bool(overwrite)
        t0 = time()
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._prepare_image)(
                subject,
                anat_file,
                self.subject_dir,
                compress_orig=compress_orig,
                overwrite=overwrite,
            )
            for subject, anat_file in tqdm(
                zip(self.subjects, self.anat_files),
                total=len(self.subjects),
                desc="Conforming images"
            )
        )
        if self.verbose:
            skipped = sum(1 for result in results if result and result.get("skipped_existing"))
            if skipped:
                print(f"[scalesurfer] reusing existing prepared images for {skipped:,} subject(s)")
            self._log_timing("prepare_images", time() - t0)

    # Torch models
    def _predict_volume(self, x, *, patch_chunk_size=64, volume_dtype=None):
        """Predict aparc+aseg volumes."""
        aparc_aseg_pred = self.model_volume.predict_volume_fast(
            x.unsqueeze(1),
            patch_chunk_size=patch_chunk_size,
        )
        return aparc_aseg_pred


    def predict_volumes(
        self,
        batch_size: int | None = None,
        *,
        subjects: list[str] | None = None,
        volume_dtype=None,
        patch_chunk_size=64,
        overwrite: bool | None = None,
        write=True
    ):
        """
        Predict aparc+aseg for all subject and save.

        Parameters
        ----------
        volume_dtype : str or torch.dtype, optional
            One-off dtype override for the volume model and input batch.
        subjects : list[str], optional
            Optional subset of subjects to process.
        patch_chunk_size : int, default=64
            Classifier chunk size in 16x16x16-patch equivalents after the encoder.
            Smaller values reduce peak logit memory; larger values reduce loop overhead.
            This chunks the tensor: (batch_size, 256, 256, 256, n_classes).
            Use None to classify all patches in one chunk.
        overwrite : bool, optional
            Overrides self.overwrite. When False and write=True, existing
            mri/aparc+aseg.pt outputs with matching provenance are reused.
        write: bool, optional, default: True
            Writes to disk if True. Otherwise, stores predictions in
            self.predicted_volumes in subject order.
        """
        volume_dtype = self.volume_dtype if volume_dtype is None else _normalize_volume_dtype(volume_dtype)
        overwrite = self.overwrite if overwrite is None else bool(overwrite)
        subject_to_index = {str(subject): idx for idx, subject in enumerate(self.subjects)}
        if subjects is None:
            selected_indices = list(range(len(self.subjects)))
        else:
            missing_subjects = [str(subject) for subject in subjects if str(subject) not in subject_to_index]
            if missing_subjects:
                raise ValueError(f"Unknown subject(s): {missing_subjects[:5]}")
            selected_indices = [subject_to_index[str(subject)] for subject in subjects]

        predict_indices = selected_indices
        if write and not overwrite:
            predict_indices = []
            skipped = []
            for isub in selected_indices:
                subject = str(self.subjects[isub])
                out_path = self.subject_dir / subject / "mri" / "aparc+aseg.pt"
                if self._check_existing_output_version(
                    subject,
                    output_path=out_path,
                    provenance_filename=_VOLUME_PROVENANCE_FILENAME,
                    requested_fs_version=self.fs_version,
                    output_label="aparc+aseg tensor",
                ):
                    skipped.append(subject)
                else:
                    predict_indices.append(isub)
            if skipped and self.verbose:
                print(f"[scalesurfer] reusing existing aparc+aseg tensors for {len(skipped):,} subject(s)")

        if write:
            self.predicted_volumes = None
        else:
            self.predicted_volumes = None
        if not predict_indices:
            if self.verbose:
                print("[scalesurfer] predict_volumes: all requested outputs already exist")
            return

        model = self.model_volume  # trigger lazy load only when work remains
        if next(model.parameters()).dtype != volume_dtype:
            model.to(device=self.device, dtype=volume_dtype)
        pred_save_dtype = _dense_label_save_dtype(model.n_classes)
        if not write:
            self.predicted_volumes = torch.empty(
                (len(predict_indices), 256, 256, 256),
                dtype=pred_save_dtype,
                device=self.device,
            )

        self._sync_device()
        t0 = time()
        bs = batch_size if batch_size is not None else self.batch_size
        profile = self._profile_enabled()
        executor = None
        pending_writes = []
        writer_max_pending = max(1, int(self.writer_queue_size))
        if write and self.async_writes:
            executor_cls = ThreadPoolExecutor if self.writer_backend == "thread" else ProcessPoolExecutor
            executor = executor_cls(max_workers=self.writer_workers)
            if self.verbose:
                print(
                    "[scalesurfer] predict_volumes: async volume writes enabled "
                    f"backend={self.writer_backend} "
                    f"workers={self.writer_workers} "
                    f"queue_size={writer_max_pending}"
                )

        def drain_pending_writes(*, block: bool = False) -> None:
            nonlocal pending_writes
            if not pending_writes:
                return
            done, not_done = wait(
                pending_writes,
                return_when=ALL_COMPLETED if block else FIRST_COMPLETED,
            )
            for future in done:
                future.result()
            pending_writes = list(not_done)

        def submit_volume_write(
            dense_label: torch.Tensor,
            out_path: Path,
            subject: str,
            save_dtype: torch.dtype | None = None,
        ) -> None:
            if executor is None:
                _write_volume_prediction_job(
                    dense_label,
                    out_path,
                    self.subject_dir,
                    subject,
                    int(self.fs_version),
                    self.chkpt_path_volume,
                    save_dtype,
                )
                return
            pending_writes.append(
                executor.submit(
                    _write_volume_prediction_job,
                    dense_label,
                    out_path,
                    self.subject_dir,
                    subject,
                    int(self.fs_version),
                    self.chkpt_path_volume,
                    save_dtype,
                )
            )
            if len(pending_writes) >= writer_max_pending:
                drain_pending_writes(block=False)

        volume_dataset = _VolumeOrigDataset(self.subject_dir, [str(s) for s in self.subjects], predict_indices)
        volume_loader = self._make_inference_loader(volume_dataset, batch_size=bs)
        non_blocking_h2d = bool(self.pin_memory and str(self.device).startswith("cuda"))
        out_offset = 0
        try:
            for batch_no, batch in enumerate(
                self._tqdm(
                    volume_loader,
                    total=len(volume_loader),
                    desc="Predicting volumes"
                )
            ):
                if profile:
                    self._sync_device()
                    batch_t0 = perf_counter()

                batch_subjects = [str(subject) for subject in batch["subject"]]
                n_subjects = len(batch_subjects)
                if profile:
                    load_t0 = perf_counter()
                X = batch["orig"].to(
                    device=self.device,
                    dtype=volume_dtype,
                    non_blocking=non_blocking_h2d,
                )
                if profile:
                    self._sync_device()
                    alloc_sec = 0.0
                    load_sec = perf_counter() - load_t0

                # Predict
                if profile:
                    predict_t0 = perf_counter()
                aparc_aseg_pred = self._predict_volume(
                    X,
                    patch_chunk_size=patch_chunk_size,
                    volume_dtype=volume_dtype,
                )
                if profile:
                    self._sync_device()
                    predict_sec = perf_counter() - predict_t0
                    store_t0 = perf_counter()
                if write:
                    aparc_aseg_cpu = aparc_aseg_pred.detach().to(device="cpu", dtype=pred_save_dtype)
                    for i, subject in enumerate(batch_subjects):
                        out_path = self.subject_dir / subject / "mri" / "aparc+aseg.pt"
                        submit_volume_write(aparc_aseg_cpu[i], out_path, subject)
                else:
                    self.predicted_volumes[out_offset:out_offset + n_subjects] = aparc_aseg_pred.to(dtype=pred_save_dtype)
                    out_offset += n_subjects
                if profile:
                    self._sync_device()
                    store_sec = perf_counter() - store_t0
                    total_sec = perf_counter() - batch_t0
                    print(
                        "[scalesurfer] batch "
                        f"{batch_no}: alloc={alloc_sec:.3f}s "
                        f"load_copy={load_sec:.3f}s "
                        f"predict={predict_sec:.3f}s "
                        f"store={store_sec:.3f}s "
                        f"total={total_sec:.3f}s"
                    )
            drain_pending_writes(block=True)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        self._sync_device()
        if self.verbose:
            self._log_timing("predict_volumes", time() - t0)

    def predict_stats(
        self,
        subjects: list[str] | None = None,
        *,
        fs_version: int | str | None = None,
        checkpoint_path: str | Path | None = None,
        batch_size: int | None = None,
        return_format: str = "wide",
        seg_is_dense: bool | str = "auto",
        local_files_only: bool = False,
        overwrite: bool | None = None,
        write: bool = True,
    ):
        """Predict FreeSurfer-like stats and write subject-level stats outputs.

        The returned wide dataframe has one row per subject. It is also stored
        on `self.df_stats`; the long table is stored on `self.df_stats_long`.
        """
        t0 = time()
        stats_fs_version = _normalize_fs_version(self.fs_version if fs_version is None else fs_version)
        overwrite = self.overwrite if overwrite is None else bool(overwrite)
        subject_list = [str(s) for s in (self.subjects if subjects is None else subjects)]
        known_subjects = {str(s) for s in self.subjects}
        unknown = [subject for subject in subject_list if subject not in known_subjects]
        if unknown:
            raise ValueError(f"Unknown subject(s): {unknown[:5]}")

        reusable_subjects: list[str] = []
        predict_subjects: list[str] = []
        if write and not overwrite:
            for subject in subject_list:
                long_path = self.subject_dir / subject / "stats" / PREDICTED_STATS_LONG_FILENAME
                if self._check_existing_output_version(
                    subject,
                    output_path=long_path,
                    provenance_filename=PREDICTED_STATS_PROVENANCE_FILENAME,
                    requested_fs_version=stats_fs_version,
                    output_label="predicted stats",
                ):
                    reusable_subjects.append(subject)
                else:
                    predict_subjects.append(subject)
        else:
            predict_subjects = list(subject_list)

        frames = []
        if reusable_subjects:
            frames.append(
                load_stats_features(
                    self.subject_dir,
                    subjects=reusable_subjects,
                    return_format="long",
                    prefer_sidecar=True,
                )
            )
            if self.verbose:
                print(f"[scalesurfer] reusing existing predicted stats for {len(reusable_subjects):,} subject(s)")

        if predict_subjects:
            missing_tensors = []
            for subject in predict_subjects:
                mri_dir = self.subject_dir / subject / "mri"
                orig_path = mri_dir / "orig.pt"
                seg_path = mri_dir / "aparc+aseg.pt"
                if not orig_path.exists() or not seg_path.exists():
                    missing_tensors.extend(str(path) for path in (orig_path, seg_path) if not path.exists())
                    continue
                if not self._check_existing_output_version(
                    subject,
                    output_path=seg_path,
                    provenance_filename=_VOLUME_PROVENANCE_FILENAME,
                    requested_fs_version=self.fs_version,
                    output_label="aparc+aseg tensor",
                ):
                    raise ValueError(
                        f"Existing aparc+aseg tensor for subject {subject!r} could not be reused. "
                        "Run surfer.predict_volumes(overwrite=True) before predict_stats(), "
                        "or use a fresh subjects_dir."
                    )
            if missing_tensors:
                raise FileNotFoundError(
                    "Missing required ScaleSurfer tensor(s): "
                    + ", ".join(missing_tensors[:10])
                    + ". Run prepare_images() and predict_volumes() first."
                )

            predictor = self.stats_predictor(
                fs_version=stats_fs_version,
                checkpoint_path=checkpoint_path,
                local_files_only=local_files_only,
            )
            bs = batch_size if batch_size is not None else self.batch_size
            stats_dataset = _StatsTensorDataset(self.subject_dir, predict_subjects)
            stats_loader = self._make_inference_loader(stats_dataset, batch_size=bs)
            iterator = self._tqdm(stats_loader, total=len(stats_loader), desc="Predicting stats")

            executor = None
            pending_writes = []
            writer_max_pending = max(1, int(self.writer_queue_size))
            if write and self.async_writes:
                executor_cls = ThreadPoolExecutor if self.writer_backend == "thread" else ProcessPoolExecutor
                executor = executor_cls(max_workers=self.writer_workers)
                if self.verbose:
                    print(
                        "[scalesurfer] predict_stats: async stats writes enabled "
                        f"backend={self.writer_backend} "
                        f"workers={self.writer_workers} "
                        f"queue_size={writer_max_pending}"
                    )

            def drain_pending_stats_writes(*, block: bool = False) -> None:
                nonlocal pending_writes
                if not pending_writes:
                    return
                done, not_done = wait(
                    pending_writes,
                    return_when=ALL_COMPLETED if block else FIRST_COMPLETED,
                )
                for future in done:
                    future.result()
                pending_writes = list(not_done)

            def submit_stats_write(subject_frame: pd.DataFrame) -> None:
                subject_frame = subject_frame.copy()
                if executor is None:
                    _write_stats_prediction_job(
                        subject_frame,
                        self.subject_dir,
                        stats_fs_version,
                        predictor.checkpoint_path,
                    )
                    return
                pending_writes.append(
                    executor.submit(
                        _write_stats_prediction_job,
                        subject_frame,
                        self.subject_dir,
                        stats_fs_version,
                        predictor.checkpoint_path,
                    )
                )
                if len(pending_writes) >= writer_max_pending:
                    drain_pending_stats_writes(block=False)

            new_frames = []
            try:
                for batch in iterator:
                    batch_subjects = [str(subject) for subject in batch["subject"]]
                    batch_long = predictor.predict_tensors(
                        batch["t1"],
                        batch["seg"],
                        subjects=batch_subjects,
                        seg_is_dense=seg_is_dense,
                        return_format="long",
                    )
                    new_frames.append(batch_long)
                    if write:
                        for _, subject_frame in batch_long.groupby("subject", sort=False):
                            submit_stats_write(subject_frame)
                drain_pending_stats_writes(block=True)
            finally:
                if executor is not None:
                    executor.shutdown(wait=True)

            if new_frames:
                frames.append(pd.concat(new_frames, ignore_index=True))

        long_df = pd.concat(frames, ignore_index=True) if frames else load_stats_features(
            self.subject_dir,
            subjects=subject_list,
            return_format="long",
            prefer_sidecar=True,
        )
        if subject_list:
            order = {subject: idx for idx, subject in enumerate(subject_list)}
            long_df = long_df.assign(_subject_order=long_df["subject"].map(order)).sort_values(
                ["_subject_order", "stats_name", "region", "measure", "target"]
            ).drop(columns="_subject_order").reset_index(drop=True)

        self.df_stats_long = long_df
        self.df_stats = stats_long_to_wide(long_df, fill_value=0.0)

        if self.verbose:
            self._log_timing("predict_stats", time() - t0, n=len(subject_list))
        if return_format == "long":
            return self.df_stats_long
        if return_format == "wide":
            return self.df_stats
        raise ValueError("return_format must be 'long' or 'wide'")

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

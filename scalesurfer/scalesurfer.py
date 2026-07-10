"""High-level public inference API."""
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path
from time import perf_counter, time

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from scalesurfer.inference._datasets import StatsTensorDataset, VolumeOrigDataset
from scalesurfer.inference._io import write_stats_prediction_job, write_volume_prediction_job
from scalesurfer.inference._models import (
    dense_label_save_dtype,
    load_volume_model_config,
    load_volume_state_dict,
    normalize_fs_version,
    normalize_volume_dtype,
    resolve_volume_checkpoint_path,
)
from scalesurfer.inference._runtime import (
    ScaleSurferRuntime,
    default_inference_num_workers,
    normalize_writer_backend,
)
from scalesurfer.inference._settings import VOLUME_PROVENANCE_FILENAME
from scalesurfer.models.volume import TransUNet3D
from scalesurfer.stats import (
    PREDICTED_STATS_LONG_FILENAME,
    PREDICTED_STATS_PROVENANCE_FILENAME,
    ScaleSurferStatsPredictor,
    load_stats_features,
    stats_long_to_wide,
)
from scalesurfer.training.volume.config import DEVICE



class ScaleSurfer(ScaleSurferRuntime):

    def __init__(
        self,
        anat_files,
        subjects,
        subject_dir,
        *,
        batch_size=1,
        n_jobs_cpu=1,
        fs_version=7,
        device=None,
        volume_dtype="float32",
        progress=True,
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
        self.fs_version = normalize_fs_version(fs_version)
        self.conform_backend = conform_backend
        self.compress_orig = bool(compress_orig)
        self.async_writes = bool(async_writes)
        self.writer_backend = normalize_writer_backend(writer_backend)
        self.writer_workers = max(1, int(writer_workers))
        self.writer_queue_size = max(1, int(writer_queue_size))
        self.inference_num_workers = (
            default_inference_num_workers(self.n_jobs)
            if inference_num_workers is None
            else max(0, int(inference_num_workers))
        )
        self.prefetch_factor = None if prefetch_factor is None else max(1, int(prefetch_factor))
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers)
        self.overwrite = overwrite
        self.verbose = verbose
        self.volume_dtype = normalize_volume_dtype(volume_dtype)

        self._model_volume = None
        self._stats_predictors = {}
        self.predicted_volumes = None
        self.df_stats = None
        self.df_stats_long = None
        self.device = DEVICE if device is None else device

        self.chkpt_path_volume = resolve_volume_checkpoint_path(self.fs_version)

        assert len(anat_files) == len(subjects), "anat_files and subjects must have the same length"
        self.prepare_directories()
        self.progress = bool(progress)
        self._tqdm = tqdm if self.progress else (lambda iterable, *args, **kwargs: iterable)


    def free(self) -> None:
        """Free memory."""
        self._empty_cache()
        self._model_volume = None
        self._stats_predictors = {}
        self.df_stats = None
        self.df_stats_long = None

    @property
    def model_volume(self):
        """Lazy load volumetric model."""
        if self._model_volume is None:
            self._model_volume = TransUNet3D(**load_volume_model_config(self.chkpt_path_volume))
            self._model_volume.load_state_dict(load_volume_state_dict(self.chkpt_path_volume))
            self._model_volume.to(device=self.device, dtype=self.volume_dtype)
            self._model_volume.eval()
        return self._model_volume


    def stats_predictor(
        self,
        fs_version: int | str | None = None,
        *,
        checkpoint_path: str | Path | None = None,
        local_files_only: bool = False,
    ) -> ScaleSurferStatsPredictor:
        """Lazy load a stats prediction model."""
        key = (
            normalize_fs_version(self.fs_version if fs_version is None else fs_version),
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
        volume_dtype = self.volume_dtype if volume_dtype is None else normalize_volume_dtype(volume_dtype)
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
                    provenance_filename=VOLUME_PROVENANCE_FILENAME,
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
        pred_save_dtype = dense_label_save_dtype(model.n_classes)
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
                write_volume_prediction_job(
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
                    write_volume_prediction_job,
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

        volume_dataset = VolumeOrigDataset(self.subject_dir, [str(s) for s in self.subjects], predict_indices)
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
        stats_fs_version = normalize_fs_version(self.fs_version if fs_version is None else fs_version)
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
                    provenance_filename=VOLUME_PROVENANCE_FILENAME,
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
            stats_dataset = StatsTensorDataset(self.subject_dir, predict_subjects)
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
                    write_stats_prediction_job(
                        subject_frame,
                        self.subject_dir,
                        stats_fs_version,
                        predictor.checkpoint_path,
                    )
                    return
                pending_writes.append(
                    executor.submit(
                        write_stats_prediction_job,
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
        x = torch.load(x_path, map_location="cpu").to(torch.float32)
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

from __future__ import annotations

from dataclasses import dataclass
import json
import random
import re
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from .api import prepare_data_pipeline
from .config import (
    DATA_CFG,
    DEVICE,
    MODEL_CFG,
    PATCH_SIZE,
    SEED,
    TRAIN_CFG,
    build_runtime_cfgs,
)
from .metrics import build_eval_plot_bundle, collect_test_metrics_fast_cached
from .volume.model import TransUNet3D
from .train import evaluate_runtime, init_training_runtime, step_scheduler_epoch, train_step_runtime
from .utils import auto_discover_pairs, gpu_mem_gb


_DS_RE = re.compile(r"(ds\d{6,})", re.IGNORECASE)


def extract_dataset_id(path: str | Path) -> str | None:
    """Extract OpenNeuro-like dataset id (e.g., ds004731) from any path-like string."""
    m = _DS_RE.search(str(path))
    if m is None:
        return None
    return m.group(1).lower()


def fs_major_from_version(version: str | int | float | None) -> int | None:
    """Map version string like '7.3.2' to major int 7. Return None if unknown."""
    if version is None:
        return None
    if isinstance(version, bool):
        return None
    if isinstance(version, int):
        return int(version) if int(version) > 0 else None
    if isinstance(version, float):
        iv = int(version)
        return iv if iv > 0 else None

    s = str(version).strip()
    if not s:
        return None
    m = re.search(r"(\d+)", s)
    if m is None:
        return None
    major = int(m.group(1))
    return major if major > 0 else None


def load_fs_version_map(json_path: str | Path) -> dict[str, int | None]:
    """
    Load dataset -> FreeSurfer major-version map.

    Input values can be full versions (e.g., "6.0.1") or null.
    Returned keys are lowercased dataset ids.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {json_path}, got {type(raw)}")

    out: dict[str, int | None] = {}
    for k, v in raw.items():
        ds = str(k).strip().lower()
        out[ds] = fs_major_from_version(v)
    return out


def _split_counts(n: int, ratios=(0.8, 0.1, 0.1)) -> tuple[int, int, int]:
    tr, va, te = (float(ratios[0]), float(ratios[1]), float(ratios[2]))
    if n <= 0:
        return 0, 0, 0

    n_val = int(round(n * va))
    n_test = int(round(n * te))

    # Keep tiny datasets usable while still creating eval splits when possible.
    if n >= 3:
        n_val = max(1, n_val)
        n_test = max(1, n_test)

    if n_val + n_test >= n:
        overflow = n_val + n_test - (n - 1)
        while overflow > 0 and (n_val > 0 or n_test > 0):
            if n_val >= n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
            overflow -= 1

    n_train = n - n_val - n_test
    return n_train, n_val, n_test


def split_pairs_reproducible(
    x_files: list[str],
    y_files: list[str],
    ratios=(0.8, 0.1, 0.1),
    seed: int = SEED,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    """Pair-preserving reproducible split into train/val/test."""
    if len(x_files) != len(y_files):
        raise ValueError("x_files and y_files must have the same length")

    pairs = list(zip(x_files, y_files))
    rng = random.Random(int(seed))
    rng.shuffle(pairs)

    n_train, n_val, n_test = _split_counts(len(pairs), ratios=ratios)

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train : n_train + n_val]
    test_pairs = pairs[n_train + n_val : n_train + n_val + n_test]

    x_tr, y_tr = zip(*train_pairs) if train_pairs else ([], [])
    x_va, y_va = zip(*val_pairs) if val_pairs else ([], [])
    x_te, y_te = zip(*test_pairs) if test_pairs else ([], [])

    return list(x_tr), list(y_tr), list(x_va), list(y_va), list(x_te), list(y_te)


def version_coverage_report(
    x_files: list[str],
    fs_major_by_dataset: dict[str, int | None],
    default_major: int | None = None,
) -> pd.DataFrame:
    """Dataset-level version coverage for a list of tensor paths."""
    rows = []
    for x in x_files:
        ds = extract_dataset_id(x)
        major = None
        if ds is not None:
            major = fs_major_by_dataset.get(ds)
        if major is None:
            major = default_major
        rows.append({
            "dataset": ds if ds is not None else "<missing>",
            "major": major,
            "path": str(x),
        })

    if not rows:
        return pd.DataFrame(columns=["dataset", "major", "n_samples"])

    df = pd.DataFrame(rows)
    out = (
        df.groupby(["dataset", "major"], dropna=False, as_index=False)
        .agg(n_samples=("path", "count"))
        .sort_values(["major", "dataset"], na_position="first")
        .reset_index(drop=True)
    )
    return out


def validate_training_versions(
    x_train_files: list[str],
    fs_major_by_dataset: dict[str, int | None],
    *,
    expected_major: int | None = None,
    default_major: int | None = None,
) -> pd.DataFrame:
    """
    Ensure each training sample has a valid version mapping.

    - If expected_major is provided, all training samples must map to that major.
    - If default_major is provided, missing datasets inherit that major.
    """
    report = version_coverage_report(
        x_files=x_train_files,
        fs_major_by_dataset=fs_major_by_dataset,
        default_major=default_major,
    )

    if len(report) == 0:
        raise ValueError("Training split is empty.")

    missing = report[report["major"].isna()]
    if len(missing):
        ds_list = sorted(str(x) for x in missing["dataset"].unique().tolist())
        raise ValueError(
            "Some training datasets have no valid FreeSurfer version: "
            + ", ".join(ds_list[:20])
        )

    if expected_major is not None:
        bad = report[report["major"].astype(int) != int(expected_major)]
        if len(bad):
            ds_list = sorted(str(x) for x in bad["dataset"].unique().tolist())
            raise ValueError(
                f"Training split contains datasets outside FS v{int(expected_major)}: "
                + ", ".join(ds_list[:20])
            )

    return report



def build_versioned_splits_from_existing_split(
    base_split: dict[str, list[str]],
    fs_major_by_dataset: dict[str, int | None],
    *,
    target_majors=(5, 6, 7),
    default_major_for_unmapped: int | None = None,
    error_on_unassigned: bool = True,
) -> dict[int, dict[str, list[str]]]:
    """
    Partition an already-defined train/val/test split into FS-major subsets.

    This preserves the original split assignment exactly (no re-splitting).
    """
    required = ("x_train", "y_train", "x_val", "y_val", "x_test", "y_test")
    for key in required:
        if key not in base_split:
            raise KeyError(f"Missing split key: {key}")

    if len(base_split["x_train"]) != len(base_split["y_train"]):
        raise ValueError("base_split train x/y lengths differ")
    if len(base_split["x_val"]) != len(base_split["y_val"]):
        raise ValueError("base_split val x/y lengths differ")
    if len(base_split["x_test"]) != len(base_split["y_test"]):
        raise ValueError("base_split test x/y lengths differ")

    majors = [int(m) for m in target_majors]
    out: dict[int, dict[str, list[str]]] = {
        m: {"x_train": [], "y_train": [], "x_val": [], "y_val": [], "x_test": [], "y_test": []}
        for m in majors
    }

    unassigned: list[tuple[str, str, str, str | None, int | None]] = []

    def _assign(split_name: str, xs: list[str], ys: list[str]) -> None:
        x_key = f"x_{split_name}"
        y_key = f"y_{split_name}"
        for x, y in zip(xs, ys):
            ds = extract_dataset_id(x)
            mapped = fs_major_by_dataset.get(ds) if ds is not None else None
            if mapped is None:
                mapped = default_major_for_unmapped
            if mapped in majors:
                out[int(mapped)][x_key].append(x)
                out[int(mapped)][y_key].append(y)
            else:
                unassigned.append((split_name, str(x), str(y), ds, mapped))

    _assign("train", base_split["x_train"], base_split["y_train"])
    _assign("val", base_split["x_val"], base_split["y_val"])
    _assign("test", base_split["x_test"], base_split["y_test"])

    if unassigned and bool(error_on_unassigned):
        preview = "\n".join(
            f"split={s} ds={ds} mapped={mapped} x={x}"
            for s, x, _, ds, mapped in unassigned[:10]
        )
        raise ValueError(
            f"{len(unassigned)} samples were not assigned to target_majors={majors}. "
            "Set default_major_for_unmapped, include the major in target_majors, "
            "or disable error_on_unassigned.\n"
            + preview
        )

    return out


def build_v8_split_from_root(
    tensors_root: str | Path,
    *,
    seed: int = SEED,
    ratios=(0.8, 0.1, 0.1),
) -> dict[str, list[str]]:
    """Reproducible train/val/test split from a tensor root where all samples are FS v8."""
    root = Path(tensors_root)
    x_all, y_all = auto_discover_pairs(root)
    if not x_all:
        raise ValueError(f"No tensor pairs found under {root}")

    x_tr, y_tr, x_va, y_va, x_te, y_te = split_pairs_reproducible(
        x_all,
        y_all,
        ratios=ratios,
        seed=int(seed),
    )
    return {
        "x_train": x_tr,
        "y_train": y_tr,
        "x_val": x_va,
        "y_val": y_va,
        "x_test": x_te,
        "y_test": y_te,
    }


def discover_gcloud_file_map(gcloud_root: str | Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    """
    Build convert.py file_map from gcloud FreeSurfer outputs.

    Keys are normalized to '<subject_token>/mri' so outputs become:
      tensors_gcloud/<subject_token>/mri/{orig.pt, aparc+aseg.pt}
    """
    root = Path(gcloud_root)
    aparc_files = sorted(root.rglob("aparc+aseg.mgz"))

    file_map: dict[str, dict[str, str]] = {}
    missing_raw: list[str] = []
    used_keys: set[str] = set()

    for aparc in aparc_files:
        orig = aparc.with_name("orig.mgz")
        if not orig.exists():
            missing_raw.append(str(aparc))
            continue

        subject_token = aparc.parent.parent.name
        key = f"{subject_token}/mri"
        if key in used_keys:
            suffix = 2
            while f"{key}__dup{suffix}" in used_keys:
                suffix += 1
            key = f"{key}__dup{suffix}"
        used_keys.add(key)

        file_map[key] = {
            "orig": str(orig),
            "aparc+aseg": str(aparc),
        }

    return file_map, missing_raw


def convert_gcloud_mgz_to_tensors(
    gcloud_root: str | Path,
    out_root: str | Path = "tensors_gcloud",
    *,
    n_jobs: int = -1,
) -> dict[str, int | str]:
    """
    Convert gcloud mgz pairs to .pt tensors using existing convert.py pipeline.

    Existing outputs are preserved (convert.py skips already-saved pairs).
    """
    from .convert import convert_file_map_to_pt

    file_map, missing_raw = discover_gcloud_file_map(gcloud_root)
    if not file_map:
        raise ValueError(f"No valid mgz pairs found under {gcloud_root}")

    results = convert_file_map_to_pt(
        file_map=file_map,
        out_root=out_root,
        n_jobs=int(n_jobs),
        unsafe_int8=False,
    )

    ok = 0
    failed = 0
    skipped = 0
    for r in results:
        if bool(r.get("ok", False)):
            ok += 1
            if bool(r.get("skipped_existing", False)):
                skipped += 1
        else:
            failed += 1

    return {
        "gcloud_root": str(Path(gcloud_root).resolve()),
        "out_root": str(Path(out_root).resolve()),
        "pairs_discovered": int(len(file_map)),
        "missing_raw_for_aparc": int(len(missing_raw)),
        "converted_ok": int(ok),
        "converted_failed": int(failed),
        "converted_skipped_existing": int(skipped),
    }


def make_unique_run_dir(base_dir: str | Path, run_name: str) -> Path:
    """Create a unique directory under base_dir without overwriting existing outputs."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = base / f"{run_name}_{stamp}"
    idx = 1
    while candidate.exists():
        candidate = base / f"{run_name}_{stamp}_{idx:02d}"
        idx += 1

    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


@dataclass
class FSExperimentResult:
    run_name: str
    fs_major: int
    run_dir: Path
    checkpoint_path: Path
    n_train: int
    n_val: int
    n_test: int
    warmup_steps: int
    cosine_total_steps: int
    best_epoch: int
    best_val_ce: float
    final_test_ce: float | None
    history_df: pd.DataFrame
    sample_metrics_df: pd.DataFrame
    region_metrics_df: pd.DataFrame
    timing_df: pd.DataFrame
    plot_bundle: dict


def _default_model_overrides() -> dict:
    # Match the successful cosine notebook profile.
    return {
        "channels": (12, 20, 32, 48, 64, 96),
        "transformer_depth": 2,
        "n_heads": 4,
        "dropout": 0.0,
        "positional_encoding": "sincos",
    }


def _default_train_overrides(checkpoint_path: Path) -> dict:
    return {
        "epochs": 5,
        "effective_batch_size": 1,
        "initial_micro_batch_size": 1,
        "patch_chunk_size": 192,
        "quick_val_every_steps": 0,
        "quick_val_batches": 1,
        "early_stopping_patience": 0,
        "lr": 1e-3,
        "min_lr": 1e-6,
        "lr_scheduler": "cosine_warmup",
        "warmup_steps": 0,
        "cosine_total_steps": None,
        "weight_decay": 5e-5,
        "target_max_vram_gb": 24.0,
        "num_workers": 8,
        "prefetch_factor": 2,
        "spatial_window": None,
        "spatial_stride": None,
        "enforce_global_attention": True,
        "checkpoint_path": str(checkpoint_path),
        "max_train_steps": 0,
        "max_train_hours": 0.0,
    }


def run_short_finetune_experiment(
    *,
    run_name: str,
    fs_major: int,
    split: dict[str, list[str]],
    base_checkpoint_path: str | Path,
    checkpoint_base_dir: str | Path,
    fs_major_by_dataset: dict[str, int | None],
    data_overrides: dict | None = None,
    model_overrides: dict | None = None,
    train_overrides: dict | None = None,
    seed: int = SEED,
    device: str = DEVICE,
    patch_size: tuple = PATCH_SIZE,
    default_major_for_unmapped: int | None = None,
) -> FSExperimentResult:
    """
    Run a compact fine-tuning experiment with unique outputs and cosine+warmup.

    - Starts from base_checkpoint_path
    - Trains at most 5 epochs by default
    - Evaluates only on provided test split
    - Uses shared DATA_CFG cache_dir by default (override via `data_overrides`)
    - Optional `default_major_for_unmapped` lets non-ds* paths be treated as a chosen major
    """
    required = ["x_train", "y_train", "x_val", "y_val", "x_test", "y_test"]
    for k in required:
        if k not in split:
            raise KeyError(f"split is missing key: {k}")

    if len(split["x_train"]) == 0:
        raise ValueError(f"{run_name}: empty training split")

    validate_training_versions(
        split["x_train"],
        fs_major_by_dataset,
        expected_major=int(fs_major),
        default_major=(
            int(default_major_for_unmapped)
            if default_major_for_unmapped is not None
            else (int(fs_major) if int(fs_major) == 8 else None)
        ),
    )

    run_dir = make_unique_run_dir(checkpoint_base_dir, run_name)
    ckpt_path = run_dir / "transunet3d_best.pt"

    user_data_overrides = dict(data_overrides or {})
    user_model_overrides = dict(model_overrides or {})
    user_train_overrides = dict(train_overrides or {})

    data_cfg_overrides = {
        "x_train_files": list(split["x_train"]),
        "y_train_files": list(split["y_train"]),
        "x_val_files": list(split["x_val"]),
        "y_val_files": list(split["y_val"]),
        "x_test_files": list(split["x_test"]),
        "y_test_files": list(split["y_test"]),
        "group_split_enabled": False,
        "cache_rebuild": False,
    }
    data_cfg_overrides.update(user_data_overrides)

    model_cfg_overrides = _default_model_overrides()
    model_cfg_overrides.update(user_model_overrides)

    train_cfg_overrides = _default_train_overrides(ckpt_path)
    train_cfg_overrides.update(user_train_overrides)

    data_cfg, model_cfg, train_cfg, _changed = build_runtime_cfgs(
        data_overrides=data_cfg_overrides,
        model_overrides=model_cfg_overrides,
        train_overrides=train_cfg_overrides,
    )

    pipeline = prepare_data_pipeline(
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        seed=int(seed),
        device=device,
        patch_size=patch_size,
    )

    epochs = int(train_cfg["epochs"])
    steps_per_epoch = max(1, len(pipeline.train_loader))

    if str(train_cfg.get("lr_scheduler", "plateau")).strip().lower() in {
        "cosine_warmup",
        "cosine",
        "cosine_with_warmup",
    }:
        total_steps = max(1, epochs * steps_per_epoch)
        warmup_steps = int(train_cfg.get("warmup_steps", 0))
        if warmup_steps <= 0:
            # ~10% of the full schedule, at least 1 step.
            warmup_steps = max(1, int(round(0.10 * total_steps)))
        warmup_steps = min(max(1, warmup_steps), max(1, total_steps - 1))
        train_cfg["warmup_steps"] = int(warmup_steps)
        train_cfg["cosine_total_steps"] = int(total_steps)
    else:
        warmup_steps = 0
        total_steps = 0

    model = TransUNet3D(
        n_classes=pipeline.n_classes,
        in_channels=int(model_cfg["in_channels"]),
        base_shape=tuple(int(v) for v in pipeline.base_volume_shape),
        patch_size=patch_size,
        channels=tuple(int(v) for v in model_cfg["channels"]),
        transformer_depth=int(model_cfg["transformer_depth"]),
        n_heads=int(model_cfg["n_heads"]),
        dropout=float(model_cfg["dropout"]),
        positional_encoding=str(model_cfg["positional_encoding"]),
        task_type=str(model_cfg.get("task_type", "classification")),
    ).to(device)

    base_ckpt = torch.load(base_checkpoint_path, map_location="cpu")
    base_state = base_ckpt.get("model_state", base_ckpt)
    missing, unexpected = model.load_state_dict(base_state, strict=False)
    if missing:
        print(f"[{run_name}] missing keys while loading base checkpoint: {len(missing)}")
    if unexpected:
        print(f"[{run_name}] unexpected keys while loading base checkpoint: {len(unexpected)}")

    runtime = init_training_runtime(model, train_cfg)

    target_vram = train_cfg.get("target_max_vram_gb", None)
    patience = int(train_cfg.get("early_stopping_patience", 0))
    auto_increase_micro_batch = bool(train_cfg.get("auto_increase_micro_batch", False))
    effective_batch_size = int(pipeline.effective_batch_size)

    best_val = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        model.train()
        train_num, train_den = 0.0, 0
        epoch_hit_oom = False

        pbar = tqdm(
            pipeline.train_loader,
            total=len(pipeline.train_loader),
            desc=f"{run_name} | epoch {epoch:03d} train",
            leave=False,
        )
        for x_cpu, y_cpu in pbar:
            batch_num, batch_den, hit_oom = train_step_runtime(model, runtime, x_cpu, y_cpu)
            epoch_hit_oom = epoch_hit_oom or bool(hit_oom)

            train_num += float(batch_num)
            train_den += int(batch_den)

            alloc_gb, peak_gb = gpu_mem_gb()
            if device == "cuda" and target_vram is not None and alloc_gb > float(target_vram):
                if runtime.runtime_micro_bs > 1:
                    runtime.runtime_micro_bs = max(1, runtime.runtime_micro_bs // 2)
                elif runtime.runtime_patch_chunk > 16:
                    runtime.runtime_patch_chunk = max(16, runtime.runtime_patch_chunk // 2)

            pbar.set_postfix(
                loss=f"{(train_num / max(1, train_den)):.4f}",
                lr=f"{runtime.optimizer.param_groups[0]['lr']:.2e}",
                mb=runtime.runtime_micro_bs,
                pc=runtime.runtime_patch_chunk,
                vram=f"{alloc_gb:.1f}/{peak_gb:.1f}G",
            )
        pbar.close()

        train_loss = train_num / max(1, train_den)
        if pipeline.val_loader is not None and len(pipeline.val_loader) > 0:
            val_loss = evaluate_runtime(
                model,
                pipeline.val_loader,
                runtime,
                desc=f"{run_name} | epoch {epoch:03d} val",
            )
        else:
            val_loss = train_loss

        step_scheduler_epoch(runtime, val_loss)

        epoch_sec = float(time.time() - epoch_t0)
        alloc_gb, peak_gb = gpu_mem_gb()
        history.append(
            {
                "epoch": int(epoch),
                "train_ce": float(train_loss),
                "val_ce": float(val_loss),
                "lr": float(runtime.optimizer.param_groups[0]["lr"]),
                "micro_bs": int(runtime.runtime_micro_bs),
                "patch_chunk": int(runtime.runtime_patch_chunk),
                "window": tuple(runtime.runtime_window) if runtime.runtime_window is not None else None,
                "sec": epoch_sec,
                "alloc_gb": float(alloc_gb),
                "peak_gb": float(peak_gb),
            }
        )

        print(
            f"[{run_name}] epoch {epoch:03d} | train_ce={train_loss:.4f} | "
            f"val_ce={val_loss:.4f} | lr={runtime.optimizer.param_groups[0]['lr']:.2e} | "
            f"mb={runtime.runtime_micro_bs} | pc={runtime.runtime_patch_chunk} | "
            f"vram={alloc_gb:.1f}/{peak_gb:.1f}GB | t={epoch_sec:.1f}s"
        )

        if val_loss < best_val:
            best_val = float(val_loss)
            best_epoch = int(epoch)
            epochs_no_improve = 0

            torch.save(
                {
                    "epoch": best_epoch,
                    "best_val": best_val,
                    "model_state": model.state_dict(),
                    "optimizer_state": runtime.optimizer.state_dict(),
                    "scheduler_state": runtime.scheduler.state_dict(),
                    "scaler_state": runtime.scaler.state_dict() if runtime.scaler.is_enabled() else None,
                    "model_cfg": model_cfg,
                    "train_cfg": train_cfg,
                    "data_cfg": data_cfg,
                    "patch_size": patch_size,
                    "base_volume_shape": pipeline.base_volume_shape,
                    "n_classes": pipeline.n_classes,
                    "runtime_micro_bs": runtime.runtime_micro_bs,
                    "runtime_patch_chunk": runtime.runtime_patch_chunk,
                    "runtime_window": runtime.runtime_window,
                    "history": history,
                    "base_checkpoint_path": str(Path(base_checkpoint_path).resolve()),
                },
                ckpt_path,
            )
            print(f"[{run_name}] saved best checkpoint -> {ckpt_path} (epoch={best_epoch}, val_ce={best_val:.4f})")
        else:
            epochs_no_improve += 1

        if auto_increase_micro_batch and (not epoch_hit_oom) and runtime.runtime_micro_bs < effective_batch_size:
            runtime.runtime_micro_bs = min(effective_batch_size, runtime.runtime_micro_bs * 2)

        if patience > 0 and epochs_no_improve >= patience:
            print(f"[{run_name}] early stopping at epoch {epoch:03d} (patience={patience})")
            break

    if ckpt_path.exists():
        best_ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state"], strict=False)

    final_test_ce = None
    if pipeline.test_loader is not None and len(pipeline.test_loader) > 0:
        final_test_ce = float(
            evaluate_runtime(
                model,
                pipeline.test_loader,
                runtime,
                desc=f"{run_name} | final test",
            )
        )

    eval_results_cache_dir = Path(
        data_cfg.get(
            "eval_results_cache_dir",
            DATA_CFG.get("eval_results_cache_dir", ".eval_cache/test_set/results"),
        )
    )

    sample_metrics_df, region_metrics_df, timing_df = collect_test_metrics_fast_cached(
        model=model,
        pipeline=pipeline,
        class_values=pipeline.class_values,
        patch_chunk_size=int(train_cfg["patch_chunk_size"]),
        compute_boundary=False,
        region_df=None,
        device=device,
        results_cache_dir=eval_results_cache_dir,
        reuse_results_cache=True,
    )
    bundle = build_eval_plot_bundle(
        sample_metrics_df=sample_metrics_df,
        region_metrics_df=region_metrics_df,
        method="model",
        tissue_as_percent=True,
    )

    history_df = pd.DataFrame(history)
    history_df.to_csv(run_dir / "history.csv", index=False)
    sample_metrics_df.to_csv(run_dir / "sample_metrics.csv", index=False)
    region_metrics_df.to_csv(run_dir / "region_metrics.csv", index=False)
    timing_df.to_csv(run_dir / "timing.csv", index=False)

    pd.DataFrame(
        {
            "run_name": [run_name],
            "fs_major": [int(fs_major)],
            "n_train": [int(len(split["x_train"]))],
            "n_val": [int(len(split["x_val"]))],
            "n_test": [int(len(split["x_test"]))],
            "warmup_steps": [int(warmup_steps)],
            "cosine_total_steps": [int(total_steps)],
            "best_epoch": [int(best_epoch)],
            "best_val_ce": [float(best_val)],
            "final_test_ce": [float(final_test_ce) if final_test_ce is not None else float("nan")],
            "run_dir": [str(run_dir.resolve())],
            "checkpoint_path": [str(ckpt_path.resolve())],
        }
    ).to_csv(run_dir / "summary.csv", index=False)

    return FSExperimentResult(
        run_name=run_name,
        fs_major=int(fs_major),
        run_dir=run_dir,
        checkpoint_path=ckpt_path,
        n_train=int(len(split["x_train"])),
        n_val=int(len(split["x_val"])),
        n_test=int(len(split["x_test"])),
        warmup_steps=int(warmup_steps),
        cosine_total_steps=int(total_steps),
        best_epoch=int(best_epoch),
        best_val_ce=float(best_val),
        final_test_ce=(None if final_test_ce is None else float(final_test_ce)),
        history_df=history_df,
        sample_metrics_df=sample_metrics_df,
        region_metrics_df=region_metrics_df,
        timing_df=timing_df,
        plot_bundle=bundle,
    )

from copy import deepcopy
import random
import torch
from pathlib import Path

# Repro + device
SEED = 1337
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

MODULE_PATH = Path(__file__).resolve().parent

PATCH_SIZE = (16, 16, 16)

DATA_PATH = MODULE_PATH.parent / "data"

DATA_CFG = {
    # Use explicit lists, .txt/.lst, or .json list. If empty, auto-discovery under tensors/** is used.
    "x_train_files": [],
    "y_train_files": [],

    # Optional explicit validation/test files. If omitted, group/random split is used.
    "x_val_files": [],
    "y_val_files": [],
    "x_test_files": [],
    "y_test_files": [],

    # Grouped split config (group = tensors/{x}/...). Disabled by default.
    # Use explicit x/y val/test lists or set this True when needed.
    "group_split_enabled": False,
    "group_split_root": "tensors",
    "split_ratios": (0.8, 0.1, 0.1),
    "split_seed": SEED,

    # Fallback global split if grouped split is disabled
    "val_fraction": 0.1,

    # Optional caps for debugging
    "max_train_samples": None,
    "max_val_samples": None,

    # If None, inferred from label_values/label scan below.
    "n_classes": None,

    # Optional explicit sparse label ids (e.g. [0, 2, 3, 4, 1001, ...]).
    # If empty and use_default_aparc_aseg_labels=True, use the canonical aparc+aseg ID set.
    # Otherwise scan a bounded number of y-files to build a sparse->dense LUT.
    "label_values": [],
    "use_default_aparc_aseg_labels": True,
    "label_scan_max_files": 64,

    # Keep only this many x/y files opened in-memory at once.
    "max_open_files": 1,

    # Preprocessed cache to remove CPU bottlenecks at train time.
    "cache_enabled": True,
    # Shared repo-level cache (absolute path) so behavior is stable across cwd
    # and notebooks/scripts reuse the same preprocessed tensors.
    "cache_dir": str(MODULE_PATH.parent / ".tensor_cache_preproc"),
    # Shared eval-results cache (pickled metric frames) across runs/checkpoints.
    "eval_results_cache_dir": str(MODULE_PATH.parent / ".eval_cache" / "test_set" / "results"),
    "cache_rebuild": False,
    # If True, validate manifest entries against the filesystem on startup.
    # Default False keeps repeat runs fast and trusts the saved pair mapping.
    "cache_recheck": False,
    "cache_apply_label_lut": True,
    "cache_zscore_x": True,
    "cache_x_dtype": "float16",
    "cache_y_dtype": "int16",
    # Behavior when source tensors cannot be loaded: "skip" or "raise".
    "cache_on_load_error": "skip",
    # Behavior when labels are not covered by current LUT during cache build: "skip" or "raise".
    "cache_on_label_miss": "skip",
    # If False, previously skipped files are not retried on every run.
    "cache_retry_skipped": False,
}

MODEL_CFG = {
    "in_channels": 1,
    "channels": (16, 24, 48, 72, 96, 128),
    "transformer_depth": 2,
    "n_heads": 4,
    "dropout": 0.1,
    # "learned" keeps prior behavior; "sincos" uses deterministic 3D coordinates.
    "positional_encoding": "learned",
    # "classification" for discrete labels, "regression" for continuous volumes.
    "task_type": "classification",
}

TRAIN_CFG = {
    "epochs": 200,
    "effective_batch_size": 2,
    "initial_micro_batch_size": 1,
    "patch_chunk_size": 64,

    # Use windows for VRAM safety. None => full volume.
    "spatial_window": (96, 96, 96),
    "spatial_stride": None,
    # If True, keep full-volume/global attention under OOM by reducing only
    # micro-batch and patch chunk sizes (never shrinking spatial_window).
    "enforce_global_attention": False,

    "lr": 3e-4,
    "min_lr": 1e-6,
    # Scheduler: "plateau" (default) or "cosine_warmup".
    "lr_scheduler": "plateau",
    # Plateau scheduler knobs.
    "plateau_factor": 0.5,
    "plateau_patience": 1,
    # Cosine+warmup scheduler knobs (used only when lr_scheduler="cosine_warmup").
    "warmup_steps": 0,
    # Must be set for cosine_warmup (typically epochs * len(train_loader)).
    "cosine_total_steps": None,
    "weight_decay": 1e-4,
    "grad_clip_norm": 1.0,
    "target_max_vram_gb": 31.0,

    "num_workers": 4,
    "pin_memory": True,
    "prefetch_factor": 2,

    # Frequent validation reporting
    "quick_val_every_steps": 50,
    "quick_val_batches": 4,

    # Checkpoint / stopping
    "checkpoint_path": "checkpoints/transunet3d_best.pt",
    "early_stopping_patience": 20,

    # If an epoch runs cleanly (no OOM), try increasing micro-batch.
    "auto_increase_micro_batch": False,
    # Optional hard limits for notebook/runtime loops (<=0 disables).
    "max_train_steps": 0,
    "max_train_hours": 0.0,
}


def copy_cfg(cfg):
    return deepcopy(cfg)


def cfg_with_overrides(default_cfg, overrides=None):
    cfg = deepcopy(default_cfg)
    changed = {}
    if overrides:
        for key, value in overrides.items():
            if key not in cfg:
                raise KeyError(f"Unknown cfg key: {key}")
            if cfg[key] != value:
                cfg[key] = value
                changed[key] = value
    return cfg, changed


def build_runtime_cfgs(data_overrides=None, model_overrides=None, train_overrides=None):
    data_cfg, data_changed = cfg_with_overrides(DATA_CFG, data_overrides)
    model_cfg, model_changed = cfg_with_overrides(MODEL_CFG, model_overrides)
    train_cfg, train_changed = cfg_with_overrides(TRAIN_CFG, train_overrides)
    changed = {
        "data": data_changed,
        "model": model_changed,
        "train": train_changed,
    }
    return data_cfg, model_cfg, train_cfg, changed

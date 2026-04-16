from dataclasses import dataclass
from pathlib import Path

import torch

from .config import DATA_CFG as DEFAULT_DATA_CFG
from .config import DEVICE, PATCH_SIZE, SEED, TRAIN_CFG as DEFAULT_TRAIN_CFG
from .config import MODULE_PATH
from .config import build_runtime_cfgs
from .data import (
    build_cache_for_pairs,
    build_dataset,
    build_label_lut,
    default_aparc_aseg_label_values,
    infer_label_values,
    limit_dataset,
    make_loader,
    resolve_paths,
    split_pairs_by_group,
)
from .utils import ensure_divisible
from . import data as data_mod
from . import train as train_mod
from . import utils as utils_mod


@dataclass
class DataPipeline:
    train_ds: object
    val_ds: object
    test_ds: object
    train_loader: object
    val_loader: object
    test_loader: object
    class_values: torch.Tensor
    label_lut: torch.Tensor
    n_classes: int
    inferred_n_classes: int
    label_source: str
    cache_enabled: bool
    cache_dir: Path
    cached_built_total: int
    cached_reused_total: int
    cached_skipped_total: int
    effective_batch_size: int
    base_volume_shape: tuple
    runtime_label_lut: torch.Tensor | None
    runtime_zscore_x: bool
    data_cfg: dict
    train_cfg: dict


class _DenseToFsLabelDataset:
    def __init__(self, base_ds, class_values):
        self.base_ds = base_ds
        self.class_values = torch.as_tensor(class_values, dtype=torch.int64).cpu()

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, y = self.base_ds[idx]
        y_dense = torch.as_tensor(y, dtype=torch.int64)
        y_fs = self.class_values[y_dense]
        return x, y_fs.contiguous()


def sync_runtime_modules(data_cfg, train_cfg, seed=SEED, device=DEVICE, patch_size=PATCH_SIZE):
    data_mod.DATA_CFG = data_cfg
    data_mod.TRAIN_CFG = train_cfg
    data_mod.SEED = int(seed)
    data_mod.DEVICE = str(device)
    data_mod.PATCH_SIZE = tuple(int(v) for v in patch_size)

    utils_mod.DEVICE = str(device)
    utils_mod.PATCH_SIZE = tuple(int(v) for v in patch_size)

    train_mod.DEVICE = str(device)
    train_mod.PATCH_SIZE = tuple(int(v) for v in patch_size)


def prepare_data_pipeline(data_cfg=None, train_cfg=None, seed=SEED, device=DEVICE, patch_size=PATCH_SIZE, _validate_shape=True):
    if data_cfg is None:
        data_cfg = DEFAULT_DATA_CFG
    if train_cfg is None:
        train_cfg = DEFAULT_TRAIN_CFG

    sync_runtime_modules(
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        seed=seed,
        device=device,
        patch_size=patch_size,
    )

    x_all = resolve_paths(data_cfg["x_train_files"])
    y_all = resolve_paths(data_cfg["y_train_files"])
    if not x_all or not y_all:
        raise ValueError("No training files found. Set DATA_CFG['x_train_files'] and DATA_CFG['y_train_files'].")
    if len(x_all) != len(y_all):
        raise ValueError("DATA_CFG x/y file lists must have same length")

    x_val_exp = resolve_paths(data_cfg["x_val_files"]) if data_cfg["x_val_files"] else []
    y_val_exp = resolve_paths(data_cfg["y_val_files"]) if data_cfg["y_val_files"] else []
    x_test_exp = resolve_paths(data_cfg["x_test_files"]) if data_cfg.get("x_test_files") else []
    y_test_exp = resolve_paths(data_cfg["y_test_files"]) if data_cfg.get("y_test_files") else []

    if x_val_exp or y_val_exp or x_test_exp or y_test_exp:
        if len(x_val_exp) != len(y_val_exp):
            raise ValueError("Explicit val x/y lists must have same length")
        if len(x_test_exp) != len(y_test_exp):
            raise ValueError("Explicit test x/y lists must have same length")
        x_train_files, y_train_files = x_all, y_all
        x_val_files, y_val_files = x_val_exp, y_val_exp
        x_test_files, y_test_files = x_test_exp, y_test_exp
    else:
        if data_cfg.get("group_split_enabled", True):
            ratios = tuple(data_cfg.get("split_ratios", (0.8, 0.1, 0.1)))
            x_train_files, y_train_files, x_val_files, y_val_files, x_test_files, y_test_files = split_pairs_by_group(
                x_all,
                y_all,
                root=data_cfg.get("group_split_root", "tensors"),
                ratios=ratios,
                seed=data_cfg.get("split_seed", seed),
            )
        else:
            n_total = len(x_all)
            n_val = max(1, int(round(n_total * float(data_cfg.get("val_fraction", 0.1)))))
            n_val = min(n_val, n_total - 1)
            g = torch.Generator().manual_seed(int(seed))
            perm = torch.randperm(n_total, generator=g).tolist()
            val_idx = set(perm[:n_val])

            x_train_files, y_train_files, x_val_files, y_val_files = [], [], [], []
            for i, (x, y) in enumerate(zip(x_all, y_all)):
                if i in val_idx:
                    x_val_files.append(x)
                    y_val_files.append(y)
                else:
                    x_train_files.append(x)
                    y_train_files.append(y)
            x_test_files, y_test_files = [], []

    if not x_train_files:
        raise ValueError("Train split is empty")

    all_y_files = y_train_files + y_val_files + y_test_files
    label_values_cfg = data_cfg.get("label_values")
    if label_values_cfg:
        label_values = [int(v) for v in label_values_cfg]
        label_source = "DATA_CFG['label_values']"
    elif bool(data_cfg.get("use_default_aparc_aseg_labels", True)):
        label_values = default_aparc_aseg_label_values()
        label_source = "default aparc+aseg labels"
    else:
        scan_max = int(data_cfg.get("label_scan_max_files", 64))
        if scan_max <= 0:
            raise ValueError("Set DATA_CFG['label_values'] or DATA_CFG['label_scan_max_files'] > 0")
        label_values, scanned_files = infer_label_values(
            all_y_files,
            max_files=scan_max,
            seed=data_cfg.get("split_seed", seed) + 17,
        )
        label_source = f"scan({scanned_files} y-files)"

    if 0 not in label_values:
        label_values = [0] + label_values

    class_values, label_lut = build_label_lut(label_values)
    inferred_n_classes = int(class_values.numel())

    cache_enabled = bool(data_cfg.get("cache_enabled", True))
    cache_rebuild = bool(data_cfg.get("cache_rebuild", False))
    cache_recheck = bool(data_cfg.get("cache_recheck", False))
    cache_apply_label_lut = bool(data_cfg.get("cache_apply_label_lut", True))
    cache_dir = Path(data_cfg.get("cache_dir", MODULE_PATH.parent / ".tensor_cache_preproc"))

    cached_built_total = 0
    cached_reused_total = 0
    cached_skipped_total = 0
    if cache_enabled:
        x_train_cache, y_train_cache, built_train, reused_train, skipped_train = build_cache_for_pairs(
            x_train_files,
            y_train_files,
            cache_dir=cache_dir / "train",
            label_lut=label_lut,
            rebuild=cache_rebuild,
            recheck=cache_recheck,
        )
        cached_built_total += int(built_train)
        cached_reused_total += int(reused_train)
        cached_skipped_total += int(skipped_train)
        if x_val_files:
            x_val_cache, y_val_cache, built_val, reused_val, skipped_val = build_cache_for_pairs(
                x_val_files,
                y_val_files,
                cache_dir=cache_dir / "val",
                label_lut=label_lut,
                rebuild=cache_rebuild,
                recheck=cache_recheck,
            )
            cached_built_total += int(built_val)
            cached_reused_total += int(reused_val)
            cached_skipped_total += int(skipped_val)
        else:
            x_val_cache, y_val_cache = [], []
        if x_test_files:
            x_test_cache, y_test_cache, built_test, reused_test, skipped_test = build_cache_for_pairs(
                x_test_files,
                y_test_files,
                cache_dir=cache_dir / "test",
                label_lut=label_lut,
                rebuild=cache_rebuild,
                recheck=cache_recheck,
            )
            cached_built_total += int(built_test)
            cached_reused_total += int(reused_test)
            cached_skipped_total += int(skipped_test)
        else:
            x_test_cache, y_test_cache = [], []
    else:
        x_train_cache, y_train_cache = x_train_files, y_train_files
        x_val_cache, y_val_cache = x_val_files, y_val_files
        x_test_cache, y_test_cache = x_test_files, y_test_files

    if not x_train_cache:
        raise ValueError(
            "No usable train files after cache filtering. "
            "Set DATA_CFG['label_values'] explicitly or set DATA_CFG['cache_on_label_miss']='raise' to debug."
        )

    runtime_label_lut = None if (cache_enabled and cache_apply_label_lut) else label_lut
    runtime_zscore_x = not (cache_enabled and bool(data_cfg.get("cache_zscore_x", True)))

    train_ds_full = build_dataset(
        x_train_cache,
        y_train_cache,
        data_cfg["max_open_files"],
        zscore_x=runtime_zscore_x,
        label_lut=runtime_label_lut,
    )
    val_ds_full = build_dataset(
        x_val_cache,
        y_val_cache,
        data_cfg["max_open_files"],
        zscore_x=runtime_zscore_x,
        label_lut=runtime_label_lut,
    ) if x_val_cache else None
    test_ds_full = build_dataset(
        x_test_cache,
        y_test_cache,
        data_cfg["max_open_files"],
        zscore_x=runtime_zscore_x,
        label_lut=runtime_label_lut,
    ) if x_test_cache else None

    train_ds = limit_dataset(train_ds_full, data_cfg["max_train_samples"], seed=seed)
    val_ds = limit_dataset(val_ds_full, data_cfg["max_val_samples"], seed=seed + 1) if val_ds_full is not None else None
    test_ds = test_ds_full

    if data_cfg["n_classes"] is None:
        n_classes = inferred_n_classes
    else:
        n_classes = int(data_cfg["n_classes"])
        if n_classes < inferred_n_classes:
            raise ValueError(f"DATA_CFG['n_classes']={n_classes} is smaller than inferred classes={inferred_n_classes}")

    if n_classes < 2:
        raise ValueError(f"n_classes must be >=2, got {n_classes}")

    effective_batch_size = int(train_cfg["effective_batch_size"])
    if effective_batch_size < 1:
        raise ValueError(f"TRAIN_CFG['effective_batch_size'] must be >=1, got {effective_batch_size}")

    train_loader = make_loader(
        train_ds,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
        prefetch_factor=train_cfg["prefetch_factor"],
    )
    val_loader = make_loader(
        val_ds,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
        prefetch_factor=train_cfg["prefetch_factor"],
    ) if val_ds is not None else None
    test_loader = make_loader(
        test_ds,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
        prefetch_factor=train_cfg["prefetch_factor"],
    ) if test_ds is not None else None

    if _validate_shape:
        first_x_batch, first_y_batch = next(iter(train_loader))
        base_volume_shape = tuple(int(v) for v in first_x_batch.shape[1:])
        ensure_divisible(base_volume_shape, patch_size)
        del first_x_batch, first_y_batch
    else:
        base_volume_shape = ()

    return DataPipeline(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_values=class_values,
        label_lut=label_lut,
        n_classes=int(n_classes),
        inferred_n_classes=int(inferred_n_classes),
        label_source=label_source,
        cache_enabled=cache_enabled,
        cache_dir=cache_dir,
        cached_built_total=int(cached_built_total),
        cached_reused_total=int(cached_reused_total),
        cached_skipped_total=int(cached_skipped_total),
        effective_batch_size=int(effective_batch_size),
        base_volume_shape=base_volume_shape,
        runtime_label_lut=runtime_label_lut,
        runtime_zscore_x=runtime_zscore_x,
        data_cfg=data_cfg,
        train_cfg=train_cfg,
    )


def prepare_loaders(
    data_root_or_cfg=None,
    train_cfg=None,
    seed=SEED,
    device=DEVICE,
    patch_size=PATCH_SIZE,
    batch_size=None,
    from_dense_labels=False,
):
    """Return `(train_loader, val_loader, test_loader)`.

    `data_root_or_cfg` may be:
    - `str` / `Path`: tensor root containing `rawavg.pt` and `aparc+aseg.pt`
    - `dict`: full data config for `prepare_data_pipeline`
    - `None`: use the default data config
    If `from_dense_labels=True`, decode dense target labels back to FreeSurfer ids.
    """
    train_overrides = dict(train_cfg) if train_cfg is not None else {}
    if batch_size is not None:
        train_overrides["effective_batch_size"] = int(batch_size)

    if isinstance(data_root_or_cfg, (str, Path)):
        tensors_root = Path(data_root_or_cfg).expanduser().resolve()
        from .experiments import build_v8_split_from_root

        split = build_v8_split_from_root(tensors_root=tensors_root, seed=seed)

        data_cfg, _, built_train_cfg, _ = build_runtime_cfgs(
            data_overrides={
                "x_train_files": list(split["x_train"]),
                "y_train_files": list(split["y_train"]),
                "x_val_files": list(split["x_val"]),
                "y_val_files": list(split["y_val"]),
                "x_test_files": list(split["x_test"]),
                "y_test_files": list(split["y_test"]),
                "group_split_enabled": False,
                "cache_enabled": False,
                "cache_rebuild": False,
                "cache_recheck": False,
            },
            train_overrides=train_overrides,
        )
        train_cfg = built_train_cfg
    else:
        data_cfg = data_root_or_cfg
        if data_cfg is not None:
            data_cfg = dict(data_cfg)
            data_cfg["cache_enabled"] = False
            data_cfg["cache_rebuild"] = False
            data_cfg["cache_recheck"] = False
        if train_overrides:
            _, _, train_cfg, _ = build_runtime_cfgs(train_overrides=train_overrides)

    pipeline = prepare_data_pipeline(
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        seed=seed,
        device=device,
        patch_size=patch_size,
        _validate_shape=False,
    )

    if bool(from_dense_labels):
        effective_batch_size = int(pipeline.train_cfg["effective_batch_size"])
        loader_kwargs = {
            "num_workers": pipeline.train_cfg["num_workers"],
            "pin_memory": pipeline.train_cfg["pin_memory"],
            "prefetch_factor": pipeline.train_cfg["prefetch_factor"],
        }
        train_ds = _DenseToFsLabelDataset(pipeline.train_ds, pipeline.class_values)
        val_ds = _DenseToFsLabelDataset(pipeline.val_ds, pipeline.class_values) if pipeline.val_ds is not None else None
        test_ds = _DenseToFsLabelDataset(pipeline.test_ds, pipeline.class_values) if pipeline.test_ds is not None else None
        train_loader = make_loader(train_ds, batch_size=effective_batch_size, shuffle=True, **loader_kwargs)
        val_loader = (
            make_loader(val_ds, batch_size=effective_batch_size, shuffle=False, **loader_kwargs)
            if val_ds is not None else None
        )
        test_loader = (
            make_loader(test_ds, batch_size=effective_batch_size, shuffle=False, **loader_kwargs)
            if test_ds is not None else None
        )
        return train_loader, val_loader, test_loader

    return pipeline.train_loader, pipeline.val_loader, pipeline.test_loader


def summarize_data_pipeline(pipeline, data_cfg=None):
    if data_cfg is None:
        data_cfg = pipeline.data_cfg

    print(f"train samples: {len(pipeline.train_ds)}")
    print(f"val samples:   {len(pipeline.val_ds) if pipeline.val_ds is not None else 0}")
    print(f"test samples:  {len(pipeline.test_ds) if pipeline.test_ds is not None else 0}")
    print(f"volume shape (padded): {pipeline.base_volume_shape}")
    print(f"inferred classes: {pipeline.inferred_n_classes}")
    print(f"n_classes used:  {pipeline.n_classes}")
    print(f"label source:    {pipeline.label_source}")
    print(f"cache enabled:   {pipeline.cache_enabled}")
    if pipeline.cache_enabled:
        print(f"cache dir:       {pipeline.cache_dir}")
        print(f"cache x dtype:   {data_cfg.get('cache_x_dtype', 'float16')}")
        print(f"cache y dtype:   {data_cfg.get('cache_y_dtype', 'int16')}")
        print(f"cache built now: {pipeline.cached_built_total}")
        print(f"cache reused:    {pipeline.cached_reused_total}")
        print(f"cache skipped:   {pipeline.cached_skipped_total}")
    print(f"effective batch size: {pipeline.effective_batch_size}")
    print(f"max_open_files: {data_cfg['max_open_files']}")

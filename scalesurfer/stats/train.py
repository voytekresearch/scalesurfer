from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .models import StatsPredictionModel, build_stats_model
from .utils import (
    group_for_target,
    load_or_build_pooled_feature_cache,
    make_stats_feature_loader,
    make_stats_loader,
    pooled_feature_cache_path,
    save_json,
    strip_compile_prefix,
)


def masked_huber_loss(
    predictions: dict[str, torch.Tensor],
    y: dict[str, torch.Tensor],
    mask: dict[str, torch.Tensor],
    *,
    delta: float = 1.0,
    reduction: str = "scalar_mean",
    group_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, int]:
    reduction = str(reduction)
    if reduction not in {"scalar_mean", "group_mean"}:
        raise ValueError("reduction must be 'scalar_mean' or 'group_mean'")
    total = None
    denom = 0
    group_losses = []
    group_denoms = []
    first_pred = next(iter(predictions.values()))
    for group, pred in predictions.items():
        target = y[group].to(device=pred.device, dtype=pred.dtype)
        valid = mask[group].to(device=pred.device)
        if not valid.any():
            continue

        if not torch.isfinite(pred).all():
            raise FloatingPointError(f"Non-finite predictions in group={group}")

        if not torch.isfinite(target[valid]).all():
            raise FloatingPointError(f"Non-finite valid targets in group={group}")

        loss = F.huber_loss(pred, target, delta=float(delta), reduction="none")
        selected = loss[valid]
        if reduction == "group_mean":
            weight = 1.0 if group_weights is None else float(group_weights.get(group, 1.0))
            group_losses.append(selected.mean() * weight)
            group_denoms.append(weight)
        else:
            group_sum = selected.sum()
            total = group_sum if total is None else total + group_sum
            denom += int(selected.numel())
    if reduction == "group_mean":
        if not group_losses:
            return torch.zeros((), device=first_pred.device), 0
        weight_sum = max(float(sum(group_denoms)), 1e-8)
        batch_size = int(first_pred.shape[0]) if first_pred.ndim else 1
        return torch.stack(group_losses).sum() / weight_sum, batch_size
    if total is None:
        return torch.zeros((), device=first_pred.device), 0
    return total / max(1, denom), denom


def run_stats_epoch(
    model: StatsPredictionModel,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer | None,
    device: str | torch.device,
    desc: str,
    delta: float = 1.0,
    amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    loss_reduction: str = "scalar_mean",
    group_loss_weights: dict[str, float] | None = None,
) -> float:
    train = optimizer is not None
    model.train(mode=train)
    total, denom = 0.0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    use_amp = bool(amp) and torch.device(device).type == "cuda"
    with context:
        for batch in tqdm(loader, desc=desc, leave=False):
            if train:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                if "h" in batch:
                    h = batch["h"].to(device=device, non_blocking=True)
                    predictions = model.forward_from_features(h)
                else:
                    x = batch["x"].to(device=device, non_blocking=True)
                    seg = batch["seg"].to(device=device, non_blocking=True)

                    # predictions = model(x, seg)
                    h = model.extract_features(x, seg)

                    if not torch.isfinite(h).all():
                        bad = ~torch.isfinite(h)
                        rows = batch["rows"]
                        bad_samples = [rows[i]["sample_id"] for i in torch.where(bad.any(dim=1))[0].detach().cpu().tolist()]
                        raise FloatingPointError(
                            f"Non-finite extracted features. "
                            f"bad_samples={bad_samples[:5]}, "
                            f"h_min={h[torch.isfinite(h)].min().item() if torch.isfinite(h).any() else None}, "
                            f"h_max={h[torch.isfinite(h)].max().item() if torch.isfinite(h).any() else None}"
                        )

                    predictions = model.forward_from_features(h)

                    for g, p in predictions.items():
                        if not torch.isfinite(p).all():
                            head_bad = [
                                name for name, param in model.heads[g].named_parameters()
                                if not torch.isfinite(param).all()
                            ]
                            raise FloatingPointError(
                                f"Non-finite predictions in group={g}; "
                                f"head_bad_params={head_bad}; "
                                f"h_min={h.min().item()}, h_max={h.max().item()}"
                            )


                loss, n = masked_huber_loss(
                    predictions,
                    batch["y"],
                    batch["mask"],
                    delta=delta,
                    reduction=loss_reduction,
                    group_weights=group_loss_weights,
                )
            if train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total += float(loss.detach().cpu()) * max(1, int(n))
            denom += max(1, int(n))
    return total / max(1, denom)


@dataclass
class TrainResult:
    stage: str
    run_dir: Path
    checkpoint_path: Path
    history: pd.DataFrame
    summary: pd.DataFrame
    target_metrics: pd.DataFrame


def evaluate_stats_model(
    model: StatsPredictionModel,
    loader: DataLoader,
    *,
    target_stats: pd.DataFrame,
    cols_by_group: dict[str, list[str]],
    split_name: str,
    device: str | torch.device,
    amp: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    stats_idx = target_stats.set_index("target")
    target_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    accum: dict[str, dict[str, list[float]]] = {}

    use_amp = bool(amp) and torch.device(device).type == "cuda"
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval {split_name}", leave=False):
            with torch.amp.autocast("cuda", enabled=use_amp):
                if "h" in batch:
                    h = batch["h"].to(device=device, non_blocking=True)
                    preds = model.forward_from_features(h)
                else:
                    x = batch["x"].to(device=device, non_blocking=True)
                    seg = batch["seg"].to(device=device, non_blocking=True)
                    preds = model(x, seg)
            for group, cols in cols_by_group.items():
                pred_norm = preds[group].detach().cpu().numpy()
                y_norm = batch["y"][group].detach().cpu().numpy()
                valid = batch["mask"][group].detach().cpu().numpy().astype(bool)
                mean = stats_idx.loc[cols, "mean"].to_numpy(dtype=np.float32)
                std = stats_idx.loc[cols, "std"].to_numpy(dtype=np.float32)
                pred = pred_norm * std[None, :] + mean[None, :]
                truth = y_norm * std[None, :] + mean[None, :]
                for j, target in enumerate(cols):
                    mask_j = valid[:, j]
                    if not mask_j.any():
                        continue
                    err = pred[mask_j, j] - truth[mask_j, j]
                    denom = np.maximum(np.abs(truth[mask_j, j]), 1e-6)
                    bucket = accum.setdefault(target, {"err": [], "abs_pct": [], "norm_abs": []})
                    bucket["err"].extend(float(v) for v in err)
                    bucket["abs_pct"].extend(float(v) for v in np.abs(err) / denom * 100.0)
                    bucket["norm_abs"].extend(float(v) for v in np.abs(pred_norm[mask_j, j] - y_norm[mask_j, j]))
                per_sample_abs = np.abs(pred_norm - y_norm)
                for i, sample_id in enumerate(batch["sample_id"]):
                    if valid[i].any():
                        sample_rows.append(
                            {
                                "split": split_name,
                                "sample_id": sample_id,
                                "group": group,
                                "normalized_mae": float(per_sample_abs[i][valid[i]].mean()),
                                "n_targets": int(valid[i].sum()),
                            }
                        )

    for target, values in accum.items():
        err = np.asarray(values["err"], dtype=np.float64)
        abs_pct = np.asarray(values["abs_pct"], dtype=np.float64)
        norm_abs = np.asarray(values["norm_abs"], dtype=np.float64)
        target_rows.append(
            {
                "split": split_name,
                "target": target,
                "group": group_for_target(target),
                "n": int(err.size),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "bias": float(np.mean(err)),
                "median_abs_percent_error": float(np.median(abs_pct)),
                "normalized_mae": float(np.mean(norm_abs)),
            }
        )

    return pd.DataFrame(sample_rows), pd.DataFrame(target_rows)


def checkpoint_payload(
    model: StatsPredictionModel,
    *,
    epoch: int,
    best_val_loss: float,
    target_stats: pd.DataFrame,
    cols_by_group: dict[str, list[str]],
    config: dict[str, object],
    history: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "model_state": model.state_dict(),
        "target_stats": target_stats.to_dict("records"),
        "columns_by_group": cols_by_group,
        "label_ids": list(model.label_ids),
        "pool_features": list(model.pool_features),
        "pool_stat_names": list(model.pool_stat_names),
        "label_size_feature_names": list(model.label_size_feature_names),
        "feature_schema": str(model.feature_schema),
        "input_dim": int(model.input_dim),
        "config": config,
        "history": history,
    }


def _load_stats_training_checkpoint(
    model: StatsPredictionModel,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
    load_encoder: bool = True,
    checkpoint: dict | None = None,
) -> dict:
    ckpt = checkpoint if checkpoint is not None else torch.load(Path(checkpoint_path), map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Expected stats training checkpoint dict: {checkpoint_path}")
    state = strip_compile_prefix(ckpt["model_state"])
    if not bool(load_encoder):
        state = {k: v for k, v in state.items() if not k.startswith("encoder.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load_stats_training_checkpoint] missing keys: {len(missing)}")
    if unexpected:
        print(f"[load_stats_training_checkpoint] unexpected keys: {len(unexpected)}")
    return ckpt


def train_stats_stage(
    *,
    stage: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_matrix: pd.DataFrame,
    target_stats: pd.DataFrame,
    cols_by_group: dict[str, list[str]],
    segmentation_checkpoint: str | Path,
    out_dir: str | Path,
    init_stats_checkpoint: str | Path | None = None,
    load_init_encoder: bool = True,
    label_ids: Iterable[int] | None = None,
    pool_features: tuple[str, ...] = ("enc2", "enc3", "enc4", "z"),
    hidden: int = 256,
    dropout: float = 0.1,
    freeze_encoder: bool = True,
    epochs: int = 5,
    batch_size: int = 1,
    num_workers: int = 0,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
    amp: bool = False,
    cache_features: bool = False,
    feature_cache_dir: str | Path | None = None,
    feature_batch_size: int | None = None,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    delta: float = 1.0,
    loss_reduction: str = "scalar_mean",
    group_loss_weights: dict[str, float] | None = None,
    device: str | torch.device = "cpu",
) -> TrainResult:
    out_dir = Path(out_dir)
    run_dir = out_dir / f"{stage}_{time.strftime('%Y%m%d_%H%M%S')}"
    suffix = 1
    while run_dir.exists():
        run_dir = out_dir / f"{stage}_{time.strftime('%Y%m%d_%H%M%S')}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = run_dir / "stats_model_best.pt"

    out_dims = {group: len(cols) for group, cols in cols_by_group.items()}
    model, seg_ckpt = build_stats_model(
        segmentation_checkpoint=segmentation_checkpoint,
        out_dims=out_dims,
        label_ids=label_ids,
        pool_features=pool_features,
        hidden=hidden,
        dropout=dropout,
        device=device,
    )
    if init_stats_checkpoint is not None:
        _load_stats_training_checkpoint(model, init_stats_checkpoint, device=device, load_encoder=bool(load_init_encoder))
    model.freeze_encoder(bool(freeze_encoder))

    feature_cache_path = None
    feature_payload = None
    if bool(cache_features):
        if not bool(freeze_encoder):
            print("[stats] cache_features=True ignored because freeze_encoder=False")
        else:
            cache_root = Path(feature_cache_dir) if feature_cache_dir is not None else out_dir / "pooled_feature_cache"
            cache_samples = pd.concat([train_df, val_df, test_df], ignore_index=True).drop_duplicates("sample_id")
            feature_cache_path = pooled_feature_cache_path(
                cache_dir=cache_root,
                stage=stage,
                segmentation_checkpoint=segmentation_checkpoint,
                sample_ids=cache_samples["sample_id"].astype(str).tolist(),
                pool_features=pool_features,
                label_ids=model.label_ids,
                feature_schema=model.feature_schema,
            )
            feature_payload = load_or_build_pooled_feature_cache(
                model,
                cache_samples,
                cache_path=feature_cache_path,
                target_matrix=target_matrix,
                target_stats=target_stats,
                cols_by_group=cols_by_group,
                batch_size=int(feature_batch_size or batch_size),
                num_workers=num_workers,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
                amp=amp,
                device=device,
            )

    if feature_payload is not None:
        train_loader = make_stats_feature_loader(
            train_df,
            target_matrix,
            target_stats,
            cols_by_group,
            feature_payload=feature_payload,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = make_stats_feature_loader(
            val_df,
            target_matrix,
            target_stats,
            cols_by_group,
            feature_payload=feature_payload,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
        test_loader = make_stats_feature_loader(
            test_df,
            target_matrix,
            target_stats,
            cols_by_group,
            feature_payload=feature_payload,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )
    else:
        train_loader = make_stats_loader(
            train_df,
            target_matrix,
            target_stats,
            cols_by_group,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
        val_loader = make_stats_loader(
            val_df,
            target_matrix,
            target_stats,
            cols_by_group,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
        test_loader = make_stats_loader(
            test_df,
            target_matrix,
            target_stats,
            cols_by_group,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(epochs)))
    use_amp = bool(amp) and torch.device(device).type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = {
        "stage": stage,
        "segmentation_checkpoint": str(Path(segmentation_checkpoint).resolve()),
        "init_stats_checkpoint": None if init_stats_checkpoint is None else str(Path(init_stats_checkpoint).resolve()),
        "load_init_encoder": bool(load_init_encoder),
        "freeze_encoder": bool(freeze_encoder),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "persistent_workers": bool(persistent_workers) if persistent_workers is not None else None,
        "prefetch_factor": None if prefetch_factor is None else int(prefetch_factor),
        "amp": bool(use_amp),
        "cache_features": feature_payload is not None,
        "feature_cache_path": None if feature_cache_path is None else str(Path(feature_cache_path).resolve()),
        "feature_batch_size": None if feature_batch_size is None else int(feature_batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "delta": float(delta),
        "loss_reduction": str(loss_reduction),
        "group_loss_weights": group_loss_weights,
        "pool_features": list(pool_features),
        "pool_stat_names": list(model.pool_stat_names),
        "label_size_feature_names": list(model.label_size_feature_names),
        "feature_schema": str(model.feature_schema),
        "hidden": int(hidden),
        "dropout": float(dropout),
        "segmentation_epoch": seg_ckpt.get("epoch"),
    }
    save_json(run_dir / "config.json", config)
    train_df.to_csv(run_dir / "train_samples.csv", index=False)
    val_df.to_csv(run_dir / "val_samples.csv", index=False)
    test_df.to_csv(run_dir / "test_samples.csv", index=False)

    history: list[dict[str, object]] = []
    best_val = float("inf")
    best_epoch = 0
    for epoch in range(1, int(epochs) + 1):
        train_loss = run_stats_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            desc=f"{stage} epoch {epoch:03d} train",
            delta=delta,
            amp=use_amp,
            scaler=scaler,
            loss_reduction=loss_reduction,
            group_loss_weights=group_loss_weights,
        )
        val_loss = run_stats_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            desc=f"{stage} epoch {epoch:03d} val",
            delta=delta,
            amp=use_amp,
            loss_reduction=loss_reduction,
            group_loss_weights=group_loss_weights,
        )
        scheduler.step()
        row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(f"[{stage}] epoch {epoch:03d} train={train_loss:.4f} val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = float(val_loss)
            best_epoch = int(epoch)
            torch.save(
                checkpoint_payload(
                    model,
                    epoch=best_epoch,
                    best_val_loss=best_val,
                    target_stats=target_stats,
                    cols_by_group=cols_by_group,
                    config=config,
                    history=history,
                ),
                checkpoint_path,
            )
            print(f"[{stage}] saved best checkpoint -> {checkpoint_path}")

    if checkpoint_path.exists():
        _load_stats_training_checkpoint(model, checkpoint_path, device=device, load_encoder=True)

    history_df = pd.DataFrame(history)
    history_df.to_csv(run_dir / "history.csv", index=False)

    sample_frames = []
    target_frames = []
    for split_name, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        sample_df, target_df = evaluate_stats_model(
            model,
            loader,
            target_stats=target_stats,
            cols_by_group=cols_by_group,
            split_name=split_name,
            device=device,
            amp=use_amp,
        )
        sample_frames.append(sample_df)
        target_frames.append(target_df)

    sample_metrics = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame()
    target_metrics = pd.concat(target_frames, ignore_index=True) if target_frames else pd.DataFrame()
    sample_metrics.to_csv(run_dir / "sample_metrics.csv", index=False)
    target_metrics.to_csv(run_dir / "target_metrics.csv", index=False)

    summary = (
        target_metrics.groupby(["split", "group"], as_index=False)
        .agg(
            n_targets=("target", "nunique"),
            n_values=("n", "sum"),
            normalized_mae=("normalized_mae", "mean"),
            median_abs_percent_error=("median_abs_percent_error", "median"),
        )
        if not target_metrics.empty
        else pd.DataFrame()
    )
    summary.insert(0, "stage", stage)
    summary.insert(1, "best_epoch", best_epoch)
    summary.insert(2, "best_val_loss", best_val)
    summary.to_csv(run_dir / "summary.csv", index=False)

    return TrainResult(
        stage=stage,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        history=history_df,
        summary=summary,
        target_metrics=target_metrics,
    )


def latest_stage_checkpoint(results: dict[str, TrainResult], stage: str) -> Path:
    if stage not in results:
        raise KeyError(f"Missing TrainResult for stage: {stage}")
    return results[stage].checkpoint_path

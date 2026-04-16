from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def training_summary_frame(training_results: dict[str, object]) -> pd.DataFrame:
    rows = []
    for target_kind, result in dict(training_results).items():
        history = getattr(result, "history", pd.DataFrame())
        metric_name = str(getattr(result, "metric_name", "metric"))
        best_val_metric = None
        if history is not None and not history.empty:
            if "val_metric" in history:
                best_val_metric = float(history["val_metric"].min())
            elif "val_ce" in history:
                best_val_metric = float(history["val_ce"].min())
        rows.append(
            {
                "target_kind": target_kind,
                "checkpoint_path": str(getattr(result, "checkpoint_path", "")),
                "trained_this_run": bool(getattr(result, "trained", False)),
                "metric_name": metric_name,
                "test_metric": getattr(result, "test_metric", getattr(result, "test_ce", None)),
                "epochs_logged": int(len(history)),
                "best_val_metric": best_val_metric,
            }
        )
    return pd.DataFrame(rows)


def training_history_frame(training_results: dict[str, object]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for target_kind, result in dict(training_results).items():
        history = getattr(result, "history", pd.DataFrame())
        if history is None or history.empty:
            continue
        frame = history.copy()
        frame["target_kind"] = target_kind
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def prediction_summary_frame(inference_result: object) -> pd.DataFrame:
    rows = []
    for target_kind, result in dict(getattr(inference_result, "prediction_results", {})).items():
        tensor = getattr(result, "tensor")
        rows.append(
            {
                "target_kind": target_kind,
                "source": getattr(result, "source", ""),
                "checkpoint_path": None if getattr(result, "checkpoint_path", None) is None else str(result.checkpoint_path),
                "load_sec": float(getattr(result, "load_sec", 0.0)),
                "predict_sec": float(getattr(result, "predict_sec", 0.0)),
                "tensor_shape": tuple(int(v) for v in tensor.shape),
            }
        )
    return pd.DataFrame(rows)


def flatten_stats_measure_errors(stats_dict: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for relpath, payload in dict(stats_dict).items():
        measures = payload.get("measures", pd.DataFrame())
        if measures is not None and not measures.empty:
            frame = measures.copy()
            frame["stats_relpath"] = relpath
            frame["kind"] = "measure"
            rows.append(frame)
        table = payload.get("table", pd.DataFrame())
        if table is not None and not table.empty:
            frame = table.copy()
            frame["stats_relpath"] = relpath
            frame["kind"] = "table"
            rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def evaluation_tables(inference_result: object) -> dict[str, pd.DataFrame]:
    evaluation = getattr(inference_result, "evaluation", None) or {}
    return {
        "volumes": evaluation.get("volumes", pd.DataFrame()),
        "surfaces": evaluation.get("surfaces", pd.DataFrame()),
        "morphometry": evaluation.get("morphometry", pd.DataFrame()),
        "stats_flat": flatten_stats_measure_errors(evaluation.get("stats", {})),
    }


def plot_training_histories(training_results: dict[str, object]):
    history = training_history_frame(training_results)
    if history.empty:
        return None
    targets = list(history["target_kind"].unique())
    fig, axes = plt.subplots(len(targets), 1, figsize=(10, max(3, 2.8 * len(targets))), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, target_kind in zip(axes, targets):
        frame = history[history["target_kind"] == target_kind]
        metric_name = str(frame["metric_name"].dropna().iloc[0]) if "metric_name" in frame and not frame["metric_name"].dropna().empty else "metric"
        train_col = "train_metric" if "train_metric" in frame else "train_ce"
        val_col = "val_metric" if "val_metric" in frame else "val_ce"
        ax.plot(frame["epoch"], frame[train_col], marker="o", label=f"train_{metric_name}")
        if val_col in frame:
            ax.plot(frame["epoch"], frame[val_col], marker="s", label=f"val_{metric_name}")
        ax.set_title(target_kind)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name.upper())
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    return fig


def plot_inference_timings(inference_result: object):
    timings = pd.DataFrame(
        [
            {"stage": stage, "seconds": seconds}
            for stage, seconds in dict(getattr(inference_result, "timings_sec", {})).items()
        ]
    )
    if timings.empty:
        return None
    timings = timings.sort_values("seconds", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(timings))))
    ax.barh(timings["stage"], timings["seconds"], color="#4063d8")
    ax.set_xlabel("Seconds")
    ax.set_title("End-to-End Inference Timing")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_volume_metrics(volumes_df: pd.DataFrame):
    if volumes_df is None or volumes_df.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(volumes_df["relpath"], volumes_df["macro_dice_nonzero"], color="#2f855a")
    axes[0].set_title("Final Volume Dice")
    axes[0].set_ylabel("Macro Dice")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(volumes_df["relpath"], volumes_df["voxel_accuracy"], color="#dd6b20")
    axes[1].set_title("Final Volume Accuracy")
    axes[1].set_ylabel("Voxel Accuracy")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_surface_metrics(surfaces_df: pd.DataFrame):
    if surfaces_df is None or surfaces_df.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(surfaces_df["relpath"], surfaces_df["symmetric_mean_mm"], color="#6b46c1")
    axes[0].set_title("Surface Distance")
    axes[0].set_ylabel("Symmetric Mean (mm)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(surfaces_df["relpath"], surfaces_df["symmetric_hausdorff_mm"], color="#c53030")
    axes[1].set_title("Surface Hausdorff")
    axes[1].set_ylabel("Hausdorff (mm)")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_morphometry_metrics(morphometry_df: pd.DataFrame):
    if morphometry_df is None or morphometry_df.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(morphometry_df["relpath"], morphometry_df["corr"], color="#3182ce")
    axes[0].set_title("Morphometry Correlation")
    axes[0].set_ylabel("Correlation")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(morphometry_df["relpath"], morphometry_df["rmse"], color="#d69e2e")
    axes[1].set_title("Morphometry RMSE")
    axes[1].set_ylabel("RMSE")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_stats_measure_errors(stats_df: pd.DataFrame, *, top_k: int = 20):
    if stats_df is None or stats_df.empty or "abs_percent_error" not in stats_df.columns:
        return None
    frame = stats_df[stats_df["kind"] == "measure"].copy()
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["abs_percent_error"])
    if frame.empty:
        return None
    frame = frame.sort_values("abs_percent_error", ascending=False).head(int(top_k))
    labels = [f"{Path(rel).name}:{field}" for rel, field in zip(frame["stats_relpath"], frame["field"], strict=False)]
    fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(frame))))
    ax.barh(labels[::-1], frame["abs_percent_error"].to_numpy()[::-1], color="#e53e3e")
    ax.set_xlabel("Absolute Percent Error")
    ax.set_title("Top Stats Measure Errors")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


__all__ = [
    "evaluation_tables",
    "flatten_stats_measure_errors",
    "plot_inference_timings",
    "plot_morphometry_metrics",
    "plot_stats_measure_errors",
    "plot_surface_metrics",
    "plot_training_histories",
    "plot_volume_metrics",
    "prediction_summary_frame",
    "training_history_frame",
    "training_summary_frame",
]

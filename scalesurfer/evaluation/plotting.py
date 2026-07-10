import matplotlib.pyplot as plt
import numpy as np

def plot_region_dice(bundle, ax=None, ylim=None):
    """Plot per-region Dice scores."""
    region_plot_df = bundle["region_plot_df"]

    if ax is None:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 24))
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    colors = plt.cm.viridis(norm(region_plot_df["dice"].values))

    ax.barh(
        region_plot_df["label"],
        region_plot_df["dice"],
        color=colors,
        edgecolor="none"
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Dice")
    ax.set_ylabel("Region")
    ax.set_title(f"Per-Region Dice")
    mean_dice = float(region_plot_df["dice"].mean()) if len(region_plot_df) else np.nan
    median_dice = float(region_plot_df["dice"].median()) if len(region_plot_df) else np.nan
    if np.isfinite(mean_dice):
        ax.axvline(mean_dice, color="black", linestyle="--", linewidth=1.2, label=f"mean={mean_dice:.3f}")
    if np.isfinite(median_dice):
        ax.axvline(median_dice, color="gray", linestyle=":", linewidth=1.2, label=f"median={median_dice:.3f}")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower right")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    #cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    #cbar.set_label("Dice")
    #plt.tight_layout()
    if ylim is None:
        plt.ylim(-1, 107)
    else:
        plt.ylim(*ylim)

def plot_tissue_dice(bundle, ax=None):
    """Tissue Dice distribution."""
    tissue_long_df = bundle["tissue_long_df"]
    order = ["CSF", "GM", "WM", "FG"]
    data = [tissue_long_df.loc[tissue_long_df["tissue"] == t, "dice_plot"].values for t in order]
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    positions = np.arange(1, len(order) + 1)
    vp = ax.violinplot(data, positions=positions, widths=0.8, showmeans=False, showmedians=False, showextrema=False)
    for body, color in zip(vp["bodies"], ["C0", "C1", "C2", "C3"]):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.55)
    ax.boxplot(
        data,
        positions=positions,
        widths=0.20,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "black", "linewidth": 1.2},
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.0},
        whiskerprops={"color": "black", "linewidth": 1.0},
        capprops={"color": "black", "linewidth": 1.0},
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(order)
    is_percent = bool(len(tissue_long_df) and str(tissue_long_df["dice_unit"].iloc[0]) == "%")
    ax.set_ylabel("Dice score (%)" if is_percent else "Dice score")
    if is_percent:
        ax.set_ylim(0, 100)
    ax.set_title(f"Tissue Dice Distribution")
    ax.grid(axis="y", alpha=0.2)

    plt.ylim(65, 100)
    plt.tight_layout()

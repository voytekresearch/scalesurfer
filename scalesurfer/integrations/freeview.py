"""FreeView command builder."""

from pathlib import Path

def build_freeview_command(
    subject_dir: str | Path,
    pred_surfaces: dict[str, Path] | None = None,
    *,
    show_orig: bool = True,
    show_aparc: bool = True,
    pred_edge_color: str = "yellow",
    gt_edge_color_white: str = "red",
    gt_edge_color_pial: str = "cyan",
    pred_curv_files: dict[str, Path] | None = None,
) -> list[str]:
    """
    Build a freeview command showing GT (red/cyan) and predicted surfaces.

    GT surfaces include their ``?h.curv`` overlay automatically if the file
    exists in the subject's ``surf/`` directory.  Pass ``pred_curv_files``
    (as returned by ``compute_curvature_files``) to give predicted surfaces the
    same green/red curvature colouring.
    """
    subject_dir = Path(subject_dir)
    surf_dir = subject_dir / "surf"
    mri_dir = subject_dir / "mri"

    cmd = ["freeview"]
    if show_orig and (mri_dir / "orig.mgz").exists():
        cmd += ["-v", str(mri_dir / "orig.mgz")]
    if show_aparc and (mri_dir / "aparc+aseg.mgz").exists():
        cmd += [str(mri_dir / "aparc+aseg.mgz")]
    cmd += ["-f"]

    gt_colors = {
        "lh.white": gt_edge_color_white,
        "rh.white": gt_edge_color_white,
        "lh.pial": gt_edge_color_pial,
        "rh.pial": gt_edge_color_pial,
    }
    for name, color in gt_colors.items():
        p = surf_dir / name
        if p.exists():
            hemi = name.split(".")[0]
            entry = f"{p}:edgecolor={color}:edgethickness=1"
            curv_p = surf_dir / f"{hemi}.curv"
            if curv_p.exists():
                entry += f":overlay={curv_p}"
            cmd.append(entry)

    if pred_surfaces:
        for name, p in pred_surfaces.items():
            if p is not None and Path(p).exists():
                entry = f"{p}:edgecolor={pred_edge_color}:edgethickness=2"
                if pred_curv_files and name in pred_curv_files:
                    curv_p = Path(pred_curv_files[name])
                    if curv_p.exists():
                        entry += f":overlay={curv_p}"
                cmd.append(entry)

    return cmd

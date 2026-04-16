"""Helpers for FreeSurfer."""
from __future__ import annotations
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


HEMIS = ("lh", "rh")
STAGES = ("annot", "derived-surfs", "ribbon", "volumes", "stats")
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
)
REQUIRED_PREDICTION_RELPATHS = (
    "mri/aseg.presurf.mgz",
    "surf/lh.white",
    "surf/rh.white",
    "surf/lh.pial",
    "surf/rh.pial",
)
OPTIONAL_PREDICTION_RELPATHS = (
    "mri/aparc+aseg.mgz",
    "mri/norm.mgz",
    "mri/brainmask.mgz",
    "mri/transforms/talairach.xfm",
    "label/lh.aparc.annot",
    "label/rh.aparc.annot",
    "label/lh.cortex.label",
    "label/rh.cortex.label",
)
CORE_OUTPUT_RELPATHS = (
    "label/lh.aparc.annot",
    "label/lh.cortex.label",
    "label/rh.aparc.annot",
    "label/rh.cortex.label",
    "label/aparc.annot.ctab",
    "surf/lh.smoothwm",
    "surf/lh.inflated",
    "surf/lh.sulc",
    "surf/lh.curv",
    "surf/lh.area",
    "surf/lh.curv.pial",
    "surf/lh.area.pial",
    "surf/lh.thickness",
    "surf/lh.area.mid",
    "surf/lh.volume",
    "surf/rh.smoothwm",
    "surf/rh.inflated",
    "surf/rh.sulc",
    "surf/rh.curv",
    "surf/rh.area",
    "surf/rh.curv.pial",
    "surf/rh.area.pial",
    "surf/rh.thickness",
    "surf/rh.area.mid",
    "surf/rh.volume",
    "mri/lh.ribbon.mgz",
    "mri/rh.ribbon.mgz",
    "mri/ribbon.mgz",
    "mri/aseg.presurf.hypos.mgz",
    "mri/aseg.mgz",
    "mri/aparc+aseg.mgz",
    "mri/wmparc.mgz",
    "stats/lh.curv.stats",
    "stats/lh.aparc.stats",
    "stats/lh.aparc.pial.stats",
    "stats/rh.curv.stats",
    "stats/rh.aparc.stats",
    "stats/rh.aparc.pial.stats",
    "stats/aseg.stats",
    "stats/wmparc.stats",
)
OPTIONAL_OUTPUT_RELPATHS = (
    "stats/brainvol.stats",
    "stats/aparc+aseg.stats",
)


@dataclass(frozen=True)
class SubjectPaths:
    subject_dir: Path
    subjects_dir: Path
    subject_id: str
    mri_dir: Path
    surf_dir: Path
    label_dir: Path
    stats_dir: Path
    scripts_dir: Path
    transforms_dir: Path

    @classmethod
    def from_subject_dir(cls, subject_dir: str | Path) -> "SubjectPaths":
        subject_dir = Path(subject_dir).expanduser().resolve()
        return cls(
            subject_dir=subject_dir,
            subjects_dir=subject_dir.parent,
            subject_id=subject_dir.name,
            mri_dir=subject_dir / "mri",
            surf_dir=subject_dir / "surf",
            label_dir=subject_dir / "label",
            stats_dir=subject_dir / "stats",
            scripts_dir=subject_dir / "scripts",
            transforms_dir=subject_dir / "mri" / "transforms",
        )


@dataclass(frozen=True)
class PredictionBundle:
    root: Path
    aseg_presurf: Path
    lh_white: Path
    rh_white: Path
    lh_pial: Path
    rh_pial: Path
    aparc_aseg: Path | None
    norm: Path | None
    brainmask: Path | None
    talairach_xfm: Path | None
    lh_aparc_annot: Path | None
    rh_aparc_annot: Path | None
    lh_cortex_label: Path | None
    rh_cortex_label: Path | None

    @classmethod
    def from_root(cls, root: str | Path) -> "PredictionBundle":
        root = Path(root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Predictions directory does not exist: {root}")

        def _req(rel: str) -> Path:
            path = root / rel
            if not path.exists():
                raise FileNotFoundError(f"Required prediction is missing: {path}")
            return path

        def _opt(rel: str) -> Path | None:
            path = root / rel
            return path if path.exists() else None

        return cls(
            root=root,
            aseg_presurf=_req(REQUIRED_PREDICTION_RELPATHS[0]),
            lh_white=_req(REQUIRED_PREDICTION_RELPATHS[1]),
            rh_white=_req(REQUIRED_PREDICTION_RELPATHS[2]),
            lh_pial=_req(REQUIRED_PREDICTION_RELPATHS[3]),
            rh_pial=_req(REQUIRED_PREDICTION_RELPATHS[4]),
            aparc_aseg=_opt("mri/aparc+aseg.mgz"),
            norm=_opt("mri/norm.mgz"),
            brainmask=_opt("mri/brainmask.mgz"),
            talairach_xfm=_opt("mri/transforms/talairach.xfm"),
            lh_aparc_annot=_opt("label/lh.aparc.annot"),
            rh_aparc_annot=_opt("label/rh.aparc.annot"),
            lh_cortex_label=_opt("label/lh.cortex.label"),
            rh_cortex_label=_opt("label/rh.cortex.label"),
        )


@dataclass(frozen=True)
class TailConfig:
    subject: SubjectPaths
    bundle: PredictionBundle
    freesurfer_home: Path
    fs_license: Path | None
    threads: int
    link_mode: str
    dry_run: bool
    force: bool
    run_autorecon1: bool
    input_t1: Path | None
    stages: tuple[str, ...]
    annot_ctab: Path
    aparc_projdist_mm: float
    brainvol_stats: bool
    aparc_aseg_stats: bool


def required_prediction_relpaths() -> tuple[str, ...]:
    return REQUIRED_PREDICTION_RELPATHS


def optional_prediction_relpaths() -> tuple[str, ...]:
    return OPTIONAL_PREDICTION_RELPATHS


def final_output_relpaths(
    *,
    include_brainvol_stats: bool = False,
    include_aparc_aseg_stats: bool = False,
) -> tuple[str, ...]:
    outputs = list(CORE_OUTPUT_RELPATHS)
    if include_brainvol_stats:
        outputs.append("stats/brainvol.stats")
    if include_aparc_aseg_stats:
        outputs.append("stats/aparc+aseg.stats")
    return tuple(outputs)


class CommandRunner:
    def __init__(self, config: TailConfig) -> None:
        self.config = config
        self.log_path = config.subject.scripts_dir / "predicted.log"
        self.cmd_path = config.subject.scripts_dir / "predicted.cmd"

    def _script_prelude(self, cwd: Path) -> list[str]:
        cfg = self.config
        lines = [
            f"export FREESURFER_HOME={shlex.quote(str(cfg.freesurfer_home))}",
            f"source {shlex.quote(str(cfg.freesurfer_home / 'SetUpFreeSurfer.sh'))}",
            "setup_rc=$?",
            'if [ "$setup_rc" -ne 0 ]; then exit "$setup_rc"; fi',
            f"export SUBJECTS_DIR={shlex.quote(str(cfg.subject.subjects_dir))}",
        ]
        if cfg.fs_license is not None:
            lines.append(f"export FS_LICENSE={shlex.quote(str(cfg.fs_license))}")
        for env_var in THREAD_ENV_VARS:
            lines.append(f"export {env_var}={int(cfg.threads)}")
        lines.append(f"cd {shlex.quote(str(cwd))}")
        lines.append("cd_rc=$?")
        lines.append('if [ "$cd_rc" -ne 0 ]; then exit "$cd_rc"; fi')
        return lines

    def maybe_run(
        self,
        args: Iterable[str],
        *,
        cwd: Path,
        outputs: Iterable[Path] | None = None,
        description: str,
    ) -> None:
        outputs = [Path(x) for x in (outputs or ())]
        if outputs and not self.config.force and all(path.exists() for path in outputs):
            print(f"[skip] {description}", file=sys.stderr)
            return

        argv = [str(x) for x in args]
        command_str = shlex.join(argv)
        print(f"[run] {description}", file=sys.stderr)
        print(command_str, file=sys.stderr)

        if self.config.dry_run:
            return

        self.config.subject.scripts_dir.mkdir(parents=True, exist_ok=True)
        with self.cmd_path.open("a", encoding="utf-8") as cmd_f:
            cmd_f.write(command_str + "\n")

        script = "\n".join([*self._script_prelude(cwd), command_str])
        with self.log_path.open("a", encoding="utf-8") as log_f:
            log_f.write(f"\n# {description} @ {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_f.write(command_str + "\n")
            log_f.flush()
            proc = subprocess.run(
                ["bash", "-lc", script],
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if proc.returncode != 0:
            raise RuntimeError(
                f"{description} failed with exit code {proc.returncode}. "
                f"See {self.log_path}."
            )


def resolve_freesurfer_home(value: str | Path | None) -> Path:
    candidate = value or os.environ.get("FREESURFER_HOME")
    if not candidate:
        raise FileNotFoundError("Set --freesurfer-home or FREESURFER_HOME.")
    path = Path(candidate).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"FREESURFER_HOME does not exist: {path}")
    if not (path / "SetUpFreeSurfer.sh").exists():
        raise FileNotFoundError(f"Missing SetUpFreeSurfer.sh under {path}")
    return path


def resolve_fs_license_path(
    fs_license: str | Path | None,
    *,
    freesurfer_home: Path,
    allow_missing: bool,
) -> Path | None:
    candidates: list[Path] = []
    if fs_license is not None:
        candidates.append(Path(fs_license).expanduser())

    env_license = os.environ.get("FS_LICENSE")
    if env_license:
        candidates.append(Path(env_license).expanduser())

    home = Path.home()
    candidates.extend(
        [
            home / "license.txt",
            home / ".license",
            home / "Downloads" / "license.txt",
            home / "Documents" / "license.txt",
            freesurfer_home / "license.txt",
            freesurfer_home / ".license",
        ]
    )

    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists() and resolved.is_file():
            return resolved

    if allow_missing:
        return None
    raise FileNotFoundError("No FreeSurfer license file found. Set --fs-license or FS_LICENSE.")


def ensure_subject_layout(paths: SubjectPaths) -> None:
    for path in (
        paths.subject_dir,
        paths.mri_dir,
        paths.surf_dir,
        paths.label_dir,
        paths.stats_dir,
        paths.scripts_dir,
        paths.transforms_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)


def install_file(src: Path, dst: Path, *, link_mode: str, force: bool) -> None:
    src_resolved = src.expanduser().resolve()
    dst_resolved = dst.expanduser().resolve(strict=False)
    if src_resolved == dst_resolved:
        return

    if dst.exists() or dst.is_symlink():
        if not force:
            return
        _remove_existing(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)

    if link_mode == "symlink":
        dst.symlink_to(src)
    elif link_mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported link mode: {link_mode}")


def maybe_symlink(target: Path, link_path: Path, *, force: bool) -> None:
    if link_path.exists() or link_path.is_symlink():
        if not force:
            return
        _remove_existing(link_path)
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target.name)


def _normalized_path(path: Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _prediction_install_targets(config: TailConfig) -> set[Path]:
    subject = config.subject
    bundle = config.bundle

    paths = {
        subject.mri_dir / "aseg.presurf.mgz",
        subject.surf_dir / "lh.white",
        subject.surf_dir / "rh.white",
        subject.surf_dir / "lh.pial",
        subject.surf_dir / "rh.pial",
        subject.surf_dir / "lh.pial.T1",
        subject.surf_dir / "rh.pial.T1",
    }

    optional_targets = [
        (bundle.aparc_aseg, subject.mri_dir / "aparc+aseg.mgz"),
        (bundle.norm, subject.mri_dir / "norm.mgz"),
        (bundle.brainmask, subject.mri_dir / "brainmask.mgz"),
        (bundle.talairach_xfm, subject.transforms_dir / "talairach.xfm"),
        (bundle.lh_aparc_annot, subject.label_dir / "lh.aparc.annot"),
        (bundle.rh_aparc_annot, subject.label_dir / "rh.aparc.annot"),
        (bundle.lh_cortex_label, subject.label_dir / "lh.cortex.label"),
        (bundle.rh_cortex_label, subject.label_dir / "rh.cortex.label"),
    ]
    for src, dst in optional_targets:
        if src is not None:
            paths.add(dst)
    return {_normalized_path(path) for path in paths}


def _autorecon1_outputs(config: TailConfig) -> set[Path]:
    if not config.run_autorecon1:
        return set()
    subject = config.subject
    return {
        _normalized_path(subject.mri_dir / "orig" / "001.mgz"),
        _normalized_path(subject.mri_dir / "rawavg.mgz"),
        _normalized_path(subject.mri_dir / "orig.mgz"),
    }


def _stage_outputs(config: TailConfig, stage: str) -> set[Path]:
    subject = config.subject
    if stage == "annot":
        return {
            _normalized_path(subject.label_dir / f"{hemi}.aparc.annot") for hemi in HEMIS
        } | {
            _normalized_path(subject.label_dir / f"{hemi}.cortex.label") for hemi in HEMIS
        } | {
            _normalized_path(subject.surf_dir / f"{hemi}.aparc.bootstrap.mgh") for hemi in HEMIS
        }
    if stage == "derived-surfs":
        paths: set[Path] = set()
        for hemi in HEMIS:
            for suffix in (
                "smoothwm",
                "inflated",
                "sulc",
                "curv",
                "area",
                "curv.pial",
                "area.pial",
                "thickness",
                "area.mid",
                "volume",
            ):
                paths.add(_normalized_path(subject.surf_dir / f"{hemi}.{suffix}"))
        return paths
    if stage == "ribbon":
        return {
            _normalized_path(subject.mri_dir / "lh.ribbon.mgz"),
            _normalized_path(subject.mri_dir / "rh.ribbon.mgz"),
            _normalized_path(subject.mri_dir / "ribbon.mgz"),
            _normalized_path(subject.mri_dir / "aseg.presurf.hypos.mgz"),
        }
    if stage == "volumes":
        paths = {
            _normalized_path(subject.mri_dir / "aseg.mgz"),
            _normalized_path(subject.mri_dir / "aparc+aseg.mgz"),
            _normalized_path(subject.mri_dir / "wmparc.mgz"),
        }
        if config.brainvol_stats:
            paths.add(_normalized_path(subject.stats_dir / "brainvol.stats"))
        return paths
    if stage == "stats":
        paths: set[Path] = {
            _normalized_path(subject.stats_dir / "aseg.stats"),
            _normalized_path(subject.stats_dir / "wmparc.stats"),
        }
        if config.aparc_aseg_stats:
            paths.add(_normalized_path(subject.stats_dir / "aparc+aseg.stats"))
        for hemi in HEMIS:
            paths.add(_normalized_path(subject.stats_dir / f"{hemi}.curv.stats"))
            paths.add(_normalized_path(subject.stats_dir / f"{hemi}.aparc.stats"))
            paths.add(_normalized_path(subject.stats_dir / f"{hemi}.aparc.pial.stats"))
        return paths
    raise ValueError(f"Unsupported stage: {stage}")


def _planned_available_paths(config: TailConfig, *, through_stage: str | None = None) -> set[Path]:
    if not config.dry_run:
        return set()

    available = _prediction_install_targets(config) | _autorecon1_outputs(config)
    for stage in STAGES:
        if stage not in config.stages:
            continue
        if through_stage is not None and stage == through_stage:
            break
        available |= _stage_outputs(config, stage)
    return available


def path_available(path: Path, *, planned: Iterable[Path] = ()) -> bool:
    normalized = _normalized_path(path)
    if path.exists():
        return True
    return normalized in {_normalized_path(candidate) for candidate in planned}


def maybe_run_autorecon1(config: TailConfig, runner: CommandRunner) -> None:
    if not config.run_autorecon1:
        return

    subject = config.subject
    orig_001 = subject.mri_dir / "orig" / "001.mgz"
    rawavg = subject.mri_dir / "rawavg.mgz"
    orig = subject.mri_dir / "orig.mgz"

    if not config.force and orig_001.exists() and rawavg.exists() and orig.exists():
        print("[skip] autorecon1 already present", file=sys.stderr)
        return

    if config.input_t1 is None:
        raise ValueError("--run-autorecon1 requires --input-t1")

    args = [
        "recon-all",
        "-sd",
        str(subject.subjects_dir),
        "-s",
        subject.subject_id,
    ]
    if not orig_001.exists() or config.force:
        args.extend(["-i", str(config.input_t1)])
    args.append("-autorecon1")
    if config.threads > 1:
        args.extend(["-parallel", "-openmp", str(int(config.threads))])

    runner.maybe_run(
        args,
        cwd=subject.subject_dir,
        outputs=[rawavg, orig],
        description="recon-all -autorecon1",
    )


def install_prediction_bundle(config: TailConfig) -> None:
    subject = config.subject
    bundle = config.bundle

    installs: list[tuple[Path, Path]] = [
        (bundle.aseg_presurf, subject.mri_dir / "aseg.presurf.mgz"),
        (bundle.lh_white, subject.surf_dir / "lh.white"),
        (bundle.rh_white, subject.surf_dir / "rh.white"),
        (bundle.lh_pial, subject.surf_dir / "lh.pial"),
        (bundle.rh_pial, subject.surf_dir / "rh.pial"),
    ]

    optional_installs = [
        (bundle.aparc_aseg, subject.mri_dir / "aparc+aseg.mgz"),
        (bundle.norm, subject.mri_dir / "norm.mgz"),
        (bundle.brainmask, subject.mri_dir / "brainmask.mgz"),
        (bundle.talairach_xfm, subject.transforms_dir / "talairach.xfm"),
        (bundle.lh_aparc_annot, subject.label_dir / "lh.aparc.annot"),
        (bundle.rh_aparc_annot, subject.label_dir / "rh.aparc.annot"),
        (bundle.lh_cortex_label, subject.label_dir / "lh.cortex.label"),
        (bundle.rh_cortex_label, subject.label_dir / "rh.cortex.label"),
    ]

    if config.dry_run:
        for src, dst in installs:
            print(f"[install] {src} -> {dst}", file=sys.stderr)
        for src, dst in optional_installs:
            if src is not None:
                print(f"[install] {src} -> {dst}", file=sys.stderr)
        print(
            f"[install] {subject.surf_dir / 'lh.pial'} -> {subject.surf_dir / 'lh.pial.T1'} (symlink)",
            file=sys.stderr,
        )
        print(
            f"[install] {subject.surf_dir / 'rh.pial'} -> {subject.surf_dir / 'rh.pial.T1'} (symlink)",
            file=sys.stderr,
        )
        return

    for src, dst in installs:
        install_file(src, dst, link_mode=config.link_mode, force=config.force)
    for src, dst in optional_installs:
        if src is not None:
            install_file(src, dst, link_mode=config.link_mode, force=config.force)

    maybe_symlink(subject.surf_dir / "lh.pial", subject.surf_dir / "lh.pial.T1", force=config.force)
    maybe_symlink(subject.surf_dir / "rh.pial", subject.surf_dir / "rh.pial.T1", force=config.force)


def require_paths(paths: Iterable[Path], *, message: str, planned: Iterable[Path] = ()) -> None:
    planned_paths = {_normalized_path(path) for path in planned}
    missing = [
        str(path)
        for path in paths
        if not Path(path).exists() and _normalized_path(Path(path)) not in planned_paths
    ]
    if missing:
        raise FileNotFoundError(f"{message}: " + ", ".join(missing))


def build_annotation_stage(config: TailConfig, runner: CommandRunner) -> None:
    subject = config.subject
    bundle = config.bundle
    planned = _planned_available_paths(config, through_stage="annot")
    require_paths(
        [subject.mri_dir / "orig.mgz", subject.surf_dir / "lh.white", subject.surf_dir / "rh.white"],
        message="Annotation bootstrap requires a prepared subject geometry and white surfaces",
        planned=planned,
    )

    has_aparc_aseg = bundle.aparc_aseg is not None or path_available(
        subject.mri_dir / "aparc+aseg.mgz",
        planned=planned,
    )
    has_bootstrap_annots = all(
        path_available(subject.label_dir / f"{hemi}.aparc.annot", planned=planned) for hemi in HEMIS
    )
    if not has_aparc_aseg and not has_bootstrap_annots:
        raise FileNotFoundError(
            "No predicted mri/aparc+aseg.mgz is available to bootstrap lh/rh.aparc.annot."
        )

    for hemi in HEMIS:
        annot = subject.label_dir / f"{hemi}.aparc.annot"
        surfseg = subject.surf_dir / f"{hemi}.aparc.bootstrap.mgh"
        if config.force or not path_available(annot, planned=planned):
            runner.maybe_run(
                [
                    "mri_vol2surf",
                    "--mov",
                    str(subject.mri_dir / "aparc+aseg.mgz"),
                    "--regheader",
                    subject.subject_id,
                    "--hemi",
                    hemi,
                    "--surf",
                    "white",
                    "--interp",
                    "nearest",
                    "--projdist",
                    str(float(config.aparc_projdist_mm)),
                    "--o",
                    str(surfseg),
                ],
                cwd=subject.subject_dir,
                outputs=[surfseg],
                description=f"{hemi} volume-to-surface bootstrap",
            )
            runner.maybe_run(
                [
                    "mris_seg2annot",
                    "--seg",
                    str(surfseg),
                    "--ctab",
                    str(config.annot_ctab),
                    "--s",
                    subject.subject_id,
                    "--h",
                    hemi,
                    "--o",
                    str(annot),
                ],
                cwd=subject.subject_dir,
                outputs=[annot],
                description=f"{hemi} bootstrap annotation",
            )

        cortex_label = subject.label_dir / f"{hemi}.cortex.label"
        if config.force or not path_available(cortex_label, planned=planned):
            runner.maybe_run(
                ["make_cortex_label", "--s", subject.subject_id, f"--{hemi}"],
                cwd=subject.subject_dir,
                outputs=[cortex_label],
                description=f"{hemi} cortex label",
            )


def build_derived_surface_stage(config: TailConfig, runner: CommandRunner) -> None:
    subject = config.subject
    planned = _planned_available_paths(config, through_stage="derived-surfs")
    require_paths(
        [
            subject.surf_dir / "lh.white",
            subject.surf_dir / "rh.white",
            subject.surf_dir / "lh.pial",
            subject.surf_dir / "rh.pial",
        ],
        message="Derived surfaces require predicted white and pial surfaces",
        planned=planned,
    )

    for hemi in HEMIS:
        white = subject.surf_dir / f"{hemi}.white"
        pial = subject.surf_dir / f"{hemi}.pial"
        smoothwm = subject.surf_dir / f"{hemi}.smoothwm"
        inflated = subject.surf_dir / f"{hemi}.inflated"

        runner.maybe_run(
            ["mris_smooth", "-n", "3", "-nw", str(white), str(smoothwm)],
            cwd=subject.subject_dir,
            outputs=[smoothwm],
            description=f"{hemi} smoothwm",
        )
        runner.maybe_run(
            ["mris_inflate", str(smoothwm), str(inflated)],
            cwd=subject.subject_dir,
            outputs=[inflated, subject.surf_dir / f"{hemi}.sulc"],
            description=f"{hemi} inflated and sulc",
        )
        runner.maybe_run(
            [
                "mris_place_surface",
                "--curv-map",
                str(white),
                "2",
                "10",
                str(subject.surf_dir / f"{hemi}.curv"),
            ],
            cwd=subject.subject_dir,
            outputs=[subject.surf_dir / f"{hemi}.curv"],
            description=f"{hemi} white curvature map",
        )
        runner.maybe_run(
            [
                "mris_place_surface",
                "--area-map",
                str(white),
                str(subject.surf_dir / f"{hemi}.area"),
            ],
            cwd=subject.subject_dir,
            outputs=[subject.surf_dir / f"{hemi}.area"],
            description=f"{hemi} white area map",
        )
        runner.maybe_run(
            [
                "mris_place_surface",
                "--curv-map",
                str(pial),
                "2",
                "10",
                str(subject.surf_dir / f"{hemi}.curv.pial"),
            ],
            cwd=subject.subject_dir,
            outputs=[subject.surf_dir / f"{hemi}.curv.pial"],
            description=f"{hemi} pial curvature map",
        )
        runner.maybe_run(
            [
                "mris_place_surface",
                "--area-map",
                str(pial),
                str(subject.surf_dir / f"{hemi}.area.pial"),
            ],
            cwd=subject.subject_dir,
            outputs=[subject.surf_dir / f"{hemi}.area.pial"],
            description=f"{hemi} pial area map",
        )
        runner.maybe_run(
            [
                "mris_place_surface",
                "--thickness",
                str(white),
                str(pial),
                "20",
                "5",
                str(subject.surf_dir / f"{hemi}.thickness"),
            ],
            cwd=subject.subject_dir,
            outputs=[subject.surf_dir / f"{hemi}.thickness"],
            description=f"{hemi} thickness",
        )
        runner.maybe_run(
            ["vertexvol", "--s", subject.subject_id, f"--{hemi}", "--th3"],
            cwd=subject.subject_dir,
            outputs=[
                subject.surf_dir / f"{hemi}.area.mid",
                subject.surf_dir / f"{hemi}.volume",
            ],
            description=f"{hemi} vertex volume",
        )


def build_ribbon_stage(config: TailConfig, runner: CommandRunner) -> None:
    subject = config.subject
    planned = _planned_available_paths(config, through_stage="ribbon")
    require_paths(
        [
            subject.mri_dir / "aseg.presurf.mgz",
            subject.surf_dir / "lh.white",
            subject.surf_dir / "rh.white",
            subject.surf_dir / "lh.pial",
            subject.surf_dir / "rh.pial",
        ],
        message="Ribbon stage requires aseg.presurf and final surfaces",
        planned=planned,
    )

    runner.maybe_run(
        [
            "mris_volmask",
            "--aseg_name",
            "aseg.presurf",
            "--label_left_white",
            "2",
            "--label_left_ribbon",
            "3",
            "--label_right_white",
            "41",
            "--label_right_ribbon",
            "42",
            "--save_ribbon",
            "--parallel",
            subject.subject_id,
        ],
        cwd=subject.subject_dir,
        outputs=[
            subject.mri_dir / "lh.ribbon.mgz",
            subject.mri_dir / "rh.ribbon.mgz",
            subject.mri_dir / "ribbon.mgz",
        ],
        description="cortical ribbon volume mask",
    )
    runner.maybe_run(
        [
            "mri_relabel_hypointensities",
            str(subject.mri_dir / "aseg.presurf.mgz"),
            str(subject.surf_dir),
            str(subject.mri_dir / "aseg.presurf.hypos.mgz"),
        ],
        cwd=subject.subject_dir,
        outputs=[subject.mri_dir / "aseg.presurf.hypos.mgz"],
        description="relabel hypointensities",
    )


def build_volume_stage(config: TailConfig, runner: CommandRunner) -> None:
    subject = config.subject
    planned = _planned_available_paths(config, through_stage="volumes")
    require_paths(
        [
            subject.mri_dir / "aseg.presurf.hypos.mgz",
            subject.mri_dir / "ribbon.mgz",
            subject.label_dir / "lh.cortex.label",
            subject.label_dir / "rh.cortex.label",
            subject.label_dir / "lh.aparc.annot",
            subject.label_dir / "rh.aparc.annot",
            subject.surf_dir / "lh.white",
            subject.surf_dir / "rh.white",
            subject.surf_dir / "lh.pial",
            subject.surf_dir / "rh.pial",
        ],
        message="Volume stage requires ribbon, labels, annotations, and final surfaces",
        planned=planned,
    )

    runner.maybe_run(
        [
            "mri_surf2volseg",
            "--o",
            str(subject.mri_dir / "aseg.mgz"),
            "--i",
            str(subject.mri_dir / "aseg.presurf.hypos.mgz"),
            "--fix-presurf-with-ribbon",
            str(subject.mri_dir / "ribbon.mgz"),
            "--threads",
            str(int(config.threads)),
            "--lh-cortex-mask",
            str(subject.label_dir / "lh.cortex.label"),
            "--lh-white",
            str(subject.surf_dir / "lh.white"),
            "--lh-pial",
            str(subject.surf_dir / "lh.pial"),
            "--rh-cortex-mask",
            str(subject.label_dir / "rh.cortex.label"),
            "--rh-white",
            str(subject.surf_dir / "rh.white"),
            "--rh-pial",
            str(subject.surf_dir / "rh.pial"),
        ],
        cwd=subject.subject_dir,
        outputs=[subject.mri_dir / "aseg.mgz"],
        description="final aseg volume",
    )

    if config.brainvol_stats:
        runner.maybe_run(
            ["mri_brainvol_stats", "--subject", subject.subject_id],
            cwd=subject.subject_dir,
            outputs=[subject.stats_dir / "brainvol.stats"],
            description="brain volume cache stats",
        )

    runner.maybe_run(
        [
            "mri_surf2volseg",
            "--o",
            str(subject.mri_dir / "aparc+aseg.mgz"),
            "--label-cortex",
            "--i",
            str(subject.mri_dir / "aseg.mgz"),
            "--threads",
            str(int(config.threads)),
            "--lh-annot",
            str(subject.label_dir / "lh.aparc.annot"),
            "1000",
            "--lh-cortex-mask",
            str(subject.label_dir / "lh.cortex.label"),
            "--lh-white",
            str(subject.surf_dir / "lh.white"),
            "--lh-pial",
            str(subject.surf_dir / "lh.pial"),
            "--rh-annot",
            str(subject.label_dir / "rh.aparc.annot"),
            "2000",
            "--rh-cortex-mask",
            str(subject.label_dir / "rh.cortex.label"),
            "--rh-white",
            str(subject.surf_dir / "rh.white"),
            "--rh-pial",
            str(subject.surf_dir / "rh.pial"),
        ],
        cwd=subject.subject_dir,
        outputs=[subject.mri_dir / "aparc+aseg.mgz"],
        description="regenerated aparc+aseg volume",
    )

    runner.maybe_run(
        [
            "mri_surf2volseg",
            "--o",
            str(subject.mri_dir / "wmparc.mgz"),
            "--label-wm",
            "--i",
            str(subject.mri_dir / "aparc+aseg.mgz"),
            "--threads",
            str(int(config.threads)),
            "--lh-annot",
            str(subject.label_dir / "lh.aparc.annot"),
            "3000",
            "--lh-cortex-mask",
            str(subject.label_dir / "lh.cortex.label"),
            "--lh-white",
            str(subject.surf_dir / "lh.white"),
            "--lh-pial",
            str(subject.surf_dir / "lh.pial"),
            "--rh-annot",
            str(subject.label_dir / "rh.aparc.annot"),
            "4000",
            "--rh-cortex-mask",
            str(subject.label_dir / "rh.cortex.label"),
            "--rh-white",
            str(subject.surf_dir / "rh.white"),
            "--rh-pial",
            str(subject.surf_dir / "rh.pial"),
        ],
        cwd=subject.subject_dir,
        outputs=[subject.mri_dir / "wmparc.mgz"],
        description="wmparc volume",
    )


def _aseg_stats_command(config: TailConfig, *, planned: Iterable[Path] = ()) -> list[str]:
    subject = config.subject
    cmd = [
        "mri_segstats",
        "--seg",
        str(subject.mri_dir / "aseg.mgz"),
        "--sum",
        str(subject.stats_dir / "aseg.stats"),
        "--pv",
        str(subject.mri_dir / "norm.mgz"),
        "--empty",
        "--brainmask",
        str(subject.mri_dir / "brainmask.mgz"),
        "--brain-vol-from-seg",
        "--excludeid",
        "0",
        "--excl-ctxgmwm",
        "--subcortgray",
        "--in",
        str(subject.mri_dir / "norm.mgz"),
        "--in-intensity-name",
        "norm",
        "--in-intensity-units",
        "MR",
        "--ctab",
        str(config.freesurfer_home / "ASegStatsLUT.txt"),
        "--subject",
        subject.subject_id,
    ]
    if path_available(subject.mri_dir / "ribbon.mgz", planned=planned):
        cmd.append("--supratent")
    if path_available(subject.transforms_dir / "talairach.xfm", planned=planned):
        cmd.append("--etiv")
    if path_available(subject.surf_dir / "lh.white", planned=planned) and path_available(
        subject.surf_dir / "rh.white",
        planned=planned,
    ):
        cmd.append("--surf-wm-vol")
        if path_available(subject.surf_dir / "lh.pial", planned=planned) and path_available(
            subject.surf_dir / "rh.pial",
            planned=planned,
        ):
            cmd.extend(["--surf-ctx-vol", "--totalgray"])
    return cmd


def _wmparc_stats_command(config: TailConfig, *, planned: Iterable[Path] = ()) -> list[str]:
    subject = config.subject
    cmd = [
        "mri_segstats",
        "--seg",
        str(subject.mri_dir / "wmparc.mgz"),
        "--sum",
        str(subject.stats_dir / "wmparc.stats"),
        "--pv",
        str(subject.mri_dir / "norm.mgz"),
        "--excludeid",
        "0",
        "--brainmask",
        str(subject.mri_dir / "brainmask.mgz"),
        "--in",
        str(subject.mri_dir / "norm.mgz"),
        "--in-intensity-name",
        "norm",
        "--in-intensity-units",
        "MR",
        "--subject",
        subject.subject_id,
        "--surf-wm-vol",
        "--ctab",
        str(config.freesurfer_home / "WMParcStatsLUT.txt"),
    ]
    if path_available(subject.transforms_dir / "talairach.xfm", planned=planned):
        cmd.append("--etiv")
    return cmd


def _aparc_aseg_stats_command(config: TailConfig, *, planned: Iterable[Path] = ()) -> list[str]:
    subject = config.subject
    cmd = [
        "mri_segstats",
        "--seg",
        str(subject.mri_dir / "aparc+aseg.mgz"),
        "--sum",
        str(subject.stats_dir / "aparc+aseg.stats"),
        "--pv",
        str(subject.mri_dir / "norm.mgz"),
        "--excludeid",
        "0",
        "--ctab-default",
        "--empty",
        "--brain-vol-from-seg",
        "--brainmask",
        str(subject.mri_dir / "brainmask.mgz"),
        "--in",
        str(subject.mri_dir / "norm.mgz"),
        "--in-intensity-name",
        "norm",
        "--in-intensity-units",
        "MR",
        "--subject",
        subject.subject_id,
    ]
    if path_available(subject.transforms_dir / "talairach.xfm", planned=planned):
        cmd.append("--etiv")
    return cmd


def ensure_aparc_stats_ctab(config: TailConfig) -> Path:
    subject_ctab = config.subject.label_dir / "aparc.annot.ctab"
    if config.dry_run:
        print(f"[install] {config.annot_ctab} -> {subject_ctab}", file=sys.stderr)
        return subject_ctab
    if subject_ctab.exists() and not config.force:
        return subject_ctab
    subject_ctab.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.annot_ctab, subject_ctab)
    return subject_ctab


def _aparc_stats_ctab(config: TailConfig, *, planned: Iterable[Path] = ()) -> Path:
    subject_ctab = config.subject.label_dir / "aparc.annot.ctab"
    if path_available(subject_ctab, planned=planned):
        return subject_ctab
    return config.annot_ctab


def build_stats_stage(config: TailConfig, runner: CommandRunner) -> None:
    subject = config.subject
    planned = _planned_available_paths(config, through_stage="stats")
    ensure_aparc_stats_ctab(config)
    aparc_ctab = _aparc_stats_ctab(
        config,
        planned=planned | {_normalized_path(subject.label_dir / "aparc.annot.ctab")},
    )
    require_paths(
        [
            subject.mri_dir / "norm.mgz",
            subject.mri_dir / "brainmask.mgz",
            subject.mri_dir / "aseg.mgz",
            subject.mri_dir / "wmparc.mgz",
            subject.surf_dir / "lh.white",
            subject.surf_dir / "rh.white",
            subject.surf_dir / "lh.pial",
            subject.surf_dir / "rh.pial",
            subject.surf_dir / "lh.sulc",
            subject.surf_dir / "rh.sulc",
            subject.surf_dir / "lh.curv",
            subject.surf_dir / "rh.curv",
            subject.label_dir / "lh.aparc.annot",
            subject.label_dir / "rh.aparc.annot",
            subject.label_dir / "lh.cortex.label",
            subject.label_dir / "rh.cortex.label",
        ],
        message="Stats stage requires norm/brainmask, final volumes, and surface derivatives",
        planned=planned,
    )

    for hemi in HEMIS:
        runner.maybe_run(
            [
                "mris_curvature_stats",
                "-m",
                "--writeCurvatureFiles",
                "-G",
                "-o",
                str(subject.stats_dir / f"{hemi}.curv.stats"),
                "-F",
                "smoothwm",
                subject.subject_id,
                hemi,
                "curv",
                "sulc",
            ],
            cwd=subject.subject_dir,
            outputs=[subject.stats_dir / f"{hemi}.curv.stats"],
            description=f"{hemi} curvature stats",
        )

        runner.maybe_run(
            [
                "mris_anatomical_stats",
                "-mgz",
                "-cortex",
                str(subject.label_dir / f"{hemi}.cortex.label"),
                "-f",
                str(subject.stats_dir / f"{hemi}.aparc.stats"),
                "-b",
                "-a",
                str(subject.label_dir / f"{hemi}.aparc.annot"),
                "-c",
                str(aparc_ctab),
                subject.subject_id,
                hemi,
                "white",
            ],
            cwd=subject.subject_dir,
            outputs=[subject.stats_dir / f"{hemi}.aparc.stats"],
            description=f"{hemi} aparc white stats",
        )
        runner.maybe_run(
            [
                "mris_anatomical_stats",
                "-mgz",
                "-cortex",
                str(subject.label_dir / f"{hemi}.cortex.label"),
                "-f",
                str(subject.stats_dir / f"{hemi}.aparc.pial.stats"),
                "-b",
                "-a",
                str(subject.label_dir / f"{hemi}.aparc.annot"),
                "-c",
                str(aparc_ctab),
                subject.subject_id,
                hemi,
                "pial",
            ],
            cwd=subject.subject_dir,
            outputs=[subject.stats_dir / f"{hemi}.aparc.pial.stats"],
            description=f"{hemi} aparc pial stats",
        )

    runner.maybe_run(
        _aseg_stats_command(config, planned=planned),
        cwd=subject.subject_dir,
        outputs=[subject.stats_dir / "aseg.stats"],
        description="aseg stats",
    )
    runner.maybe_run(
        _wmparc_stats_command(config, planned=planned),
        cwd=subject.subject_dir,
        outputs=[subject.stats_dir / "wmparc.stats"],
        description="wmparc stats",
    )
    if config.aparc_aseg_stats:
        runner.maybe_run(
            _aparc_aseg_stats_command(config, planned=planned),
            cwd=subject.subject_dir,
            outputs=[subject.stats_dir / "aparc+aseg.stats"],
            description="aparc+aseg stats",
        )


def build_config(
    *,
    subject_dir: str | Path,
    predictions_dir: str | Path,
    freesurfer_home: str | Path | None = None,
    fs_license: str | Path | None = None,
    threads: int | None = None,
    link_mode: str = "symlink",
    dry_run: bool = False,
    force: bool = False,
    run_autorecon1: bool = False,
    input_t1: str | Path | None = None,
    stages: Iterable[str] | None = None,
    aparc_projdist_mm: float = 1.0,
    brainvol_stats: bool = False,
    aparc_aseg_stats: bool = False,
) -> TailConfig:
    subject = SubjectPaths.from_subject_dir(subject_dir)
    bundle = PredictionBundle.from_root(predictions_dir)
    fs_home = resolve_freesurfer_home(freesurfer_home)
    stage_values = tuple(stages or ("all",))
    stage_tuple = tuple(STAGES if "all" in stage_values else stage_values)
    annot_ctab = fs_home / "FreeSurferColorLUT.txt"
    if not annot_ctab.exists():
        raise FileNotFoundError(f"Could not find annotation color table: {annot_ctab}")

    resolved_input_t1 = None
    if input_t1 is not None:
        resolved_input_t1 = Path(input_t1).expanduser().resolve()

    default_threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))
    return TailConfig(
        subject=subject,
        bundle=bundle,
        freesurfer_home=fs_home,
        fs_license=resolve_fs_license_path(
            fs_license,
            freesurfer_home=fs_home,
            allow_missing=bool(dry_run),
        ),
        threads=max(1, int(default_threads if threads is None else threads)),
        link_mode=str(link_mode),
        dry_run=bool(dry_run),
        force=bool(force),
        run_autorecon1=bool(run_autorecon1),
        input_t1=resolved_input_t1,
        stages=stage_tuple,
        annot_ctab=annot_ctab,
        aparc_projdist_mm=float(aparc_projdist_mm),
        brainvol_stats=bool(brainvol_stats),
        aparc_aseg_stats=bool(aparc_aseg_stats),
    )


def run(config: TailConfig) -> None:
    ensure_subject_layout(config.subject)
    runner = CommandRunner(config)

    maybe_run_autorecon1(config, runner)
    install_prediction_bundle(config)

    if "annot" in config.stages:
        build_annotation_stage(config, runner)
    if "derived-surfs" in config.stages:
        build_derived_surface_stage(config, runner)
    if "ribbon" in config.stages:
        build_ribbon_stage(config, runner)
    if "volumes" in config.stages:
        build_volume_stage(config, runner)
    if "stats" in config.stages:
        build_stats_stage(config, runner)



__all__ = [
    "CORE_OUTPUT_RELPATHS",
    "CommandRunner",
    "HEMIS",
    "OPTIONAL_OUTPUT_RELPATHS",
    "OPTIONAL_PREDICTION_RELPATHS",
    "PredictionBundle",
    "REQUIRED_PREDICTION_RELPATHS",
    "SubjectPaths",
    "STAGES",
    "THREAD_ENV_VARS",
    "TailConfig",
    "build_annotation_stage",
    "build_derived_surface_stage",
    "build_ribbon_stage",
    "build_stats_stage",
    "build_config",
    "build_volume_stage",
    "ensure_subject_layout",
    "final_output_relpaths",
    "install_prediction_bundle",
    "maybe_run_autorecon1",
    "optional_prediction_relpaths",
    "required_prediction_relpaths",
    "resolve_freesurfer_home",
    "resolve_fs_license_path",
    "run",
]

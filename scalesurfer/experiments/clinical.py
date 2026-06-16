from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from tqdm.auto import tqdm


DEFAULT_C_GRID = tuple(float(x) for x in np.logspace(-2, 2, 9))
DEFAULT_SCENARIOS = (
    "features",
    "residualized_features",
)

DEFAULT_CORE_EXCLUDED_MEASURES = ("MeanCurv", "GausCurv", "FoldInd", "CurvInd")
DEFAULT_MALE_VALUES = {"m", "male", "man", "1"}
DEFAULT_FEMALE_VALUES = {"f", "female", "woman", "2"}


@dataclass
class ClinicalExperimentConfig:
    experiment_name: str
    output_dir: str | Path
    feature_source: str = "scalesurfer_stats_predicted"
    disease_context: str = ""
    positive_label_name: str = "Disease"
    negative_label_name: str = "Healthy"
    scenarios: tuple[str, ...] = DEFAULT_SCENARIOS
    primary_scenario: str = "features"
    confound_cols: tuple[str, ...] = ()
    continuous_confound_cols: tuple[str, ...] = ()
    categorical_confound_cols: tuple[str, ...] = ()
    stratify_cols: tuple[str, ...] = ()
    cv_splits: int = 5
    cv_repeats: int = 1
    inner_splits: int = 5
    tune_c: bool = True
    c_grid: tuple[float, ...] = DEFAULT_C_GRID
    class_weight: str | dict | None = "balanced"
    max_iter: int = 10_000
    n_jobs: int | None = -1
    logistic_regression_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"penalty": "l1", "solver": "liblinear"}
    )
    seed: int = 42
    test_size: float = 0.20
    top_n_features: int = 35
    progress: bool = True


@dataclass
class ClinicalExperimentResult:
    config: ClinicalExperimentConfig
    model_df: pd.DataFrame
    feature_cols: list[str]
    cv_metrics: pd.DataFrame
    cv_scores: pd.DataFrame
    oof_metrics: pd.DataFrame
    oof_predictions: pd.DataFrame
    holdout_metrics: pd.DataFrame
    coefficients: pd.DataFrame
    feature_importance: pd.DataFrame
    confound_control_summary: pd.DataFrame
    c_table: pd.DataFrame
    output_paths: dict[str, Path] = field(default_factory=dict)

    def scenario(self, name: str) -> dict[str, pd.DataFrame]:
        return {
            "cv_metrics": self.cv_metrics[self.cv_metrics["scenario"].eq(name)].copy(),
            "cv_scores": self.cv_scores[self.cv_scores["scenario"].eq(name)].copy(),
            "oof_metrics": self.oof_metrics[self.oof_metrics["scenario"].eq(name)].copy(),
            "oof_predictions": self.oof_predictions[self.oof_predictions["scenario"].eq(name)].copy(),
            "feature_importance": self.feature_importance[self.feature_importance["scenario"].eq(name)].copy(),
        }


@dataclass
class ClinicalStatsWorkflowConfig:
    """Notebook-facing configuration for the shared clinical stats workflow."""

    experiment_name: str
    output_dir: str | Path
    feature_source: str = "scalesurfer_stats_predicted"
    disease_context: str = ""
    positive_label_name: str = "Disease"
    negative_label_name: str = "Healthy"
    stats_feature_preset: str = "core_no_curv_no_wmparc"
    age_cols: tuple[str, ...] = ("age", "AGE", "Age")
    sex_cols: tuple[str, ...] = ("sex", "gender", "SEX", "GENDER")
    study_col: str | None = "dataset_id"
    scanner_cols: tuple[str, ...] = (
        "ScannerSerialNumber",
        "T1_SCANNER_MANUFACTURER",
        "T1_SCANNER_MODEL",
        "T1_FIELD_STRENGTH",
    )
    protocol_cols: tuple[str, ...] = ("T1_PROTOCOL_PHASE",)
    feature_confound_cols: tuple[str, ...] = ()
    remove_feature_confounds_from_features: bool = True
    include_study_confound: bool = True
    include_scanner_confound: bool = False
    scenarios: tuple[str, ...] = DEFAULT_SCENARIOS
    primary_scenario: str = "features"
    stratify_cols: tuple[str, ...] = ("study_id",)
    cv_splits: int = 5
    cv_repeats: int = 5
    inner_splits: int = 5
    tune_c: bool = True
    c_grid: tuple[float, ...] = DEFAULT_C_GRID
    class_weight: str | dict | None = "balanced"
    max_iter: int = 10_000
    n_jobs: int | None = -1
    logistic_regression_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"penalty": "l1", "solver": "liblinear"}
    )
    seed: int = 42
    test_size: float = 0.20
    top_n_features: int = 35
    progress: bool = True


@dataclass
class ClinicalStatsWorkflowResult:
    """Result bundle returned by the shared notebook-facing workflow."""

    workflow_config: ClinicalStatsWorkflowConfig
    experiment_config: ClinicalExperimentConfig
    result: ClinicalExperimentResult
    model_df: pd.DataFrame
    all_feature_cols: list[str]
    feature_cols: list[str]
    confound_cols: tuple[str, ...]
    continuous_confound_cols: tuple[str, ...]
    categorical_confound_cols: tuple[str, ...]
    feature_confound_cols: tuple[str, ...]
    usable_confound_terms: tuple[str, ...]
    label_col: str = "y"
    subject_col: str = "subject"

    @property
    def excluded_feature_count(self) -> int:
        return int(len(self.all_feature_cols) - len(self.feature_cols))


def clean_feature_frame(model_df: pd.DataFrame, feature_cols: Sequence[str]) -> tuple[pd.DataFrame, list[str]]:
    """Return numeric feature columns, dropping missing/non-numeric/all-missing columns."""
    cols = [col for col in feature_cols if col in model_df.columns]
    numeric_cols = [col for col in cols if pd.api.types.is_numeric_dtype(model_df[col])]
    frame = model_df[numeric_cols].replace([np.inf, -np.inf], np.nan).copy()
    all_missing = frame.columns[frame.isna().all()].tolist()
    if all_missing:
        frame = frame.drop(columns=all_missing)
    return frame, frame.columns.tolist()


def make_confounds_available(model_df: pd.DataFrame, confound_cols: Iterable[str]) -> list[str]:
    """Keep requested confounds that are present and not entirely missing."""
    out: list[str] = []
    for col in confound_cols:
        if col not in model_df.columns:
            continue
        series = model_df[col]
        if series.notna().any():
            out.append(col)
    return out


def canonicalize_clinical_covariates(
    model_df: pd.DataFrame,
    *,
    age_cols: Sequence[str] = ("age", "AGE", "Age"),
    sex_cols: Sequence[str] = ("sex", "gender", "SEX", "GENDER"),
    study_col: str | None = "dataset_id",
    scanner_cols: Sequence[str] = (
        "ScannerSerialNumber",
        "T1_SCANNER_MANUFACTURER",
        "T1_SCANNER_MODEL",
        "T1_FIELD_STRENGTH",
    ),
    protocol_cols: Sequence[str] = ("T1_PROTOCOL_PHASE",),
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Add canonical covariates used by the shared clinical-control experiments.

    The raw clinical tables often encode the same concept several ways. This
    function keeps the raw columns intact, but adds a small set of harmonized
    columns with stable names:

    - ``age_years``: first available numeric age column.
    - ``sex_canonical``: male/female/other from any sex/gender-like column.
    - ``study_id``: study, site, or dataset identifier.
    - ``scanner_id``: combined scanner/protocol string when locally available.

    These columns are meant for confound controls. They
    are intentionally conservative; label-like columns such as diagnosis subtype
    or subject record IDs are never inferred as confounds here.
    """
    out = model_df if inplace else model_df.copy()

    age = pd.Series(np.nan, index=out.index, dtype="float64")
    for col in age_cols:
        if col not in out.columns:
            continue
        candidate = pd.to_numeric(out[col], errors="coerce")
        age = age.where(age.notna(), candidate)
    out["age_years"] = age

    def _clean_sex_value(value: object) -> object:
        if pd.isna(value):
            return pd.NA
        text = str(value).strip().lower()
        if text in DEFAULT_MALE_VALUES:
            return "male"
        if text in DEFAULT_FEMALE_VALUES:
            return "female"
        if text in {"", "nan", "none", "unknown", "<missing>"}:
            return pd.NA
        return "other"

    sex = pd.Series(pd.NA, index=out.index, dtype="string")
    for col in sex_cols:
        if col not in out.columns:
            continue
        candidate = out[col].map(_clean_sex_value).astype("string")
        sex = sex.where(sex.notna(), candidate)
    out["sex_canonical"] = sex

    if study_col is not None and study_col in out.columns:
        study = out[study_col].astype("string").fillna("<missing>")
    else:
        study = pd.Series("study", index=out.index, dtype="string")
    out["study_id"] = study

    scanner_parts = []
    for col in tuple(scanner_cols) + tuple(protocol_cols):
        if col not in out.columns:
            continue
        values = out[col].astype("string").str.strip()
        values = values.mask(values.isin(["", "nan", "None", "NA", "N/A"]), pd.NA)
        scanner_parts.append(values)
    if scanner_parts:
        scanner = scanner_parts[0].copy()
        for part in scanner_parts[1:]:
            scanner = scanner.str.cat(part, sep="|", na_rep="")
            scanner = scanner.str.strip("|")
            scanner = scanner.mask(scanner.eq(""), pd.NA)
        out["scanner_id"] = scanner.astype("string")
    else:
        out["scanner_id"] = pd.Series(pd.NA, index=out.index, dtype="string")

    return out


def clean_confound_column_sets(
    model_df: pd.DataFrame,
    *,
    include_study: bool = True,
    include_scanner: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return available canonical confound columns as all/continuous/categorical tuples."""
    continuous = [col for col in ("age_years",) if col in model_df.columns and model_df[col].notna().any()]
    categorical_candidates = ["sex_canonical"]
    if include_study:
        categorical_candidates.append("study_id")
    if include_scanner:
        categorical_candidates.append("scanner_id")
    categorical = [
        col
        for col in categorical_candidates
        if col in model_df.columns and model_df[col].notna().any() and model_df[col].nunique(dropna=True) > 0
    ]
    confounds = tuple(continuous + categorical)
    return confounds, tuple(continuous), tuple(categorical)


def select_stats_feature_columns(
    feature_cols: Sequence[str],
    *,
    preset: str = "all",
    disease_context: str = "",
    excluded_measures: Sequence[str] = DEFAULT_CORE_EXCLUDED_MEASURES,
) -> list[str]:
    """
    Select a pre-specified FreeSurfer-stats feature family.

    This is deliberately rule based rather than label driven. The default
    notebook preset, ``core_no_curv_no_wmparc``, removes curvature/folding
    targets that were weakly predicted by the stats model and omits ``wmparc``
    white-matter parcellation volumes, leaving a compact cortical/subcortical
    morphometry set. Because the rule is fixed before classifier fitting, it
    does not leak outcome information across folds.
    """
    preset_key = str(preset).strip().lower()
    cols = list(dict.fromkeys(str(col) for col in feature_cols))
    if preset_key in {"", "all", "none"}:
        return cols

    excluded = {str(x) for x in excluded_measures}

    def keep_core(col: str) -> bool:
        parsed = parse_stats_feature_name(col)
        return parsed["measure"] not in excluded

    if preset_key in {"core", "core_no_curv", "core_morphometry"}:
        return [col for col in cols if keep_core(col)]
    if preset_key in {"core_no_curv_no_wmparc", "core_morphometry_no_wmparc"}:
        return [col for col in cols if keep_core(col) and not str(col).startswith("wmparc__")]
    if preset_key in {"volume_thickness", "volume_thickness_global"}:
        measures = {"Volume_mm3", "GrayVol", "SurfArea", "ThickAvg", "ThickStd", "NumVert"}
        return [
            col
            for col in cols
            if (parse_stats_feature_name(col)["measure"] in measures or "__global__" in str(col))
            and keep_core(col)
        ]
    if preset_key in {"literature", "literature_core"}:
        return [
            col
            for col in cols
            if keep_core(col) and literature_relevance(col, disease_context) != "not pre-specified"
        ]
    raise ValueError(
        "Unknown stats feature preset "
        f"{preset!r}. Expected all, core_no_curv, core_no_curv_no_wmparc, "
        "volume_thickness_global, or literature_core."
    )


def metric_row(
    y_true: Sequence[int],
    score: Sequence[float],
    pred: Sequence[int],
    *,
    split: str,
    scenario: str,
    fold: int | None = None,
    positive_label_name: str = "Disease",
    negative_label_name: str = "Healthy",
) -> dict[str, object]:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = np.asarray(pred, dtype=int)
    has_two_classes = np.unique(y_true).size == 2
    return {
        "scenario": scenario,
        "split": split,
        "fold": fold,
        "n": int(len(y_true)),
        f"n_{negative_label_name.lower()}": int(np.sum(y_true == 0)),
        f"n_{positive_label_name.lower()}": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
        "n_positive": int(np.sum(y_true == 1)),
        "auc": float(roc_auc_score(y_true, score)) if has_two_classes else np.nan,
        "average_precision": float(average_precision_score(y_true, score)) if has_two_classes else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)) if has_two_classes else np.nan,
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
    }


def stratified_group_kfold_indices(
    y: Sequence[int],
    groups: Sequence[object],
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split each group by class, then combine matching fold ids across groups."""
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups).astype(str)
    unique_groups = np.unique(groups)
    if unique_groups.size == 0:
        raise ValueError("No groups available for grouped CV.")

    min_per_class = []
    for group in unique_groups:
        idx = np.flatnonzero(groups == group)
        labels, counts = np.unique(y[idx], return_counts=True)
        if labels.size != 2:
            raise ValueError(f"{group}: expected both classes, got {dict(zip(labels, counts))}")
        min_per_class.append(int(counts.min()))

    n_splits = min(int(n_splits), min(min_per_class))
    if n_splits < 2:
        raise ValueError("Need at least two samples per class in every group for grouped CV.")

    rng = np.random.default_rng(int(seed))
    test_folds: list[list[int]] = [[] for _ in range(n_splits)]
    for group in unique_groups:
        group_idx = np.flatnonzero(groups == group)
        for class_value in np.unique(y[group_idx]):
            class_idx = group_idx[y[group_idx] == class_value].copy()
            rng.shuffle(class_idx)
            for fold, fold_idx in enumerate(np.array_split(class_idx, n_splits)):
                test_folds[fold].extend(fold_idx.tolist())

    all_idx = np.arange(len(y))
    splits = []
    for fold_idx in test_folds:
        test_idx = np.array(sorted(fold_idx), dtype=int)
        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=True)
        splits.append((train_idx, test_idx))
    return splits


def make_cv_splits(
    y: Sequence[int],
    model_df: pd.DataFrame,
    *,
    stratify_cols: Sequence[str] = (),
    n_splits: int = 5,
    repeats: int = 1,
    seed: int = 42,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    y = np.asarray(y, dtype=int)
    stratify_cols = [col for col in stratify_cols if col in model_df.columns]
    split_rows: list[tuple[int, np.ndarray, np.ndarray]] = []

    if stratify_cols:
        groups = _combined_strata(model_df, stratify_cols)
        for repeat in range(int(repeats)):
            try:
                splits = stratified_group_kfold_indices(
                    y,
                    groups,
                    n_splits=int(n_splits),
                    seed=int(seed) + repeat,
                )
            except ValueError:
                break
            for fold, (train_idx, test_idx) in enumerate(splits, start=1 + repeat * len(splits)):
                split_rows.append((fold, train_idx, test_idx))
        if split_rows:
            return split_rows

    cv = RepeatedStratifiedKFold(
        n_splits=int(n_splits),
        n_repeats=int(repeats),
        random_state=int(seed),
    )
    for fold, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(y)), y), start=1):
        split_rows.append((fold, train_idx, test_idx))
    return split_rows


def _combined_strata(model_df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    if not cols:
        return np.repeat("all", len(model_df))
    parts = []
    for col in cols:
        value = model_df[col].astype("string").fillna("<missing>").to_numpy()
        parts.append(value)
    out = parts[0].astype(str)
    for part in parts[1:]:
        out = np.char.add(np.char.add(out, "__"), part.astype(str))
    return out


def _inner_cv_splits(
    y_train: np.ndarray,
    train_frame: pd.DataFrame,
    *,
    stratify_cols: Sequence[str],
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if stratify_cols:
        try:
            return [
                (tr, va)
                for _fold, tr, va in make_cv_splits(
                    y_train,
                    train_frame.reset_index(drop=True),
                    stratify_cols=stratify_cols,
                    n_splits=n_splits,
                    repeats=1,
                    seed=seed,
                )
            ]
        except ValueError:
            pass
    n_splits = min(int(n_splits), int(np.bincount(y_train).min()))
    if n_splits < 2:
        raise ValueError("Need at least two samples per class for inner CV.")
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y_train)), y_train))


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _fit_confound_matrix(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    confound_cols: Sequence[str],
    continuous_cols: Sequence[str] = (),
    categorical_cols: Sequence[str] = (),
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    confound_cols = make_confounds_available(train_frame, confound_cols)
    if not confound_cols:
        return np.zeros((len(train_frame), 0), dtype=np.float64), np.zeros((len(test_frame), 0), dtype=np.float64), []

    continuous = [col for col in continuous_cols if col in confound_cols]
    categorical = [col for col in categorical_cols if col in confound_cols]
    unspecified = [col for col in confound_cols if col not in set(continuous) | set(categorical)]
    for col in unspecified:
        if pd.api.types.is_numeric_dtype(train_frame[col]):
            continuous.append(col)
        else:
            categorical.append(col)

    train_blocks = []
    test_blocks = []
    names: list[str] = []

    if continuous:
        train_num = train_frame[continuous].apply(pd.to_numeric, errors="coerce")
        test_num = test_frame[continuous].apply(pd.to_numeric, errors="coerce")
        med = train_num.median(axis=0).fillna(0.0)
        train_blocks.append(train_num.fillna(med).to_numpy(dtype=np.float64))
        test_blocks.append(test_num.fillna(med).to_numpy(dtype=np.float64))
        names.extend([f"confound__{col}" for col in continuous])

    if categorical:
        train_cat = train_frame[categorical].astype("string").fillna("<missing>")
        test_cat = test_frame[categorical].astype("string").fillna("<missing>")
        encoder = _one_hot_encoder()
        train_oh = encoder.fit_transform(train_cat)
        test_oh = encoder.transform(test_cat)
        if hasattr(train_oh, "toarray"):
            train_oh = train_oh.toarray()
            test_oh = test_oh.toarray()
        train_blocks.append(np.asarray(train_oh, dtype=np.float64))
        test_blocks.append(np.asarray(test_oh, dtype=np.float64))
        names.extend([f"confound__{name}" for name in encoder.get_feature_names_out(categorical).tolist()])

    if not train_blocks:
        return np.zeros((len(train_frame), 0), dtype=np.float64), np.zeros((len(test_frame), 0), dtype=np.float64), []
    train_matrix = np.column_stack(train_blocks)
    test_matrix = np.column_stack(test_blocks)

    # Constant confound columns are no-ops. In particular, residualizing against
    # an intercept-only design just mean-centers features, which is then undone
    # by the later standardization path and can make residualized predictions
    # exactly match the raw feature model.
    var = np.nanvar(train_matrix, axis=0)
    keep = np.isfinite(var) & (var > 0.0)
    if not np.any(keep):
        return np.zeros((len(train_frame), 0), dtype=np.float64), np.zeros((len(test_frame), 0), dtype=np.float64), []
    kept_names = [name for name, use in zip(names, keep) if bool(use)]
    return train_matrix[:, keep], test_matrix[:, keep], kept_names


def _numeric_feature_arrays(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    feature_cols: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cols = [col for col in feature_cols if col in train_frame.columns]
    train = train_frame[cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
    test = test_frame[cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
    return train, test, list(cols)


def _impute_numeric(
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = np.nanmedian(train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    train_out = np.where(np.isfinite(train), train, med)
    test_out = np.where(np.isfinite(test), test, med)
    return train_out, test_out, med


def _fit_numeric_transform(
    train: np.ndarray,
    test: np.ndarray,
    names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, np.ndarray]]:
    train_imp, test_imp, med = _impute_numeric(train, test)
    var = np.nanvar(train_imp, axis=0)
    keep = np.isfinite(var) & (var > 0.0)
    if not np.any(keep):
        raise ValueError("All candidate columns are constant after imputation.")
    train_keep = train_imp[:, keep]
    test_keep = test_imp[:, keep]
    mean = train_keep.mean(axis=0)
    std = train_keep.std(axis=0)
    std = np.where((std > 0) & np.isfinite(std), std, 1.0)
    kept_names = [str(name) for name, use in zip(names, keep) if bool(use)]
    state = {"median": med, "keep": keep, "mean": mean, "std": std}
    return (train_keep - mean) / std, (test_keep - mean) / std, kept_names, state


def _residualize_features(
    feature_train: np.ndarray,
    feature_test: np.ndarray,
    confound_train: np.ndarray,
    confound_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    feat_train, feat_test, _ = _impute_numeric(feature_train, feature_test)
    if confound_train.shape[1] == 0:
        return feat_train, feat_test
    c_train = np.column_stack([np.ones(len(confound_train)), confound_train])
    c_test = np.column_stack([np.ones(len(confound_test)), confound_test])
    beta = np.linalg.pinv(c_train) @ feat_train
    return feat_train - c_train @ beta, feat_test - c_test @ beta


def _scenario_raw_matrices(
    model_df: pd.DataFrame,
    feature_cols: Sequence[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    config: ClinicalExperimentConfig,
    scenario: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_frame = model_df.iloc[train_idx]
    test_frame = model_df.iloc[test_idx]
    feature_train, feature_test, feature_names = _numeric_feature_arrays(train_frame, test_frame, feature_cols)
    confound_train, confound_test, confound_names = _fit_confound_matrix(
        train_frame,
        test_frame,
        confound_cols=config.confound_cols,
        continuous_cols=config.continuous_confound_cols,
        categorical_cols=config.categorical_confound_cols,
    )

    if scenario == "features":
        return feature_train, feature_test, feature_names
    if scenario == "residualized_features":
        if confound_train.shape[1] == 0:
            return feature_train, feature_test, feature_names
        train_resid, test_resid = _residualize_features(feature_train, feature_test, confound_train, confound_test)
        return train_resid, test_resid, feature_names
    raise ValueError(f"Unknown scenario: {scenario}")


def _normalized_logreg_kwargs(config: ClinicalExperimentConfig, *, for_cv: bool = False) -> dict[str, Any]:
    logreg_kwargs = dict(config.logistic_regression_kwargs or {})
    reserved = {"C", "Cs", "class_weight", "max_iter", "random_state", "n_jobs"}
    if for_cv:
        reserved |= {"cv", "refit", "scoring"}
    overlap = sorted(reserved & set(logreg_kwargs))
    if overlap:
        raise ValueError(
            "Set LogisticRegression model-selection parameters via ClinicalExperimentConfig, "
            f"not logistic_regression_kwargs: {overlap}"
        )

    legacy_penalty = logreg_kwargs.pop("penalty", None)
    legacy_penalty_key = str(legacy_penalty).strip().lower() if legacy_penalty is not None else ""
    legacy_l1_ratio = None
    if legacy_penalty_key in {"l1", "l2"}:
        legacy_l1_ratio = 1.0 if legacy_penalty_key == "l1" else 0.0
    elif legacy_penalty_key in {"", "deprecated"}:
        legacy_l1_ratio = None
    elif legacy_penalty_key == "elasticnet":
        if "l1_ratio" not in logreg_kwargs and "l1_ratios" not in logreg_kwargs:
            raise ValueError("penalty='elasticnet' requires l1_ratio or l1_ratios in logistic_regression_kwargs.")
    elif legacy_penalty is not None:
        logreg_kwargs["penalty"] = legacy_penalty

    if for_cv:
        if "l1_ratio" in logreg_kwargs and "l1_ratios" in logreg_kwargs:
            raise ValueError("Use only one of l1_ratio or l1_ratios in logistic_regression_kwargs.")
        if legacy_l1_ratio is not None and "l1_ratio" not in logreg_kwargs and "l1_ratios" not in logreg_kwargs:
            logreg_kwargs["l1_ratios"] = (legacy_l1_ratio,)
        if "l1_ratio" in logreg_kwargs:
            value = logreg_kwargs.pop("l1_ratio")
            if value is not None:
                logreg_kwargs["l1_ratios"] = (float(value),)
        if "l1_ratios" in logreg_kwargs and logreg_kwargs["l1_ratios"] is not None:
            value = logreg_kwargs["l1_ratios"]
            if np.isscalar(value):
                logreg_kwargs["l1_ratios"] = (float(value),)
            else:
                logreg_kwargs["l1_ratios"] = tuple(float(x) for x in value)
    else:
        if "l1_ratio" in logreg_kwargs and "l1_ratios" in logreg_kwargs:
            raise ValueError("Use only one of l1_ratio or l1_ratios in logistic_regression_kwargs.")
        if legacy_l1_ratio is not None and "l1_ratio" not in logreg_kwargs and "l1_ratios" not in logreg_kwargs:
            logreg_kwargs["l1_ratio"] = legacy_l1_ratio
        if "l1_ratios" in logreg_kwargs:
            value = logreg_kwargs.pop("l1_ratios")
            if value is not None:
                ratios = (float(value),) if np.isscalar(value) else tuple(float(x) for x in value)
                if len(ratios) > 1:
                    raise ValueError("Multiple l1_ratios require tune_c=True so LogisticRegressionCV can select one.")
                if ratios:
                    logreg_kwargs["l1_ratio"] = ratios[0]

    return logreg_kwargs


def _fit_logreg(x_train: np.ndarray, y_train: np.ndarray, *, C: float, config: ClinicalExperimentConfig, seed: int) -> LogisticRegression:
    logreg_kwargs = _normalized_logreg_kwargs(config, for_cv=False)
    params = {
        "C": float(C),
        "class_weight": config.class_weight,
        "max_iter": int(config.max_iter),
        "random_state": int(seed),
    }
    params.update(logreg_kwargs)
    model = LogisticRegression(**params)
    model.fit(x_train, y_train)
    return model


def _logreg_cv_fit_count(
    y_train: np.ndarray,
    train_frame: pd.DataFrame,
    config: ClinicalExperimentConfig,
    *,
    seed: int,
) -> int:
    c_grid = tuple(float(c) for c in config.c_grid)
    if not config.tune_c or len(c_grid) <= 1:
        return 0

    inner_splits = _inner_cv_splits(
        y_train,
        train_frame,
        stratify_cols=config.stratify_cols,
        n_splits=config.inner_splits,
        seed=seed,
    )
    logreg_kwargs = _normalized_logreg_kwargs(config, for_cv=True)
    n_l1_ratios = len(logreg_kwargs.get("l1_ratios", (None,)))
    return int(len(c_grid) * len(inner_splits) * n_l1_ratios)


def _logreg_cv_score_table(model: LogisticRegressionCV) -> pd.DataFrame:
    if not getattr(model, "scores_", None):
        return pd.DataFrame()

    positive_class = model.classes_[-1]
    scores = model.scores_.get(positive_class)
    if scores is None:
        scores = next(iter(model.scores_.values()))
    scores = np.asarray(scores, dtype=np.float64)
    cs = np.asarray(model.Cs_, dtype=np.float64)
    selected_c = float(np.ravel(model.C_)[0])
    selected_l1_ratio = getattr(model, "l1_ratio_", None)
    if selected_l1_ratio is not None:
        selected_l1_ratio = np.ravel(selected_l1_ratio)[0]
        selected_l1_ratio = None if selected_l1_ratio is None else float(selected_l1_ratio)

    rows: list[dict[str, object]] = []
    if scores.ndim == 2:
        for c_idx, C in enumerate(cs):
            fold_scores = scores[:, c_idx]
            rows.append(
                {
                    "C": float(C),
                    "l1_ratio": np.nan,
                    "mean_inner_auc": float(np.nanmean(fold_scores)),
                    "std_inner_auc": float(np.nanstd(fold_scores, ddof=1)) if fold_scores.size > 1 else 0.0,
                    "mean_inner_average_precision": np.nan,
                    "n_inner_folds": int(fold_scores.size),
                    "selected": bool(np.isclose(float(C), selected_c)),
                }
            )
    elif scores.ndim == 3:
        l1_ratios = getattr(model, "l1_ratios_", None)
        if l1_ratios is None:
            l1_ratios = np.arange(scores.shape[2], dtype=np.float64)
        l1_ratios = np.asarray(l1_ratios, dtype=np.float64)
        for c_idx, C in enumerate(cs):
            for ratio_idx, l1_ratio in enumerate(l1_ratios):
                fold_scores = scores[:, c_idx, ratio_idx]
                rows.append(
                    {
                        "C": float(C),
                        "l1_ratio": float(l1_ratio),
                        "mean_inner_auc": float(np.nanmean(fold_scores)),
                        "std_inner_auc": float(np.nanstd(fold_scores, ddof=1)) if fold_scores.size > 1 else 0.0,
                        "mean_inner_average_precision": np.nan,
                        "n_inner_folds": int(fold_scores.size),
                        "selected": bool(
                            np.isclose(float(C), selected_c)
                            and selected_l1_ratio is not None
                            and np.isclose(float(l1_ratio), selected_l1_ratio)
                        ),
                    }
                )
    else:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(["mean_inner_auc", "C", "l1_ratio"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


def _fit_logreg_with_optional_cv(
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_frame: pd.DataFrame,
    config: ClinicalExperimentConfig,
    *,
    seed: int,
    progress_bar: Any | None = None,
) -> tuple[LogisticRegression | LogisticRegressionCV, float, pd.DataFrame]:
    c_grid = tuple(float(c) for c in config.c_grid)
    if not config.tune_c or len(c_grid) <= 1:
        selected_c = float(c_grid[0])
        model = _fit_logreg(x_train, y_train, C=selected_c, config=config, seed=seed)
        return model, selected_c, pd.DataFrame()

    inner_splits = _inner_cv_splits(
        y_train,
        train_frame.reset_index(drop=True),
        stratify_cols=config.stratify_cols,
        n_splits=config.inner_splits,
        seed=seed,
    )
    logreg_kwargs = _normalized_logreg_kwargs(config, for_cv=True)
    params = {
        "Cs": list(c_grid),
        "cv": inner_splits,
        "scoring": "roc_auc",
        "class_weight": config.class_weight,
        "max_iter": int(config.max_iter),
        "random_state": int(seed),
        "refit": True,
        "n_jobs": config.n_jobs,
    }
    if "use_legacy_attributes" in LogisticRegressionCV().get_params(deep=False):
        params["use_legacy_attributes"] = True
    params.update(logreg_kwargs)
    model = LogisticRegressionCV(**params)
    model.fit(x_train, y_train)
    if progress_bar is not None:
        n_l1_ratios = len(logreg_kwargs.get("l1_ratios", (None,)))
        progress_bar.update(int(len(c_grid) * len(inner_splits) * n_l1_ratios))
    c_table = _logreg_cv_score_table(model)
    selected_c = float(np.ravel(model.C_)[0])
    return model, selected_c, c_table


def _c_tuning_fit_count(
    y: np.ndarray,
    model_df: pd.DataFrame,
    splits: Sequence[tuple[int, np.ndarray, np.ndarray]],
    config: ClinicalExperimentConfig,
    *,
    n_scenarios: int,
) -> int:
    c_grid = tuple(float(c) for c in config.c_grid)
    if not config.tune_c or len(c_grid) <= 1:
        return 0
    total_per_scenario = 0
    for _fold, train_idx, _test_idx in splits:
        total_per_scenario += _logreg_cv_fit_count(
            y[train_idx],
            model_df.iloc[train_idx],
            config,
            seed=config.seed + 1000 * int(_fold),
        )
    return int(n_scenarios) * int(total_per_scenario)


def parse_stats_feature_name(feature: str) -> dict[str, str]:
    parts = str(feature).split("__")
    source = parts[0] if len(parts) > 0 else ""
    region = parts[1] if len(parts) > 1 else ""
    measure = parts[2] if len(parts) > 2 else ""
    hemi = ""
    if source.startswith("lh."):
        hemi = "lh"
    elif source.startswith("rh."):
        hemi = "rh"
    elif region.startswith("Left-") or region.startswith("wm-lh-"):
        hemi = "lh"
    elif region.startswith("Right-") or region.startswith("wm-rh-"):
        hemi = "rh"
    family = source.split(".")[0]
    return {"source": source, "family": family, "region": region, "measure": measure, "hemi": hemi}


def literature_relevance(feature: str, disease_context: str = "") -> str:
    text = str(feature).lower()
    disease = str(disease_context).lower()
    if "alzheimer" in disease or disease in {"ad", "adni", "dementia"}:
        high = [
            "hippocampus",
            "entorhinal",
            "parahippocampal",
            "inferiortemporal",
            "middletemporal",
            "temporalpole",
            "fusiform",
            "precuneus",
            "posteriorcingulate",
            "amygdala",
            "inf-lat-vent",
            "lateral-ventricle",
            "ventricle",
        ]
        global_terms = ["cortexvol", "totalgrayvol", "brainsegvol", "meanthickness", "etiv"]
        if any(term in text for term in high):
            return "AD literature-consistent"
        if any(term in text for term in global_terms):
            return "AD global atrophy/size"
        return "not pre-specified"
    if "schiz" in disease or "psychosis" in disease:
        high = [
            "lateral-ventricle",
            "inf-lat-vent",
            "ventricle",
            "hippocampus",
            "amygdala",
            "thalamus",
            "caudate",
            "putamen",
            "pallidum",
            "accumbens",
            "superiortemporal",
            "middletemporal",
            "insula",
            "anteriorcingulate",
            "rostralmiddlefrontal",
            "caudalmiddlefrontal",
            "superiorfrontal",
            "cortexvol",
            "meanthickness",
        ]
        if any(term in text for term in high):
            return "schizophrenia literature-consistent"
        return "not pre-specified"
    if "parkinson" in disease or disease in {"pd", "ppmi"}:
        subcortical = [
            "putamen",
            "caudate",
            "pallidum",
            "thalamus",
            "accumbens",
            "brain-stem",
            "brainstem",
            "cerebellum",
        ]
        cortical_cognitive = [
            "hippocampus",
            "amygdala",
            "entorhinal",
            "parahippocampal",
            "inferiortemporal",
            "middletemporal",
            "temporalpole",
            "insula",
            "precentral",
            "postcentral",
            "paracentral",
            "superiorfrontal",
            "rostralmiddlefrontal",
            "caudalmiddlefrontal",
            "anteriorcingulate",
            "posteriorcingulate",
        ]
        global_terms = [
            "cortexvol",
            "totalgrayvol",
            "brainsegvol",
            "meanthickness",
            "ventricle",
            "etiv",
        ]
        if any(term in text for term in subcortical):
            return "PD subcortical/brainstem/cerebellar"
        if any(term in text for term in cortical_cognitive):
            return "PD cortical/cognitive-motor"
        if any(term in text for term in global_terms):
            return "PD global atrophy/size"
        return "not pre-specified"
    return "not pre-specified"


def aggregate_feature_importance(
    coefficients: pd.DataFrame,
    *,
    disease_context: str = "",
) -> pd.DataFrame:
    if coefficients.empty:
        return pd.DataFrame()
    feature_rows = coefficients[~coefficients["feature"].astype(str).str.startswith("confound__")].copy()
    if feature_rows.empty:
        return pd.DataFrame()
    out = (
        feature_rows.assign(abs_coef=lambda frame: frame["coef"].abs(), coef_sign=lambda frame: np.sign(frame["coef"]))
        .groupby(["scenario", "feature"], as_index=False)
        .agg(
            coef=("coef", "mean"),
            coef_std=("coef", "std"),
            abs_coef=("abs_coef", "mean"),
            abs_coef_std=("abs_coef", "std"),
            sign_consistency=("coef_sign", lambda s: float(np.abs(np.nanmean(s)))),
            n_folds=("fold", "nunique"),
        )
        .sort_values(["scenario", "abs_coef"], ascending=[True, False])
        .reset_index(drop=True)
    )
    parsed = pd.DataFrame([parse_stats_feature_name(feature) for feature in out["feature"]])
    out = pd.concat([out, parsed], axis=1)
    out["literature_relevance"] = out["feature"].map(lambda x: literature_relevance(x, disease_context))
    return out


def run_clinical_binary_experiment(
    model_df: pd.DataFrame,
    feature_cols: Sequence[str],
    config: ClinicalExperimentConfig,
    *,
    label_col: str = "y",
    subject_col: str = "subject",
) -> ClinicalExperimentResult:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = safe_tag(config.feature_source)

    model_df = model_df.copy().reset_index(drop=True)
    if label_col not in model_df.columns:
        raise KeyError(f"label_col {label_col!r} is not in model_df")
    if subject_col not in model_df.columns:
        model_df[subject_col] = np.arange(len(model_df)).astype(str)

    feature_frame, feature_cols_clean = clean_feature_frame(model_df, feature_cols)
    model_df[feature_cols_clean] = feature_frame
    y = model_df[label_col].to_numpy(dtype=np.int64)
    subjects = model_df[subject_col].astype(str).to_numpy()

    scenarios = _usable_scenarios(config, model_df)
    splits = make_cv_splits(
        y,
        model_df,
        stratify_cols=config.stratify_cols,
        n_splits=config.cv_splits,
        repeats=config.cv_repeats,
        seed=config.seed,
    )

    cv_metric_rows = []
    cv_score_frames = []
    oof_frames = []
    coef_rows = []
    c_rows = []
    holdout_rows = []

    t0 = time.perf_counter()
    c_tuning_total = _c_tuning_fit_count(y, model_df, splits, config, n_scenarios=len(scenarios))
    c_progress = tqdm(
        total=c_tuning_total,
        desc="Tuning C (inner CV)",
        unit="fit",
        disable=(not config.progress or c_tuning_total == 0),
    )
    try:
        for scenario in scenarios:
            oof_score = np.full(len(model_df), np.nan, dtype=float)
            oof_counts = np.zeros(len(model_df), dtype=int)
            for fold, train_idx, test_idx in splits:
                if c_tuning_total:
                    c_progress.set_postfix({"scenario": scenario, "fold": fold}, refresh=False)
                x_train_raw, x_test_raw, names = _scenario_raw_matrices(
                    model_df,
                    feature_cols_clean,
                    train_idx,
                    test_idx,
                    config,
                    scenario,
                )
                x_train, x_test, kept_names, _state = _fit_numeric_transform(x_train_raw, x_test_raw, names)
                model, selected_c, c_table = _fit_logreg_with_optional_cv(
                    x_train,
                    y[train_idx],
                    model_df.iloc[train_idx],
                    config,
                    seed=config.seed + 1000 * fold,
                    progress_bar=c_progress,
                )
                if not c_table.empty:
                    c_rows.append(c_table.assign(scenario=scenario, fold=fold))

                score = model.predict_proba(x_test)[:, 1]
                pred = (score >= 0.5).astype(int)

                oof_score[test_idx] = np.nan_to_num(oof_score[test_idx], nan=0.0) + score
                oof_counts[test_idx] += 1

                row = metric_row(
                    y[test_idx],
                    score,
                    pred,
                    split="cv",
                    scenario=scenario,
                    fold=fold,
                    positive_label_name=config.positive_label_name,
                    negative_label_name=config.negative_label_name,
                )
                row["selected_c"] = float(selected_c)
                if not c_table.empty:
                    row["best_inner_auc"] = float(c_table["mean_inner_auc"].max())
                cv_metric_rows.append(row)

                score_frame = model_df.iloc[test_idx].copy()
                score_frame = score_frame[[col for col in _score_metadata_cols(model_df, subject_col) if col in score_frame.columns]]
                score_frame["scenario"] = scenario
                score_frame["fold"] = fold
                score_frame["y"] = y[test_idx]
                score_frame["score"] = score
                score_frame["pred"] = pred
                score_frame["selected_c"] = float(selected_c)
                cv_score_frames.append(score_frame)

                coef_rows.extend(
                    {
                        "scenario": scenario,
                        "fold": fold,
                        "feature": feature,
                        "coef": float(coef),
                        "selected_c": float(selected_c),
                    }
                    for feature, coef in zip(kept_names, model.coef_.ravel())
                )
            mean_oof_score = oof_score / np.maximum(oof_counts, 1)
            oof_pred = (mean_oof_score >= 0.5).astype(int)
            oof_metric = metric_row(
                y,
                mean_oof_score,
                oof_pred,
                split="oof",
                scenario=scenario,
                fold=None,
                positive_label_name=config.positive_label_name,
                negative_label_name=config.negative_label_name,
            )
            oof_frames.append(
                model_df[[col for col in _score_metadata_cols(model_df, subject_col) if col in model_df.columns]]
                .assign(scenario=scenario, y=y, score=mean_oof_score, pred=oof_pred)
            )
            cv_metric_rows.append(oof_metric)

            holdout_rows.append(
                _run_holdout(
                    model_df,
                    feature_cols_clean,
                    y,
                    config,
                    scenario=scenario,
                    subject_col=subject_col,
                )
            )
    finally:
        c_progress.close()

    cv_metrics = pd.DataFrame(cv_metric_rows)
    oof_metrics = cv_metrics[cv_metrics["split"].eq("oof")].reset_index(drop=True)
    cv_metrics = cv_metrics[~cv_metrics["split"].eq("oof")].reset_index(drop=True)
    cv_scores = pd.concat(cv_score_frames, ignore_index=True) if cv_score_frames else pd.DataFrame()
    oof_predictions = pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame()
    holdout_metrics = pd.DataFrame(holdout_rows)
    coefficients = pd.DataFrame(coef_rows)
    feature_importance = aggregate_feature_importance(coefficients, disease_context=config.disease_context)
    c_table = pd.concat(c_rows, ignore_index=True) if c_rows else pd.DataFrame()
    confound_summary = _confound_control_summary(oof_metrics, config.primary_scenario)

    output_paths = _write_experiment_outputs(
        output_dir,
        tag,
        model_df,
        cv_metrics,
        cv_scores,
        oof_metrics,
        oof_predictions,
        holdout_metrics,
        coefficients,
        feature_importance,
        confound_summary,
        c_table,
    )
    output_paths.update(
        _write_experiment_plots(
            output_dir,
            tag,
            config,
            cv_scores,
            oof_predictions,
        )
    )

    elapsed = time.perf_counter() - t0
    print(
        f"[{config.experiment_name}] finished {len(scenarios)} scenario(s), "
        f"{len(splits)} CV fold(s) each in {elapsed:.1f}s"
    )
    return ClinicalExperimentResult(
        config=config,
        model_df=model_df,
        feature_cols=feature_cols_clean,
        cv_metrics=cv_metrics,
        cv_scores=cv_scores,
        oof_metrics=oof_metrics,
        oof_predictions=oof_predictions,
        holdout_metrics=holdout_metrics,
        coefficients=coefficients,
        feature_importance=feature_importance,
        confound_control_summary=confound_summary,
        c_table=c_table,
        output_paths=output_paths,
    )


def run_clinical_stats_workflow(
    model_df: pd.DataFrame,
    feature_cols: Sequence[str],
    config: ClinicalStatsWorkflowConfig,
    *,
    label_col: str = "y",
    subject_col: str = "subject",
) -> ClinicalStatsWorkflowResult:
    """
    Run the shared clinical classification workflow used by the notebooks.

    Dataset-specific notebook code should stop once it has built ``model_df``,
    ``feature_cols``, and a binary label column. This function then applies the
    common covariate harmonization, feature preset, confound controls, repeated
    CV, holdout sanity check, plots, and output writing.
    """
    prepared_df = canonicalize_clinical_covariates(
        model_df,
        age_cols=config.age_cols,
        sex_cols=config.sex_cols,
        study_col=config.study_col,
        scanner_cols=config.scanner_cols,
        protocol_cols=config.protocol_cols,
    )
    all_feature_cols = [col for col in feature_cols if col in prepared_df.columns]
    feature_confound_cols = tuple(
        col
        for col in config.feature_confound_cols
        if col in prepared_df.columns and pd.api.types.is_numeric_dtype(prepared_df[col])
    )
    selected_feature_cols = select_stats_feature_columns(
        all_feature_cols,
        preset=config.stats_feature_preset,
        disease_context=config.disease_context,
    )
    if config.remove_feature_confounds_from_features and feature_confound_cols:
        feature_confound_set = set(feature_confound_cols)
        selected_feature_cols = [col for col in selected_feature_cols if col not in feature_confound_set]
    confound_cols, continuous_cols, categorical_cols = clean_confound_column_sets(
        prepared_df,
        include_study=config.include_study_confound,
        include_scanner=config.include_scanner_confound,
    )
    if feature_confound_cols:
        confound_cols = tuple(dict.fromkeys(tuple(confound_cols) + feature_confound_cols))
        continuous_cols = tuple(dict.fromkeys(tuple(continuous_cols) + feature_confound_cols))
    experiment_config = ClinicalExperimentConfig(
        experiment_name=config.experiment_name,
        output_dir=config.output_dir,
        feature_source=config.feature_source,
        disease_context=config.disease_context,
        positive_label_name=config.positive_label_name,
        negative_label_name=config.negative_label_name,
        scenarios=config.scenarios,
        primary_scenario=config.primary_scenario,
        confound_cols=confound_cols,
        continuous_confound_cols=continuous_cols,
        categorical_confound_cols=categorical_cols,
        stratify_cols=config.stratify_cols,
        cv_splits=config.cv_splits,
        cv_repeats=config.cv_repeats,
        inner_splits=config.inner_splits,
        tune_c=config.tune_c,
        c_grid=config.c_grid,
        class_weight=config.class_weight,
        max_iter=config.max_iter,
        n_jobs=config.n_jobs,
        logistic_regression_kwargs=dict(config.logistic_regression_kwargs or {}),
        seed=config.seed,
        test_size=config.test_size,
        top_n_features=config.top_n_features,
        progress=config.progress,
    )
    _usable_confound_matrix, _usable_confound_test, usable_confound_terms = _fit_confound_matrix(
        prepared_df,
        prepared_df,
        confound_cols=confound_cols,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
    )
    result = run_clinical_binary_experiment(
        prepared_df,
        selected_feature_cols,
        experiment_config,
        label_col=label_col,
        subject_col=subject_col,
    )
    return ClinicalStatsWorkflowResult(
        workflow_config=config,
        experiment_config=experiment_config,
        result=result,
        model_df=prepared_df,
        all_feature_cols=all_feature_cols,
        feature_cols=selected_feature_cols,
        confound_cols=confound_cols,
        continuous_confound_cols=continuous_cols,
        categorical_confound_cols=categorical_cols,
        feature_confound_cols=feature_confound_cols,
        usable_confound_terms=tuple(usable_confound_terms),
        label_col=label_col,
        subject_col=subject_col,
    )


def display_clinical_stats_workflow(
    workflow: ClinicalStatsWorkflowResult,
) -> None:
    """Display the compact notebook report for a clinical stats workflow."""
    try:
        from IPython.display import display
    except ImportError:  # pragma: no cover - normal notebooks have IPython.
        display = print

    result = workflow.result
    print(
        f"Analysis table: {len(workflow.model_df):,} subjects x "
        f"{len(workflow.feature_cols):,} selected features"
    )

    print()
    print("Confounds used:")
    print(f"  continuous: {workflow.continuous_confound_cols}")
    print(f"  categorical: {workflow.categorical_confound_cols}")
    print(f"  feature-derived: {workflow.feature_confound_cols}")
    print(f"  encoded terms: {len(workflow.usable_confound_terms)}")

    print()
    print("OOF performance:")
    scenarios = ["features", "residualized_features"]
    metric_cols = ["scenario", "n", "auc", "average_precision", "balanced_accuracy", "f1", "selected_c"]
    oof_cols = [col for col in metric_cols if col in result.oof_metrics.columns]
    display(result.oof_metrics[result.oof_metrics["scenario"].isin(scenarios)][oof_cols].reset_index(drop=True))

    if not result.holdout_metrics.empty:
        print()
        print("Holdout performance:")
        holdout_cols = [col for col in metric_cols if col in result.holdout_metrics.columns]
        display(result.holdout_metrics[result.holdout_metrics["scenario"].isin(scenarios)][holdout_cols].reset_index(drop=True))

    print()
    print("CSV outputs:")
    for name, path in result.output_paths.items():
        if Path(path).suffix.lower() == ".csv":
            print(f"  {name}: {path}")


def _usable_scenarios(config: ClinicalExperimentConfig, model_df: pd.DataFrame) -> list[str]:
    confound_train, _confound_test, _confound_names = _fit_confound_matrix(
        model_df,
        model_df,
        confound_cols=config.confound_cols,
        continuous_cols=config.continuous_confound_cols,
        categorical_cols=config.categorical_confound_cols,
    )
    has_usable_confounds = confound_train.shape[1] > 0
    scenarios = []
    for scenario in config.scenarios:
        if scenario == "residualized_features" and not has_usable_confounds:
            continue
        scenarios.append(scenario)
    if config.primary_scenario not in scenarios:
        scenarios.insert(0, config.primary_scenario)
    return list(dict.fromkeys(scenarios))


def _score_metadata_cols(model_df: pd.DataFrame, subject_col: str) -> list[str]:
    preferred = [
        subject_col,
        "PTID",
        "participant_id",
        "dataset_id",
        "clinical_label",
        "binary_label",
        "dx_name",
        "label",
        "group",
        "schz_group",
        "session_id",
        "image",
    ]
    return [col for col in preferred if col in model_df.columns]


def _run_holdout(
    model_df: pd.DataFrame,
    feature_cols: Sequence[str],
    y: np.ndarray,
    config: ClinicalExperimentConfig,
    *,
    scenario: str,
    subject_col: str,
) -> dict[str, object]:
    strata_cols = [col for col in config.stratify_cols if col in model_df.columns]
    if strata_cols:
        strata = pd.Series(_combined_strata(model_df, strata_cols), index=model_df.index).astype(str) + "__" + pd.Series(y).astype(str)
        if strata.value_counts().min() < 2:
            strata = pd.Series(y, index=model_df.index)
    else:
        strata = pd.Series(y, index=model_df.index)
    train_idx, test_idx = train_test_split(
        np.arange(len(model_df)),
        test_size=float(config.test_size),
        stratify=strata,
        random_state=int(config.seed),
    )
    x_train_raw, x_test_raw, names = _scenario_raw_matrices(
        model_df,
        feature_cols,
        train_idx,
        test_idx,
        config,
        scenario,
    )
    x_train, x_test, _kept_names, _state = _fit_numeric_transform(x_train_raw, x_test_raw, names)
    model, selected_c, _table = _fit_logreg_with_optional_cv(
        x_train,
        y[train_idx],
        model_df.iloc[train_idx],
        config,
        seed=config.seed + 50_000,
    )
    score = model.predict_proba(x_test)[:, 1]
    pred = (score >= 0.5).astype(int)
    row = metric_row(
        y[test_idx],
        score,
        pred,
        split="holdout",
        scenario=scenario,
        fold=None,
        positive_label_name=config.positive_label_name,
        negative_label_name=config.negative_label_name,
    )
    row["selected_c"] = selected_c
    tn, fp, fn, tp = confusion_matrix(y[test_idx], pred, labels=[0, 1]).ravel()
    row.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return row


def _confound_control_summary(oof_metrics: pd.DataFrame, primary_scenario: str) -> pd.DataFrame:
    cols = ["scenario", "auc", "average_precision", "balanced_accuracy", "accuracy", "f1", "precision", "recall"]
    out = oof_metrics[[col for col in cols if col in oof_metrics.columns]].copy()
    if out.empty:
        return out
    primary_auc = out.loc[out["scenario"].eq(primary_scenario), "auc"]
    baseline = float(primary_auc.iloc[0]) if len(primary_auc) else np.nan
    out["delta_auc_vs_primary"] = out["auc"] - baseline
    return out.sort_values("auc", ascending=False).reset_index(drop=True)


def _write_experiment_outputs(
    output_dir: Path,
    tag: str,
    model_df: pd.DataFrame,
    cv_metrics: pd.DataFrame,
    cv_scores: pd.DataFrame,
    oof_metrics: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    holdout_metrics: pd.DataFrame,
    coefficients: pd.DataFrame,
    feature_importance: pd.DataFrame,
    confound_summary: pd.DataFrame,
    c_table: pd.DataFrame,
) -> dict[str, Path]:
    outputs = {
        "model_rows": output_dir / f"{tag}_model_rows.csv",
        "cv_metrics": output_dir / f"{tag}_cv_metrics.csv",
        "cv_scores": output_dir / f"{tag}_cv_scores.csv",
        "oof_metrics": output_dir / f"{tag}_oof_metrics.csv",
        "oof_predictions": output_dir / f"{tag}_oof_predictions.csv",
        "holdout_metrics": output_dir / f"{tag}_holdout_metrics.csv",
        "logreg_coefficients": output_dir / f"{tag}_logreg_coefficients.csv",
        "feature_importance": output_dir / f"{tag}_feature_importance.csv",
        "confound_control_summary": output_dir / f"{tag}_confound_control_summary.csv",
        "inner_c_grid": output_dir / f"{tag}_logreg_inner_c_grid.csv",
    }
    model_df.to_csv(outputs["model_rows"], index=False)
    cv_metrics.to_csv(outputs["cv_metrics"], index=False)
    cv_scores.to_csv(outputs["cv_scores"], index=False)
    oof_metrics.to_csv(outputs["oof_metrics"], index=False)
    oof_predictions.to_csv(outputs["oof_predictions"], index=False)
    holdout_metrics.to_csv(outputs["holdout_metrics"], index=False)
    coefficients.to_csv(outputs["logreg_coefficients"], index=False)
    feature_importance.to_csv(outputs["feature_importance"], index=False)
    confound_summary.to_csv(outputs["confound_control_summary"], index=False)
    if not c_table.empty:
        c_table.to_csv(outputs["inner_c_grid"], index=False)
    return outputs


def _write_experiment_plots(
    output_dir: Path,
    tag: str,
    config: ClinicalExperimentConfig,
    cv_scores: pd.DataFrame,
    oof_predictions: pd.DataFrame,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for scenario in pd.unique(cv_scores["scenario"]) if not cv_scores.empty else []:
        scenario_tag = safe_tag(str(scenario))
        roc_path = output_dir / f"{tag}_{scenario_tag}_cv_roc.png"
        plot_cv_roc(cv_scores, oof_predictions, scenario=scenario, title=f"{config.experiment_name}: {scenario} ROC", save_path=roc_path)
        outputs[f"{scenario}_roc"] = roc_path
    return outputs


def plot_cv_roc(
    cv_scores: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    *,
    scenario: str,
    title: str,
    save_path: str | Path | None = None,
):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    data = cv_scores[cv_scores["scenario"].eq(scenario)]
    for fold, fold_df in data.groupby("fold"):
        if fold_df["y"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(fold_df["y"], fold_df["score"])
        auc = roc_auc_score(fold_df["y"], fold_df["score"])
        ax.plot(fpr, tpr, alpha=0.65, linewidth=1.1, label=f"fold {fold}: AUC={auc:.3f}")
    oof = oof_predictions[oof_predictions["scenario"].eq(scenario)]
    if len(oof) and oof["y"].nunique() == 2:
        fpr, tpr, _ = roc_curve(oof["y"], oof["score"])
        auc = roc_auc_score(oof["y"], oof["score"])
        ax.plot(fpr, tpr, color="black", linewidth=2.4, label=f"OOF: AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.35)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def safe_tag(value: str) -> str:
    text = str(value).strip().replace("/", "_")
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() or ch in {"_", "-", "."} else "_")
    return "".join(out).strip("_") or "experiment"


__all__ = [
    "ClinicalExperimentConfig",
    "ClinicalExperimentResult",
    "ClinicalStatsWorkflowConfig",
    "ClinicalStatsWorkflowResult",
    "aggregate_feature_importance",
    "canonicalize_clinical_covariates",
    "clean_feature_frame",
    "clean_confound_column_sets",
    "display_clinical_stats_workflow",
    "literature_relevance",
    "make_confounds_available",
    "make_cv_splits",
    "metric_row",
    "parse_stats_feature_name",
    "plot_cv_roc",
    "run_clinical_binary_experiment",
    "run_clinical_stats_workflow",
    "safe_tag",
    "select_stats_feature_columns",
    "stratified_group_kfold_indices",
]

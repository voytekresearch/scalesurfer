from __future__ import annotations

from pathlib import Path
from typing import Sequence

from scalesurfer.experiments.clinical import ClinicalStatsWorkflowConfig


CLINICAL_STATS_FEATURE_SOURCE = "scalesurfer_stats_predicted"
CLINICAL_STATS_FEATURE_PRESET = "all"
CLINICAL_STATS_FEATURE_CONFOUND_COLS = ("aseg__global__eTIV",)
CLINICAL_STATS_FS_VERSION = 7
CLINICAL_STATS_SEED = 12345

CLINICAL_STATS_SCENARIOS = ("features", "residualized_features")
CLINICAL_STATS_PRIMARY_SCENARIO = "features"
CLINICAL_STATS_STRATIFY_COLS = ("study_id",)

CLINICAL_STATS_INCLUDE_STUDY_CONFOUND = False
CLINICAL_STATS_INCLUDE_SCANNER_CONFOUND = False
CLINICAL_STATS_REMOVE_FEATURE_CONFOUNDS_FROM_FEATURES = True

CLINICAL_STATS_CV_SPLITS = 5
CLINICAL_STATS_CV_REPEATS = 1
CLINICAL_STATS_INNER_SPLITS = 5
CLINICAL_STATS_TUNE_C = True
CLINICAL_STATS_C_GRID = (0.01, 0.1, 1.0, 10.0)
CLINICAL_STATS_TEST_SIZE = 0.20
CLINICAL_STATS_TOP_N_FEATURES = 35
CLINICAL_STATS_CLASS_WEIGHT = "balanced"
CLINICAL_STATS_LOGISTIC_REGRESSION_KWARGS = {
    "solver": "saga",
    "l1_ratio": 1.0,
}
CLINICAL_STATS_N_JOBS = -1
CLINICAL_STATS_PROGRESS = True


def make_clinical_stats_workflow_config(
    *,
    experiment_name: str,
    output_dir: str | Path,
    disease_context: str,
    positive_label_name: str,
    negative_label_name: str,
    age_cols: Sequence[str],
    sex_cols: Sequence[str],
    study_col: str | None,
    scanner_cols: Sequence[str] = (),
    protocol_cols: Sequence[str] = (),
) -> ClinicalStatsWorkflowConfig:
    """Create the shared clinical-stats workflow config used by all notebooks."""
    return ClinicalStatsWorkflowConfig(
        experiment_name=experiment_name,
        output_dir=output_dir,
        feature_source=CLINICAL_STATS_FEATURE_SOURCE,
        disease_context=disease_context,
        positive_label_name=positive_label_name,
        negative_label_name=negative_label_name,
        stats_feature_preset=CLINICAL_STATS_FEATURE_PRESET,
        age_cols=tuple(age_cols),
        sex_cols=tuple(sex_cols),
        study_col=study_col,
        scanner_cols=tuple(scanner_cols),
        protocol_cols=tuple(protocol_cols),
        feature_confound_cols=CLINICAL_STATS_FEATURE_CONFOUND_COLS,
        remove_feature_confounds_from_features=CLINICAL_STATS_REMOVE_FEATURE_CONFOUNDS_FROM_FEATURES,
        include_study_confound=CLINICAL_STATS_INCLUDE_STUDY_CONFOUND,
        include_scanner_confound=CLINICAL_STATS_INCLUDE_SCANNER_CONFOUND,
        scenarios=CLINICAL_STATS_SCENARIOS,
        primary_scenario=CLINICAL_STATS_PRIMARY_SCENARIO,
        stratify_cols=CLINICAL_STATS_STRATIFY_COLS,
        cv_splits=CLINICAL_STATS_CV_SPLITS,
        cv_repeats=CLINICAL_STATS_CV_REPEATS,
        inner_splits=CLINICAL_STATS_INNER_SPLITS,
        tune_c=CLINICAL_STATS_TUNE_C,
        c_grid=CLINICAL_STATS_C_GRID,
        logistic_regression_kwargs=dict(CLINICAL_STATS_LOGISTIC_REGRESSION_KWARGS),
        class_weight=CLINICAL_STATS_CLASS_WEIGHT,
        seed=CLINICAL_STATS_SEED,
        test_size=CLINICAL_STATS_TEST_SIZE,
        top_n_features=CLINICAL_STATS_TOP_N_FEATURES,
        n_jobs=CLINICAL_STATS_N_JOBS,
        progress=CLINICAL_STATS_PROGRESS,
    )


__all__ = [
    "CLINICAL_STATS_C_GRID",
    "CLINICAL_STATS_FEATURE_CONFOUND_COLS",
    "CLINICAL_STATS_FEATURE_PRESET",
    "CLINICAL_STATS_FEATURE_SOURCE",
    "CLINICAL_STATS_FS_VERSION",
    "CLINICAL_STATS_LOGISTIC_REGRESSION_KWARGS",
    "CLINICAL_STATS_N_JOBS",
    "CLINICAL_STATS_SCENARIOS",
    "CLINICAL_STATS_SEED",
    "make_clinical_stats_workflow_config",
]

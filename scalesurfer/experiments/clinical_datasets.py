from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from scalesurfer.stats import load_stats_feature_matrix


SCHIZOPHRENIA_HEALTHY_LABELS = {"control", "old control", "young control", "hc", "con"}
SCHIZOPHRENIA_DISEASE_LABELS = {
    "schizophrenia",
    "chronic sz",
    "early sz",
    "schizoaffective",
    "schz",
    "scz",
    "avh+",
    "avh-",
}

OPENNEURO_SCHIZOPHRENIA_DATASETS = {
    "ds000030": {
        "label_col": "diagnosis",
        "healthy": {"CONTROL"},
        "disease": {"SCHZ"},
    },
    "ds004302": {
        "label_col": "group",
        "healthy": {"HC"},
        "disease": {"AVH+", "AVH-"},
    },
    "ds000115": {
        "label_col": "condit",
        "healthy": {"CON"},
        "disease": {"SCZ"},
    },
}


def bids_label(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"^sub-", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^ses-", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^A-Za-z0-9]+", "", text)
    if not text:
        raise ValueError(f"Cannot build BIDS label from {value!r}")
    return text


def image_id_from_bids_path(path: str | Path) -> str:
    name = Path(path).name
    match = re.search(r"_run-([^_]+)_", name)
    if match:
        return match.group(1)
    if name.endswith(".nii.gz"):
        return name[:-7]
    return Path(name).stem


def load_prepared_bids_images(
    dataset_root: str | Path,
    *,
    dataset_id: str,
    subject_prefix: str | None = None,
) -> pd.DataFrame:
    """Load prepared BIDS T1w images from ``data/<dataset>/bids``.

    The preferred source is the ``bids_manifest.csv`` written by
    ``scripts/prepare_clinical_bids.py``. If the manifest is unavailable, the
    BIDS tree is indexed directly.
    """
    dataset_root = Path(dataset_root)
    manifest_path = dataset_root / "bids_manifest.csv"
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        if df.empty:
            return pd.DataFrame(columns=["subject_id", "dataset_id", "participant_id", "session_id", "image_id", "image"])
        df = df.rename(columns={"bids_path": "image"}).copy()
        df["dataset_id"] = dataset_id
        df["participant_id"] = df["participant_id"].map(bids_label)
        df["session_id"] = df["session_id"].map(bids_label)
        df["image_id"] = df["image_id"].map(bids_label)
    else:
        bids_dir = dataset_root / "bids"
        rows = []
        for image in sorted(bids_dir.glob("sub-*/ses-*/anat/*_T1w.nii*")):
            subject = image.parts[-4].removeprefix("sub-")
            session = image.parts[-3].removeprefix("ses-")
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "participant_id": bids_label(subject),
                    "session_id": bids_label(session),
                    "image_id": bids_label(image_id_from_bids_path(image)),
                    "image": str(image),
                }
            )
        df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["subject_id", "dataset_id", "participant_id", "session_id", "image_id", "image"])

    prefix = subject_prefix or dataset_id
    df["subject_id"] = (
        prefix
        + "__sub-"
        + df["participant_id"].astype(str)
        + "__ses-"
        + df["session_id"].astype(str)
        + "__run-"
        + df["image_id"].astype(str)
    )
    return df[["subject_id", "dataset_id", "participant_id", "session_id", "image_id", "image"]].drop_duplicates()


def ptid_from_adni_bids_subject(bids_subject_id: object) -> str:
    text = str(bids_subject_id).strip().removeprefix("sub-")
    match = re.fullmatch(r"(\d{3})S(\d+)", text)
    if match is None:
        raise ValueError(f"Cannot parse ADNI BIDS subject id {bids_subject_id!r}")
    return f"{match.group(1)}_S_{int(match.group(2)):04d}"


def load_adni_bids_images(
    bids_dir: str | Path,
    *,
    dataset_id: str = "adni",
    subject_prefix: str = "adni",
) -> pd.DataFrame:
    rows = []
    bids_dir = Path(bids_dir)
    for image in sorted(bids_dir.glob("sub-*/ses-*/anat/*_T1w.nii*")):
        bids_subject = image.parts[-4]
        bids_session = image.parts[-3]
        participant_id = ptid_from_adni_bids_subject(bids_subject)
        session_id = bids_label(bids_session.removeprefix("ses-"))
        image_id = bids_label(image_id_from_bids_path(image))
        rows.append(
            {
                "subject_id": f"{subject_prefix}__sub-{bids_label(participant_id)}__ses-{session_id}__run-{image_id}",
                "dataset_id": dataset_id,
                "participant_id": participant_id,
                "session_id": session_id,
                "image_id": image_id,
                "image": str(image),
            }
        )
    return pd.DataFrame(rows).drop_duplicates()


def load_classifier_table(
    csv_path: str | Path,
    *,
    dataset_id: str,
    positive_groups: Sequence[str],
    negative_groups: Sequence[str] = ("Control", "CN"),
    positive_name: str = "Disease",
    negative_name: str = "Healthy",
) -> pd.DataFrame:
    """Parse an IDA-style classifier CSV into canonical clinical metadata."""
    raw = pd.read_csv(csv_path)
    rename = {
        "Image Data ID": "image_id",
        "Subject": "participant_id",
        "Group": "clinical_label",
        "Sex": "sex",
        "Age": "age",
        "Visit": "visit",
        "Modality": "modality",
        "Description": "sequence_description",
        "Type": "image_type",
        "Acq Date": "acq_date",
        "Format": "image_format",
        "Downloaded": "downloaded",
    }
    df = raw.rename(columns=rename).copy()
    df["dataset_id"] = dataset_id
    df["participant_id"] = df["participant_id"].astype(str).str.strip()
    df["image_id"] = df["image_id"].map(bids_label)
    df["age"] = pd.to_numeric(df.get("age"), errors="coerce")
    adni_site = df["participant_id"].astype(str).str.extract(r"^(\d{3})_?S_?", expand=False)
    df["site_id"] = adni_site.where(adni_site.notna(), dataset_id)

    pos = {str(x).strip().lower() for x in positive_groups}
    neg = {str(x).strip().lower() for x in negative_groups}
    label_key = df["clinical_label"].astype(str).str.strip().str.lower()
    df["binary_label"] = np.where(label_key.isin(pos), positive_name, np.where(label_key.isin(neg), negative_name, pd.NA))
    df = df[df["binary_label"].notna()].copy()
    df["y"] = (df["binary_label"] == positive_name).astype(int)
    return df


def build_classifier_bids_cohort(
    *,
    dataset_root: str | Path,
    classifier_csv: str | Path,
    dataset_id: str,
    positive_groups: Sequence[str],
    negative_groups: Sequence[str] = ("Control", "CN"),
    positive_name: str = "Disease",
    negative_name: str = "Healthy",
    subject_prefix: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    images = load_prepared_bids_images(dataset_root, dataset_id=dataset_id, subject_prefix=subject_prefix)
    clinical = load_classifier_table(
        classifier_csv,
        dataset_id=dataset_id,
        positive_groups=positive_groups,
        negative_groups=negative_groups,
        positive_name=positive_name,
        negative_name=negative_name,
    )
    if images.empty:
        return images, clinical.iloc[0:0].copy()

    merged = images.merge(
        clinical,
        on=["dataset_id", "participant_id", "image_id"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        merged = images.merge(
            clinical.drop(columns=["image_id"]).drop_duplicates(["dataset_id", "participant_id"]),
            on=["dataset_id", "participant_id"],
            how="inner",
            validate="many_to_one",
        )
        merged["image_id"] = merged["image_id_x"]
        merged = merged.drop(columns=[col for col in ("image_id_x", "image_id_y") if col in merged.columns])

    clinical_inputs = merged[["subject_id", "dataset_id", "participant_id", "session_id", "image_id", "image"]].copy()
    clinical_participants = merged.copy()
    return clinical_inputs.reset_index(drop=True), clinical_participants.reset_index(drop=True)


def build_adni_classifier_bids_cohort(
    *,
    bids_dir: str | Path,
    classifier_csv: str | Path,
    dataset_id: str = "adni",
    positive_groups: Sequence[str] = ("AD",),
    negative_groups: Sequence[str] = ("CN",),
    positive_name: str = "AD",
    negative_name: str = "CN",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    images = load_adni_bids_images(bids_dir, dataset_id=dataset_id, subject_prefix=dataset_id)
    clinical = load_classifier_table(
        classifier_csv,
        dataset_id=dataset_id,
        positive_groups=positive_groups,
        negative_groups=negative_groups,
        positive_name=positive_name,
        negative_name=negative_name,
    )
    if images.empty:
        return images, clinical.iloc[0:0].copy()
    merged = images.merge(
        clinical.drop(columns=["image_id"]).drop_duplicates(["dataset_id", "participant_id"]),
        on=["dataset_id", "participant_id"],
        how="inner",
        validate="many_to_one",
    )
    clinical_inputs = merged[["subject_id", "dataset_id", "participant_id", "session_id", "image_id", "image"]].copy()
    clinical_participants = merged.copy()
    return clinical_inputs.reset_index(drop=True), clinical_participants.reset_index(drop=True)


def _read_demographics(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not df.empty and str(df.iloc[0].get("Anonymized ID", "")).strip().upper() == "ID":
        df = df.iloc[1:].copy()
    return df


def load_schizophrenia_demographics(path: str | Path, *, dataset_id: str) -> pd.DataFrame:
    df = _read_demographics(path).copy()
    df = df.rename(
        columns={
            "Anonymized ID": "participant_id",
            "Subject Type": "clinical_label",
            "Current Age": "age",
            "Gender": "sex",
            "Visit": "visit",
            "Sub Study Label": "sub_study_label",
        }
    )
    df["dataset_id"] = dataset_id
    df["participant_id"] = df["participant_id"].map(bids_label)
    df["age"] = pd.to_numeric(df.get("age"), errors="coerce")
    label_key = df["clinical_label"].astype(str).str.strip().str.lower()
    df["binary_label"] = np.where(
        label_key.isin(SCHIZOPHRENIA_DISEASE_LABELS),
        "Disease",
        np.where(label_key.isin(SCHIZOPHRENIA_HEALTHY_LABELS), "Healthy", pd.NA),
    )
    df["schz_group"] = np.where(df["binary_label"].eq("Disease"), "SCHZ", np.where(df["binary_label"].eq("Healthy"), "HC", pd.NA))
    return df[df["binary_label"].notna()].copy()


def build_local_schizophrenia_bids_cohort(
    *,
    dataset_root: str | Path,
    demographics_csv: str | Path,
    dataset_id: str,
    subject_prefix: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    images = load_prepared_bids_images(dataset_root, dataset_id=dataset_id, subject_prefix=subject_prefix)
    clinical = load_schizophrenia_demographics(demographics_csv, dataset_id=dataset_id)
    if images.empty:
        return images, clinical.iloc[0:0].copy()
    merged = images.merge(
        clinical,
        on=["dataset_id", "participant_id"],
        how="inner",
        validate="many_to_many",
    )
    if "visit" in merged.columns:
        visit_key = merged["visit"].astype(str).str.lower()
        session_key = merged["session_id"].astype(str).str.lower()
        baseline_match = (
            (visit_key.eq("b") & session_key.isin(["1", "visit1"]))
            | (visit_key.eq("bsgbaseline") & session_key.eq("bsgbaseline"))
        )
        if baseline_match.any():
            merged = merged[baseline_match].copy()
    clinical_inputs = merged[["subject_id", "dataset_id", "participant_id", "session_id", "image_id", "image"]].drop_duplicates().copy()
    clinical_participants = merged.drop_duplicates("subject_id").copy()
    return clinical_inputs.reset_index(drop=True), clinical_participants.reset_index(drop=True)


def build_openneuro_schizophrenia_cohort(
    *,
    notebook_dir: str | Path,
    dataset_ids: Sequence[str] = ("ds000030", "ds004302", "ds000115"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the local OpenNeuro schizophrenia cohorts already staged by notebooks."""
    notebook_dir = Path(notebook_dir)
    input_frames = []
    participant_frames = []
    for dataset_id in dataset_ids:
        if dataset_id not in OPENNEURO_SCHIZOPHRENIA_DATASETS:
            raise ValueError(f"Unknown OpenNeuro schizophrenia dataset {dataset_id!r}")
        cfg = OPENNEURO_SCHIZOPHRENIA_DATASETS[dataset_id]
        participant_path = notebook_dir / f"{dataset_id}_participants.tsv"
        image_root = notebook_dir / f"{dataset_id}_t1w"
        if not participant_path.exists() or not image_root.exists():
            continue
        participants = pd.read_csv(participant_path, sep="\t").copy()
        participants["dataset_id"] = dataset_id
        participants["participant_id"] = participants["participant_id"].astype(str)
        label_col = str(cfg["label_col"])
        label = participants[label_col].astype(str).str.strip()
        participants["clinical_label"] = label
        participants["label_source_col"] = label_col
        participants["binary_label"] = np.where(
            label.isin(cfg["disease"]),
            "Disease",
            np.where(label.isin(cfg["healthy"]), "Healthy", pd.NA),
        )
        participants["schz_group"] = np.where(
            participants["binary_label"].eq("Disease"),
            "SCHZ",
            np.where(participants["binary_label"].eq("Healthy"), "HC", pd.NA),
        )
        participants = participants[participants["binary_label"].notna()].copy()

        rows = []
        for image in sorted(image_root.glob("sub-*/anat/*_T1w.nii*")):
            participant_id = image.parts[-3]
            session_id = "none"
            rows.append(
                {
                    "subject_id": f"{dataset_id}__{participant_id}__ses-{session_id}",
                    "dataset_id": dataset_id,
                    "participant_id": participant_id,
                    "session_id": session_id,
                    "image_id": "T1w",
                    "image": str(image),
                }
            )
        inputs = pd.DataFrame(rows)
        merged = inputs.merge(participants, on=["dataset_id", "participant_id"], how="inner", validate="one_to_one")
        input_frames.append(inputs[inputs["subject_id"].isin(merged["subject_id"])])
        participant_frames.append(merged)

    input_df = pd.concat(input_frames, ignore_index=True) if input_frames else pd.DataFrame()
    participant_df = pd.concat(participant_frames, ignore_index=True) if participant_frames else pd.DataFrame()
    return input_df, participant_df


def concat_cohorts(cohorts: Sequence[tuple[pd.DataFrame, pd.DataFrame]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    inputs = [cohort[0] for cohort in cohorts if not cohort[0].empty]
    participants = [cohort[1] for cohort in cohorts if not cohort[1].empty]
    input_df = pd.concat(inputs, ignore_index=True) if inputs else pd.DataFrame()
    participant_df = pd.concat(participants, ignore_index=True) if participants else pd.DataFrame()
    return input_df, participant_df


def build_stats_model_df(
    *,
    subjects_dir: str | Path,
    clinical_inputs: pd.DataFrame,
    clinical_participants: pd.DataFrame,
    fill_value: float = 0.0,
    label_col: str = "y",
    positive_label: str = "Disease",
    negative_label: str = "Healthy",
) -> tuple[pd.DataFrame, list[str]]:
    subject_ids = clinical_inputs["subject_id"].astype(str).tolist()
    features, feature_cols = load_stats_feature_matrix(
        subjects_dir,
        subjects=subject_ids,
        fill_value=fill_value,
    )

    meta = clinical_participants.rename(columns={"subject_id": "subject"}).copy()
    if "subject" not in meta.columns and "subject_id" in clinical_participants.columns:
        meta["subject"] = clinical_participants["subject_id"]
    meta = meta.drop_duplicates("subject")

    model_df = meta.merge(features, on="subject", how="inner", validate="one_to_one")
    if label_col in model_df.columns:
        model_df["y"] = model_df[label_col].astype(int)
    elif "binary_label" in model_df.columns:
        model_df["y"] = model_df["binary_label"].map({negative_label: 0, positive_label: 1}).astype(int)
    else:
        raise ValueError("Expected either an explicit label column or binary_label in clinical_participants.")
    model_df[feature_cols] = model_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(fill_value)
    return model_df, feature_cols


def summarize_cohort(model_df: pd.DataFrame) -> pd.DataFrame:
    cols = [col for col in ("dataset_id", "binary_label") if col in model_df.columns]
    if not cols:
        return pd.DataFrame({"n": [len(model_df)]})
    return model_df.groupby(cols, dropna=False).size().rename("n").reset_index()

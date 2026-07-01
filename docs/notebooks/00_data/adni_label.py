from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd
from pandas.errors import PerformanceWarning

warnings.simplefilter("ignore", PerformanceWarning)

TABLE_DIR = Path("adni_tables")
OUT_DIR = Path("adni_out")
OUT_DIR.mkdir(exist_ok=True)

BASELINE_VISITS = {
    # Clinical baseline is VISCODE2 == bl. Baseline MRI is often stored as
    # screening/scmri in the image tables, so keep those as baseline MRI visits.
    "bl": 0,
    "scmri": 1,
    "sc": 2,
}

FS_TABLES = [
    ("UCSFFSX7.csv", 0),
    ("UCSFFSX6.csv", 1),
    ("UCSFFSX51.csv", 2),
    ("UCSFFSX.csv", 3),
]

REQUESTED_FREESURFER_FILES = (
    "stats/*.stats; mri/orig.mgz; mri/rawavg.mgz; "
    "mri/aparc+aseg.mgz; FreeSurfer version/build info"
)
BATCH_SIZE = 100


def read_table(name: str) -> pd.DataFrame:
    path = TABLE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required ADNI table: {path}")

    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.upper()
    return df


def clean_id_value(value):
    if pd.isna(value):
        return pd.NA

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return pd.NA

    text = text.removeprefix("I").removeprefix("S")
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def clean_id_series(series: pd.Series) -> pd.Series:
    return series.map(clean_id_value).astype("string")


def image_search_terms(series: pd.Series) -> pd.Series:
    return clean_id_series(series)


def series_search_terms(series: pd.Series) -> pd.Series:
    return clean_id_series(series)


def normalize_visit(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def visit_priority(series: pd.Series) -> pd.Series:
    return normalize_visit(series).map(BASELINE_VISITS).fillna(99).astype(int)


def r_day_to_iso(series: pd.Series) -> pd.Series:
    date = pd.to_datetime(
        pd.to_numeric(series, errors="coerce"),
        unit="D",
        origin="1970-01-01",
    )
    return date.dt.strftime("%Y-%m-%d").astype("string")


def normalize_diagnosis(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").map({
        1: "CN",
        2: "MCI",
        3: "AD",
    })

    text = series.astype("string").str.strip().str.upper()
    text_map = {
        "CN": "CN",
        "COGNITIVELY NORMAL": "CN",
        "NORMAL": "CN",
        "MCI": "MCI",
        "DEMENTIA": "AD",
        "AD": "AD",
        "ALZHEIMER DISEASE": "AD",
        "ALZHEIMER'S DISEASE": "AD",
    }
    return numeric.fillna(text.map(text_map)).astype("string")


def nonempty_unique(values: pd.Series) -> list[str]:
    ids = [value for value in clean_id_series(values).dropna().unique().tolist()]
    return sorted(ids, key=lambda value: int(value) if value.isdigit() else value)


def unique_text_values(values: pd.Series) -> list[str]:
    ids = [str(value).strip() for value in values.dropna().astype(str).unique().tolist()]
    return sorted(value for value in ids if value)


def chunks(values: list[str], size: int = BATCH_SIZE) -> list[list[str]]:
    return [values[start:start + size] for start in range(0, len(values), size)]


def write_batch_file(values: list[str], stem: str) -> None:
    rows = [
        {
            "batch": index,
            "n": len(batch),
            "values": ",".join(batch),
        }
        for index, batch in enumerate(chunks(values), start=1)
    ]
    pd.DataFrame(rows).to_csv(OUT_DIR / f"{stem}.csv", index=False)
    (OUT_DIR / f"{stem}.txt").write_text(
        "\n".join(row["values"] for row in rows) + ("\n" if rows else "")
    )


def write_image_id_list(df: pd.DataFrame, column: str, stem: str) -> None:
    ids = nonempty_unique(df[column])
    pd.DataFrame({
        "imageuid": ids,
        "image_search_term": ids,
    }).to_csv(OUT_DIR / f"{stem}.csv", index=False)
    (OUT_DIR / f"{stem}.txt").write_text("\n".join(ids) + ("\n" if ids else ""))
    (OUT_DIR / f"{stem}_search_terms.txt").write_text(
        "\n".join(ids) + ("\n" if ids else "")
    )
    write_batch_file(ids, f"{stem}_batches")
    write_batch_file(ids, f"{stem}_search_term_batches")


def write_series_id_list(df: pd.DataFrame, column: str, stem: str) -> None:
    ids = nonempty_unique(df[column])
    pd.DataFrame({
        "loni_series_id": ids,
        "series_search_term": ids,
    }).to_csv(OUT_DIR / f"{stem}.csv", index=False)
    (OUT_DIR / f"{stem}.txt").write_text("\n".join(ids) + ("\n" if ids else ""))
    (OUT_DIR / f"{stem}_search_terms.txt").write_text(
        "\n".join(ids) + ("\n" if ids else "")
    )
    write_batch_file(ids, f"{stem}_batches")
    write_batch_file(ids, f"{stem}_search_term_batches")


def build_baseline_labels() -> tuple[pd.DataFrame, pd.DataFrame]:
    dx = read_table("DXSUM.csv")
    visit = normalize_visit(dx["VISCODE2"] if "VISCODE2" in dx.columns else dx["VISCODE"])
    dx = dx[visit.eq("bl")].copy()

    dx["RID"] = clean_id_series(dx["RID"])
    dx["PTID"] = dx["PTID"].astype("string")
    dx["DX_EXAMDATE"] = dx["EXAMDATE"]
    dx["DX_EXAMDATE_ISO"] = r_day_to_iso(dx["EXAMDATE"])
    dx["DX_VISCODE"] = dx["VISCODE"].astype("string")
    dx["DX_VISCODE2"] = dx["VISCODE2"].astype("string")
    dx["dx_name"] = normalize_diagnosis(dx["DIAGNOSIS"])
    dx["label"] = dx["dx_name"].map({"CN": 0, "AD": 1}).astype("Int64")

    keep = [
        "RID",
        "PTID",
        "DX_VISCODE",
        "DX_VISCODE2",
        "DX_EXAMDATE",
        "DX_EXAMDATE_ISO",
        "DIAGNOSIS",
        "dx_name",
        "label",
        "DXCONFID",
        "DXDDUE",
        "ORIGPROT",
        "COLPROT",
    ]
    keep = [col for col in keep if col in dx.columns]

    labels = (
        dx[dx["dx_name"].notna()]
        .sort_values(["RID", "DX_EXAMDATE"])
        .drop_duplicates("RID", keep="first")[keep]
        .copy()
    )
    labels_ad_cn = labels[labels["dx_name"].isin(["CN", "AD"])].copy()
    return labels, labels_ad_cn


def build_t1w_candidates() -> tuple[pd.DataFrame, pd.DataFrame]:
    qc = read_table("MRIQC.csv")
    qc = qc[qc["SERIESTYPE"].astype("string").str.strip().str.lower().eq("t1w")].copy()
    qc["PTID"] = qc["PARTICIPANTID"].astype("string")
    qc["T1_IMAGEUID"] = clean_id_series(qc["IMAGE_ID"])
    qc["T1_IMAGE_SEARCH"] = image_search_terms(qc["IMAGE_ID"])
    qc["T1_LONI_SERIES"] = clean_id_series(qc["LONISERIES"])
    qc["T1_SERIES_SEARCH"] = series_search_terms(qc["LONISERIES"])
    qc["T1_VISIT"] = normalize_visit(qc["VISCODE2"])
    qc["T1_STUDYDATE"] = qc["STUDYDATE"]
    qc["T1_STUDYDATE_ISO"] = r_day_to_iso(qc["STUDYDATE"])
    qc["T1_SERIES_DESCRIPTION"] = qc["SERIESDESCRIPTION"].astype("string")
    qc["T1_SERIES_NUMBER"] = pd.to_numeric(qc["SERIESNUMBER"], errors="coerce")
    qc["T1_SERIES_TIME"] = pd.to_numeric(qc["SERIESTIME"], errors="coerce")
    qc["T1_PROTOCOL_PHASE"] = qc["MRIPROTOCOLPHASE"].astype("string")
    qc["T1_ACQUISITION_TYPE"] = qc["ACQUISITIONTYPE"].astype("string")
    qc["T1_SCANNER_MANUFACTURER"] = qc["SCANNERMANUFACTURER"].astype("string")
    qc["T1_SCANNER_MODEL"] = qc["SCANNERMODEL"].astype("string")
    qc["T1_FIELD_STRENGTH"] = pd.to_numeric(qc["MAGNETICFIELDSTRENGTH"], errors="coerce")
    qc["T1_SERIES_INSTANCE_UID"] = qc["SERIESINSTANCEUID"].astype("string")
    qc["T1_STUDY_INSTANCE_UID"] = qc["STUDYINSTANCEUID"].astype("string")
    qc = qc[qc["T1_VISIT"].isin(BASELINE_VISITS) & qc["T1_IMAGEUID"].notna()].copy()

    rank = read_table("MRIMPRANK.csv")
    rank["PTID"] = rank["PTID"].astype("string")
    rank["T1_VISIT"] = normalize_visit(rank["VISCODE"])
    rank["T1_LONI_SERIES"] = clean_id_series(rank["LONIUID"])
    rank["T1_RANK"] = pd.to_numeric(rank["RANK"], errors="coerce")
    rank["T1_RANK_SCAN"] = rank["SCAN"].astype("string")
    rank = rank[
        ["PTID", "T1_VISIT", "T1_LONI_SERIES", "T1_RANK", "T1_RANK_SCAN"]
    ].drop_duplicates(["PTID", "T1_VISIT", "T1_LONI_SERIES"])

    qc = qc.merge(rank, on=["PTID", "T1_VISIT", "T1_LONI_SERIES"], how="left")
    qc["T1_VISIT_PRIORITY"] = visit_priority(qc["T1_VISIT"])
    qc["T1_RANK_SORT"] = qc["T1_RANK"].fillna(99)
    qc["T1_REPEAT_SORT"] = qc["T1_SERIES_DESCRIPTION"].str.contains(
        "repeat|repea", case=False, na=False, regex=True
    ).astype(int)
    qc["T1_STUDYDATE_SORT"] = pd.to_numeric(qc["T1_STUDYDATE"], errors="coerce").fillna(10**9)
    qc["T1_IMAGE_SORT"] = pd.to_numeric(qc["T1_IMAGEUID"], errors="coerce").fillna(10**12)

    sort_cols = [
        "PTID",
        "T1_VISIT_PRIORITY",
        "T1_RANK_SORT",
        "T1_REPEAT_SORT",
        "T1_STUDYDATE_SORT",
        "T1_IMAGE_SORT",
    ]
    qc = qc.sort_values(sort_cols).copy()
    selected = qc.drop_duplicates("PTID", keep="first").copy()

    output_cols = [
        "PTID",
        "T1_IMAGEUID",
        "T1_IMAGE_SEARCH",
        "T1_LONI_SERIES",
        "T1_SERIES_SEARCH",
        "T1_VISIT",
        "T1_STUDYDATE",
        "T1_STUDYDATE_ISO",
        "T1_RANK",
        "T1_RANK_SCAN",
        "T1_SERIES_DESCRIPTION",
        "T1_SERIES_NUMBER",
        "T1_SERIES_TIME",
        "T1_PROTOCOL_PHASE",
        "T1_ACQUISITION_TYPE",
        "T1_SCANNER_MANUFACTURER",
        "T1_SCANNER_MODEL",
        "T1_FIELD_STRENGTH",
        "T1_SERIES_INSTANCE_UID",
        "T1_STUDY_INSTANCE_UID",
    ]
    return qc[output_cols].copy(), selected[output_cols].copy()


def fs_version_info(table_name: str, fsver: pd.Series, version: pd.Series) -> pd.Series:
    inferred = {
        "UCSFFSX7": "FreeSurfer 7.x table",
        "UCSFFSX6": "FreeSurfer 6.x table",
        "UCSFFSX51": "FreeSurfer 5.1 table",
        "UCSFFSX": "legacy UCSFFSX table",
    }.get(table_name, table_name)

    info = fsver.astype("string")
    info = info.mask(info.isna() | info.str.lower().eq("nan"), inferred)
    version_text = version.astype("string")
    return info + "; VERSION=" + version_text.fillna("")


def build_freesurfer_candidates(labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_ptids = labels[["RID", "PTID"]].drop_duplicates()
    tables: list[pd.DataFrame] = []

    for table_file, table_priority in FS_TABLES:
        path = TABLE_DIR / table_file
        if not path.exists():
            continue

        fs = pd.read_csv(path, low_memory=False).copy()
        fs.columns = fs.columns.str.upper()
        fs["RID"] = clean_id_series(fs["RID"])
        fs["FS_TABLE"] = path.stem
        fs["FS_TABLE_PRIORITY"] = table_priority
        fs["FS_PTID"] = fs["PTID"].astype("string") if "PTID" in fs.columns else pd.NA

        missing_ptid = fs["FS_PTID"].isna()
        if missing_ptid.any():
            fs = fs.merge(label_ptids, on="RID", how="left", suffixes=("", "_LABEL"))
            fs["FS_PTID"] = fs["FS_PTID"].fillna(fs["PTID"])

        visit2 = fs["VISCODE2"] if "VISCODE2" in fs.columns else pd.Series(pd.NA, index=fs.index)
        fs["FS_VISIT"] = normalize_visit(visit2.fillna(fs["VISCODE"]))
        fs["FS_EXAMDATE"] = fs["EXAMDATE"]
        fs["FS_EXAMDATE_ISO"] = r_day_to_iso(fs["EXAMDATE"])
        fs["FS_RUNDATE"] = fs["RUNDATE"] if "RUNDATE" in fs.columns else pd.NA
        fs["FS_RUNDATE_ISO"] = r_day_to_iso(fs["FS_RUNDATE"])
        fs["FS_IMAGEUID"] = clean_id_series(fs["IMAGEUID"])
        fs["FS_IMAGE_SEARCH"] = image_search_terms(fs["IMAGEUID"])
        fs["FS_LONI_SERIES"] = (
            clean_id_series(fs["LONIUID"]) if "LONIUID" in fs.columns else pd.Series(pd.NA, index=fs.index)
        )
        fs["FS_SERIES_SEARCH"] = series_search_terms(fs["FS_LONI_SERIES"])
        fs["FS_STATUS"] = fs["STATUS"].astype("string") if "STATUS" in fs.columns else pd.NA
        fs["FS_OVERALLQC"] = fs["OVERALLQC"].astype("string") if "OVERALLQC" in fs.columns else pd.NA
        fs["FSVER_RAW"] = fs["FSVER"] if "FSVER" in fs.columns else pd.Series(pd.NA, index=fs.index)
        fs["VERSION_RAW"] = fs["VERSION"] if "VERSION" in fs.columns else pd.Series(pd.NA, index=fs.index)
        fs["FS_VERSION_INFO"] = fs_version_info(path.stem, fs["FSVER_RAW"], fs["VERSION_RAW"])
        fs["FS_REQUESTED_FILES"] = REQUESTED_FREESURFER_FILES
        fs["FS_FILE_LEVEL_IDS_IN_TABLES"] = False

        fs = fs[fs["FS_VISIT"].isin(BASELINE_VISITS) & fs["FS_IMAGEUID"].notna()].copy()
        tables.append(fs)

    if not tables:
        empty = pd.DataFrame()
        return empty, empty

    fs = pd.concat(tables, ignore_index=True, sort=False)
    status_rank = {"complete": 0, "partial": 1}
    qc_rank = {"pass": 0, "partial": 1, "hippocampus only": 2, "fail": 3}
    fs["FS_VISIT_PRIORITY"] = visit_priority(fs["FS_VISIT"])
    fs["FS_STATUS_SORT"] = (
        fs["FS_STATUS"].astype("string").str.strip().str.lower().map(status_rank).fillna(9).astype(int)
    )
    fs["FS_QC_SORT"] = (
        fs["FS_OVERALLQC"].astype("string").str.strip().str.lower().map(qc_rank).fillna(8).astype(int)
    )
    fs["FS_IMAGE_SORT"] = pd.to_numeric(fs["FS_IMAGEUID"], errors="coerce").fillna(10**12)
    fs = fs.sort_values([
        "RID",
        "FS_VISIT_PRIORITY",
        "FS_STATUS_SORT",
        "FS_QC_SORT",
        "FS_TABLE_PRIORITY",
        "FS_IMAGE_SORT",
    ]).copy()
    selected = fs.drop_duplicates("RID", keep="first").copy()
    return fs, selected


def manifest_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "RID",
        "PTID",
        "dx_name",
        "label",
        "DIAGNOSIS",
        "DX_VISCODE",
        "DX_VISCODE2",
        "DX_EXAMDATE_ISO",
        "T1_IMAGEUID",
        "T1_IMAGE_SEARCH",
        "T1_LONI_SERIES",
        "T1_SERIES_SEARCH",
        "T1_VISIT",
        "T1_STUDYDATE_ISO",
        "T1_RANK",
        "T1_SERIES_DESCRIPTION",
        "T1_PROTOCOL_PHASE",
        "T1_SCANNER_MANUFACTURER",
        "T1_SCANNER_MODEL",
        "T1_FIELD_STRENGTH",
        "FS_IMAGEUID",
        "FS_IMAGE_SEARCH",
        "FS_LONI_SERIES",
        "FS_SERIES_SEARCH",
        "FS_VISIT",
        "FS_EXAMDATE_ISO",
        "FS_RUNDATE_ISO",
        "FS_TABLE",
        "FS_VERSION_INFO",
        "FS_STATUS",
        "FS_OVERALLQC",
        "FS_REQUESTED_FILES",
        "FS_FILE_LEVEL_IDS_IN_TABLES",
        "DOWNLOAD_IMAGEUID",
        "DOWNLOAD_IMAGE_SEARCH",
        "DOWNLOAD_IMAGE_SOURCE",
    ]
    return [col for col in preferred if col in df.columns]


def build_manifest(
    labels: pd.DataFrame,
    t1_selected: pd.DataFrame,
    fs_selected: pd.DataFrame,
) -> pd.DataFrame:
    manifest = labels.merge(t1_selected, on="PTID", how="left")
    fs_cols = [
        "RID",
        "FS_PTID",
        "FS_IMAGEUID",
        "FS_IMAGE_SEARCH",
        "FS_LONI_SERIES",
        "FS_SERIES_SEARCH",
        "FS_VISIT",
        "FS_EXAMDATE",
        "FS_EXAMDATE_ISO",
        "FS_RUNDATE",
        "FS_RUNDATE_ISO",
        "FS_TABLE",
        "FS_VERSION_INFO",
        "FS_STATUS",
        "FS_OVERALLQC",
        "FS_REQUESTED_FILES",
        "FS_FILE_LEVEL_IDS_IN_TABLES",
    ]
    fs_cols = [col for col in fs_cols if col in fs_selected.columns]
    manifest = manifest.merge(fs_selected[fs_cols], on="RID", how="left")

    manifest["DOWNLOAD_IMAGEUID"] = manifest["T1_IMAGEUID"].fillna(manifest["FS_IMAGEUID"])
    manifest["DOWNLOAD_IMAGE_SEARCH"] = image_search_terms(manifest["DOWNLOAD_IMAGEUID"])
    manifest["DOWNLOAD_IMAGE_SOURCE"] = "MRIQC selected baseline T1w image_id"
    fallback = manifest["T1_IMAGEUID"].isna() & manifest["FS_IMAGEUID"].notna()
    manifest.loc[fallback, "DOWNLOAD_IMAGE_SOURCE"] = "UCSF FreeSurfer source IMAGEUID fallback"
    manifest.loc[manifest["DOWNLOAD_IMAGEUID"].isna(), "DOWNLOAD_IMAGE_SOURCE"] = pd.NA

    return manifest[manifest_columns(manifest)].copy()


def attach_labels_to_candidates(
    labels: pd.DataFrame,
    t1_candidates: pd.DataFrame,
    fs_candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_cols = [
        "RID",
        "PTID",
        "dx_name",
        "label",
        "DIAGNOSIS",
        "DX_VISCODE",
        "DX_VISCODE2",
        "DX_EXAMDATE_ISO",
    ]
    label_cols = [col for col in label_cols if col in labels.columns]

    t1 = labels[label_cols].merge(t1_candidates, on="PTID", how="inner")
    fs = labels[label_cols].merge(fs_candidates, on="RID", how="inner", suffixes=("", "_FS"))
    return t1, fs


def write_readme(
    all_manifest: pd.DataFrame,
    ad_cn_manifest: pd.DataFrame,
    fs_candidates: pd.DataFrame,
) -> None:
    lines = [
        "ADNI baseline download manifest",
        "",
        "Key files:",
        "- adni_baseline_all_participants_manifest.csv: one baseline diagnosis row per RID, all CN/MCI/AD labels.",
        "- adni_baseline_ad_vs_cn_manifest.csv: AD-vs-CN subset.",
        "- t1w_imageuid_download_list.txt: selected baseline T1w MRIQC image_id values.",
        "- t1w_imageuid_download_list_batches.txt: paste-ready comma-separated batches of 100 numeric T1w IMAGEID values.",
        "- adni_baseline_ad_vs_cn_ptid_batches.txt: paste-ready comma-separated batches of 100 subject IDs for Advanced Image Search.",
        "- freesurfer_ad_vs_cn_ptid_batches.txt: paste-ready subject-ID batches restricted to participants with matched UCSF FreeSurfer rows.",
        "- freesurfer_source_imageuid_download_list.txt: selected UCSF FreeSurfer source IMAGEUID values.",
        "- baseline_download_imageuid_list.txt: selected T1w image ID when available, otherwise the FreeSurfer source IMAGEUID fallback.",
        "",
        "Important limitation:",
        "The ADNIMERGE/UCSF CSV tables provide FreeSurfer summary, QC, version, source IMAGEUID, and LONI series metadata.",
        "They do not provide file-level download paths for stats/*.stats, mri/orig.mgz, mri/rawavg.mgz, or mri/aparc+aseg.mgz.",
        "Use the manifest columns PTID, RID, FS_IMAGEUID, FS_IMAGE_SEARCH, FS_LONI_SERIES, and FS_SERIES_SEARCH to look for matching",
        "FreeSurfer derivative packages in ADNI/IDA. If those packages are not exposed in the archive, rerunning FreeSurfer from the T1w images",
        "or contacting ADNI support is the next step.",
        "",
        "Web interface notes checked 2026-06-08:",
        "- Use numeric IMAGEID values, not I-prefixed values.",
        "- ADNI FAQ: CSV files may include unique LONI IMAGEID values that can be used to search the IDA image repository.",
        "- ADNI Ask the Experts documents the scalable image route: Advanced Image Search -> Subject -> comma-separated subject IDs.",
        "- IDA User Manual, June 2026: Simple Image Search returns original/unprocessed images; processed images require Advanced Image Search.",
        "- IDA User Manual, June 2026: images must be added to a Data Collection before download.",
        "",
        "References:",
        "- https://adni.loni.usc.edu/help-faqs/faqs/",
        "- https://adni.loni.usc.edu/support/experts-knowledge-base/question/?QID=1711",
        "- https://ida.loni.usc.edu/explore/jsp/support/IDA_User_Manual.pdf",
        "",
        "Counts:",
        f"- all baseline labelled participants: {len(all_manifest)}",
        f"- all with selected T1w image_id: {all_manifest['T1_IMAGEUID'].notna().sum()}",
        f"- all with selected FreeSurfer source IMAGEUID: {all_manifest['FS_IMAGEUID'].notna().sum()}",
        f"- AD/CN baseline participants: {len(ad_cn_manifest)}",
        f"- AD/CN with selected T1w image_id: {ad_cn_manifest['T1_IMAGEUID'].notna().sum()}",
        f"- AD/CN with selected FreeSurfer source IMAGEUID: {ad_cn_manifest['FS_IMAGEUID'].notna().sum()}",
        "",
        "Selected FreeSurfer tables in the candidate pool:",
    ]
    if "FS_TABLE" in fs_candidates.columns:
        for table, count in fs_candidates["FS_TABLE"].value_counts(dropna=False).items():
            lines.append(f"- {table}: {count}")
    (OUT_DIR / "README_adni_download_manifest.txt").write_text("\n".join(lines) + "\n")


labels_all, labels_ad_cn = build_baseline_labels()
t1_candidates, t1_selected = build_t1w_candidates()
fs_candidates, fs_selected = build_freesurfer_candidates(labels_all)

all_manifest = build_manifest(labels_all, t1_selected, fs_selected)
ad_cn_manifest = build_manifest(labels_ad_cn, t1_selected, fs_selected)
all_t1_candidates, all_fs_candidates = attach_labels_to_candidates(
    labels_all,
    t1_candidates,
    fs_candidates,
)
ad_cn_t1_candidates, ad_cn_fs_candidates = attach_labels_to_candidates(
    labels_ad_cn,
    t1_candidates,
    fs_candidates,
)

all_manifest.to_csv(OUT_DIR / "adni_baseline_all_participants_manifest.csv", index=False)
ad_cn_manifest.to_csv(OUT_DIR / "adni_baseline_ad_vs_cn_manifest.csv", index=False)
ad_cn_manifest.to_csv(OUT_DIR / "adni_baseline_ad_vs_cn_with_freesurfer.csv", index=False)

all_t1_candidates.to_csv(OUT_DIR / "adni_baseline_all_participants_t1w_candidates.csv", index=False)
all_fs_candidates.to_csv(OUT_DIR / "adni_baseline_all_participants_freesurfer_candidates.csv", index=False)
ad_cn_t1_candidates.to_csv(OUT_DIR / "adni_baseline_ad_vs_cn_t1w_candidates.csv", index=False)
ad_cn_fs_candidates.to_csv(OUT_DIR / "adni_baseline_ad_vs_cn_freesurfer_candidates.csv", index=False)

write_image_id_list(ad_cn_manifest, "T1_IMAGEUID", "t1w_imageuid_download_list")
write_image_id_list(ad_cn_manifest, "T1_IMAGEUID", "imageuid_download_list")
write_image_id_list(ad_cn_manifest, "FS_IMAGEUID", "freesurfer_source_imageuid_download_list")
write_image_id_list(ad_cn_manifest, "DOWNLOAD_IMAGEUID", "baseline_download_imageuid_list")
write_series_id_list(ad_cn_manifest, "T1_LONI_SERIES", "t1w_loni_series_list")
write_series_id_list(ad_cn_manifest, "FS_LONI_SERIES", "freesurfer_source_loni_series_list")
write_batch_file(unique_text_values(ad_cn_manifest["PTID"]), "adni_baseline_ad_vs_cn_ptid_batches")
write_batch_file(unique_text_values(all_manifest["PTID"]), "adni_baseline_all_participants_ptid_batches")
write_batch_file(
    unique_text_values(ad_cn_manifest.loc[ad_cn_manifest["FS_IMAGEUID"].notna(), "PTID"]),
    "freesurfer_ad_vs_cn_ptid_batches",
)
(OUT_DIR / "freesurfer_image_processing_descriptions.txt").write_text(
    "\n".join([
        "FreeSurfer Cross-Sectional Processing aparc+aseg",
        "FreeSurfer Cross-Sectional Processing aseg",
        "FreeSurfer Cross-Sectional Processing brainmask",
        "FreeSurfer Cross-Sectional Processing orig",
        "FreeSurfer Cross-Sectional Processing rawavg",
    ]) + "\n"
)

feature_cols = [
    col for col in fs_selected.columns
    if col.startswith("ST") and pd.api.types.is_numeric_dtype(fs_selected[col])
]
fs_features = labels_ad_cn[["RID", "PTID", "dx_name", "label"]].merge(
    fs_selected[["RID", "FS_TABLE", "FS_IMAGEUID", "FS_VERSION_INFO"] + feature_cols],
    on="RID",
    how="inner",
)
fs_features.to_csv(OUT_DIR / "freesurfer_features_ad_vs_cn.csv", index=False)

write_readme(all_manifest, ad_cn_manifest, fs_candidates)

print("\nDONE")
print("\nBaseline labels:")
print(labels_all["dx_name"].value_counts())
print("\nAD/CN manifest:")
print(ad_cn_manifest["dx_name"].value_counts())
print(f"Subjects: {len(ad_cn_manifest)}")
print(f"Selected T1w image IDs: {ad_cn_manifest['T1_IMAGEUID'].notna().sum()}")
print(f"Selected FreeSurfer source IMAGEUIDs: {ad_cn_manifest['FS_IMAGEUID'].notna().sum()}")
print(f"FreeSurfer feature rows: {len(fs_features)}")

print("\nWrote:")
for path in [
    OUT_DIR / "adni_baseline_all_participants_manifest.csv",
    OUT_DIR / "adni_baseline_ad_vs_cn_manifest.csv",
    OUT_DIR / "adni_baseline_ad_vs_cn_t1w_candidates.csv",
    OUT_DIR / "adni_baseline_ad_vs_cn_freesurfer_candidates.csv",
    OUT_DIR / "t1w_imageuid_download_list.txt",
    OUT_DIR / "t1w_imageuid_download_list_batches.txt",
    OUT_DIR / "adni_baseline_ad_vs_cn_ptid_batches.txt",
    OUT_DIR / "freesurfer_ad_vs_cn_ptid_batches.txt",
    OUT_DIR / "freesurfer_source_imageuid_download_list.txt",
    OUT_DIR / "baseline_download_imageuid_list.txt",
    OUT_DIR / "README_adni_download_manifest.txt",
]:
    print(f"  {path}")

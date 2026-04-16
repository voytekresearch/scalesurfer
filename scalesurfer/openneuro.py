from __future__ import annotations

import gzip
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from threading import local
from typing import Any, Callable, Iterable, Optional, Sequence

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)

try:
    from tqdm.notebook import tqdm
except Exception:  # pragma: no cover
    def tqdm(x=None, total=None, desc=None, leave=True):
        return x if x is not None else range(total or 0)


_BUCKET = "openneuro.org"
_REGION = "us-east-1"
_thread_local = local()

_DATASET_RE = re.compile(r"^(ds\d+)/")
_SUB_RE = re.compile(r"/(sub-[^/]+)")
_SES_RE = re.compile(r"/(ses-[^/]+)")
_T1W_RE = re.compile(r"_T1w\.nii(\.gz)?$", re.IGNORECASE)
_NUMBERED_MGZ_RE = re.compile(r"^\d{3}\.mgz$", re.IGNORECASE)
_FS_ROOT_RE = re.compile(r"^(.*?)/(?:mri|surf|stats|label|scripts|touch|tmp|bem)(?:/|$)")

PATTERNS = [
    re.compile(r"stable-pub-v(?P<ver>[4-8]\.\d+(?:\.\d+)?)", re.I),
    re.compile(r"freesurfer[^\n]*?-v(?P<ver>[4-8]\.\d+(?:\.\d+)?)", re.I),
    re.compile(r"freesurfer[^\n]*?-(?P<ver>[4-8]\.\d+(?:\.\d+)?)", re.I),
    re.compile(r"\bv(?P<ver>[4-8]\.\d+(?:\.\d+)?)\b", re.I),
    re.compile(r"\b(?P<ver>[4-8]\.\d+(?:\.\d+)?)\b"),
]


def _safe_mkdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _thread_s3_client():
    client = getattr(_thread_local, "s3_client", None)
    if client is None:
        client = boto3.client(
            "s3",
            region_name=_REGION,
            config=Config(
                signature_version=UNSIGNED,
                retries={"max_attempts": 10, "mode": "standard"},
            ),
        )
        _thread_local.s3_client = client
    return client


@dataclass(frozen=True)
class OpenNeuroObject:
    dataset_id: str
    key: str
    size: int
    last_modified: str


@dataclass(frozen=True)
class AparcT1wPair:
    dataset_id: str
    subject: Optional[str]
    session: Optional[str]
    aparc_key: str
    t1w_candidates: tuple[str, ...]


@dataclass(frozen=True)
class AparcOrigPair:
    dataset_id: str
    subject: Optional[str]
    session: Optional[str]
    aparc_key: str
    orig_key: Optional[str]


@dataclass(frozen=True)
class AparcFsInputPair:
    dataset_id: str
    subject: Optional[str]
    session: Optional[str]
    aparc_key: str
    fs_subject_root: str

    orig_keys: tuple[str, ...]
    raw_keys: tuple[str, ...]
    rawavg_keys: tuple[str, ...]
    numbered_mgz_keys: tuple[str, ...]

    preferred_orig_key: Optional[str]
    preferred_raw_key: Optional[str]
    preferred_rawavg_key: Optional[str]
    preferred_numbered_mgz_key: Optional[str]


class OpenNeuroIndex:
    def __init__(self) -> None:
        self.objects: list[OpenNeuroObject] = []
        self._built = False

        self._aparc_objects_cache: Optional[list[OpenNeuroObject]] = None
        self._raw_t1w_objects_cache: Optional[list[OpenNeuroObject]] = None
        self._fs_subject_file_index: Optional[dict[str, dict[str, list[str]]]] = None

    @staticmethod
    def _s3_client():
        return boto3.client(
            "s3",
            region_name=_REGION,
            config=Config(
                signature_version=UNSIGNED,
                retries={"max_attempts": 10, "mode": "standard"},
            ),
        )

    @staticmethod
    def _extract_dataset_id(key: str) -> Optional[str]:
        m = _DATASET_RE.match(key)
        return m.group(1) if m else None

    @staticmethod
    def _extract_subject(key: str) -> Optional[str]:
        m = _SUB_RE.search("/" + key)
        return m.group(1) if m else None

    @staticmethod
    def _extract_session(key: str) -> Optional[str]:
        m = _SES_RE.search("/" + key)
        return m.group(1) if m else None

    @staticmethod
    def _basename(key: str) -> str:
        return PurePosixPath(key).name

    @staticmethod
    def _is_aparc_aseg(key: str) -> bool:
        return OpenNeuroIndex._basename(key) == "aparc+aseg.mgz"

    @staticmethod
    def _is_raw_t1w(key: str) -> bool:
        p = PurePosixPath(key)
        parts = p.parts

        if len(parts) < 3:
            return False
        if not parts[0].startswith("ds"):
            return False
        if "derivatives" in parts or "sourcedata" in parts:
            return False
        if "anat" not in parts:
            return False

        return bool(_T1W_RE.search(p.name))

    @staticmethod
    def _is_orig_mgz(key: str) -> bool:
        return OpenNeuroIndex._basename(key).lower() == "orig.mgz"

    @staticmethod
    def _is_raw_mgz(key: str) -> bool:
        return OpenNeuroIndex._basename(key).lower() == "raw.mgz"

    @staticmethod
    def _is_rawavg_mgz(key: str) -> bool:
        return OpenNeuroIndex._basename(key).lower() == "rawavg.mgz"

    @staticmethod
    def _is_numbered_mgz(key: str) -> bool:
        return bool(_NUMBERED_MGZ_RE.match(OpenNeuroIndex._basename(key)))

    @staticmethod
    def _fs_subject_root_from_aparc(aparc_key: str) -> str:
        marker = "/mri/aparc+aseg.mgz"
        if marker not in aparc_key:
            raise ValueError(f"Not an aparc+aseg path: {aparc_key}")
        return aparc_key.rsplit(marker, 1)[0]

    @staticmethod
    def _extract_fs_subject_root_from_any_fs_key(key: str) -> Optional[str]:
        m = _FS_ROOT_RE.match(key)
        return m.group(1) if m else None

    @staticmethod
    def _is_real_subject_root(fs_subject_root: str) -> bool:
        return PurePosixPath(fs_subject_root).name.startswith("sub-")

    @staticmethod
    def _select_unique(candidates: list[str]) -> Optional[str]:
        c = sorted(set(candidates))
        return c[0] if len(c) == 1 else None

    @staticmethod
    def _select_preferred_numbered_mgz(candidates: list[str]) -> Optional[str]:
        c = sorted(set(candidates))
        c001 = [k for k in c if k.endswith("/001.mgz")]
        if len(c001) == 1:
            return c001[0]
        if len(c) == 1:
            return c[0]
        return None

    def _invalidate_indexes(self) -> None:
        self._aparc_objects_cache = None
        self._raw_t1w_objects_cache = None
        self._fs_subject_file_index = None

    def build(self, prefix: str = "", force_refresh: bool = False) -> None:
        if self._built and not force_refresh and not prefix:
            return

        s3 = self._s3_client()
        paginator = s3.get_paginator("list_objects_v2")

        out: list[OpenNeuroObject] = []
        for page in paginator.paginate(Bucket=_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                dataset_id = self._extract_dataset_id(key)
                if dataset_id is None:
                    continue
                out.append(
                    OpenNeuroObject(
                        dataset_id=dataset_id,
                        key=key,
                        size=int(obj.get("Size", 0)),
                        last_modified=str(obj.get("LastModified", "")),
                    )
                )

        self.objects = out
        self._built = True
        self._invalidate_indexes()

    def save_cache(self, cache_path: str | Path) -> None:
        payload = [asdict(x) for x in self.objects]
        with gzip.open(cache_path, "wt", encoding="utf-8") as f:
            json.dump(payload, f)

    def load_cache(self, cache_path: str | Path) -> None:
        with gzip.open(cache_path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
        self.objects = [OpenNeuroObject(**row) for row in payload]
        self._built = True
        self._invalidate_indexes()

    def list_aparc_aseg(self, include_templates: bool = False) -> list[OpenNeuroObject]:
        if self._aparc_objects_cache is None:
            self._aparc_objects_cache = [obj for obj in self.objects if self._is_aparc_aseg(obj.key)]

        if include_templates:
            return list(self._aparc_objects_cache)

        out: list[OpenNeuroObject] = []
        for obj in self._aparc_objects_cache:
            fs_subject_root = self._fs_subject_root_from_aparc(obj.key)
            if self._is_real_subject_root(fs_subject_root):
                out.append(obj)
        return out

    def list_raw_t1w(self) -> list[OpenNeuroObject]:
        if self._raw_t1w_objects_cache is None:
            self._raw_t1w_objects_cache = [obj for obj in self.objects if self._is_raw_t1w(obj.key)]
        return self._raw_t1w_objects_cache

    def list_orig_mgz(self) -> list[OpenNeuroObject]:
        return [obj for obj in self.objects if self._is_orig_mgz(obj.key)]

    def list_raw_mgz(self) -> list[OpenNeuroObject]:
        return [obj for obj in self.objects if self._is_raw_mgz(obj.key)]

    def list_rawavg_mgz(self) -> list[OpenNeuroObject]:
        return [obj for obj in self.objects if self._is_rawavg_mgz(obj.key)]

    def list_numbered_mgz(self) -> list[OpenNeuroObject]:
        return [obj for obj in self.objects if self._is_numbered_mgz(obj.key)]

    def pair_aparc_with_t1w(self) -> list[AparcT1wPair]:
        aparc_files = self.list_aparc_aseg(include_templates=False)
        t1w_files = self.list_raw_t1w()

        t1w_index: dict[tuple[str, Optional[str], Optional[str]], list[str]] = {}
        t1w_subject_fallback: dict[tuple[str, Optional[str]], list[str]] = {}

        for obj in t1w_files:
            sub = self._extract_subject(obj.key)
            ses = self._extract_session(obj.key)

            t1w_index.setdefault((obj.dataset_id, sub, ses), []).append(obj.key)
            t1w_subject_fallback.setdefault((obj.dataset_id, sub), []).append(obj.key)

        out: list[AparcT1wPair] = []
        for obj in aparc_files:
            sub = self._extract_subject(obj.key)
            ses = self._extract_session(obj.key)

            candidates = t1w_index.get((obj.dataset_id, sub, ses), [])
            if not candidates:
                candidates = t1w_subject_fallback.get((obj.dataset_id, sub), [])

            out.append(
                AparcT1wPair(
                    dataset_id=obj.dataset_id,
                    subject=sub,
                    session=ses,
                    aparc_key=obj.key,
                    t1w_candidates=tuple(sorted(candidates)),
                )
            )

        return out

    def _build_fs_subject_file_index(self) -> dict[str, dict[str, list[str]]]:
        if self._fs_subject_file_index is not None:
            return self._fs_subject_file_index

        index: dict[str, dict[str, list[str]]] = {}

        for obj in self.objects:
            root = self._extract_fs_subject_root_from_any_fs_key(obj.key)
            if root is None:
                continue

            bucket = index.setdefault(root, {"orig": [], "raw": [], "rawavg": [], "numbered": []})

            key = obj.key
            if self._is_orig_mgz(key):
                bucket["orig"].append(key)
            if self._is_raw_mgz(key):
                bucket["raw"].append(key)
            if self._is_rawavg_mgz(key):
                bucket["rawavg"].append(key)
            if self._is_numbered_mgz(key):
                bucket["numbered"].append(key)

        for root in index:
            for name in index[root]:
                index[root][name].sort()

        self._fs_subject_file_index = index
        return index

    def pair_aparc_with_fs_inputs(self, include_templates: bool = False) -> list[AparcFsInputPair]:
        aparc_files = self.list_aparc_aseg(include_templates=include_templates)
        fs_index = self._build_fs_subject_file_index()

        out: list[AparcFsInputPair] = []

        for obj in aparc_files:
            fs_subject_root = self._fs_subject_root_from_aparc(obj.key)
            files = fs_index.get(fs_subject_root, {"orig": [], "raw": [], "rawavg": [], "numbered": []})

            orig_keys = files["orig"]
            raw_keys = files["raw"]
            rawavg_keys = files["rawavg"]
            numbered_mgz_keys = files["numbered"]

            out.append(
                AparcFsInputPair(
                    dataset_id=obj.dataset_id,
                    subject=self._extract_subject(obj.key),
                    session=self._extract_session(obj.key),
                    aparc_key=obj.key,
                    fs_subject_root=fs_subject_root,
                    orig_keys=tuple(orig_keys),
                    raw_keys=tuple(raw_keys),
                    rawavg_keys=tuple(rawavg_keys),
                    numbered_mgz_keys=tuple(numbered_mgz_keys),
                    preferred_orig_key=self._select_unique(orig_keys),
                    preferred_raw_key=self._select_unique(raw_keys),
                    preferred_rawavg_key=self._select_unique(rawavg_keys),
                    preferred_numbered_mgz_key=self._select_preferred_numbered_mgz(numbered_mgz_keys),
                )
            )

        return out

    def pair_aparc_with_orig(self) -> list[AparcOrigPair]:
        fs_pairs = self.pair_aparc_with_fs_inputs(include_templates=False)
        return [
            AparcOrigPair(
                dataset_id=p.dataset_id,
                subject=p.subject,
                session=p.session,
                aparc_key=p.aparc_key,
                orig_key=p.preferred_orig_key,
            )
            for p in fs_pairs
        ]


def build_or_load_index(
    cache_path: Optional[str | Path] = None,
    force_refresh: bool = False,
    prefix: str = "",
) -> OpenNeuroIndex:
    idx = OpenNeuroIndex()

    if cache_path and not force_refresh:
        try:
            idx.load_cache(cache_path)
            return idx
        except FileNotFoundError:
            pass

    idx.build(prefix=prefix, force_refresh=True)

    if cache_path:
        idx.save_cache(cache_path)

    return idx


@dataclass(frozen=True)
class DownloadItem:
    pair_index: int
    source_attr: str
    s3_key: str
    local_path: Path
    cached: bool


@dataclass(frozen=True)
class DownloadFailure:
    pair_index: int
    source_attr: str
    s3_key: str
    error_code: str
    message: str


@dataclass(frozen=True)
class DownloadReport:
    downloaded: tuple[DownloadItem, ...]
    skipped_missing_attr: tuple[tuple[int, str], ...]
    skipped_empty_value: tuple[tuple[int, str], ...]
    skipped_duplicate: tuple[tuple[int, str, str], ...]
    failed: tuple[DownloadFailure, ...]


def _as_key_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for x in value:
            if isinstance(x, str) and x:
                out.append(x)
        return out
    return []


def _local_path_for_key(key: str, out_root: str | Path, preserve_s3_path: bool) -> Path:
    root = Path(out_root)
    return root / key if preserve_s3_path else root / Path(key).name


def _retry_download_file(key: str, local_path: Path, retries: int) -> None:
    s3 = _thread_s3_client()
    local_path.parent.mkdir(parents=True, exist_ok=True)

    last_err: Exception | None = None
    for _ in range(max(1, retries)):
        try:
            s3.download_file(_BUCKET, key, str(local_path))
            return
        except (EndpointConnectionError, ConnectTimeoutError, ReadTimeoutError) as e:
            last_err = e
            continue
        except ClientError:
            raise
        except BotoCoreError as e:
            last_err = e
            continue

    if last_err is not None:
        raise last_err


def _download_task(
    pair_index: int,
    attr: str,
    key: str,
    out_root: str | Path,
    preserve_s3_path: bool,
    retries: int,
    skip_existing: bool,
) -> tuple[str, DownloadItem | DownloadFailure]:
    local_path = _local_path_for_key(key, out_root, preserve_s3_path)

    if skip_existing and local_path.exists() and local_path.stat().st_size > 0:
        return (
            "downloaded",
            DownloadItem(
                pair_index=pair_index,
                source_attr=attr,
                s3_key=key,
                local_path=local_path,
                cached=True,
            ),
        )

    try:
        _retry_download_file(key, local_path, retries=retries)
        return (
            "downloaded",
            DownloadItem(
                pair_index=pair_index,
                source_attr=attr,
                s3_key=key,
                local_path=local_path,
                cached=False,
            ),
        )
    except ClientError as e:
        if local_path.exists() and local_path.stat().st_size == 0:
            try:
                local_path.unlink()
            except Exception:
                pass

        code = str(e.response.get("Error", {}).get("Code", "ClientError"))
        msg = str(e.response.get("Error", {}).get("Message", str(e)))
        return (
            "failed",
            DownloadFailure(
                pair_index=pair_index,
                source_attr=attr,
                s3_key=key,
                error_code=code,
                message=msg,
            ),
        )
    except Exception as e:
        if local_path.exists() and local_path.stat().st_size == 0:
            try:
                local_path.unlink()
            except Exception:
                pass

        return (
            "failed",
            DownloadFailure(
                pair_index=pair_index,
                source_attr=attr,
                s3_key=key,
                error_code=type(e).__name__,
                message=str(e),
            ),
        )


def fetch_openneuro_files(
    pairs: Sequence[Any],
    out_root: str | Path,
    which: Sequence[str] = ("aparc_key", "preferred_rawavg_key"),
    *,
    preserve_s3_path: bool = True,
    deduplicate_keys: bool = True,
    retries: int = 3,
    show_progress: bool = True,
    skip_existing: bool = True,
    max_workers: int = 8,
) -> DownloadReport:
    root = _safe_mkdir(out_root)

    downloaded: list[DownloadItem] = []
    skipped_missing_attr: list[tuple[int, str]] = []
    skipped_empty_value: list[tuple[int, str]] = []
    skipped_duplicate: list[tuple[int, str, str]] = []
    failed: list[DownloadFailure] = []

    seen_keys: set[str] = set()
    plan: list[tuple[int, str, str]] = []

    for pair_index, pair in enumerate(pairs):
        for attr in which:
            if not hasattr(pair, attr):
                skipped_missing_attr.append((pair_index, attr))
                continue

            keys = _as_key_list(getattr(pair, attr))
            if not keys:
                skipped_empty_value.append((pair_index, attr))
                continue

            for key in keys:
                if deduplicate_keys and key in seen_keys:
                    skipped_duplicate.append((pair_index, attr, key))
                    continue
                seen_keys.add(key)
                plan.append((pair_index, attr, key))

    if not plan:
        return DownloadReport(
            downloaded=tuple(),
            skipped_missing_attr=tuple(skipped_missing_attr),
            skipped_empty_value=tuple(skipped_empty_value),
            skipped_duplicate=tuple(skipped_duplicate),
            failed=tuple(),
        )

    pbar = tqdm(total=len(plan), desc="Downloading OpenNeuro files") if show_progress else None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                _download_task,
                pair_index,
                attr,
                key,
                root,
                preserve_s3_path,
                retries,
                skip_existing,
            ): (pair_index, attr, key)
            for pair_index, attr, key in plan
        }

        for fut in as_completed(futures):
            kind, payload = fut.result()
            if kind == "downloaded":
                downloaded.append(payload)  # type: ignore[arg-type]
            else:
                failed.append(payload)  # type: ignore[arg-type]

            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    return DownloadReport(
        downloaded=tuple(downloaded),
        skipped_missing_attr=tuple(skipped_missing_attr),
        skipped_empty_value=tuple(skipped_empty_value),
        skipped_duplicate=tuple(skipped_duplicate),
        failed=tuple(failed),
    )


def fetch_openneuro_files_with_selector(
    pairs: Sequence[Any],
    out_root: str | Path,
    selector: Callable[[Any], Iterable[str | None]],
    *,
    preserve_s3_path: bool = True,
    deduplicate_keys: bool = True,
    retries: int = 3,
    show_progress: bool = True,
    skip_existing: bool = True,
    max_workers: int = 8,
) -> DownloadReport:
    class _Box:
        __slots__ = ("custom_keys",)

        def __init__(self, keys: Iterable[str | None]):
            self.custom_keys = tuple(k for k in keys if isinstance(k, str) and k)

    boxed = [_Box(selector(p)) for p in pairs]
    return fetch_openneuro_files(
        boxed,
        out_root=out_root,
        which=("custom_keys",),
        preserve_s3_path=preserve_s3_path,
        deduplicate_keys=deduplicate_keys,
        retries=retries,
        show_progress=show_progress,
        skip_existing=skip_existing,
        max_workers=max_workers,
    )


def _get_objects(idx_or_objects):
    return idx_or_objects.objects if hasattr(idx_or_objects, "objects") else idx_or_objects


def get_all_build_stamp_paths(
    idx_or_objects: Any,
    dataset_ids: Iterable[str] | None = None,
) -> dict[str, list[str]]:
    objects = _get_objects(idx_or_objects)

    out: dict[str, list[str]] = {}
    if dataset_ids is not None:
        out = {str(ds): [] for ds in dataset_ids}

    for obj in objects:
        key = getattr(obj, "key", None)
        if not isinstance(key, str):
            continue

        if not key.lower().endswith("/build-stamp.txt"):
            continue

        parts = key.split("/")
        if not parts:
            continue

        dataset_id = parts[0]
        if not dataset_id.startswith("ds"):
            continue

        out.setdefault(dataset_id, [])
        out[dataset_id].append(key)

    for ds in out:
        out[ds] = sorted(set(out[ds]))

    return out


def get_first_build_stamp_path_per_dataset(
    idx_or_objects: Any,
    dataset_ids: Iterable[str] | None = None,
    prefer_subject_paths: bool = True,
) -> dict[str, str | None]:
    all_paths = get_all_build_stamp_paths(idx_or_objects, dataset_ids=dataset_ids)

    out: dict[str, str | None] = {}
    for ds, paths in all_paths.items():
        if not paths:
            out[ds] = None
            continue

        if prefer_subject_paths:
            sub_paths = [p for p in paths if "/sub-" in p]
            out[ds] = sorted(sub_paths)[0] if sub_paths else sorted(paths)[0]
        else:
            out[ds] = sorted(paths)[0]

    return out


def make_build_stamp_fetch_records(
    idx_or_objects: Any,
    dataset_ids: Iterable[str] | None = None,
    prefer_subject_paths: bool = True,
) -> list[dict[str, str | None]]:
    first_map = get_first_build_stamp_path_per_dataset(
        idx_or_objects,
        dataset_ids=dataset_ids,
        prefer_subject_paths=prefer_subject_paths,
    )
    return [{"dataset_id": ds, "build_stamp_key": key} for ds, key in first_map.items()]


def _download_one(
    dataset_id: str,
    s3_key: str,
    cache_root: str | Path,
    retries: int = 3,
    skip_existing: bool = True,
):
    root = Path(cache_root)
    local_path = root / s3_key
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and local_path.exists() and local_path.stat().st_size > 0:
        return {
            "dataset_id": dataset_id,
            "s3_key": s3_key,
            "local_path": str(local_path),
            "cached": True,
            "ok": True,
            "error": None,
        }

    s3 = _thread_s3_client()

    last_err = None
    for _ in range(max(1, retries)):
        try:
            s3.download_file(_BUCKET, s3_key, str(local_path))
            return {
                "dataset_id": dataset_id,
                "s3_key": s3_key,
                "local_path": str(local_path),
                "cached": False,
                "ok": True,
                "error": None,
            }
        except ClientError as e:
            code = str(e.response.get("Error", {}).get("Code", "ClientError"))
            msg = str(e.response.get("Error", {}).get("Message", str(e)))
            return {
                "dataset_id": dataset_id,
                "s3_key": s3_key,
                "local_path": str(local_path),
                "cached": False,
                "ok": False,
                "error": f"{code}: {msg}",
            }
        except (EndpointConnectionError, ConnectTimeoutError, ReadTimeoutError, BotoCoreError) as e:
            last_err = e
            continue
        except Exception as e:
            return {
                "dataset_id": dataset_id,
                "s3_key": s3_key,
                "local_path": str(local_path),
                "cached": False,
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
            }

    return {
        "dataset_id": dataset_id,
        "s3_key": s3_key,
        "local_path": str(local_path),
        "cached": False,
        "ok": False,
        "error": f"{type(last_err).__name__}: {last_err}" if last_err else "unknown error",
    }


def fetch_build_stamp_cache(
    first_build_stamp: dict[str, Optional[str]],
    cache_root: str | Path = "fs_build_stamp_cache/files",
    max_workers: int = 16,
    retries: int = 3,
    skip_existing: bool = True,
    show_progress: bool = True,
):
    root = _safe_mkdir(cache_root)

    jobs: list[tuple[str, str]] = []
    missing_key: dict[str, Optional[str]] = {}

    for ds, key in first_build_stamp.items():
        ds = str(ds)
        if key is None or not str(key).strip():
            missing_key[ds] = None
            continue
        jobs.append((ds, str(key)))

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                _download_one,
                dataset_id=ds,
                s3_key=key,
                cache_root=root,
                retries=retries,
                skip_existing=skip_existing,
            ): (ds, key)
            for ds, key in jobs
        }

        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(futures), desc="Fetching build-stamp.txt")

        for fut in iterator:
            results.append(fut.result())

    local_path_by_dataset: dict[str, Optional[str]] = {str(ds): None for ds in first_build_stamp.keys()}
    text_by_dataset: dict[str, Optional[str]] = {str(ds): None for ds in first_build_stamp.keys()}
    success: dict[str, dict] = {}
    failed: dict[str, dict] = {}

    for row in results:
        ds = row["dataset_id"]
        if row["ok"]:
            local_path_by_dataset[ds] = row["local_path"]
            try:
                text_by_dataset[ds] = Path(row["local_path"]).read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                text_by_dataset[ds] = f"<<READ_ERROR: {type(e).__name__}: {e}>>"
            success[ds] = row
        else:
            failed[ds] = row

    return {
        "local_path_by_dataset": local_path_by_dataset,
        "text_by_dataset": text_by_dataset,
        "success": success,
        "failed": failed,
        "missing_key": missing_key,
    }


def normalize_fs_version(ver: str) -> str | None:
    m = re.fullmatch(r"([4-8])\.(\d+)(?:\.(\d+))?", ver.strip())
    if not m:
        return None
    major, minor, patch = m.group(1), m.group(2), m.group(3) or "0"
    return f"{major}.{minor}.{patch}"


def extract_fs_version(text: str | None) -> str | None:
    if not text:
        return None
    for pat in PATTERNS:
        m = pat.search(text)
        if m:
            return normalize_fs_version(m.group("ver"))
    return None


def extract_fs_versions(text_by_dataset: dict[str, str | None]) -> dict[str, str | None]:
    return {str(ds): extract_fs_version(text) for ds, text in text_by_dataset.items()}


def list_all_files(root_dir: str | Path) -> list[str]:
    return [str(p.resolve()) for p in Path(root_dir).rglob("*") if p.is_file()]


def build_fs_file_map_from_cache(root_dir: str | Path = "openneuro_cache") -> dict[str, dict[str, str | None]]:
    files = list_all_files(root_dir)
    file_map: dict[str, dict[str, str | None]] = {}
    for f in files:
        s = f.split("/")
        k = "/".join(s[:-1])
        name = s[-1].split(".")[0]

        if k not in file_map:
            file_map[k] = {"rawavg": None, "aparc+aseg": None}

        file_map[k][name] = f
    return file_map


def extend_file_map_with_hcp(file_map: dict[str, dict[str, str | None]], hcp_root: str | Path = "hcp_filt") -> dict[str, dict[str, str | None]]:
    files = list_all_files(hcp_root)
    files = [
        f
        for f in files
        if "T1w/" in f and (f.endswith("T1w_acpc_dc_restore.nii.gz") or f.endswith("aparc+aseg.nii.gz"))
    ]

    out = dict(file_map)
    for f in files:
        s = f.split("/")
        k = "/".join(s[:-1])
        name = s[-1].split(".")[0]

        if k not in out:
            out[k] = {"rawavg": None, "aparc+aseg": None}

        if name == "T1w_acpc_dc_restore":
            name = "rawavg"
        out[k][name] = f

    return out


def filter_complete_file_map(file_map: dict[str, dict[str, str | None]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for k, v in file_map.items():
        rawavg = v.get("rawavg")
        aparc = v.get("aparc+aseg")
        if rawavg is not None and aparc is not None:
            out[k] = {"rawavg": str(rawavg), "aparc+aseg": str(aparc)}
    return out


def load_fs_file_map_json(path: str | Path = "fs_file_map.json") -> dict[str, dict[str, str | None]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}, got {type(data)}")
    return data


def save_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def convert_file_map_to_tensors(
    file_map: dict[str, dict[str, str]],
    out_root: str | Path = "tensors",
    n_jobs: int = -1,
    unsafe_int8: bool = False,
):
    from convert import convert_file_map_to_pt

    return convert_file_map_to_pt(
        file_map=file_map,
        out_root=out_root,
        n_jobs=n_jobs,
        unsafe_int8=unsafe_int8,
    )


def prepare_images_if_needed(*args, **kwargs):
    from convert import prepare_images_if_needed as _fn

    return _fn(*args, **kwargs)


def prepare_arrays_if_needed(*args, **kwargs):
    from convert import prepare_arrays_if_needed as _fn

    return _fn(*args, **kwargs)


def debug_prepare_images_report(*args, **kwargs):
    from convert import debug_prepare_images_report as _fn

    return _fn(*args, **kwargs)


__all__ = [
    "OpenNeuroObject",
    "AparcT1wPair",
    "AparcOrigPair",
    "AparcFsInputPair",
    "OpenNeuroIndex",
    "build_or_load_index",
    "DownloadItem",
    "DownloadFailure",
    "DownloadReport",
    "fetch_openneuro_files",
    "fetch_openneuro_files_with_selector",
    "get_all_build_stamp_paths",
    "get_first_build_stamp_path_per_dataset",
    "make_build_stamp_fetch_records",
    "fetch_build_stamp_cache",
    "normalize_fs_version",
    "extract_fs_version",
    "extract_fs_versions",
    "list_all_files",
    "build_fs_file_map_from_cache",
    "extend_file_map_with_hcp",
    "filter_complete_file_map",
    "load_fs_file_map_json",
    "save_json",
    "convert_file_map_to_tensors",
    "prepare_images_if_needed",
    "prepare_arrays_if_needed",
    "debug_prepare_images_report",
]

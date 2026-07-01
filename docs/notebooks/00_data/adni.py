
# PKG = Path(r"/mnt/c/Users/ryanp/Downloads/ADNIMERGE2.tar.gz")

from pathlib import Path
import tarfile
import tempfile

import pandas as pd
import rdata


# -----------------------
# Edit this path
# -----------------------
PKG = Path(r"/mnt/c/Users/ryanp/Downloads/ADNIMERGE2.tar.gz")

OUT = Path("adni_tables")
OUT.mkdir(exist_ok=True)

# Keep this focused. ADRS is not needed for your first AD vs CN image experiment.
WANTED = {
    # baseline subject labels
    "ADSL",

    # optional diagnosis tables
    "DXSUM",
    "RS",

    # MRI metadata / QC
    "MRIMETA",
    "MRI3META",
    "MRIMPPRO",
    "MRIMPRANK",
    "MRIQC",

    # FreeSurfer feature tables, if present
    "UCSFFSX7",
    "UCSFFSX6",
    "UCSFFSX51",
    "UCSFFSX",
}


def clean_name(name: str) -> str:
    return name.upper().replace(".", "_").replace("-", "_")


def to_dataframe(obj):
    """Best-effort conversion to pandas DataFrame."""
    if isinstance(obj, pd.DataFrame):
        return obj

    # rdata sometimes returns objects that can be coerced cleanly.
    try:
        return pd.DataFrame(obj)
    except Exception:
        return None


exported = []
failed = []

with tarfile.open(PKG, "r:gz") as tar:
    members = [
        m for m in tar.getmembers()
        if "/data/" in m.name
        and m.name.lower().endswith((".rda", ".rdata", ".rds"))
    ]

    print(f"Found {len(members)} R data files inside package.")

    for m in members:
        stem = clean_name(Path(m.name).stem)

        if stem not in WANTED:
            continue

        print(f"\nReading {m.name}")

        with tar.extractfile(m) as src:
            if src is None:
                continue

            suffix = Path(m.name).suffix

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(src.read())
                tmp_path = Path(tmp.name)

        try:
            # Returns dict-like object: object_name -> object
            result = rdata.read_rda(tmp_path)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            failed.append((m.name, str(e)))
            continue

        for obj_name, obj in result.items():
            df = to_dataframe(obj)

            if df is None:
                print(f"  skipped {obj_name}: could not convert to DataFrame")
                continue

            out_name = clean_name(obj_name) if obj_name else stem
            out_file = OUT / f"{out_name}.csv"

            df.to_csv(out_file, index=False)
            exported.append(out_file)

            print(f"  exported {out_file} shape={df.shape}")

print("\nExported:")
for f in exported:
    print(" ", f)

if failed:
    print("\nFailed:")
    for name, err in failed:
        print(" ", name, "->", err)
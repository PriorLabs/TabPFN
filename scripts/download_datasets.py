"""
Download and prepare insurance classification datasets for the TabPFN vs GLM benchmark.

Datasets sourced:
1. coil2000        — COIL 2000 (Dutch insurance company product purchase), OpenML ID 298
                     9,822 rows · 85 features · binary: caravan insurance ownership
2. ausprivauto0405 — Australian Private Auto Insurance 2004–05, CASdatasets (GitHub mirror)
                     67,856 rows · 7 features · binary: ClaimOcc (claim occurred)
3. freMTPL2freq_binary — French MTPL frequency binarised from existing freMTPL2freq.csv
                     678,013 rows (sampled to FREMTPL_SAMPLE_SIZE) · binary: ClaimNb > 0

The 4th dataset (eudirectlapse.csv) is already present in data/raw/.

Run from the repo root:
    python scripts/download_datasets.py
"""

import sys
import urllib.request
import tempfile
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

FREMTPL_SAMPLE_SIZE = 50_000   # rows to sample from the 678K freMTPL2freq
RANDOM_SEED = 42

CASDATASETS_BASE = "https://raw.githubusercontent.com/dutangc/CASdatasets/master/data/"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _status(msg: str) -> None:
    print(f"  {msg}", flush=True)


def _skip(path: Path) -> bool:
    if path.exists():
        _status(f"Already exists, skipping: {path.name}")
        return True
    return False


# ---------------------------------------------------------------------------
# Dataset 1: COIL 2000 (Dutch insurance — caravan insurance ownership)
# ---------------------------------------------------------------------------
def download_coil2000() -> None:
    out = RAW_DIR / "coil2000.csv"
    if _skip(out):
        return

    print("Downloading COIL 2000 from OpenML (ID 298)…")
    from sklearn.datasets import fetch_openml

    bundle = fetch_openml(data_id=298, as_frame=True, parser="auto")
    df = bundle.frame
    # Target column is CARAVAN (0/1 int stored as int64 after parser='auto')
    df["CARAVAN"] = df["CARAVAN"].astype(int)
    df.to_csv(out, index=False)
    pos = int(df["CARAVAN"].sum())
    _status(f"Saved {len(df):,} rows · {len(df.columns)-1} features · "
            f"{pos:,} positive ({pos/len(df):.1%}) → {out.name}")


# ---------------------------------------------------------------------------
# Dataset 2: ausprivauto0405 (Australian vehicle insurance — claim occurrence)
# ---------------------------------------------------------------------------
def download_ausprivauto0405() -> None:
    out = RAW_DIR / "ausprivauto0405.csv"
    if _skip(out):
        return

    print("Downloading ausprivauto0405 from CASdatasets GitHub mirror…")
    try:
        import pyreadr
    except ImportError:
        print("  pyreadr not installed. Run: pip install pyreadr")
        sys.exit(1)

    url = CASDATASETS_BASE + "ausprivauto0405.rda"
    tmp = tempfile.mktemp(suffix=".rda")
    try:
        urllib.request.urlretrieve(url, tmp)
        result = pyreadr.read_r(tmp)
        df = result[list(result.keys())[0]]

        # Convert categoricals to string so CSV is clean
        for col in df.select_dtypes("category").columns:
            df[col] = df[col].astype(str)

        # Drop ClaimNb and ClaimAmount — keep only features + binary target
        # ClaimOcc is the classification target (1 = at least one claim)
        df = df.drop(columns=["ClaimNb", "ClaimAmount"])
        df.to_csv(out, index=False)
        pos = int(df["ClaimOcc"].sum())
        _status(f"Saved {len(df):,} rows · {len(df.columns)-1} features · "
                f"{pos:,} positive ({pos/len(df):.1%}) → {out.name}")
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# Dataset 3: freMTPL2freq binarised (French MTPL — claim indicator)
# ---------------------------------------------------------------------------
def process_fremtpl2freq_binary() -> None:
    out = RAW_DIR / "freMTPL2freq_binary.csv"
    if _skip(out):
        return

    src = RAW_DIR / "freMTPL2freq.csv"
    if not src.exists():
        print(f"  Source not found: {src}. Skipping freMTPL2freq_binary.")
        return

    print(f"Processing freMTPL2freq → binary (sampling {FREMTPL_SAMPLE_SIZE:,} rows)…")
    df = pd.read_csv(src)

    # Binarise: did any claim occur?
    df["ClaimIndicator"] = (df["ClaimNb"] > 0).astype(int)
    df = df.drop(columns=["IDpol", "ClaimNb"])   # drop ID and raw count

    # Stratified sample to keep class proportions
    rng = np.random.default_rng(RANDOM_SEED)
    pos_idx = df.index[df["ClaimIndicator"] == 1]
    neg_idx = df.index[df["ClaimIndicator"] == 0]
    pos_rate = len(pos_idx) / len(df)
    n_pos = round(FREMTPL_SAMPLE_SIZE * pos_rate)
    n_neg = FREMTPL_SAMPLE_SIZE - n_pos

    sampled_idx = np.concatenate([
        rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False),
        rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False),
    ])
    rng.shuffle(sampled_idx)
    df_sample = df.loc[sampled_idx].reset_index(drop=True)

    df_sample.to_csv(out, index=False)
    pos = int(df_sample["ClaimIndicator"].sum())
    _status(f"Saved {len(df_sample):,} rows · {len(df_sample.columns)-1} features · "
            f"{pos:,} positive ({pos/len(df_sample):.1%}) → {out.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("Insurance Dataset Downloader")
    print(f"Output directory: {RAW_DIR}")
    print("=" * 60)

    download_coil2000()
    download_ausprivauto0405()
    process_fremtpl2freq_binary()

    print()
    print("Done. Files in data/raw/:")
    for f in sorted(RAW_DIR.iterdir()):
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name:<35} {size_kb:>8,} KB")


if __name__ == "__main__":
    main()

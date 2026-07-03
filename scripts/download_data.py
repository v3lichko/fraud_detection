"""
Downloads the Credit Card Fraud Detection dataset (Dal Pozzolo et al., ULB)
from its OpenML mirror (dataset id 1597) and saves it as data/raw/creditcard.csv.

Usage:
    python scripts/download_data.py
"""
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ARFF_PATH = RAW_DIR / "creditcard.arff"
CSV_PATH = RAW_DIR / "creditcard.csv"

# OpenML file id for the "creditcard" dataset (version 1, did=1597).
OPENML_URL = "https://api.openml.org/data/v1/download/1673544/creditcard.arff"

COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]


def download_arff() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {OPENML_URL} ...")
    req = urllib.request.Request(OPENML_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as response, open(ARFF_PATH, "wb") as out_file:
        total = 0
        while chunk := response.read(1024 * 1024):
            out_file.write(chunk)
            total += len(chunk)
            print(f"\r  {total / 1e6:.1f} MB downloaded", end="")
    print()


def convert_to_csv() -> None:
    import pandas as pd

    with open(ARFF_PATH) as f:
        data_start = None
        for i, line in enumerate(f):
            if line.strip().lower() == "@data":
                data_start = i + 1
                break
    if data_start is None:
        raise RuntimeError("Could not find @data section in ARFF file")

    df = pd.read_csv(ARFF_PATH, skiprows=data_start, header=None, names=COLUMNS)
    df = df.dropna(subset=["Class"])
    df["Class"] = df["Class"].astype(str).str.strip().str.strip("'").astype(int)

    print(f"Loaded {len(df)} rows, {df['Class'].sum()} labeled as fraud "
          f"({df['Class'].mean():.4%})")

    df.to_csv(CSV_PATH, index=False)
    print(f"Saved {CSV_PATH}")
    ARFF_PATH.unlink()


def main() -> None:
    if CSV_PATH.exists():
        print(f"{CSV_PATH} already exists, skipping download.")
        return
    download_arff()
    convert_to_csv()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Failed to download dataset: {exc}", file=sys.stderr)
        sys.exit(1)

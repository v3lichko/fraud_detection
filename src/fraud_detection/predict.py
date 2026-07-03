"""Score new transactions with the trained fraud model.

Usage:
    python -m fraud_detection.predict --input path/to/transactions.csv --output path/to/scored.csv

Input CSV must have the same columns as the training data, minus `Class`
(Time, V1..V28, Amount).
"""
import argparse

import joblib
import pandas as pd

from fraud_detection.config import MODEL_PATH


def load_model(path=MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m fraud_detection.train` first."
        )
    return joblib.load(path)


def score(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    pipeline = bundle["pipeline"]
    threshold = bundle["threshold"]
    proba = pipeline.predict_proba(df)[:, 1]
    out = df.copy()
    out["fraud_probability"] = proba
    out["is_fraud_predicted"] = (proba >= threshold).astype(int)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV of transactions to score")
    parser.add_argument("--output", required=True, help="where to write scored CSV")
    args = parser.parse_args()

    bundle = load_model()
    print(f"Loaded {bundle['model_name']} (threshold={bundle['threshold']:.4f})")

    df = pd.read_csv(args.input)
    scored = score(df, bundle)
    scored.to_csv(args.output, index=False)

    n_flagged = scored["is_fraud_predicted"].sum()
    print(f"Scored {len(scored)} transactions, flagged {n_flagged} as fraud.")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

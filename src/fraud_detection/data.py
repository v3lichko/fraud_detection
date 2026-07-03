"""Loading and splitting the credit card transactions dataset."""
import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection.config import RAW_DATA_PATH, RANDOM_STATE, TARGET_COL, TEST_SIZE


def load_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(path)


def load_train_test_split(path=RAW_DATA_PATH, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Stratified split so both train and test keep the same (tiny) fraud rate."""
    df = load_data(path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

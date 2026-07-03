import numpy as np
import pandas as pd
import pytest

from fraud_detection.data import load_train_test_split
from fraud_detection.features import PCA_COLUMNS


@pytest.fixture
def toy_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 500
    data = {
        "Time": rng.integers(0, 170000, n),
        "Amount": rng.uniform(1, 500, n),
    }
    for col in PCA_COLUMNS:
        data[col] = rng.normal(size=n)
    # ~2% fraud rate, comparable order of magnitude imbalance to the real data
    data["Class"] = rng.choice([0, 1], size=n, p=[0.98, 0.02])
    path = tmp_path / "toy_creditcard.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def test_split_is_stratified(toy_csv):
    X_train, X_test, y_train, y_test = load_train_test_split(path=toy_csv, test_size=0.2)
    assert len(X_train) + len(X_test) == 500
    assert abs(y_train.mean() - y_test.mean()) < 0.05


def test_split_has_no_row_overlap(toy_csv):
    X_train, X_test, y_train, y_test = load_train_test_split(path=toy_csv, test_size=0.2)
    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_train_test_split(path=tmp_path / "does_not_exist.csv")

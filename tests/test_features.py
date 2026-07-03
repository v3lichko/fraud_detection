import numpy as np
import pandas as pd
import pytest

from fraud_detection.features import PCA_COLUMNS, add_hour_feature, build_preprocessor


def _toy_frame(n=10):
    rng = np.random.default_rng(0)
    data = {"Time": np.arange(0, n * 3600, 3600), "Amount": rng.uniform(1, 500, n)}
    for col in PCA_COLUMNS:
        data[col] = rng.normal(size=n)
    return pd.DataFrame(data)


def test_add_hour_feature_wraps_at_24():
    df = pd.DataFrame({"Time": [0, 3600, 90000]})
    out = add_hour_feature(df)
    assert out["Hour"].tolist() == [0, 1, 1]  # 90000s = 25h -> hour-of-day 1


def test_add_hour_feature_does_not_mutate_input():
    df = pd.DataFrame({"Time": [0, 3600]})
    add_hour_feature(df)
    assert "Hour" not in df.columns


def test_build_preprocessor_output_shape_and_no_nans():
    df = _toy_frame(20)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(df)
    assert transformed.shape == (20, len(PCA_COLUMNS) + 3)  # + Time, Amount, Hour
    assert not np.isnan(transformed).any()


def test_build_preprocessor_missing_column_raises():
    df = _toy_frame(5).drop(columns=["Amount"])
    with pytest.raises(ValueError, match="Amount"):
        build_preprocessor().fit_transform(df)

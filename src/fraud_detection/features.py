"""Feature engineering: derive a time-of-day signal and scale skewed numeric columns.

V1-V28 are already PCA components (anonymized, roughly standardized by the
data provider), so we leave them untouched and only scale the two raw,
heavily skewed columns plus the derived Hour feature.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler

PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
SCALED_COLUMNS = ["Time", "Amount", "Hour"]


def add_hour_feature(X: pd.DataFrame) -> pd.DataFrame:
    """`Time` is seconds elapsed since the first transaction in the dump
    (spans ~2 days). Hour-of-day is a more directly interpretable signal
    for fraud patterns (e.g. more fraud relative to volume at night)."""
    X = X.copy()
    X["Hour"] = (X["Time"] // 3600 % 24).astype(int)
    return X


def build_preprocessor() -> Pipeline:
    """Returns a Pipeline that adds the Hour feature, then scales the raw
    numeric columns with a RobustScaler (median/IQR based, so it isn't
    thrown off by the extreme outliers in Amount) while passing the
    already-scaled PCA columns through untouched."""
    scale_and_pass = ColumnTransformer(
        transformers=[
            ("scale", RobustScaler(), SCALED_COLUMNS),
            ("pca_features", "passthrough", PCA_COLUMNS),
        ]
    )
    return Pipeline(
        steps=[
            ("add_hour", FunctionTransformer(add_hour_feature)),
            ("scale_and_pass", scale_and_pass),
        ]
    )


def get_feature_names() -> list[str]:
    return SCALED_COLUMNS + PCA_COLUMNS

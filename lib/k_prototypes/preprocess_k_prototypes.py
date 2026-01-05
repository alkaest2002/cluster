from typing import TYPE_CHECKING

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator


def get_pipe(df: pd.DataFrame) -> Pipeline:
    """Create a preprocessing pipeline for numerical features.

    Args:
        df (pd.DataFrame): Input DataFrame to determine column types.

    Returns:
        Pipeline: A scikit-learn Pipeline object for preprocessing.

    """
    # Get indices of float columns
    float_cols: list[int] = [df.columns.get_loc(c) for c in df.select_dtypes(include=["float64"])]

    # Define steps for float pipeline
    float_steps: list[tuple[str, BaseEstimator]] = [
        # Make sure to remove zero-variance features
        ("variance_threshold", VarianceThreshold(threshold=0.0)),
        # Standard scaling
        ("scaler", StandardScaler()),
    ]

    # Define numerical pipeline
    float_pipe: Pipeline = Pipeline(steps=float_steps)

    # Define numerical pipeline tuple
    float_pipe_tuple: tuple[str, Pipeline, list[int]] = ("float", float_pipe, float_cols)

    # Define column transformer
    preprocessor: ColumnTransformer = ColumnTransformer(
        [
            float_pipe_tuple
        ],
        remainder="passthrough"
    )

    return make_pipeline(preprocessor)

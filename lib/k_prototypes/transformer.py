from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray


class NumericPreprocessor(BaseEstimator, TransformerMixin):
    """A preprocessing transformer for numerical features.

    This transformer handles numerical data by removing zero-variance features
    and applying standard scaling. It automatically detects float64 columns
    and applies the preprocessing pipeline only to those features.
    """

    def __init__(self, variance_threshold: float = 0.0) -> None:
        """Initialize the NumericPreprocessor.

        Args:
            variance_threshold: Threshold below which features are considered
                constant and will be removed. Default is 0.0.

        Attributes:
            variance_threshold: The variance threshold for feature selection.
            pipeline_: The fitted preprocessing pipeline.
            float_columns_: List of column names that are float64 dtype.

        """
        self.variance_threshold: float = variance_threshold
        self.pipeline_: Pipeline | None = None
        self.float_columns_: list[str] | None = None

    def fit(self, x: pd.DataFrame, y: Any = None) -> NumericPreprocessor:  # noqa: ARG002
        """Fit the transformer to the data.

        Identifies float64 columns and creates a preprocessing pipeline
        with variance thresholding and standard scaling.

        Args:
            x: Input data to fit the transformer on.
            y: Ignored. Present for API consistency.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If no float64 columns are found in the DataFrame.

        """
        # Identify float columns
        self.float_columns_ = x.select_dtypes(include=["float"]).columns.tolist()

        # Check if any float columns exist
        if not self.float_columns_:
            error_msg: str = "No float columns found in DataFrame for preprocessing"
            raise ValueError(error_msg)

        # Get column indices for ColumnTransformer
        float_col_indices: list[int] = [x.columns.get_loc(col) for col in self.float_columns_]

        # Define preprocessing steps for float columns
        float_steps: list[tuple[str, BaseEstimator]] = [
            ("variance_threshold", VarianceThreshold(threshold=self.variance_threshold)),
            ("scaler", StandardScaler()),
        ]

        # Create pipeline for float columns
        float_pipeline: Pipeline = Pipeline(steps=float_steps)

        # Create column transformer
        preprocessor: ColumnTransformer = ColumnTransformer(
            transformers=[
                ("float", float_pipeline, float_col_indices)
            ],
            remainder="passthrough",
            verbose_feature_names_out=False
        )

        # Create and fit the main pipeline
        self.pipeline_ = Pipeline([("preprocessor", preprocessor)])

        try:
            # Fit the pipeline
            self.pipeline_.fit(x)

        # Catch any exceptions during fitting
        except Exception as e:
            error_msg = f"Failed to fit preprocessing pipeline: {e!s}"
            raise RuntimeError(error_msg) from e
        else:
            return self

    def transform(self, x: pd.DataFrame) -> NDArray[Any]:
        """Transform the input data using the fitted preprocessing pipeline.

        Args:
            x: Input data to transform.

        Returns:
            Transformed data array with preprocessed numerical features.

        Raises:
            ValueError: If the transformer has not been fitted yet.
            RuntimeError: If transformation fails.

        """
        # Check if transformer is fitted
        if self.pipeline_ is None:
            error_msg: str = "This Numeric Preprocessor instance is not fitted yet"
            raise ValueError(error_msg)

        try:
            # Transform the data
            transformed_data: NDArray[Any] = self.pipeline_.transform(x)
        except Exception as e:
            error_msg = f"Failed to transform data: {e!s}"
            raise RuntimeError(error_msg) from e
        else:
            return transformed_data

    def fit_transform(self, x: pd.DataFrame, y: Any = None) -> NDArray[Any]:  # noqa: ARG002
        """Fit the transformer and transform the data in one step.

        Args:
            x: Input data to fit and transform.
            y: Ignored. Present for API consistency.

        Returns:
            Transformed data array with preprocessed numerical features.

        """
        return self.fit(x).transform(x)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Get output feature names for transformation.

        Args:
            input_features: Input feature names. If None, uses fitted feature names.

        Returns:
            List of output feature names after transformation.

        Raises:
            ValueError: If the transformer has not been fitted yet.

        """
        if self.pipeline_ is None:
            error_msg: str = "This NumericPreprocessor instance is not fitted yet"
            raise ValueError(error_msg)

        try:
            feature_names: list[str] = list(self.pipeline_.get_feature_names_out(input_features))
        except Exception as e:
            error_msg = f"Failed to get feature names: {e!s}"
            raise RuntimeError(error_msg) from e
        else:
            return feature_names

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
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
            transformer_: The fitted preprocessing transformer.
            float_columns_: List of column names that are float64 dtype.

        """
        self.variance_threshold: float = variance_threshold
        self.transformer_: ColumnTransformer | None = None
        self.float_columns_: list[str] | None = None

    @staticmethod
    def check_is_fitted_(instance: Any, attributes: list[str]) -> None:
        """Check if the estimator instance is fitted by verifying the presence of attributes.

        Args:
            instance: The estimator instance to check.
            attributes: List of attribute names that should be present if fitted.

        Returns:
            None

        Raises:
            ValueError: If any of the specified attributes are not found in the instance.

        """
        # Identify missing attributes
        missing_attrs = [attr for attr in attributes if not hasattr(instance, attr)]

        # If any attributes are missing, raise ValueError
        if missing_attrs:
            error_msg = (
                f"This {instance.__class__.__name__} instance is not fitted yet. "
                f"Missing attributes: {missing_attrs}"
            )
            raise ValueError(error_msg)

    def fit(self, x: pd.DataFrame, y: Any = None) -> NumericPreprocessor:  # noqa: ARG002
        """Fit the transformer to the data.

        Identifies float columns, creates a preprocessing pipeline
        and fits it to learn the transformation parameters.

        Args:
            x: Input data to fit the transformer on.
            y: Ignored. Present for API consistency.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If no float columns are found in the DataFrame.
            RuntimeError: If fitting the preprocessing pipeline fails.

        """
        # Identify float columns
        self.float_columns_ = x.select_dtypes(include=["float64"]).columns.tolist()

        # Check if any float columns exist
        if not self.float_columns_:
            error_msg: str = "No float64 columns found in DataFrame for preprocessing"
            raise ValueError(error_msg)

        # Define preprocessing steps for float columns
        float_steps: list[tuple[str, BaseEstimator]] = [
            ("variance_threshold", VarianceThreshold(threshold=self.variance_threshold)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]

        # Create pipeline for float columns
        float_pipeline: Pipeline = Pipeline(steps=float_steps)

        # Create column transformer using column names (more robust than indices)
        self.transformer_ = ColumnTransformer(
            transformers=[
                ("float", float_pipeline, self.float_columns_)
            ],
            remainder="passthrough",  # Keep non-float columns unchanged
            verbose_feature_names_out=False  # Avoid adding transformer names to output feature names
        )

        try:
            # Fit the transformer to learn parameters
            self.transformer_.fit(x)

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
        self.check_is_fitted_(self, ["transformer_"])

        # Assert transformer_ is not None for type checkers
        assert self.transformer_ is not None  # nosec

        try:
            # Apply the learned transformation
            transformed_data: NDArray[Any] = self.transformer_.transform(x)

        # Catch any exceptions during transformation
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
            RuntimeError: If getting feature names fails.

        """
        # Check if transformer is fitted
        self.check_is_fitted_(self, ["transformer_"])

        # Assert transformer_ is not None for type checkers
        assert self.transformer_ is not None  # nosec

        try:
            feature_names: list[str] = list(self.transformer_.get_feature_names_out(input_features))

        # Catch any exceptions during getting feature names
        except Exception as e:
            error_msg = f"Failed to get feature names: {e!s}"
            raise RuntimeError(error_msg) from e

        else:
            return feature_names

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import gower
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

if TYPE_CHECKING:
    from numpy.typing import NDArray

warnings.filterwarnings("ignore")


class GowerDistanceTransformer(BaseEstimator, TransformerMixin):
    """Computes the Gower Distance matrix for mixed-type data.

    This transformer calculates pairwise Gower distances between samples in a dataset
    containing mixed-type features (numerical and categorical). Gower distance is
    particularly useful for clustering heterogeneous data.
    """

    def __init__(self, cat_features: list[str] | None = None) -> None:
        """Initialize the GowerDistanceTransformer.

        Args:
            cat_features: Specification of categorical features by their column names.

        Attributes:
            cat_features: List of categorical feature names.
            cat_features_bool_: Boolean mask indicating which features are categorical.

        """
        self.cat_features: list[str] | None = cat_features
        self.cat_features_bool_: NDArray[np.bool_] | None = None

    def fit(self, x: pd.DataFrame, y: Any = None) -> GowerDistanceTransformer:  # noqa: ARG002
        """Fit the transformer to the data.

        Determines which features are categorical based on the cat_features parameter
        or by auto-detecting object columns in DataFrames.

        Args:
            x: Input data to fit the transformer on.
            y: Ignored. Present for API consistency.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If specified categorical features are not found in DataFrame.
            TypeError: If specified categorical features are not of object dtype.

        """
        # If categorical features are not specified
        if self.cat_features is None:

            # Auto-detect categorical features (object dtype)
            self.cat_features_bool_ = x.dtypes == "object"

        # If categorical features are specified by name
        else:

            # If any specified categorical features are missing
            if not pd.Index(self.cat_features).isin(x.columns).all():
                # Get list of missing features
                missing_features = [feat for feat in self.cat_features if feat not in x.columns]
                # Set error message
                error_msg: str = f"Categorical features not found in DataFrame: {missing_features}"
                # Raise error
                raise ValueError(error_msg)

            # If any specified categorical features are not of object dtype
            if pd.Index(self.cat_features).isin(x.select_dtypes(include=["object"]).columns).all():
                # Get list of non-object dtype features
                non_object_features = [feat for feat in self.cat_features if x[feat].dtype != "object"]
                # Set error message
                error_msg = f"Specified categorical features must be of object dtype: {non_object_features}"
                # Raise error
                raise TypeError(error_msg)

            # Create boolean mask based on provided categorical feature names
            self.cat_features_bool_ = x.columns.isin(self.cat_features)

        return self

    def transform(self, x: pd.DataFrame) -> NDArray[np.float32]:
        """Transform the input data into a Gower distance matrix.

        Args:
            x: Input data to transform into a distance matrix.

        Returns:
            Square distance matrix of shape (n_samples, n_samples) with Gower distances.

        Raises:
            RuntimeError: If Gower distance calculation fails.

        """
        try:
            # Compute Gower distance matrix
            dist_matrix: NDArray[np.floating] = gower.gower_matrix(x, cat_features=self.cat_features_bool_)

            # Handle potential numerical instability
            # Set 1.0 for any NaN distances, i.e., max distance
            dist_matrix_cleaned: NDArray[np.floating] = np.nan_to_num(dist_matrix, nan=1.0)

            # Ensure diagonal is zero
            np.fill_diagonal(dist_matrix_cleaned, 0.0)

            return dist_matrix_cleaned.astype(np.float32)

        # Catch any errors during Gower calculation
        except Exception as e:
            error_msg: str = f"Gower calculation failed: {e!s}"
            raise RuntimeError(error_msg) from e

    def fit_transform(
        self, x: pd.DataFrame | np.ndarray,
        y: Any = None  # noqa: ARG002
    ) -> NDArray[np.float32]:
        """Fit the transformer and transform the data in one step.

        Args:
            x: Input data to fit and transform.
            y: Ignored. Present for API consistency.

        Returns:
            NDArray[np.float32]: Gower distances matrix.

        """
        return self.fit(x).transform(x)

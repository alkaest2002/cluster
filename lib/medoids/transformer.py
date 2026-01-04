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

    def __init__(self, cat_features: list[str] | list[int] | list[bool] | None = None) -> None:
        """Initialize the GowerDistanceTransformer.

        Args:
            cat_features: Specification of categorical features. Can be:
                - List of column names (for DataFrames)
                - List of column indices
                - List of booleans indicating categorical columns
                - None to auto-detect object columns in DataFrames

        """
        self.cat_features: list[str] | list[int] | list[bool] | None = cat_features
        self.cat_features_bool_: NDArray | None = None
        self.n_features_: int | None = None

    def fit(self, x: pd.DataFrame | NDArray, y: Any = None) -> GowerDistanceTransformer:  # noqa: ARG002
        """Fit the transformer to the data.

        Determines which features are categorical based on the cat_features parameter
        or by auto-detecting object columns in DataFrames.

        Args:
            x: Input data to fit the transformer on.
            y: Ignored. Present for API consistency.

        Returns:
            Self for method chaining.

        """
        # If x is a DataFrame
        if isinstance(x, pd.DataFrame):
            # Determine boolean mask for categorical features
            self.n_features_ = x.shape[1]
            # If cat_features is provided
            if self.cat_features is not None:
                # If user passed a list of column names
                if self.cat_features and isinstance(self.cat_features[0], str):
                    cat_mask: pd.Series[bool] = x.columns.isin(self.cat_features)
                    self.cat_features_bool_ = cat_mask.values
                # If user passed indices
                elif self.cat_features and isinstance(self.cat_features[0], int):
                    bool_mask: NDArray = np.zeros(x.shape[1], dtype=bool)
                    bool_mask[self.cat_features] = True
                    self.cat_features_bool_ = bool_mask
                # If user passed booleans
                else:
                    self.cat_features_bool_ = np.array(self.cat_features, dtype=bool)
            # If cat_features is not provided
            else:
                # Auto-detect object columns if None provided
                dtype_mask: pd.Series[bool] = x.dtypes == "object"
                self.cat_features_bool_ = dtype_mask.values
        # If x is a numpy array
        else:
            # Fallback for numpy arrays (assumes cat_features is provided manually)
            self.cat_features_bool_ = (
                np.array(self.cat_features, dtype=bool) if self.cat_features is not None else None
            )

        return self

    def transform(self, x: pd.DataFrame | NDArray) -> NDArray[np.float32]:
        """Transform the input data into a Gower distance matrix.

        Args:
            x: Input data to transform into a distance matrix.

        Returns:
            Square distance matrix of shape (n_samples, n_samples) with Gower distances.

        Raises:
            RuntimeError: If Gower distance calculation fails.

        """
        try:
            # Convert to DataFrame if needed
            x_df: pd.DataFrame = pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x

            # Compute Gower distance matrix
            dist_matrix: NDArray[np.floating] = gower.gower_matrix(x_df, cat_features=self.cat_features_bool_)

            # Handle potential numerical instability
            # Set 1.0 for any NaN distances, i.e., max distance
            clean_matrix: NDArray[np.floating] = np.nan_to_num(dist_matrix, nan=1.0)

            # Ensure diagonal is zero
            np.fill_diagonal(clean_matrix, 0.0)

            return clean_matrix.astype(np.float32)

        # Catch any errors during Gower calculation
        except Exception as e:
            error_msg: str = f"Gower calculation failed: {e!s}"
            raise RuntimeError(error_msg) from e

    def fit_transform(self, x: pd.DataFrame | np.ndarray, y: Any = None) -> NDArray[np.float32]:  # noqa: ARG002
        """Fit the transformer and transform the data in one step.

        Args:
            x: Input data to fit and transform.
            y: Ignored. Present for API consistency.

        Returns:
            Square distance matrix of shape (n_samples, n_samples) with Gower distances.

        """
        return self.fit(x).transform(x)

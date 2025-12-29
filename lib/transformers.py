import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from typing import Any, Hashable, TypeAlias 

ArrayLike2D: TypeAlias = pd.DataFrame | np.ndarray | list[list[Any]]


class CardinalityAwareOrdinalEncoder(BaseEstimator, TransformerMixin):
    """Ordinal-encode categorical features with a strategy based on cardinality.

    This encoder chooses how many categories to keep per feature based on the
    number of unique values (cardinality):

    * Low cardinality: keep all categories.
    * Medium cardinality: keep categories meeting a minimum frequency threshold.
    * High cardinality: keep the union of (top N most frequent) and those above
      a minimum frequency percentage.

    Categories not kept are mapped to a special "__OTHER__" bucket (encoded as 0)
    when applicable. Unknown categories at transform time can either be mapped to
    `unknown_value` or raise an error depending on `handle_unknown`.

    Attributes:
        categories_: Mapping from feature name to the list of kept categories.
        strategies_: Mapping from feature name to the selected strategy name.
        encodings_: Mapping from feature name to a dict of category -> integer code.
        n_features_in_: Number of features seen during fit.
        feature_names_in_: Feature names seen during fit (as a list of strings).

    Args:
        low_card_threshold: Features with unique values <= this are treated as
            low-cardinality and all categories are kept.
        medium_card_threshold: Threshold separating medium and high cardinality.
            Features with unique values <= this and > low_card_threshold are
            treated as medium-cardinality.
        min_frequency_pct: Minimum frequency (as a fraction of samples) for a
            category to be kept in medium-cardinality features.
        min_absolute_count: Minimum absolute count for a category to be kept in
            medium-cardinality features.
        high_card_top_n: For high-cardinality features, keep the top N categories
            by frequency.
        high_card_freq_pct: For high-cardinality features, keep categories whose
            frequency is at least this fraction of samples.
        handle_unknown: If "use_encoded_value", unknown categories are encoded as
            `unknown_value`. Otherwise, unknown categories raise a ValueError.
        unknown_value: Encoded value for unknown categories at transform time
            when `handle_unknown="use_encoded_value"`.
        dtype: Output dtype for the transformed array.
    """

    def __init__(
        self,
        low_card_threshold: int = 5,
        medium_card_threshold: int = 20,
        min_frequency_pct: float = 0.005,  # 0.5%
        min_absolute_count: int = 3,
        high_card_top_n: int = 15,
        high_card_freq_pct: float = 0.01,  # 1%
        handle_unknown: str = "use_encoded_value",
        unknown_value: int = -1,
        dtype: type = np.int64,
    ) -> None:
        self.low_card_threshold: int = low_card_threshold
        self.medium_card_threshold: int = medium_card_threshold
        self.min_frequency_pct: float = min_frequency_pct
        self.min_absolute_count: int = min_absolute_count
        self.high_card_top_n: int = high_card_top_n
        self.high_card_freq_pct: float = high_card_freq_pct
        self.handle_unknown: str = handle_unknown
        self.unknown_value: int = unknown_value
        self.dtype: type = dtype

    def fit(
        self, X: ArrayLike2D, y: Any | None = None
    ) -> "CardinalityAwareOrdinalEncoder":
        """Fit the encoder on training data.

        Args:
            X: Training data of shape (n_samples, n_features). If not a DataFrame,
                it will be converted to one with generated column names.
            y: Ignored. Present for API consistency.

        Returns:
            Fitted encoder instance.
        """
        # Validate and convert input to DataFrame
        X_df: pd.DataFrame = self._validate_input(X)
        
        # Get number of samples
        n_samples: int = len(X_df)

        # Initialize storage for categories, strategies, and encodings
        self.categories_: dict[str, list[Hashable]] = {}
        self.strategies_: dict[str, str] = {}
        self.encodings_: dict[str, dict[Hashable, int]] = {}

        # Determine strategies and encodings per feature
        for col_name in X_df.columns:
            
            # Compute unique count
            n_unique: int = int(X_df[col_name].nunique())
            
            # Compute value counts
            counts = X_df[col_name].value_counts()

            # Decide strategy based on cardinality
            # Low cardinality
            if n_unique <= self.low_card_threshold:
                keep_categories = counts.index.tolist()
                self.strategies_[col_name] = "low_cardinality"
            # Medium cardinality
            elif n_unique <= self.medium_card_threshold:
                min_count: float = max(self.min_absolute_count, self.min_frequency_pct * n_samples)
                keep_categories = counts[counts >= min_count].index.tolist()
                self.strategies_[col_name] = "medium_cardinality"
            # High cardinality 
            else:
                top_n = counts.head(self.high_card_top_n).index
                freq_threshold = counts[counts >= self.high_card_freq_pct * n_samples].index
                keep_categories = list(set(top_n) | set(freq_threshold))
                self.strategies_[col_name] = "high_cardinality"

            # Store kept categories
            self.categories_[col_name] = keep_categories

            # Create encoding dict
            if len(keep_categories) < n_unique:
                encoding_dict: dict[Hashable, int] = {cat: i + 1 for i, cat in enumerate(keep_categories)}
                encoding_dict["__OTHER__"] = 0
            # All categories kept
            else:
                encoding_dict = {cat: i for i, cat in enumerate(keep_categories)}

            # Store encoding
            self.encodings_[col_name] = encoding_dict

        # Store feature info
        self.n_features_in_ = X_df.shape[1]
        self.feature_names_in_ = X_df.columns.tolist()
        
        return self

    def transform(self, X: ArrayLike2D) -> np.ndarray:
        """Transform data into ordinal-encoded numpy array.

        Args:
            X: Data to transform of shape (n_samples, n_features).

        Returns:
            Ordinal-encoded array of shape (n_samples, n_features).
        """
        # Check if fitted
        check_is_fitted(self)

        # Validate and convert input to DataFrame
        X_df: pd.DataFrame = self._validate_input(X)

        # Initialize output array
        X_transformed: np.ndarray = np.empty((X_df.shape[0], X_df.shape[1]), dtype=self.dtype)

        # Apply encoding per feature
        for col_idx, col_name in enumerate(X_df.columns):

            # Get column data
            col_data: pd.Series = X_df[col_name].copy()

            # Get encoding dict
            encoding_dict: dict[Hashable, int] = self.encodings_[col_name]
            
            # Map unknown categories to "__OTHER__" if applicable
            if "__OTHER__" in encoding_dict:
                # Create mask for known categories
                mask_known: pd.Series = col_data.isin(self.categories_[col_name])
                col_data = col_data.where(mask_known, "__OTHER__")

            # Handle unknown categories
            if self.handle_unknown == "use_encoded_value":
                encoded_col = col_data.map(encoding_dict).fillna(self.unknown_value)
            else:
                encoded_col = col_data.map(encoding_dict)
                if encoded_col.isna().any():
                    raise ValueError(f"Unknown categories found in column {col_name}")

            # Assign to output array
            X_transformed[:, col_idx] = encoded_col.astype(self.dtype)

        return X_transformed

    def inverse_transform(self, X: Any) -> pd.DataFrame:
        """Inverse transform ordinal codes back to categories.

        Args:
            X: Encoded data of shape (n_samples, n_features).

        Returns:
            DataFrame with original categories (with "__OTHER__" rendered as "other").
        """
        # Check if fitted
        check_is_fitted(self)

        # Convert input to numpy array
        X_arr = np.asarray(X)

        # Initialize output DataFrame
        X_original = pd.DataFrame(index=range(X_arr.shape[0]), columns=self.feature_names_in_)

        # Inverse map per feature
        for col_idx, col_name in enumerate(self.feature_names_in_):

            # Create reverse encoding dict
            reverse_encoding: dict[int, Hashable] = {v: k for k, v in self.encodings_[col_name].items()}

            # Map encoded values back to original categories
            col_data = pd.Series(X_arr[:, col_idx])
            
            # Perform the mapping
            X_original[col_name] = col_data.map(reverse_encoding)
            
            # Replace unknown_value with NaN
            X_original[col_name] = X_original[col_name].replace("__OTHER__", "other")

        return X_original

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        """Get output feature names.

        Args:
            input_features: Optional list of input feature names.

        Returns:
            Numpy array of output feature names.
        """
        check_is_fitted(self)
        if input_features is None:
            return np.array(self.feature_names_in_)
        return np.array(input_features)

    def _validate_input(self, X: ArrayLike2D) -> pd.DataFrame:
        """Validate and convert input to a pandas DataFrame.

        Args:
            X: 2D array-like input.

        Returns:
            Input as a DataFrame.

        Raises:
            ValueError: If X is not a 2D array-like structure.
        """
        if isinstance(X, pd.DataFrame):
            return X

        if hasattr(X, "shape") and len(getattr(X, "shape")) == 2:
            X_arr: np.ndarray = np.asarray(X)
            return pd.DataFrame(X_arr, columns=[f"feature_{i}" for i in range(X_arr.shape[1])])

        raise ValueError("Input must be a 2D array-like structure")

    def get_cardinality_info(self) -> dict[str, dict[str, Any]]:
        """Return cardinality strategy information per feature.

        Returns:
            Dict keyed by feature name with details about strategy and category counts.
        """
        check_is_fitted(self)
        info: dict[str, dict[str, Any]] = {}
        for col_name in self.feature_names_in_:
            info[col_name] = {
                "strategy": self.strategies_[col_name],
                "original_categories": len(self.categories_[col_name]),
                "encoded_categories": len(self.encodings_[col_name]),
                "has_other_category": "__OTHER__" in self.encodings_[col_name],
            }
        return info

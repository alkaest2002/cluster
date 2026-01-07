from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.base import BaseEstimator, ClusterMixin

from lib.k_prototypes.utils import k_prototypes_distance, k_prototypes_silhouette_scorer

if TYPE_CHECKING:
    from numpy.typing import NDArray

warnings.filterwarnings("ignore")


class KPrototypesWrapper(BaseEstimator, ClusterMixin):
    """Wrapper for kmodes KPrototypes to ensure Scikit-Learn compatibility.

    This wrapper provides a scikit-learn compatible interface for the k-prototypes
    clustering algorithm from the kmodes package, with integrated silhouette scoring
    using custom k-prototypes distance.

    Attributes:
        kprototypes_: KPrototypes instance from kmodes.
        n_clusters: Number of clusters.
        gamma: Weight parameter for categorical distance component.
        labels_: Cluster labels for each sample.
        cluster_centroids_: Cluster centroids (numeric and categorical).
        cost_: Final cost of the clustering.
        n_iter_: Number of iterations run.
        silhouette_score_: Silhouette score using k-prototypes distance.
        numeric_features_: List of numeric feature names.
        categorical_features_: List of categorical feature names.

    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        n_clusters: int = 2,
        gamma: float = 1.0,
        max_iter: int = 100,
        n_init: int = 10,
        init: str = "Huang",
        verbose: int = 0,
        random_state: int | None = 42,
        n_jobs: int = 1,
    ) -> None:
        """Initialize the KPrototypesWrapper.

        Args:
            n_clusters: Number of clusters to form.
            gamma: Weight parameter for categorical distance component.
            max_iter: Maximum number of iterations.
            n_init: Number of random initializations.
            init: Initialization method ('Huang' or 'Cao').
            verbose: Verbosity level.
            random_state: Random state for reproducibility.
            n_jobs: Number of parallel jobs.

        """
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.max_iter = max_iter
        self.n_init = n_init
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Initialize kmodes KPrototypes instance
        self.kprototypes_ = KPrototypes(
            n_clusters=n_clusters,
            gamma=gamma,
            max_iter=max_iter,
            n_init=n_init,
            init=init,
            verbose=verbose,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        # Attributes set during fit
        self.labels_: pd.Series = pd.Series(dtype=np.int32)
        self.cluster_centroids_: list[tuple[NDArray[np.float64], NDArray[Any]]] = []
        self.cost_: float = np.inf
        self.n_iter_: int = 0
        self.silhouette_score_: float = -1.0
        self.numeric_features_: list[str] = []
        self.categorical_features_: list[str] = []

    @property
    def inertia_(self) -> float:
        """Get the final cost as inertia for sklearn compatibility."""
        return self.cost_

    @staticmethod
    def validate_input_data_(
        numeric_df: pd.DataFrame,
        categorical_df: pd.DataFrame
    ) -> None:
        """Validate input data dimensions and types.

        Args:
            numeric_df: Numeric feature DataFrame.
            categorical_df: Categorical feature DataFrame.

        Raises:
            ValueError: If input data is invalid.

        """
        # Check for empty data
        if numeric_df.empty and categorical_df.empty:
            error_msg: str = "Both numeric and categorical data cannot be empty."
            raise ValueError(error_msg)

        # Check for consistent number of samples
        if not numeric_df.empty and not categorical_df.empty and len(numeric_df) != len(categorical_df):
            error_msg = "Numeric and categorical data must have same number of samples."
            raise ValueError(error_msg)

    def fit(
        self,
        x: tuple[pd.DataFrame, pd.DataFrame],
        y: Any = None  # noqa: ARG002
    ) -> KPrototypesWrapper:
        """Fit the k-prototypes clustering algorithm.

        Args:
            x: Tuple of (numeric_features_df, categorical_features_df).
            y: Ignored. Present for API consistency.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If input data is invalid or clustering fails.

        """
        try:
            numeric_df, categorical_df = x

            # Validate input
            self.validate_input_data_(numeric_df, categorical_df)

            # Store feature names
            self.numeric_features_ = numeric_df.columns.tolist()
            self.categorical_features_ = categorical_df.columns.tolist()

            # Convert to numpy for kmodes computation
            x_num = numeric_df.values.astype(np.float64) if not numeric_df.empty else np.array([])
            x_cat = categorical_df.values if not categorical_df.empty else np.array([])

            # Prepare data for kmodes (expects single array with categorical indices)
            if x_num.size > 0 and x_cat.size > 0:
                # Combine numeric and categorical data
                combined_data = np.column_stack([x_num, x_cat])
                categorical_indices = list(range(x_num.shape[1], combined_data.shape[1]))
            elif x_num.size > 0:
                # Only numeric data
                combined_data = x_num
                categorical_indices = []
            else:
                # Only categorical data
                combined_data = x_cat
                categorical_indices = list(range(x_cat.shape[1]))

            # Fit the model
            self.kprototypes_.fit(combined_data, categorical=categorical_indices)

            # Store results as pandas Series with original index
            n_samples = len(numeric_df) if not numeric_df.empty else len(categorical_df)
            index = numeric_df.index if not numeric_df.empty else categorical_df.index

            self.labels_ = pd.Series(
                self.kprototypes_.labels_.astype(np.int32),
                index=index,
                name="cluster"
            )
            self.cluster_centroids_ = self.kprototypes_.cluster_centroids_
            self.cost_ = self.kprototypes_.cost_
            self.n_iter_ = self.kprototypes_.n_iter_

            # Calculate silhouette score using k-prototypes distance
            if len(self.labels_.unique()) > 1:
                # Use our custom distance function for silhouette score
                distance_matrix = k_prototypes_distance(
                    x_num=x_num,
                    x_cat=x_cat,
                    gamma=self.gamma,
                    standardize_num_scales=True,
                )
                self.silhouette_score_ = k_prototypes_silhouette_scorer(self, distance_matrix)
            else:
                self.silhouette_score_ = -1.0

        except Exception:
            # Handle clustering failure gracefully
            n_samples = len(numeric_df) if not numeric_df.empty else len(categorical_df)
            index = numeric_df.index if not numeric_df.empty else categorical_df.index

            self.labels_ = pd.Series(
                np.zeros(n_samples, dtype=np.int32),
                index=index,
                name="cluster"
            )
            self.cluster_centroids_ = []
            self.cost_ = np.inf
            self.n_iter_ = 0
            self.silhouette_score_ = -1.0

        return self

    def predict(
        self,
        x: tuple[pd.DataFrame, pd.DataFrame]
    ) -> pd.Series:
        """Predict cluster labels for new data.

        Args:
            x: Tuple of (numeric_features_df, categorical_features_df).

        Returns:
            Predicted cluster labels as pandas Series.

        Raises:
            AttributeError: If the model has not been fitted yet.

        """
        if len(self.labels_) == 0:
            error_msg: str = "Model must be fitted before making predictions."
            raise AttributeError(error_msg)

        numeric_df, categorical_df = x

        # Convert to numpy for prediction
        x_num = numeric_df.values.astype(np.float64) if not numeric_df.empty else np.array([])
        x_cat = categorical_df.values if not categorical_df.empty else np.array([])

        # Prepare data same way as in fit
        if x_num.size > 0 and x_cat.size > 0:
            combined_data = np.column_stack([x_num, x_cat])
            categorical_indices = list(range(x_num.shape[1], combined_data.shape[1]))
        elif x_num.size > 0:
            combined_data = x_num
            categorical_indices = []
        else:
            combined_data = x_cat
            categorical_indices = list(range(x_cat.shape[1]))

        predictions = self.kprototypes_.predict(
            combined_data,
            categorical=categorical_indices
        ).astype(np.int32)

        # Return as pandas Series with original index
        index = numeric_df.index if not numeric_df.empty else categorical_df.index
        return pd.Series(predictions, index=index, name="cluster")

    def fit_predict(
        self,
        x: tuple[pd.DataFrame, pd.DataFrame],
        y: Any = None
    ) -> pd.Series:
        """Fit the model and predict cluster labels.

        Args:
            x: Tuple of (numeric_features_df, categorical_features_df).
            y: Ignored. Present for API consistency.

        Returns:
            Cluster labels for the input data as pandas Series.

        """
        return self.fit(x, y).labels_

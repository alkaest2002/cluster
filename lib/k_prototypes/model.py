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

    def fit(self, X: pd.DataFrame) -> KPrototypesWrapper:  # ignore[mypy-note]
        """Fit the K-Prototypes clustering to the data.

        Args:
            X: Input DataFrame with mixed numeric and categorical features.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If input data is invalid.

        """
        # Store feature information
        self.numeric_features_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features_ = X.select_dtypes(include=["object"]).columns.tolist()

        # Validate we have at least some data
        if not self.numeric_features_ and not self.categorical_features_:
            error_msg: str = "DataFrame must contain at least one numeric or categorical column."
            raise ValueError(error_msg)

        # Identify categorical column indices for kmodes
        categorical_indices: list[int] = [X.columns.get_loc(col) for col in self.categorical_features_]

        # Fit the k-prototypes model
        self.kprototypes_.fit(X.values, categorical=categorical_indices)

        # Extract results
        self.labels_ = pd.Series(self.kprototypes_.labels_, index=X.index, dtype=np.int32)
        self.cluster_centroids_ = self.kprototypes_.cluster_centroids_
        self.cost_ = float(self.kprototypes_.cost_)
        self.n_iter_ = int(self.kprototypes_.n_iter_)

        # Calculate silhouette score
        x_numeric: NDArray[np.float64] = (
            X[self.numeric_features_].values
            if self.numeric_features_
            else np.array([]).reshape(len(X), 0)
        )
        x_categorical: NDArray[Any] = (
            X[self.categorical_features_].values
            if self.categorical_features_
            else np.array([]).reshape(len(X), 0)
        )

        distance_matrix = k_prototypes_distance(
            x_num=x_numeric,
            x_cat=x_categorical,
            gamma=self.gamma,
            standardize_num_scales=True
        )

        self.silhouette_score_ = k_prototypes_silhouette_scorer(self, distance_matrix)

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict cluster labels for new data points.

        Args:
            X: Input DataFrame with mixed numeric and categorical features.

        Returns:
            Series with cluster labels for each sample.

        Raises:
            ValueError: If model is not fitted or input data is incompatible.

        """
        # Check if model is fitted
        if not hasattr(self.kprototypes_, "cluster_centroids_"):
            error_msg: str = "Model must be fitted before calling predict."
            raise ValueError(error_msg)

        # Validate feature consistency
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

        if numeric_cols != self.numeric_features_ or categorical_cols != self.categorical_features_:
            error_msg = "Input features don't match training data."
            raise ValueError(error_msg)

        # Identify categorical column indices
        categorical_indices = [X.columns.get_loc(col) for col in self.categorical_features_]

        # Predict cluster labels
        labels = self.kprototypes_.predict(X.values, categorical=categorical_indices)

        return pd.Series(labels, index=X.index, dtype=np.int32)

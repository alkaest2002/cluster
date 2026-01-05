from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from kmedoids import KMedoids
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import silhouette_score

if TYPE_CHECKING:
    from numpy.typing import NDArray

warnings.filterwarnings("ignore")


class KMedoidsWrapper(BaseEstimator, ClusterMixin):
    """Wrapper for kmedoids-python to ensure Scikit-Learn compatibility.

    This wrapper provides a scikit-learn compatible interface for the kmedoids
    clustering algorithm, particularly for use with precomputed distance matrices.

    Attributes:
        kmedoids_: KMedoids instance.
        n_clusters: Number of clusters.
        labels_: Cluster labels for each sample.
        medoid_indices_: Indices of the medoids in the dataset.
        inertia_: Sum of distances of samples to their closest medoid.
        silhouette_score_: Silhouette score of the clustering.

    Methods:
        fit: Fit the k-medoids model to the data.
        predict: Predict the closest cluster for new samples.

    """

    def __init__(
        self,
        n_clusters: int = 2,
        method: str = "fasterpam",
        init: str = "build",
        max_iter: int = 100,
        random_state: int = 42
    ) -> None:
        """Initialize the KMedoidsWrapper.

        Args:
            n_clusters: Number of clusters to form.
            method: Algorithm variant to use ('fasterpam', 'pam', etc.).
            init: Initialization method for medoids selection.
            max_iter: Maximum number of iterations.
            random_state: Random state for reproducibility.

        """
        self.kmedoids_: KMedoids = KMedoids(
            n_clusters=n_clusters,
            metric="precomputed",
            method=method,
            init=init,
            max_iter=max_iter,
            random_state=random_state
        )
        self.n_clusters: int = n_clusters
        self.labels_: NDArray[np.int32] = np.array([], dtype=np.int32)
        self.medoid_indices_: NDArray[np.int32] = np.array([], dtype=np.int32)
        self.inertia_: float | None = None
        self.silhouette_score_: float | None = None

    @staticmethod
    def validate_distance_matrix_(dist_matrix: NDArray[np.float32]) -> None:
        """Validate the distance matrix is square and non-empty.

        Args:
            dist_matrix: Distance matrix to validate.

        Raises:
            ValueError: If the distance matrix is not square or is empty.

        """
        # Check if distance matrix is empty
        if dist_matrix.size == 0:
            error_msg: str = "Distance matrix is empty."
            raise ValueError(error_msg)

        # Check if distance matrix is square
        if dist_matrix.shape[0] != dist_matrix.shape[1]:
            error_msg = "Distance matrix must be square."
            raise ValueError(error_msg)

    def fit(self, x: NDArray[np.floating], y: Any = None) -> KMedoidsWrapper:  # noqa: ARG002
        """Fit the k-medoids clustering algorithm.

        Args:
            x: Precomputed square distance matrix of shape (n_samples, n_samples).
            y: Ignored. Present for API consistency.

        Returns:
            Self for method chaining.

        """
        try:
            # Fit model
            self.kmedoids_.fit(x)

            # Store attributes
            self.labels_ = self.kmedoids_.labels_
            self.medoid_indices_ = self.kmedoids_.medoid_indices_
            self.inertia_ = self.kmedoids_.inertia_

            # Calculate Scores
            if len(np.unique(self.labels_)) > 1:
                self.silhouette_score_ = silhouette_score(x, self.labels_, metric="precomputed")
            else:
                self.silhouette_score_ = -1.0

        except Exception:
            self.labels_ = np.zeros(x.shape[0], dtype=np.int32)
            self.medoid_indices_ = np.array([], dtype=np.int32)
            self.inertia_ = np.inf
            self.silhouette_score_ = -1.0

        return self

    def predict(self, x: NDArray[np.floating]) -> Any:
        """Predict the closest cluster for new samples.

        Args:
            x: New samples to predict clusters for.

        Returns:
            Cluster labels for the new samples.

        Raises:
            AttributeError: If the model has not been fitted yet.

        """
        if not hasattr(self, "labels_") or len(self.labels_) == 0:
            error_msg: str = "Model must be fitted before making predictions."
            raise AttributeError(error_msg)

        return self.kmedoids_.predict(x)

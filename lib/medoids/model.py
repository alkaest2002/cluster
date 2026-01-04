from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from kmedoids import KMedoids
from sklearn.base import BaseEstimator, ClusterMixin

if TYPE_CHECKING:
    from numpy.typing import NDArray

warnings.filterwarnings("ignore")


class KMedoidsWrapper(BaseEstimator, ClusterMixin):
    """Wrapper for kmedoids-python to ensure Scikit-Learn compatibility.

    This wrapper provides a scikit-learn compatible interface for the kmedoids
    clustering algorithm, particularly for use with precomputed distance matrices.
    """

    def __init__(
        self,
        n_clusters: int = 3,
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
        self.n_clusters: int = n_clusters
        self.method: str = method
        self.init: str = init
        self.max_iter: int = max_iter
        self.random_state: int = random_state

        # Attributes set during fitting
        self.kmedoids_: KMedoids | None = None
        self.labels_: NDArray[np.int_] | None = None
        self.medoid_indices_: NDArray[np.int_] | None = None
        self.inertia_: float | None = None
        self.cluster_centers_: None = None

    def fit(self, x: NDArray[np.floating], y: Any = None) -> KMedoidsWrapper:  # noqa: ARG002
        """Fit the k-medoids clustering algorithm.

        Args:
            x: Precomputed square distance matrix of shape (n_samples, n_samples).
            y: Ignored. Present for API consistency.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If x is not a square matrix.

        """
        # Must be square matrix
        if x.shape[0] != x.shape[1]:
            error_msg: str = f"Input must be a square distance matrix. Got shape {x.shape}."
            raise ValueError(error_msg)

        try:
            # Instantiate KMedoids model
            self.kmedoids_ = KMedoids(
                n_clusters=self.n_clusters,
                metric="precomputed",
                method=self.method,
                init=self.init,
                max_iter=self.max_iter,
                random_state=self.random_state
            )

            # Fit model
            self.kmedoids_.fit(x)

            # Store attributes
            self.labels_ = self.kmedoids_.labels_
            self.medoid_indices_ = self.kmedoids_.medoid_indices_
            self.inertia_ = self.kmedoids_.inertia_
            self.cluster_centers_ = None  # Not applicable for k-medoids

        except Exception as e:
            self.labels_ = np.zeros(x.shape[0], dtype=np.int_)
            self.inertia_ = np.inf
            error_msg = f"KMedoids fitting failed for k={self.n_clusters}: {e!s}"
            raise RuntimeError(error_msg) from e

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
        if self.kmedoids_ is None:
            error_msg: str = "Model must be fitted before making predictions."
            raise AttributeError(error_msg)

        return self.kmedoids_.predict(x)

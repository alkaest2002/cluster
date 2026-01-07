from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import silhouette_score


def k_prototypes_distance(
    x_num: npt.ArrayLike,
    x_cat: npt.ArrayLike,
    gamma: float = 1.0,
    standardize_num_scales: bool = True,
    dtype: npt.DTypeLike = np.float32,
) -> np.ndarray:
    """Calculate pairwise k-prototypes distances for mixed data.

    Args:
        x_num: Numeric feature matrix of shape (n_samples, n_numeric_features).
        x_cat: Categorical feature matrix of shape (n_samples, n_categorical_features).
        gamma: Weight parameter for categorical distance component.
        standardize_num_scales: Whether to standardize numeric features.
        dtype: Output data type for the distance matrix.

    Returns:
        Symmetric distance matrix of shape (n_samples, n_samples).

    """
    # Handle numeric data
    x_num_array = np.asarray(x_num, dtype=np.float64)

    # Handle categorical data
    x_cat_array = np.asarray(x_cat)

    # Number of samples
    n = x_cat_array.shape[0]

    # Numeric distance computation
    if x_num_array.size > 0:
        # Standardize numeric features if required
        if standardize_num_scales:
            mu = x_num_array.mean(axis=0)
            sd = x_num_array.std(axis=0, ddof=0)
            sd[sd == 0] = 1.0
            x_standardize_num_scalesd = (x_num_array - mu) / sd
        else:
            x_standardize_num_scalesd = x_num_array

        # Vectorized squared Euclidean distance
        squared_norms = np.sum(x_standardize_num_scalesd * x_standardize_num_scalesd, axis=1, keepdims=True)
        d_num = squared_norms + squared_norms.T - 2.0 * (x_standardize_num_scalesd @ x_standardize_num_scalesd.T)
        d_num = np.maximum(d_num, 0.0)
    else:
        d_num = np.zeros((n, n), dtype=np.float64)

    # Categorical distance computation
    if x_cat_array.size > 0:
        # Ensure categorical data is 2D
        ndim_2d: int = 2

        # Ensure x_cat_array is 2D
        # In the worst case scenario, we have only one categorical feature
        # so we need to add a new axis, i.e., convert shape (n,) to (n, 1)
        x_cat_2d: np.ndarray = x_cat_array if x_cat_array.ndim == ndim_2d else x_cat_array[:, None]

        # Initialize categorical distance matrix
        d_cat: np.ndarray = np.zeros((n, n), dtype=np.float64)

        # Compute Hamming distance for each categorical feature
        for j in range(x_cat_2d.shape[1]):
            # Update categorical distance matrix
            col = x_cat_2d[:, j]
            d_cat += (col[:, None] != col[None, :]).astype(np.float64)
    else:
        d_cat = np.zeros((n, n), dtype=np.float64)

    # Combine distances
    d_combined: np.ndarray = (d_num + gamma * d_cat).astype(dtype, copy=False)

    return d_combined


def k_prototypes_silhouette_scorer(estimator: Any, d: np.ndarray, y: Any = None) -> float:  # noqa: ARG001
    """Compute the silhouette score for the clustering estimator.

    Args:
        estimator: Fitted clustering estimator with `labels_` attribute.
        d: Precomputed distance matrix.
        y: Ignored. Present for API consistency.

    Returns:
        Silhouette score as a float.

    """
    labels = estimator.labels_
    return float(silhouette_score(d, labels, metric="precomputed"))

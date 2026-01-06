import numpy as np
import numpy.typing as npt

def kproto_pairwise_dist(
    X_num: npt.ArrayLike,
    X_cat: npt.ArrayLike,
    gamma: float = 1.0,
    num_scale: bool = True,
    dtype: npt.DTypeLike = np.float32,
) -> np.ndarray:
    """Calculate pairwise k-prototypes distances for mixed data.
    
    Args:
        X_num: Numeric feature matrix of shape (n_samples, n_numeric_features).
        X_cat: Categorical feature matrix of shape (n_samples, n_categorical_features).
        gamma: Weight parameter for categorical distance component.
        num_scale: Whether to standardize numeric features.
        dtype: Output data type for the distance matrix.
    
    Returns:
        Symmetric distance matrix of shape (n_samples, n_samples).
    """
    # Handle numeric data
    X_num_array = np.asarray(X_num, dtype=np.float64)

    # Handle categorical data
    X_cat_array = np.asarray(X_cat)
    
    # Number of samples
    n = X_cat_array.shape[0]
    
    # Numeric distance computation
    if X_num_array.size > 0:
        # Standardize numeric features if required
        if num_scale:
            mu = X_num_array.mean(axis=0)
            sd = X_num_array.std(axis=0, ddof=0)
            sd[sd == 0] = 1.0
            X_num_scaled = (X_num_array - mu) / sd
        else:
            X_num_scaled = X_num_array
            
        # Vectorized squared Euclidean distance
        squared_norms = np.sum(X_num_scaled * X_num_scaled, axis=1, keepdims=True)
        D_num = squared_norms + squared_norms.T - 2.0 * (X_num_scaled @ X_num_scaled.T)
        D_num = np.maximum(D_num, 0.0)
    else:
        D_num = np.zeros((n, n), dtype=np.float64)
    
    # Categorical distance computation
    if X_cat_array.size > 0:
        
        # Ensure X_cat_array is 2D
        X_cat_2d: npt.ArrayLike = X_cat_array if X_cat_array.ndim == 2 else X_cat_array[:, None]
        
        # Initialize categorical distance matrix
        D_cat: npt.ArrayLike = np.zeros((n, n), dtype=np.float64)
        
        # Compute Hamming distance for each categorical feature
        for j in range(X_cat_2d.shape[1]):
            # Update categorical distance matrix
            col = X_cat_2d[:, j]
            D_cat += (col[:, None] != col[None, :]).astype(np.float64)
    else:
        D_cat = np.zeros((n, n), dtype=np.float64)
    
    # Combine distances
    D_combined = D_num + gamma * D_cat

    return D_combined.astype(dtype, copy=False)

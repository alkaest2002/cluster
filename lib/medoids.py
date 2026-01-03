from __future__ import annotations

import io
import warnings
from base64 import b64encode
from typing import TYPE_CHECKING, Any

import gower
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kmedoids import KMedoids
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics import silhouette_score

if TYPE_CHECKING:
    from numpy.typing import NDArray

warnings.filterwarnings("ignore")

# set matplotlib font to sans-serif
plt.rcParams["font.family"] = "sans-serif"


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


class KMedoidsAnalyzer:
    """Analyzer class for k-medoids clustering optimization and visualization.

    This class provides methods for finding the optimal number of clusters,
    plotting evaluation metrics, and analyzing the resulting medoids.
    """

    def __init__(self, cat_features: list[str] | list[int] | list[bool] | None = None) -> None:
        """Initialize the KMedoidsAnalyzer.

        Args:
            cat_features: Specification of categorical features for Gower distance.

        """
        self.cat_features: list[str] | list[int] | list[bool] | None = cat_features
        self.transformer: GowerDistanceTransformer = GowerDistanceTransformer(cat_features=cat_features)
        self.best_model_: KMedoidsWrapper | None = None
        self.best_k_: int | None = None
        self.dist_matrix_: NDArray[np.float32] | None = None
        self.results_df_: pd.DataFrame | None = None

    def run_optimization(
        self,
        df: pd.DataFrame,
        k_min: int = 2,
        k_max: int = 50
    ) -> tuple[KMedoidsWrapper, int, NDArray[np.float32], pd.DataFrame]:
        """Run k-medoids optimization across multiple k values.

        This function computes the Gower distance matrix once and then evaluates
        k-medoids clustering for different numbers of clusters, using silhouette
        score and inertia as evaluation metrics.

        Args:
            df: Input DataFrame containing the data to cluster.
            k_min: Minimum number of clusters to evaluate.
            k_max: Maximum number of clusters to evaluate.

        Returns:
            Tuple containing:
                - Best k-medoids model (highest silhouette score)
                - Best number of clusters
                - Computed distance matrix
                - DataFrame with evaluation metrics for all k values

        """
        # 1. Compute Matrix
        self.dist_matrix_ = self.transformer.fit_transform(df)

        # Initialize vars
        results: list[dict[str, int | float | None]] = []
        best_score: float = -1
        best_model: KMedoidsWrapper | None = None
        best_k: int = 0

        # 2. Iterate
        for k in range(k_min, k_max + 1):

            # Instantiate and fit model
            model: KMedoidsWrapper = KMedoidsWrapper(n_clusters=k, random_state=42)
            model.fit(self.dist_matrix_)

            # Store labels
            labels: NDArray[np.int_] | None = model.labels_

            # Calculate Scores
            if labels is not None and len(np.unique(labels)) > 1:
                sil_score: float = silhouette_score(self.dist_matrix_, labels, metric="precomputed")
            else:
                sil_score = -1.0

            results.append({
                "k": k,
                "inertia": model.inertia_,
                "silhouette": sil_score
            })

            # Track best model
            if sil_score > best_score:
                best_score = sil_score
                best_model = model
                best_k = k

        self.results_df_ = pd.DataFrame(results)

        # Ensure best_model is not None before returning
        if best_model is None:
            best_model = KMedoidsWrapper(n_clusters=k_min, random_state=42)
            best_model.fit(self.dist_matrix_)

        # Store best results
        self.best_model_ = best_model
        self.best_k_ = best_k

        return best_model, best_k, self.dist_matrix_, self.results_df_

    def plot_metrics(self, results_df: pd.DataFrame | None = None) -> str:
        """Plot inertia and silhouette scores for different k values.

        Creates side-by-side plots showing the elbow curve (inertia) and silhouette
        analysis to help determine the optimal number of clusters.

        Args:
            results_df: DataFrame containing 'k', 'inertia', and 'silhouette' columns.
                       If None, uses the results from the last optimization run.

        Returns:
            Base64-encoded SVG string of the plot.

        """
        if results_df is None:
            if self.results_df_ is None:
                err_message: str = "No results available. Run optimization first."
                raise ValueError(err_message)
            results_df = self.results_df_

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Silhouette Score
        max_silhouette_value: float = results_df["silhouette"].max()
        best_n_clusters_series: pd.Series[int] = results_df.loc[results_df["silhouette"] == max_silhouette_value, "k"]
        best_n_clusters: int = best_n_clusters_series.values[0]
        ax.plot(results_df["k"], results_df["silhouette"], "ro-")
        ax.set_title("Silhouette Analysis")
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Silhouette Score (Higher is better)")
        ax.axvline(best_n_clusters, color="r")
        ax.grid(axis="y")

        plt.tight_layout()

        # Initialize an in-memory buffer
        buffer: io.BytesIO = io.BytesIO()

        # Save figure to the buffer in SVG format then close it
        fig.savefig(buffer, format="svg", bbox_inches="tight", transparent=True, pad_inches=0.05)

        # Close figure
        plt.close(fig)

        # Encode the buffer contents to a base64 string
        encoded_svg: str = b64encode(buffer.getvalue()).decode()
        return encoded_svg

    def analyze_medoids(
        self,
        df: pd.DataFrame,
        model: KMedoidsWrapper | None = None,
        dist_matrix: NDArray[np.float32] | None = None
    ) -> tuple[float, pd.DataFrame]:
        """Analyze and print detailed information about the cluster medoids.

        Displays the overall silhouette score and detailed information about each
        cluster including size and the actual medoid (representative sample).

        Args:
            df: Original DataFrame containing the clustered data.
            model: Fitted k-medoids model. If None, uses the best model from optimization.
            dist_matrix: Precomputed distance matrix used for clustering.
                        If None, uses the matrix from optimization.

        Returns:
            tuple[float, pd.DataFrame].

        """
        # Validate model
        if model is None:
            if self.best_model_ is None:
                err_message: str = "No model available. Run optimization first."
                raise ValueError(err_message)
            model = self.best_model_

        # Validate distance matrix
        if dist_matrix is None:
            if self.dist_matrix_ is None:
                err_message = "No distance matrix available. Run optimization first."
                raise ValueError(err_message)
            dist_matrix = self.dist_matrix_

        # Extract labels
        labels: NDArray[np.int_] | None = model.labels_

        # Extract medoid indices
        medoid_indices: NDArray[np.int_] | None = model.medoid_indices_

        # If labels exist, compute silhouette score
        if labels is not None:
            score: float = silhouette_score(dist_matrix, labels, metric="precomputed")

        # Clone dataframe to avoid modifying original
        df_clone: pd.DataFrame = df.copy()

        # If labels exist
        if labels is not None:
            # Assign cluster labels
            df_clone["cluster"] = labels
            # Calculate cluster sizes
            df_clone = df_clone.assign(cluster_size=df_clone.groupby("cluster").transform("size"))

        # Return medoids only
        if medoid_indices is not None:
            # Extract medoid rows
            medoids_df: pd.DataFrame = df_clone.iloc[medoid_indices]
            # Return silhouette score and medoids DataFrame
            return score, medoids_df

        # Fallback return full analysis if medoids not found
        return score, df_clone

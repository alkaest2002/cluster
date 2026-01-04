from __future__ import annotations

import io
import warnings
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from lib.medoids.model import KMedoidsWrapper
from lib.medoids.transformer import GowerDistanceTransformer

if TYPE_CHECKING:
    from numpy.typing import NDArray

warnings.filterwarnings("ignore")

# set matplotlib font to sans-serif
plt.rcParams["font.family"] = "sans-serif"


class KMedoidsAnalyzer:
    """Analyzer class for k-medoids clustering."""

    def __init__(self, cat_features: list[str] | None = None) -> None:
        """Initialize the KMedoidsAnalyzer.

        Args:
            cat_features: Specification of categorical features for Gower distance.

        Attributes:
            transformer: GowerDistanceTransformer instance.
            best_model_: Best k-medoids model found during optimization.
            best_n_clusters_: Best number of clusters found during optimization.
            dist_matrix_: Computed Gower distance matrix.
            results_df_: DataFrame containing optimization results.

        """
        self.transformer: GowerDistanceTransformer = GowerDistanceTransformer(cat_features=cat_features)
        self.best_model_: KMedoidsWrapper | None = None
        self.best_n_clusters_: int | None = None
        self.dist_matrix_: NDArray[np.float32] | None = None
        self.results_df_: pd.DataFrame | None = None

    @staticmethod
    def validate_dataframe_(df: pd.DataFrame) -> None:
        """Validate that the DataFrame contains only supported dtypes.

        Args:
            df: DataFrame to validate.

        Raises:
            TypeError: If unsupported dtypes are found in the DataFrame.

        """
        # Identify unsupported dtypes
        unsupported_dtypes: pd.DataFrame = df.select_dtypes(exclude=["number", "object", "bool"])
        # If any unsupported dtypes are found
        if not unsupported_dtypes.empty:
            # Set error message
            error_msg: str = f"Unsupported Dtypes in DataFrame: {unsupported_dtypes.dtypes.to_dict()}"
            # Raise TypeError
            raise TypeError(error_msg)

    @staticmethod
    def fit_model_(
        n_clusters: int,
        dist_matrix: NDArray[np.float32],
        random_state: int = 42
    ) -> dict[str, Any]:
        """Fit the k-medoids model.

        Args:
            n_clusters: Number of clusters.
            dist_matrix: Precomputed distance matrix.
            random_state: Random state for reproducibility.

        Returns:
            Dictionary with n_clusters, inertia, and silhouette score.

        """
        # Instantiate model
        model: KMedoidsWrapper = KMedoidsWrapper(n_clusters=n_clusters, random_state=random_state)
        # Fit model
        model.fit(dist_matrix)
        # Store labels
        labels: NDArray[np.int_] | None = model.labels_
        # Calculate Scores
        if labels is not None and len(np.unique(labels)) > 1:
            sil_score: float = silhouette_score(dist_matrix, labels, metric="precomputed")
        else:
            sil_score = -1.0
        # Return results
        return {
            "n_clusters": n_clusters,
            "model": model,
            "inertia": model.inertia_,
            "silhouette": sil_score
        }

    def run_optimization(
        self,
        df: pd.DataFrame,
        n_clusters_min: int = 2,
        n_clusters_max: int = 50
    ) -> tuple[KMedoidsWrapper, int, NDArray[np.float32], pd.DataFrame]:
        """Run k-medoids optimization across multiple k values.

        This function computes the Gower distance matrix once and then evaluates
        k-medoids clustering for different numbers of clusters, using silhouette
        score metric.

        Args:
            df: Input DataFrame containing the data to cluster.
            n_clusters_min: Minimum number of clusters to evaluate.
            n_clusters_max: Maximum number of clusters to evaluate.

        Returns:
            Tuple containing:
                - Best k-medoids model (highest silhouette score)
                - Best number of clusters
                - Computed distance matrix
                - DataFrame with evaluation metrics for all k values

        """
        # Validate DataFrame
        self.validate_dataframe_(df)

        # 1. Compute Gower distance matrix once
        self.dist_matrix_ = self.transformer.fit_transform(df)

        # Initialize vars
        results: list[dict[str, int | float | None]] = []

        # 2. Iterate over n_clusters values
        for n_clusters in range(n_clusters_min, n_clusters_max + 1):

            # Fit model
            model: dict[str, Any] = self.fit_model_(n_clusters=n_clusters, dist_matrix=self.dist_matrix_)

            # Append to results list
            results.append(model)

        # Convert results to DataFrame
        self.results_df_ = pd.DataFrame(results)

        # Find index of max silhouette score
        best_model_idx: int = self.results_df_["silhouette"].idxmax()
        self.best_model_ = self.results_df_.loc[best_model_idx, "model"]
        self.best_n_clusters_ = self.results_df_.loc[best_model_idx, "n_clusters"]

        return self.best_model_, self.best_n_clusters_, self.dist_matrix_, self.results_df_.drop(columns=["model"])

    def plot_metrics(self, results_df: pd.DataFrame | None = None) -> str:
        """Plot silhouette scores for different k values.

        Args:
            results_df: DataFrame containing 'k' and 'silhouette' columns.
                       If None, uses the results from the last optimization run.

        Returns:
            SVG string of the plot.

        """
        # Validate results_df
        if results_df is None:
            if self.results_df_ is None:
                err_message: str = "No results available. Run optimization first."
                raise ValueError(err_message)
            results_df = self.results_df_

        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Determine best k based on max silhouette score
        max_silhouette_value: float = results_df["silhouette"].max()

        # Get best n_clusters corresponding to max silhouette score
        best_n_clusters_series: pd.Series[int] = results_df.loc[results_df["silhouette"] == max_silhouette_value, "n_clusters"]

        # There could be multiple k with the same max silhouette, take the first one
        best_n_clusters: int = best_n_clusters_series.values[0]

        # Plot silhouette scores
        ax.plot(results_df["n_clusters"], results_df["silhouette"], "ro-")
        ax.set_title("Silhouette Analysis")
        ax.set_xlabel("Number of Clusters (n_clusters)")
        ax.set_ylabel("Silhouette Score (Higher is better)")
        ax.axvline(best_n_clusters, color="r")
        ax.grid(axis="y")

        # Adjust layout
        plt.tight_layout()

        # Initialize an in-memory buffer
        buffer: io.BytesIO = io.BytesIO()

        # Save figure to the buffer in SVG format then close it
        fig.savefig(buffer, format="svg", bbox_inches="tight", transparent=True, pad_inches=0.05)

        # Close figure
        plt.close(fig)

        return buffer.getvalue().decode()

    def analyze_model(
        self,
        df: pd.DataFrame,
        model: KMedoidsWrapper | None = None,
        dist_matrix: NDArray[np.float32] | None = None
    ) -> tuple[float, pd.DataFrame]:
        """Analyze the fitted k-medoids model and return medoids and silhouette score.

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

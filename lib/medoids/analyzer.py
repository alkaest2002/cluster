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
            best_silhouette_: Best silhouette score achieved.
            dist_matrix_: Computed Gower distance matrix.
            results_df_: DataFrame containing optimization results.

        """
        self.transformer: GowerDistanceTransformer = GowerDistanceTransformer(cat_features=cat_features)
        self.best_model_: dict[str, Any] = {}
        self.best_n_clusters_: int = 0
        self.best_silhouette_: float = -1.0
        self.dist_matrix_: NDArray[np.float32] = np.array([], dtype=np.float32)
        self.results_df_: pd.DataFrame = pd.DataFrame()

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
            Dictionary with model, and silhouette score.

        """
        # Instantiate model
        model: KMedoidsWrapper = KMedoidsWrapper(
            n_clusters=n_clusters,
            random_state=random_state
        )

        # Fit model
        model.fit(dist_matrix)

        # Store labels
        labels: NDArray[np.int_] = model.labels_

        # Calculate Scores
        if len(np.unique(labels)) > 1:
            sil_score: float = silhouette_score(dist_matrix, labels, metric="precomputed")
        else:
            sil_score = -1.0

        # Return results
        return {
            "model": model,
            "silhouette": sil_score
        }

    def check_fitted_(self) -> None:
        """Check if the analyzer has been fitted.

        Raises:
            ValueError: If the analyzer has not been fitted yet.
                or no valid clustering is found.

        """
        # Check if best_model_ and dist_matrix_ are set
        if len(self.best_model_) == 0 or self.dist_matrix_.size == 0:
            error_msg: str = "Analyzer must be fitted before analysis."
            raise ValueError(error_msg)

        # Raise error if all silhouette scores are -1
        if (self.results_df_["silhouette"] == -1.0).all():
            error_msg = "No valid clustering found."
            raise ValueError(error_msg)

    def run_optimization(
        self,
        df: pd.DataFrame,
        n_clusters_min: int = 2,
        n_clusters_max: int = 50
    ) -> tuple[dict[str, Any], int, NDArray[np.float32], pd.DataFrame]:
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
                - Best k-medoids model dict (model + silhouette score)
                - Best number of clusters
                - Computed distance matrix
                - DataFrame with evaluation metrics for all n_clusters values

        """
        # Initialize results list
        results: list[dict[str, int | float | None]] = []

        # Validate DataFrame
        self.validate_dataframe_(df)

        # 1. Compute Gower distance matrix once
        self.dist_matrix_ = self.transformer.fit_transform(df)

        # 2. Iterate over n_clusters values
        for n_clusters in range(n_clusters_min, n_clusters_max + 1):

            # Fit model
            model_dict: dict[str, Any] = self.fit_model_(n_clusters=n_clusters, dist_matrix=self.dist_matrix_)

            # Append to results list
            results.append({"n_clusters": n_clusters, **model_dict})

        # Convert results to DataFrame
        self.results_df_ = pd.DataFrame(results)

        # Find index of max silhouette score
        best_model_idx: int = self.results_df_["silhouette"].idxmax()

        # Store best model and best_n_clusters
        self.best_model_ = self.results_df_.loc[best_model_idx, "model"]
        self.best_n_clusters_ = self.results_df_.loc[best_model_idx, "n_clusters"]
        self.best_silhouette_ = self.results_df_.loc[best_model_idx, "silhouette"]

        return self.best_model_, self.best_n_clusters_, self.dist_matrix_, self.results_df_.drop(columns=["model"])

    def plot_metrics(self) -> str:
        """Plot silhouette scores for different k values.

        Returns:
            SVG string of the plot.

        Raises:
            ValueError: If no results are available to plot.

        """
        # Ensure analyzer is fitted
        self.check_fitted_()

        # Use provided results_df or the stored one
        data: pd.DataFrame = self.results_df_

        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # There could be multiple k with the same max silhouette, take the first one
        best_n_clusters: int = self.best_n_clusters_

        # Plot silhouette scores
        ax.plot(data["n_clusters"], data["silhouette"], "ro-")
        ax.set_title("Silhouette Analysis")
        ax.set_xlabel("Number of Slusters")
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
    ) -> tuple[float, pd.DataFrame]:
        """Analyze the fitted k-medoids model and return silhouette score and medoids.

        Args:
            df: Original DataFrame containing the clustered data.

        Returns:
            Tuple containing:
                - Silhouette score of the best model.
                - DataFrame of medoid rows with cluster labels and sizes.

        """
        # Ensure analyzer is fitted
        self.check_fitted_()

        # Use stored best model
        best_model = self.best_model_["model"]

        # Extract labels
        labels: NDArray[np.int_] = best_model.labels_

        # Extract medoid indices
        medoid_indices: NDArray[np.int_] = best_model.medoid_indices_

        # Clone dataframe to avoid modifying original
        df_clone: pd.DataFrame = df.copy()

        # Assign cluster labels and cluster sizes
        df_clone = df_clone.assign(
            cluster_label=labels,
            cluster_size=lambda df: df.groupby("cluster_label").transform("size")
        )

        # Extract medoid rows
        medoids_df: pd.DataFrame = df_clone.iloc[medoid_indices]

        # Return silhouette score and medoids DataFrame
        return self.best_silhouette_, medoids_df

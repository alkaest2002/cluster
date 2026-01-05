import io
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from lib.k_medoids.model import KMedoidsWrapper
from lib.k_medoids.transformer import GowerDistanceTransformer


class Optimizer:
    """Optimizer class for k-medoids clustering."""

    def __init__(self, cat_features: list[str] | None = None) -> None:
        """Initialize the KMedoidsAnalyzer.

        Args:
            cat_features: Specification of categorical features for Gower distance.

        Attributes:
            df: Input DataFrame.
            results_df_: DataFrame containing optimization results.
            transformer: GowerDistanceTransformer instance.
            dist_matrix_: Computed Gower distance matrix.
            best_model_: Best k-medoids model found during optimization.
            best_silhouette_: Best silhouette score achieved.

        """
        self.df: pd.DataFrame = pd.DataFrame()
        self.results_df_: pd.DataFrame = pd.DataFrame()
        self.transformer: GowerDistanceTransformer = GowerDistanceTransformer(cat_features=cat_features)
        self.dist_matrix_: NDArray[np.float32] = np.array([], dtype=np.float32)
        self.best_model_: KMedoidsWrapper | None = None
        self.best_silhouette_: float = -1.0

    @staticmethod
    def validate_dataframe_(df: pd.DataFrame) -> None:
        """Validate that the DataFrame contains only supported dtypes.

        Args:
            df: DataFrame to validate.

        Raises:
            TypeError: If unsupported dtypes are found in the DataFrame.

        """
        # Identify unsupported dtypes in DataFrame
        unsupported_dtypes: pd.DataFrame = df.select_dtypes(exclude=["number", "object", "bool"])

        # If any unsupported dtypes are found in DataFrame
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

        # Return results
        return {
            "model": model,
            "n_clusters": n_clusters,
            "inertia": model.inertia_,
            "silhouette": model.silhouette_score_
        }

    def check_fitted_(self) -> None:
        """Check if the analyzer has been fitted.

        Raises:
            ValueError: If the analyzer has not been fitted yet.
                or no valid clustering is found.

        """
        # Check if best_model_ and dist_matrix_ are set
        if self.best_model_ is None or self.dist_matrix_.size == 0:
            error_msg: str = "Analyzer must be fitted before analysis."
            raise ValueError(error_msg)

        # Raise error if all silhouette scores are -1
        # This indicates no valid clustering was found
        if (self.results_df_["silhouette"] == -1.0).all():
            error_msg = "No valid clustering found."
            raise ValueError(error_msg)

    def optimize(
        self,
        df: pd.DataFrame,
        n_clusters_min: int = 2,
        n_clusters_max: int = 50
    ) -> tuple[dict[str, Any], NDArray[np.float32], pd.DataFrame]:
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

        # Copy DataFrame
        self.df = df.copy()

        #######################################################################################################
        # 1. Compute Gower distance matrix once
        #######################################################################################################
        self.dist_matrix_ = self.transformer.fit_transform(df)

        #######################################################################################################
        # 2. Iterate over n_clusters values
        ########################################################################################################
        for n_clusters in range(n_clusters_min, n_clusters_max + 1):

            # Fit model
            model_dict: dict[str, Any] = self.fit_model_(n_clusters=n_clusters, dist_matrix=self.dist_matrix_)

            # Append to results list
            results.append(model_dict)

        # Convert results to DataFrame
        self.results_df_ = pd.DataFrame(results)

        # Find index of max silhouette score
        best_model_idx: int = self.results_df_["silhouette"].idxmax()

        # Store best model related attributes
        self.best_model_ = self.results_df_.loc[best_model_idx, "model"]
        self.best_silhouette_ = self.results_df_.loc[best_model_idx, "silhouette"]
        self.best_model_labels_ = self.best_model_.labels_

        # Assign cluster labels and cluster sizes from best model to original DataFrame
        self.df = self.df.assign(
            cluster=self.best_model_.labels_,
            cluster_size=lambda df: df.groupby("cluster").transform("size")
        )

        return self.best_model_, self.dist_matrix_, self.results_df_.drop(columns=["model"])

    def get_plots(self) -> list[str]:
        """Plot silhouette scores for different k values.

        Returns:
            list[str]: List containing SVG strings of the plots.

        Raises:
            ValueError: If no results are available to plot.

        """
        # Ensure analyzer is fitted
        # If error is not raise, subsequent code can safely assume best_model_ is set
        self.check_fitted_()

        # Ensure best_model_ is KMedoidsWrapper for type checker
        assert isinstance(self.best_model_, KMedoidsWrapper)  # nosec

        # Get results DataFrame
        data: pd.DataFrame = self.results_df_

        # Create plot figure and axis
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        # Get best number of clusters
        best_n_clusters: int = self.best_model_.n_clusters

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

        return [buffer.getvalue().decode()]

    def get_analysis(self) -> pd.DataFrame:
        """Analyze the fitted k-medoids model and return silhouette score and medoids.

        Returns:
            pd.DataFrame: DataFrame of medoid rows with cluster labels and sizes.

        """
        # Ensure analyzer is fitted
        # If error is not raise, subsequent code can safely assume best_model_ is set
        self.check_fitted_()

        # Assert best_model is KMedoidsWrapper for type checker
        assert isinstance(self.best_model_, KMedoidsWrapper)  # nosec

        # Extract medoid indices
        medoid_indices: NDArray[np.int_] = self.best_model_.medoid_indices_

        # get medoid rows
        medoids_df: pd.DataFrame = self.df.iloc[medoid_indices].copy()

        return medoids_df

import io
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from lib.k_prototypes.model import KPrototypesWrapper


class KPrototypesOptimizer:
    """Optimizer class for k-prototypes clustering with gamma tuning."""

    def __init__(
        self,
        cat_features: list[str] | None = None,
        standardize_num_scales: bool = True
    ) -> None:
        """Initialize the KPrototypesOptimizer.

        Args:
            cat_features: List of categorical feature column names.
            standardize_num_scales: Whether to standardize numeric features.

        Attributes:
            df: Input DataFrame with cluster assignments.
            results_df_: DataFrame containing optimization results.
            cat_features: Categorical feature column names.
            standardize_num_scales: Whether to standardize numeric features.
            best_model_: Best k-prototypes model found during optimization.
            best_silhouette_: Best silhouette score achieved.
            best_gamma_: Best gamma value found.
            best_n_clusters_: Best number of clusters found.

        """
        self.df: pd.DataFrame = pd.DataFrame()
        self.results_df_: pd.DataFrame = pd.DataFrame()
        self.cat_features: list[str] = cat_features or []
        self.standardize_num_scales: bool = standardize_num_scales
        self.best_model_: KPrototypesWrapper | None = None
        self.best_silhouette_: float = -1.0
        self.best_gamma_: float = 1.0
        self.best_n_clusters_: int = 2

    @staticmethod
    def validate_dataframe_(df: pd.DataFrame) -> None:
        """Validate that the DataFrame contains only supported dtypes.

        Args:
            df: DataFrame to validate.

        Raises:
            TypeError: If unsupported dtypes are found in the DataFrame.

        """
        # Identify unsupported dtypes in DataFrame
        unsupported_dtypes: pd.DataFrame = df.select_dtypes(
            exclude=["number", "object"]
        )

        # If any unsupported dtypes are found in DataFrame
        if not unsupported_dtypes.empty:
            # Set error message
            error_msg: str = (
                f"Unsupported Dtypes in DataFrame: "
                f"{unsupported_dtypes.dtypes.to_dict()}"
            )
            # Raise TypeError
            raise TypeError(error_msg)

    def _prepare_data(
        self,
        df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare numeric and categorical DataFrames from input DataFrame.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple of (numeric_features_df, categorical_features_df).

        """
        # Get numeric columns (exclude categorical features)
        numeric_columns = [col for col in df.columns if col not in self.cat_features]

        # Get numeric DataFrame
        numeric_df = df[numeric_columns].select_dtypes(include=["number"])

        # Get categorical columns
        categorical_df = (
            df[self.cat_features]
                if self.cat_features
                else pd.DataFrame(index=df.index)
        )

        return numeric_df, categorical_df

    @staticmethod
    def fit_model_(
        n_clusters: int,
        numeric_df: pd.DataFrame,
        categorical_df: pd.DataFrame,
        gamma: float = 1.0,
        random_state: int = 42
    ) -> dict[str, Any]:
        """Fit the k-prototypes model.

        Args:
            n_clusters: Number of clusters.
            numeric_df: Numeric feature DataFrame.
            categorical_df: Categorical feature DataFrame.
            gamma: Weight parameter for categorical distance component.
            random_state: Random state for reproducibility.

        Returns:
            Dictionary with model and metrics.

        """
        # Instantiate model
        model: KPrototypesWrapper = KPrototypesWrapper(
            n_clusters=n_clusters,
            gamma=gamma,
            random_state=random_state
        )

        # Fit model
        model.fit((numeric_df, categorical_df))

        # Return results
        return {
            "model": model,
            "n_clusters": n_clusters,
            "gamma": gamma,
            "inertia": model.inertia_,
            "silhouette": model.silhouette_score_,
            "cost": model.cost_,
            "n_iter": model.n_iter_
        }

    def check_fitted_(self) -> None:
        """Check if the optimizer has been fitted.

        Raises:
            ValueError: If the optimizer has not been fitted yet
                or no valid clustering is found.

        """
        # Check if best_model_ is set
        if self.best_model_ is None:
            error_msg: str = "Optimizer must be fitted before analysis."
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
        n_clusters_max: int = 10,
        gamma_values: list[float] | None = None
    ) -> tuple[KPrototypesWrapper, pd.DataFrame]:
        """Run k-prototypes optimization across multiple k and gamma values.

        This function evaluates k-prototypes clustering for different numbers of clusters
        and gamma values, using silhouette score metric with k-prototypes distance.

        Args:
            df: Input DataFrame containing the data to cluster.
            n_clusters_min: Minimum number of clusters to evaluate.
            n_clusters_max: Maximum number of clusters to evaluate.
            gamma_values: List of gamma values to try. If None, uses default range.

        Returns:
            Tuple containing:
                - Best k-prototypes model
                - DataFrame with evaluation metrics for all combinations

        """
        # Default gamma values if not provided
        if gamma_values is None:
            gamma_values = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]

        # Initialize results list
        results: list[dict[str, int | float | None]] = []

        # Validate DataFrame
        self.validate_dataframe_(df)

        # Copy DataFrame
        self.df = df.copy()

        # Prepare numeric and categorical data
        numeric_df, categorical_df = self._prepare_data(df)

        #######################################################################################################
        # Grid search over n_clusters and gamma values
        #######################################################################################################
        best_silhouette = -1.0
        best_model = None
        best_params = {}

        for gamma in gamma_values:
            for n_clusters in range(n_clusters_min, n_clusters_max + 1):

                # Fit model
                model_dict: dict[str, Any] = self.fit_model_(
                    n_clusters=n_clusters,
                    numeric_df=numeric_df,
                    categorical_df=categorical_df,
                    gamma=gamma,
                    random_state=42
                )

                # Append to results list
                results.append(model_dict)

                # Track best model
                current_silhouette = model_dict["silhouette"]
                if current_silhouette > best_silhouette:
                    best_silhouette = current_silhouette
                    best_model = model_dict["model"]
                    best_params = {
                        "gamma": gamma,
                        "n_clusters": n_clusters,
                        "silhouette": current_silhouette
                    }

        # Convert results to DataFrame
        self.results_df_ = pd.DataFrame(results)

        # Store best model related attributes
        self.best_model_ = best_model
        self.best_silhouette_ = best_params["silhouette"]
        self.best_gamma_ = best_params["gamma"]
        self.best_n_clusters_ = best_params["n_clusters"]

        # Assert best_model_ is not None for type checker
        assert self.best_model_ is not None  # nosec

        # Assign cluster labels from best model to original DataFrame
        self.df = self.df.assign(cluster=self.best_model_.labels_)

        # Add cluster size using groupby transform
        self.df = self.df.assign(
            cluster_size=self.df.groupby("cluster")["cluster"].transform("size")
        )

        return self.best_model_, self.results_df_.drop(columns=["model"])

    def get_plots(self) -> list[str]:
        """Plot silhouette scores for different k and gamma values.

        Returns:
            list[str]: List containing SVG strings of the plots.

        Raises:
            ValueError: If no results are available to plot.

        """
        # Ensure optimizer is fitted
        self.check_fitted_()

        # Ensure best_model_ is KPrototypesWrapper for type checker
        assert isinstance(self.best_model_, KPrototypesWrapper)  # nosec

        # Get results DataFrame
        data: pd.DataFrame = self.results_df_

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        #######################################################################################################
        # Plot 1: Silhouette scores by n_clusters for best gamma
        #######################################################################################################
        best_gamma_data = data[data["gamma"] == self.best_gamma_]

        ax1.plot(
            best_gamma_data["n_clusters"],
            best_gamma_data["silhouette"],
            "ro-"
        )
        ax1.set_title(f"Silhouette Analysis (gamma = {self.best_gamma_})")
        ax1.set_xlabel("Number of Clusters")
        ax1.set_ylabel("Silhouette Score")
        ax1.axvline(
            self.best_n_clusters_,
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Best k={self.best_n_clusters_}"
        )
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

        #######################################################################################################
        # Plot 2: Heatmap of silhouette scores by n_clusters and gamma
        #######################################################################################################
        pivot_data = data.pivot(
            index="gamma",
            columns="n_clusters",
            values="silhouette"
        )

        im = ax2.imshow(
            pivot_data.values,
            cmap="viridis",
            aspect="auto",
            origin="lower"
        )
        ax2.set_title("Silhouette Score Heatmap")
        ax2.set_xlabel("Number of Clusters")
        ax2.set_ylabel("Gamma")

        # Set ticks
        ax2.set_xticks(range(len(pivot_data.columns)))
        ax2.set_xticklabels(pivot_data.columns)
        ax2.set_yticks(range(len(pivot_data.index)))
        ax2.set_yticklabels([f"{g:.2f}" for g in pivot_data.index])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label("Silhouette Score")

        # Mark best combination
        best_row = list(pivot_data.index).index(self.best_gamma_)
        best_col = list(pivot_data.columns).index(self.best_n_clusters_)
        ax2.plot(
            best_col,
            best_row,
            "r*",
            markersize=15,
            label=f"Best (k={self.best_n_clusters_}, gamma={self.best_gamma_})"
        )
        ax2.legend()

        # Adjust layout
        plt.tight_layout()

        # Initialize an in-memory buffer
        buffer: io.BytesIO = io.BytesIO()

        # Save figure to the buffer in SVG format then close it
        fig.savefig(
            buffer,
            format="svg",
            bbox_inches="tight",
            transparent=True,
            pad_inches=0.05
        )

        # Close figure
        plt.close(fig)

        # Retrieve SVG string from buffer
        buffer.seek(0)
        svg_content: str = buffer.getvalue().decode()
        buffer.close()

        return [svg_content]

    def get_analysis(self) -> pd.DataFrame:
        """Analyze the fitted k-prototypes model and return cluster centers info.

        Returns:
            pd.DataFrame: DataFrame with cluster analysis including centroids
                and representative samples.

        """
        # Ensure optimizer is fitted
        self.check_fitted_()

        # Assert best_model is KPrototypesWrapper for type checker
        assert isinstance(self.best_model_, KPrototypesWrapper)  # nosec

        # Get cluster centroids
        centroids = self.best_model_.cluster_centroids_

        analysis_data = []

        for cluster_id in range(self.best_model_.n_clusters):
            cluster_mask = self.df["cluster"] == cluster_id
            cluster_size = cluster_mask.sum()

            # Get centroid information
            if centroids and len(centroids) > cluster_id:
                numeric_centroid = (
                    centroids[cluster_id][0]
                    if len(centroids[cluster_id]) > 0
                    else None
                )
                categorical_centroid = (
                    centroids[cluster_id][1]
                    if len(centroids[cluster_id]) > 1
                    else None
                )
            else:
                numeric_centroid = None
                categorical_centroid = None

            # Find a representative point (first point in cluster)
            cluster_points = self.df[cluster_mask]
            representative_idx = (
                cluster_points.index[0]
                if len(cluster_points) > 0
                else None
            )

            analysis_data.append({
                "cluster": cluster_id,
                "size": cluster_size,
                "representative_idx": representative_idx,
                "numeric_centroid": numeric_centroid,
                "categorical_centroid": categorical_centroid
            })

        analysis_df = pd.DataFrame(analysis_data)

        # Add representative points data
        if not analysis_df.empty and analysis_df["representative_idx"].notna().any():
            representative_points = []
            for _, row in analysis_df.iterrows():
                if pd.notna(row["representative_idx"]):
                    rep_point = self.df.loc[row["representative_idx"]].to_dict()
                    rep_point["cluster"] = row["cluster"]
                    rep_point["cluster_size"] = row["size"]
                    representative_points.append(rep_point)

            if representative_points:
                return pd.DataFrame(representative_points)

        # Fallback: return cluster summary
        return analysis_df

    def get_cluster_summary(self) -> dict[str, Any]:
        """Get a summary of the clustering results.

        Returns:
            Dictionary containing clustering summary statistics.

        """
        self.check_fitted_()

        assert isinstance(self.best_model_, KPrototypesWrapper)  # nosec

        # Get cluster sizes using value_counts
        cluster_sizes = self.df["cluster"].value_counts().sort_index().to_dict()

        return {
            "best_n_clusters": self.best_n_clusters_,
            "best_gamma": self.best_gamma_,
            "best_silhouette_score": self.best_silhouette_,
            "cost": self.best_model_.cost_,
            "n_iterations": self.best_model_.n_iter_,
            "cluster_sizes": cluster_sizes,
            "total_samples": len(self.df),
            "n_numeric_features": len([
                col for col in self.df.columns
                if col not in {*self.cat_features, "cluster", "cluster_size"}
            ]),
            "n_categorical_features": len(self.cat_features)
        }

    def get_best_params(self) -> dict[str, Any]:
        """Get the best parameters found during optimization.

        Returns:
            Dictionary with best parameters.

        """
        self.check_fitted_()

        return {
            "n_clusters": self.best_n_clusters_,
            "gamma": self.best_gamma_,
            "silhouette_score": self.best_silhouette_
        }

    def get_gamma_analysis(self) -> pd.DataFrame:
        """Get analysis of gamma values performance.

        Returns:
            DataFrame with gamma performance analysis.

        """
        self.check_fitted_()

        # Group by gamma and get best silhouette for each
        gamma_analysis = (
            self.results_df_
            .groupby("gamma")["silhouette"]
            .agg(["max", "mean", "std", "count"])
            .reset_index()
        )

        gamma_analysis.columns = [
            "gamma",
            "best_silhouette",
            "mean_silhouette",
            "std_silhouette",
            "n_evaluations"
        ]

        return gamma_analysis.sort_values("best_silhouette", ascending=False)

    def get_clustered_data(self) -> pd.DataFrame:
        """Get the original data with cluster assignments.

        Returns:
            DataFrame with original data plus cluster and cluster_size columns.

        Raises:
            ValueError: If the optimizer has not been fitted yet.

        """
        self.check_fitted_()
        return self.df.copy()

    def get_cluster_profiles(self) -> pd.DataFrame:
        """Get statistical profiles for each cluster.

        Returns:
            DataFrame with cluster profiles including means, modes, and counts.

        """
        self.check_fitted_()

        profiles = []

        for cluster_id in sorted(self.df["cluster"].unique()):
            cluster_data = self.df[self.df["cluster"] == cluster_id]

            profile = {
                "cluster": cluster_id,
                "size": len(cluster_data),
                "percentage": len(cluster_data) / len(self.df) * 100
            }

            # Add numeric feature statistics
            numeric_cols = [
                col for col in cluster_data.columns
                if col not in {*self.cat_features, "cluster", "cluster_size"}
            ]
            for col in numeric_cols:
                if pd.api.types.is_numeric_dtype(cluster_data[col]):
                    profile[f"{col}_mean"] = cluster_data[col].mean()
                    profile[f"{col}_median"] = cluster_data[col].median()
                    profile[f"{col}_std"] = cluster_data[col].std()

            # Add categorical feature modes
            for col in self.cat_features:
                if col in cluster_data.columns:
                    mode_value = cluster_data[col].mode()
                    profile[f"{col}_mode"] = mode_value.iloc[0] if len(mode_value) > 0 else None

            profiles.append(profile)

        return pd.DataFrame(profiles)

import io
import warnings
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from lib.k_prototypes.model import KPrototypesWrapper

warnings.filterwarnings("ignore")


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
        """Validate that the DataFrame contains only supported dtypes."""
        if df.isnull().values.any():
            error_msg: str = "Input DataFrame contains missing values. Please handle them before proceeding."
            raise ValueError(error_msg)

        if not df.select_dtypes(exclude=["number", "object", "category"]).empty:
            error_msg = "DataFrame contains unsupported dtypes. Only numeric and categorical types are supported."
            raise TypeError(error_msg)

    def _get_categorical_indices(self, df: pd.DataFrame) -> list[int]:
        """Get indices of categorical columns in the DataFrame."""
        return [df.columns.get_loc(col) for col in self.cat_features if col in df.columns]

    def fit_model_(
        self,
        df: pd.DataFrame,
        categorical_indices: list[int],
        n_clusters: int,
        gamma: float = 1.0,
        random_state: int = 42
    ) -> dict[str, Any]:
        """Fit the k-prototypes model."""
        model = KPrototypesWrapper(
            n_clusters=n_clusters,
            gamma=gamma,
            random_state=random_state
        )

        # Fit model with DataFrame
        model.fit(df, categorical_indices, standardize=self.standardize_num_scales)

        return {
            "model": model,
            "n_clusters": n_clusters,
            "gamma": gamma,
            "inertia": model.inertia_,
            "silhouette": model.silhouette_score_,
            "cost": model.cost_,
            "n_iter": model.n_iter_
        }

    def check_is_fitted_(self) -> None:
        """Check if the optimizer has been fitted."""
        if self.best_model_ is None:
            raise ValueError("Optimizer must be fitted before analysis.")

        if (self.results_df_["silhouette"] == -1.0).all():
            raise ValueError("No valid clustering found.")

    def optimize(
        self,
        df: pd.DataFrame,
        n_clusters_min: int = 2,
        n_clusters_max: int = 10,
        gamma_values: list[float] | None = None
    ) -> tuple[KPrototypesWrapper, pd.DataFrame]:
        """Run k-prototypes optimization across multiple k and gamma values."""
        if gamma_values is None:
            gamma_values = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0]

        # Validate DataFrame
        self.validate_dataframe_(df)
        self.df = df.copy()

        # Get categorical indices once
        categorical_indices = self._get_categorical_indices(df)

        # Grid search
        results = []
        for gamma in gamma_values:
            for n_clusters in range(n_clusters_min, n_clusters_max + 1):
                model_dict = self.fit_model_(
                    df=df,
                    categorical_indices=categorical_indices,
                    n_clusters=n_clusters,
                    gamma=gamma,
                    random_state=42
                )

                results.append(model_dict)

                # Track best model
                if model_dict["silhouette"] > self.best_silhouette_:
                    self.best_model_ = model_dict["model"]
                    self.best_silhouette_ = model_dict["silhouette"]
                    self.best_gamma_ = model_dict["gamma"]
                    self.best_n_clusters_ = model_dict["n_clusters"]

        self.results_df_ = pd.DataFrame(results)

        # Add cluster assignments to original data
        assert self.best_model_ is not None
        self.df = self.df.assign(
            cluster=pd.Series(self.best_model_.labels_, index=self.df.index),
            cluster_size=lambda x: x.groupby("cluster").size().loc[x["cluster"]].values
        )

        return self.best_model_, self.results_df_.drop(columns=["model"])

    def get_plots(self) -> list[str]:
        """Plot silhouette scores for different k and gamma values."""
        self.check_is_fitted_()

        data = self.results_df_

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Silhouette scores by n_clusters for best gamma
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

        # Plot 2: Heatmap of silhouette scores by n_clusters and gamma
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

        plt.tight_layout()

        # Save to buffer
        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="svg",
            bbox_inches="tight",
            transparent=True,
            pad_inches=0.05
        )
        plt.close(fig)

        buffer.seek(0)
        svg_content = buffer.getvalue().decode()
        buffer.close()

        return [svg_content]

    def get_analysis(self) -> pd.DataFrame:
        """Analyze the fitted k-prototypes model and return cluster centers info."""
        self.check_is_fitted_()

        analysis_data = []

        for cluster_id in range(self.best_model_.n_clusters):
            cluster_mask = self.df["cluster"] == cluster_id
            cluster_size = cluster_mask.sum()

            # Get centroid information
            centroids = self.best_model_.cluster_centroids_
            if centroids and len(centroids) > cluster_id:
                numeric_centroid = (
                    centroids[cluster_id][0] if len(centroids[cluster_id]) > 0 else None
                )
                categorical_centroid = (
                    centroids[cluster_id][1] if len(centroids[cluster_id]) > 1 else None
                )
            else:
                numeric_centroid = None
                categorical_centroid = None

            # Find representative point (closest to centroid or first point)
            cluster_points = self.df[cluster_mask]
            representative_idx = (
                cluster_points.index[0] if len(cluster_points) > 0 else None
            )

            analysis_data.append({
                "cluster": cluster_id,
                "size": cluster_size,
                "percentage": (cluster_size / len(self.df)) * 100,
                "representative_idx": representative_idx,
                "numeric_centroid": numeric_centroid,
                "categorical_centroid": categorical_centroid
            })

        return pd.DataFrame(analysis_data)

    def get_cluster_summary(self) -> dict[str, Any]:
        """Get a summary of the clustering results."""
        self.check_is_fitted_()

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
        """Get the best parameters found during optimization."""
        self.check_is_fitted_()

        return {
            "n_clusters": self.best_n_clusters_,
            "gamma": self.best_gamma_,
            "silhouette_score": self.best_silhouette_
        }

    def get_gamma_analysis(self) -> pd.DataFrame:
        """Get analysis of gamma values performance."""
        self.check_is_fitted_()

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
        """Get the original data with cluster assignments."""
        self.check_is_fitted_()
        return self.df.copy()

    def get_cluster_profiles(self) -> pd.DataFrame:
        """Get statistical profiles for each cluster."""
        self.check_is_fitted_()

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
                and pd.api.types.is_numeric_dtype(cluster_data[col])
            ]

            for col in numeric_cols:
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

"""
Gower Distance + Hierarchical Clustering with scikit-learn API compatibility
Using the gower package for optimized distance computation
Assumes ordinal variables are properly coded as ordered Categorical in pandas
"""

import numpy as np
import pandas as pd
import gower
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from typing import Any
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import warnings


class GowerClusterer(BaseEstimator, ClusterMixin):
    """
    Hierarchical clustering using Gower distance via the gower package.
    
    This estimator handles mixed data types automatically via pandas dtypes and
    follows scikit-learn's API conventions.
    
    Parameters
    ----------
    linkage_method : str, default="average"
        Linkage method for hierarchical clustering. Options:
        - "single": single linkage (minimum)
        - "complete": complete linkage (maximum)
        - "average": average linkage (UPGMA)
        - "weighted": weighted average linkage (WPGMA)
        Note: "ward" is not recommended with Gower distance
    
    n_clusters : int, default=2
        Number of clusters to form. This parameter is used when calling
        fit_predict() or when extract_clusters() is called without parameters.
    
    cat_features : list of str, optional
        Explicitly specify categorical columns (overrides dtype detection).
        Use this if you have numerical codes that should be treated as categorical.
    
    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point (0-indexed, following sklearn convention).
        Available after calling fit().
    
    distance_matrix_ : ndarray of shape (n_samples, n_samples)
        The computed Gower distance matrix.
    
    linkage_matrix_ : ndarray of shape (n_samples-1, 4)
        The hierarchical clustering encoded as a linkage matrix.
    
    n_features_in_ : int
        Number of features seen during fit.
    
    feature_names_in_ : ndarray of shape (n_features_in_,), dtype=str
        Names of features seen during fit. Only available if input is a DataFrame.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gower_clustering import GowerClusterer
    >>> 
    >>> # Create sample mixed dataset
    >>> data = {
    ...     "age": [25, 30, 35, 40, 45],
    ...     "income": [50000, 60000, 70000, 80000, 90000],
    ...     "education': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master"],
    ...     "city': ['NYC', 'LA', 'NYC', 'Chicago', 'LA"]
    ... }
    >>> df = pd.DataFrame(data)
    >>> 
    >>> # Fit and predict clusters
    >>> clusterer = GowerClusterer(n_clusters=2)
    >>> labels = clusterer.fit_predict(df)
    >>> print(labels)
    [0 0 1 1 1]
    """
    
    def __init__(
        self,
        linkage_method: str = "average",
        n_clusters: int = 2,
        cat_features: list[str] | None = None
    ) -> None:
        self.linkage_method = linkage_method
        self.n_clusters = n_clusters
        self.cat_features = cat_features
    
    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "GowerClusterer":
        """
        Compute Gower distance and fit hierarchical clustering.
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Input data with mixed feature types:
            - Numerical: int, float dtypes
            - Categorical: object, category dtypes
            - Ordinal: ordered Categorical dtype (pd.CategoricalDtype(ordered=True))
            - Binary: bool dtype or 0/1 integers
        
        y : array-like of shape (n_samples,), optional
            Target values (ignored). This parameter exists for API consistency.
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Validate input
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame to handle mixed data types")
        
        if X.empty:
            raise ValueError("X cannot be empty")
        
        # Store input information
        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)
        
        # Store data copy for later use
        self._X = X.copy()
        
        # Validate linkage method
        valid_methods = ["single', 'complete', 'average', 'weighted"]
        if self.linkage_method not in valid_methods:
            raise ValueError(f"linkage_method must be one of {valid_methods}")
        
        # Compute Gower distance matrix
        try:
            self.distance_matrix_ = gower.gower_matrix(X, cat_features=self.cat_features)
        except Exception as e:
            raise ValueError(f"Failed to compute Gower distance: {str(e)}")
        
        # Convert to condensed form and compute linkage
        condensed = squareform(self.distance_matrix_, checks=False)
        
        try:
            self.linkage_matrix_ = linkage(condensed, method=self.linkage_method)
        except Exception as e:
            raise ValueError(f"Failed to compute hierarchical clustering: {str(e)}")
        
        # Extract default clusters
        self.labels_ = self._extract_clusters(self.n_clusters) - 1  # Convert to 0-indexed
        
        return self
    
    def fit_predict(self, X: pd.DataFrame, y: np.ndarray | None = None) -> np.ndarray:
        """
        Compute clustering and return cluster labels.
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Input data with mixed feature types.
        
        y : array-like of shape (n_samples,), optional
            Target values (ignored). This parameter exists for API consistency.
        
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels (0-indexed).
        """
        return self.fit(X, y).labels_
    
    def _extract_clusters(
        self,
        n_clusters: int | None = None,
        threshold: float | None = None
    ) -> np.ndarray:
        """
        Extract flat clusters from hierarchical clustering (1-indexed).
        
        Parameters
        ----------
        n_clusters : int, optional
            Number of clusters to form
        threshold : float, optional
            Distance threshold for cutting dendrogram
        
        Returns
        -------
        np.ndarray
            Cluster labels for each sample (1-indexed)
        """
        if n_clusters is not None:
            return fcluster(self.linkage_matrix_, n_clusters, criterion="maxclust")
        elif threshold is not None:
            return fcluster(self.linkage_matrix_, threshold, criterion="distance")
        else:
            raise ValueError("Must specify either n_clusters or threshold")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Note: This is a placeholder implementation. True prediction for new samples
        would require computing distances to existing clusters, which is complex
        for hierarchical clustering with Gower distance.
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            New data to predict clusters for.
        
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Predicted cluster labels.
            
        Raises
        ------
        NotImplementedError
            This method is not implemented for hierarchical clustering.
        """
        check_is_fitted(self, ["distance_matrix_', 'linkage_matrix_"])
        raise NotImplementedError(
            "Prediction for new samples is not straightforward with hierarchical "
            "clustering. Consider using fit_predict on the combined dataset."
        )
    
    def get_clusters(
        self,
        n_clusters: int | None = None,
        threshold: float | None = None
    ) -> np.ndarray:
        """
        Extract flat clusters with custom parameters.
        
        Parameters
        ----------
        n_clusters : int, optional
            Number of clusters to form
        threshold : float, optional
            Distance threshold for cutting dendrogram
        
        Returns
        -------
        np.ndarray
            Cluster labels for each sample (0-indexed, sklearn convention)
        """
        check_is_fitted(self, ["linkage_matrix_"])
        labels_1indexed = self._extract_clusters(n_clusters, threshold)
        return labels_1indexed - 1  # Convert to 0-indexed
    
    def plot_dendrogram(
        self,
        labels: list[str] | None = None,
        figsize: tuple[int, int] = (12, 6),
        color_threshold: float | None = None,
        **kwargs: Any
    ) -> plt.Figure:
        """
        Plot hierarchical clustering dendrogram.
        
        Parameters
        ----------
        labels : list of str, optional
            Sample labels for dendrogram leaves
        figsize : tuple of int, default=(12, 6)
            Figure size (width, height)
        color_threshold : float, optional
            Threshold for coloring clusters in dendrogram
        **kwargs : dict
            Additional arguments passed to scipy.cluster.hierarchy.dendrogram
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object
        """
        check_is_fitted(self, ["linkage_matrix_"])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        dendrogram_kwargs = {
            "labels": labels,
            "leaf_rotation": 90,
            "leaf_font_size": 10,
            "ax": ax
        }
        
        if color_threshold is not None:
            dendrogram_kwargs["color_threshold"] = color_threshold
        
        dendrogram_kwargs.update(kwargs)
        
        dendrogram(self.linkage_matrix_, **dendrogram_kwargs)
        
        ax.set_title(
            f"Hierarchical Clustering Dendrogram (method={self.linkage_method})",
            fontsize=14, pad=20
        )
        ax.set_xlabel("Sample Index' if labels is None else 'Sample", fontsize=12)
        ax.set_ylabel("Gower Distance", fontsize=12)
        plt.tight_layout()
        
        return fig
    
    def score(self, X: pd.DataFrame, y: np.ndarray | None = None) -> float:
        """
        Return the mean silhouette score for the clustering.
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Input data (should be the same as used for fitting)
        y : array-like of shape (n_samples,), optional
            Target values (ignored)
        
        Returns
        -------
        score : float
            Mean silhouette score
        """
        check_is_fitted(self, ["distance_matrix_', 'labels_"])
        
        from sklearn.metrics import silhouette_score
        
        if len(np.unique(self.labels_)) < 2:
            warnings.warn("Cannot compute silhouette score with less than 2 clusters")
            return -1.0
        
        return silhouette_score(
            self.distance_matrix_, self.labels_, metric="precomputed"
        )
    
    def compute_silhouette_scores(self, clusters: np.ndarray | None = None) -> dict[str, float | np.ndarray]:
        """
        Compute detailed silhouette scores for cluster quality assessment.
        
        Parameters
        ----------
        clusters : np.ndarray, optional
            Cluster assignments (0-indexed). If None, uses self.labels_
        
        Returns
        -------
        dict
            Dictionary with "mean' silhouette score and per-sample 'scores"
        """
        check_is_fitted(self, ["distance_matrix_"])
        
        if clusters is None:
            clusters = self.labels_
        
        from sklearn.metrics import silhouette_score, silhouette_samples
        
        if len(np.unique(clusters)) < 2:
            warnings.warn("Cannot compute silhouette score with less than 2 clusters")
            return {"mean': -1.0, 'scores": np.full(len(clusters), -1.0)}
        
        mean_score = silhouette_score(
            self.distance_matrix_, clusters, metric="precomputed"
        )
        sample_scores = silhouette_samples(
            self.distance_matrix_, clusters, metric="precomputed"
        )
        
        return {
            "mean": mean_score,
            "scores": sample_scores
        }
    
    def get_cluster_profiles(
        self,
        clusters: np.ndarray | None = None,
        features: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Generate summary statistics for each cluster.
        
        Parameters
        ----------
        clusters : np.ndarray, optional
            Cluster assignments (0-indexed). If None, uses self.labels_
        features : list of str, optional
            Specific features to profile (default: all numerical features)
        
        Returns
        -------
        pd.DataFrame
            Summary statistics grouped by cluster
        """
        check_is_fitted(self, ["labels_"])
        
        if not hasattr(self, "_X"):
            raise ValueError("Original data not available for profiling")
        
        if clusters is None:
            clusters = self.labels_
        
        df_clustered = self._X.copy()
        df_clustered["cluster"] = clusters
        
        if features is None:
            # Profile all numerical features by default
            features = df_clustered.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f != "cluster"]
        
        if not features:
            raise ValueError("No numerical features to profile")
        
        profiles = df_clustered.groupby("cluster")[features].agg(["mean", "std", "count"])
        
        return profiles


class GowerTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that computes Gower distance matrix for mixed data types.
    
    This transformer can be used in scikit-learn pipelines to preprocess
    mixed-type data by computing pairwise Gower distances.
    
    Parameters
    ----------
    cat_features : list of str, optional
        Explicitly specify categorical columns (overrides dtype detection).
    
    return_distance_matrix : bool, default=True
        If True, returns the full distance matrix. If False, returns the
        condensed distance vector.
    
    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    
    feature_names_in_ : ndarray of shape (n_features_in_,), dtype=str
        Names of features seen during fit.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from gower_clustering import GowerTransformer
    >>> 
    >>> # Create sample mixed dataset
    >>> data = {
    ...     "age": [25, 30, 35],
    ...     "city': ['NYC', 'LA', 'NYC"]
    ... }
    >>> df = pd.DataFrame(data)
    >>> 
    >>> # Transform to distance matrix
    >>> transformer = GowerTransformer()
    >>> distance_matrix = transformer.fit_transform(df)
    >>> print(distance_matrix.shape)
    (3, 3)
    """
    
    def __init__(
        self,
        cat_features: list[str] | None = None,
        return_distance_matrix: bool = True
    ) -> None:
        self.cat_features = cat_features
        self.return_distance_matrix = return_distance_matrix
    
    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "GowerTransformer":
        """
        Fit the transformer (no-op for Gower distance).
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Input data with mixed feature types.
        y : array-like of shape (n_samples,), optional
            Target values (ignored).
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame to handle mixed data types")
        
        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform input data to Gower distance matrix.
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Input data with mixed feature types.
        
        Returns
        -------
        distance_matrix : ndarray
            If return_distance_matrix=True: array of shape (n_samples, n_samples)
            If return_distance_matrix=False: condensed distance vector
        """
        check_is_fitted(self, ["n_features_in_"])
        
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame to handle mixed data types")
        
        try:
            distance_matrix = gower.gower_matrix(X, cat_features=self.cat_features)
        except Exception as e:
            raise ValueError(f"Failed to compute Gower distance: {str(e)}")
        
        if self.return_distance_matrix:
            return distance_matrix
        else:
            return squareform(distance_matrix, checks=False)
    
    def fit_transform(self, X: pd.DataFrame, y: np.ndarray | None = None) -> np.ndarray:
        """
        Fit transformer and transform input data.
        
        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            Input data with mixed feature types.
        y : array-like of shape (n_samples,), optional
            Target values (ignored).
        
        Returns
        -------
        distance_matrix : ndarray
            Distance matrix or condensed distance vector
        """
        return self.fit(X, y).transform(X)


def prepare_ordinal_features(
    df: pd.DataFrame,
    ordinal_config: dict[str, list[str]]
) -> pd.DataFrame:
    """
    Convert ordinal features to properly ordered Categorical dtype.
    
    This ensures the gower package treats them as ranked (numerical) rather 
    than categorical.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    ordinal_config : dict of str to list of str
        Dictionary mapping column names to ordered category lists.
        Example: {"education': ['High School', 'Bachelor', 'Master', 'PhD"]}
    
    Returns
    -------
    pd.DataFrame
        DataFrame with ordinal columns converted to ordered Categorical
    
    Examples
    --------
    >>> import pandas as pd
    >>> from gower_clustering import prepare_ordinal_features
    >>> 
    >>> df = pd.DataFrame({"education': ['Bachelor', 'PhD', 'Master"]})
    >>> ordinal_config = {"education': ['Bachelor', 'Master', 'PhD"]}
    >>> df_prepared = prepare_ordinal_features(df, ordinal_config)
    >>> print(df_prepared["education"].dtype)
    category
    >>> print(df_prepared["education"].cat.ordered)
    True
    """
    df = df.copy()
    
    for col, categories in ordinal_config.items():
        if col in df.columns:
            # Validate that all values in the column are in the categories
            missing_categories = set(df[col].dropna().unique()) - set(categories)
            if missing_categories:
                raise ValueError(
                    f"Column '{col}' contains values not in ordinal_config: "
                    f"{missing_categories}"
                )
            
            df[col] = pd.Categorical(
                df[col],
                categories=categories,
                ordered=True
            )
        else:
            warnings.warn(f"Column '{col}' not found in DataFrame")
    
    return df


def cluster_analysis_pipeline(
    df: pd.DataFrame,
    ordinal_config: dict[str, list[str]] | None = None,
    cat_features: list[str] | None = None,
    n_clusters: int = 3,
    linkage_method: str = "average",
    plot_dendrogram: bool = True,
    sample_labels: list[str] | None = None,
    compute_silhouette: bool = True
) -> dict[str, Any]:
    """
    Complete pipeline for Gower distance + hierarchical clustering.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data with mixed feature types.
        Ensure dtypes are set correctly:
        - Numerical: int, float
        - Categorical: object or category
        - Binary: bool or int (0/1)
        - Ordinal: will be set via ordinal_config
    ordinal_config : dict of str to list of str, optional
        Mapping of column names to ordered categories.
        Example: {"education': ['High School', 'Bachelor', 'Master', 'PhD"]}
    cat_features : list of str, optional
        Explicitly specify categorical columns (overrides dtype detection)
    n_clusters : int, default=3
        Number of clusters to extract
    linkage_method : str, default="average"
        Hierarchical clustering linkage method.
        Options: "single', 'complete', 'average', 'weighted"
    plot_dendrogram : bool, default=True
        Whether to plot dendrogram
    sample_labels : list of str, optional
        Labels for samples in dendrogram
    compute_silhouette : bool, default=True
        Whether to compute silhouette scores for cluster quality
    
    Returns
    -------
    dict
        Results dictionary containing:
        - "clusterer": Fitted GowerClusterer object
        - "labels": Cluster assignments (0-indexed, sklearn convention)
        - "distance_matrix": Gower distance matrix
        - "linkage_matrix": Hierarchical clustering linkage matrix
        - "silhouette": Silhouette scores (if compute_silhouette=True)
        - "df_clustered": Original dataframe with cluster assignments
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gower_clustering import cluster_analysis_pipeline
    >>> 
    >>> # Create sample dataset
    >>> np.random.seed(42)
    >>> data = {
    ...     "age": np.random.randint(18, 70, 10),
    ...     "education': np.random.choice(['High School', 'Bachelor"], 10),
    ...     "city': np.random.choice(['NYC', 'LA"], 10)
    ... }
    >>> df = pd.DataFrame(data)
    >>> 
    >>> # Run pipeline
    >>> ordinal_config = {"education': ['High School', 'Bachelor"]}
    >>> results = cluster_analysis_pipeline(
    ...     df=df,
    ...     ordinal_config=ordinal_config,
    ...     n_clusters=2,
    ...     plot_dendrogram=False
    ... )
    >>> print(len(results["labels"]))
    10
    """
    # Prepare ordinal features if specified
    if ordinal_config:
        df_processed = prepare_ordinal_features(df, ordinal_config)
    else:
        df_processed = df.copy()
    
    # Initialize and fit clusterer
    clusterer = GowerClusterer(
        linkage_method=linkage_method,
        n_clusters=n_clusters,
        cat_features=cat_features
    )
    
    labels = clusterer.fit_predict(df_processed)
    
    # Plot dendrogram
    if plot_dendrogram:
        fig = clusterer.plot_dendrogram(labels=sample_labels)
        plt.show()
    
    # Prepare results
    results = {
        "clusterer": clusterer,
        "labels": labels,
        "distance_matrix": clusterer.distance_matrix_,
        "linkage_matrix": clusterer.linkage_matrix_,
        "df_clustered": df.copy()
    }
    
    results["df_clustered']['cluster"] = labels
    
    # Compute silhouette scores
    if compute_silhouette:
        results["silhouette"] = clusterer.compute_silhouette_scores(labels)
    
    return results


# Example usage and testing
if __name__ == "__main__":
    # Create sample mixed dataset
    np.random.seed(42)
    n_samples = 50
    
    data = {
        "age": np.random.randint(18, 70, n_samples),
        "income": np.random.randint(30000, 150000, n_samples),
        "education": np.random.choice(["High School", "Bachelor", "Master", "PhD"], n_samples),
        "city": np.random.choice(["NYC", "LA", "Chicago", "Houston"], n_samples),
        "owns_car": np.random.choice([0, 1], n_samples),
        "satisfaction": np.random.choice(["Low", "Medium", "High"], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Define ordinal features with their ordering
    ordinal_config = {
        "education": ["High School", "Bachelor", "Master", "PhD"],
        "satisfaction": ["Low", "Medium", "High"]
    }
    
    # Test scikit-learn API compatibility
    print("Testing scikit-learn API compatibility...")
    
    # Prepare data
    df_prepared = prepare_ordinal_features(df, ordinal_config)
    
    # Test GowerClusterer
    clusterer = GowerClusterer(n_clusters=4, linkage_method="average")
    labels = clusterer.fit_predict(df_prepared)
    score = clusterer.score(df_prepared)
    
    print(f"Clustering completed. Labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")
    print(f"Silhouette score: {score:.3f}")
    
    # Test GowerTransformer
    transformer = GowerTransformer()
    distance_matrix = transformer.fit_transform(df_prepared)
    print(f"Distance matrix shape: {distance_matrix.shape}")
    
    # Run full pipeline
    print("\nRunning full clustering pipeline...")
    results = cluster_analysis_pipeline(
        df=df,
        ordinal_config=ordinal_config,
        n_clusters=4,
        linkage_method="average",
        plot_dendrogram=True,
        compute_silhouette=True
    )
    
    # Display results
    print("\n" + "="*60)
    print("CLUSTERING RESULTS")
    print("="*60)
    
    print(f"\nNumber of clusters: {len(np.unique(results["labels"]))}")
    print("\nCluster distribution:")
    cluster_counts = pd.Series(results["labels"]).value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        print(f"  Cluster {cluster}: {count} samples ({count/len(df)*100:.1f}%)")
    
    if "silhouette" in results:
        print(f"\nMean Silhouette Score: {results["silhouette']['mean"]:.3f}")
        print("  (Score ranges from -1 to 1; higher is better)")
    
    print("\n" + "="*60)
    print("CLUSTER PROFILES (Numerical Features)")
    print("="*60)
    profiles = results["clusterer"].get_cluster_profiles(results["labels"])
    print(profiles.round(2))
    
    print("\nScikit-learn API compatibility test completed successfully!")

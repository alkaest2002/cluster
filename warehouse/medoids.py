# warehouse/medoids.py
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gower
import warnings
from kmedoids import KMedoids
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics import silhouette_score
from typing import Any

warnings.filterwarnings("ignore")


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
        self.cat_features = cat_features
        self.cat_features_bool_: np.ndarray | None = None
        self.n_features_: int | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> GowerDistanceTransformer:
        """Fit the transformer to the data.
        
        Determines which features are categorical based on the cat_features parameter
        or by auto-detecting object columns in DataFrames.
        
        Args:
            X: Input data to fit the transformer on.
            y: Ignored. Present for API consistency.
            
        Returns:
            Self for method chaining.
        """
        # If X is a DataFrame
        if isinstance(X, pd.DataFrame):
            # Determine boolean mask for categorical features
            self.n_features_ = X.shape[1]
            # If cat_features is provided
            if self.cat_features is not None:
                # If user passed a list of column names
                if isinstance(self.cat_features[0], str):
                    self.cat_features_bool_ = X.columns.isin(self.cat_features)
                # If user passed indices
                elif isinstance(self.cat_features[0], int):
                    self.cat_features_bool_ = np.zeros(X.shape[1], dtype=bool)
                    self.cat_features_bool_[self.cat_features] = True
                # If user passed booleans
                else:
                    self.cat_features_bool_ = np.array(self.cat_features, dtype=bool)
            # If cat_features is not provided
            else:
                # Auto-detect object columns if None provided
                self.cat_features_bool_ = (X.dtypes == "object")
        # If X is a numpy array
        else:
            # Fallback for numpy arrays (assumes cat_features is provided manually)
            self.cat_features_bool_ = np.array(self.cat_features, dtype=bool) if self.cat_features is not None else None
            
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transform the input data into a Gower distance matrix.
        
        Args:
            X: Input data to transform into a distance matrix.
            
        Returns:
            Square distance matrix of shape (n_samples, n_samples) with Gower distances.
            
        Raises:
            RuntimeError: If Gower distance calculation fails.
        """
        print(f"Computing Gower Distance Matrix for {X.shape[0]} samples...")
        try:
            # Convert to DataFrame if needed
            X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

            # Compute Gower distance matrix
            dist_matrix = gower.gower_matrix(X_df, cat_features=self.cat_features_bool_)
            
            # Handle potential numerical instability
            # Set 1.0 for any NaN distances, i.e., max distance
            dist_matrix = np.nan_to_num(dist_matrix, nan=1.0)

            # Ensure diagonal is zero
            np.fill_diagonal(dist_matrix, 0.0)

            return dist_matrix.astype(np.float32)
            
        # Catch any errors during Gower calculation
        except Exception as e:
            raise RuntimeError(f"Gower calculation failed: {str(e)}")

    def fit_transform(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> np.ndarray:
        """Fit the transformer and transform the data in one step.
        
        Args:
            X: Input data to fit and transform.
            y: Ignored. Present for API consistency.
            
        Returns:
            Square distance matrix of shape (n_samples, n_samples) with Gower distances.
        """
        return self.fit(X).transform(X)


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
        self.n_clusters = n_clusters
        self.method = method
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Attributes set during fitting
        self.kmedoids_: KMedoids | None = None
        self.labels_: np.ndarray | None = None
        self.medoid_indices_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.cluster_centers_: None = None

    def fit(self, X: np.ndarray, y: Any = None) -> KMedoidsWrapper:
        """Fit the k-medoids clustering algorithm.
        
        Args:
            X: Precomputed square distance matrix of shape (n_samples, n_samples).
            y: Ignored. Present for API consistency.
            
        Returns:
            Self for method chaining.
            
        Raises:
            ValueError: If X is not a square matrix.
        """
        # Must be square matrix
        if X.shape[0] != X.shape[1]:
            raise ValueError(f"Input must be a square distance matrix. Got shape {X.shape}.")

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
            self.kmedoids_.fit(X)
            
            # Store attributes
            self.labels_ = self.kmedoids_.labels_
            self.medoid_indices_ = self.kmedoids_.medoid_indices_
            self.inertia_ = self.kmedoids_.inertia_
            self.cluster_centers_ = None  # Not applicable for k-medoids
            
        except Exception as e:
            print(f"Clustering failed for k={self.n_clusters}: {e}")
            self.labels_ = np.zeros(X.shape[0])
            self.inertia_ = np.inf
            
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the closest cluster for new samples.
        
        Args:
            X: New samples to predict clusters for.
            
        Returns:
            Cluster labels for the new samples.
            
        Raises:
            AttributeError: If the model has not been fitted yet.
        """
        if self.kmedoids_ is None:
            raise AttributeError("Model must be fitted before making predictions.")
        return self.kmedoids_.predict(X)


def run_optimization(
    df: pd.DataFrame, 
    k_min: int = 2, 
    k_max: int = 50, 
    cat_features: list[str] | list[int] | list[bool] | None = None
) -> tuple[KMedoidsWrapper, int, np.ndarray, pd.DataFrame]:
    """Run k-medoids optimization across multiple k values.
    
    This function computes the Gower distance matrix once and then evaluates
    k-medoids clustering for different numbers of clusters, using silhouette
    score and inertia as evaluation metrics.
    
    Args:
        df: Input DataFrame containing the data to cluster.
        k_min: Minimum number of clusters to evaluate.
        k_max: Maximum number of clusters to evaluate.
        cat_features: Specification of categorical features for Gower distance.
        
    Returns:
        Tuple containing:
            - Best k-medoids model (highest silhouette score)
            - Best number of clusters
            - Computed distance matrix
            - DataFrame with evaluation metrics for all k values
    """
    # 1. Compute Matrix
    transformer = GowerDistanceTransformer(cat_features=cat_features)
    dist_matrix = transformer.fit_transform(df)
    
    # Initialize vars
    results = []
    best_score = -1
    best_model = None
    best_k = 0
    
    print(f"\nOptimizing clusters from k={k_min} to {k_max}...")
    
    # 2. Iterate
    for k in range(k_min, k_max + 1):
        # Instantiate and fit model
        model = KMedoidsWrapper(n_clusters=k, random_state=42)
        model.fit(dist_matrix)
        
        # Store labels
        labels = model.labels_
        
        # Calculate Scores
        if len(np.unique(labels)) > 1:
            sil_score = silhouette_score(dist_matrix, labels, metric="precomputed")
        else:
            sil_score = -1.0
            
        results.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": sil_score
        })
        
        print(f"  k={k} | Silhouette: {sil_score:.4f} | Inertia: {model.inertia_:.2f}")
        
        # Track best model
        if sil_score > best_score:
            best_score = sil_score
            best_model = model
            best_k = k
            
    results_df = pd.DataFrame(results)
    
    return best_model, best_k, dist_matrix, results_df


def plot_metrics(results_df: pd.DataFrame) -> None:
    """Plot inertia and silhouette scores for different k values.
    
    Creates side-by-side plots showing the elbow curve (inertia) and silhouette
    analysis to help determine the optimal number of clusters.
    
    Args:
        results_df: DataFrame containing 'k', 'inertia', and 'silhouette' columns.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Elbow Curve (Inertia)
    ax1.plot(results_df['k'], results_df['inertia'], 'bo-')
    ax1.set_title('Elbow Curve (Inertia)')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia (Total Distance)')
    ax1.grid(True)
    
    # Silhouette Score
    ax2.plot(results_df['k'], results_df['silhouette'], 'ro-')
    ax2.set_title('Silhouette Analysis')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score (Higher is better)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def analyze_medoids(df: pd.DataFrame, model: KMedoidsWrapper, dist_matrix: np.ndarray) -> None:
    """Analyze and print detailed information about the cluster medoids.
    
    Displays the overall silhouette score and detailed information about each
    cluster including size and the actual medoid (representative sample).
    
    Args:
        df: Original DataFrame containing the clustered data.
        model: Fitted k-medoids model.
        dist_matrix: Precomputed distance matrix used for clustering.
    """
    print("\n" + "="*50)
    print("FINAL CLUSTER ANALYSIS")
    print("="*50)
    
    labels = model.labels_
    medoid_indices = model.medoid_indices_
    
    # Global score
    score = silhouette_score(dist_matrix, labels, metric='precomputed')
    print(f"Overall Silhouette Score: {score:.4f}")
    
    df_analysis = df.copy()
    df_analysis['Cluster'] = labels
    
    for i, idx in enumerate(medoid_indices):
        cluster_size = np.sum(labels == i)
        print(f"\nCluster {i} (Size: {cluster_size})")
        print(f"Medoid (Representative Profile):")
        print("-" * 30)
        # Print the row corresponding to the medoid
        print(df.iloc[idx])


def create_sample_data(n_samples: int = 300) -> pd.DataFrame:
    """Create sample mixed-type data for demonstration purposes.
    
    Generates a DataFrame with numerical and categorical features including
    age, income, score, gender, membership level, and activity status.
    
    Args:
        n_samples: Number of samples to generate.
        
    Returns:
        DataFrame containing the generated sample data.
    """
    # Use random seed generator for reproducibility
    np.random.seed(42)
    data = {
        "age": np.random.randint(20, 70, n_samples),
        "income": np.random.normal(60000, 15000, n_samples),
        "score": np.random.randint(1, 100, n_samples),
        "gender": np.random.choice(["Male", "Female", "Other"], n_samples),
        "membership": np.random.choice(["Basic", "Silver", "Gold"], n_samples),
        "active": np.random.choice([True, False], n_samples)
    }
    return pd.DataFrame(data)


if __name__ == "__main__":
    # 1. Create Data
    df = create_sample_data(300)
    print("Data Sample:")
    print(df.head())
    
    # 2. Identify Categorical Columns (Optional, Gower can auto-detect, but being explicit is safer)
    cat_cols = ["gender", "membership", "active"]
    
    # 3. Run Optimization (Calculates Matrix Once -> Loops k)
    best_model, best_k, dist_matrix, results_df = run_optimization(df, cat_features=cat_cols)

    # print best k
    print(f"\nBest number of clusters (k): {best_k}")
    
    # 4. Visualization
    # Note: If running in a non-GUI environment (like some servers), comment this out.
    try:
        plot_metrics(results_df)
    except:
        print("Skipping plotting (GUI not available)")

    # 5. Show detailed results for the best model
    analyze_medoids(df, best_model, dist_matrix)

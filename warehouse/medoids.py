# warehouse/medoids.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gower
import warnings
from kmedoids import KMedoids
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

class GowerDistanceTransformer(BaseEstimator, TransformerMixin):
    """
    Computes the Gower Distance matrix for mixed-type data.
    """
    def __init__(self, cat_features=None):
        """
        :param cat_features: List of column names or list of booleans indicating categorical cols.
        """
        self.cat_features = cat_features
        self.cat_features_bool_ = None

    def fit(self, X, y=None):
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

    def transform(self, X):
        print(f"Computing Gower Distance Matrix for {X.shape[0]} samples...")
        try:
            # Convert to DataFrame if needed
            X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

            # Compute Gower distance matrix
            dist_matrix = gower.gower_matrix(X_df, cat_features=self.cat_features_bool_)
            
            # Handle potential numerical instability
            dist_matrix = np.nan_to_num(dist_matrix, nan=1.0)

            # Ensure diagonal is zero
            np.fill_diagonal(dist_matrix, 0.0)

            return dist_matrix.astype(np.float32)
            
        # Catch any errors during Gower calculation
        except Exception as e:
            raise RuntimeError(f"Gower calculation failed: {str(e)}")

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class KMedoidsWrapper(BaseEstimator, ClusterMixin):
    """
    Wrapper for kmedoids-python to ensure Scikit-Learn compatibility.
    """
    def __init__(self, n_clusters=3, method="fasterpam", init="build", max_iter=100, random_state=42):
        self.n_clusters = n_clusters
        self.method = method
        self.init = init
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        X must be a precomputed distance matrix.
        """
        # Must be square matrix
        if X.shape[0] != X.shape[1]:
            raise ValueError(f"Input must be a square distance matrix. Got shape {X.shape}.")

        try:
            # Istantiate KMedoids model
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

    def predict(self, X):
        return self.kmedoids_.predict(X)


def run_optimization(df, k_min=2, k_max=50, cat_features=None):
    """
    1. Computes Distance Matrix (ONCE).
    2. Iterates through k values.
    3. Returns metrics for Elbow/Silhouette analysis.
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
        # Istantiate and fit model
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


def plot_metrics(results_df):
    """Plots Inertia and Silhouette scores side by side."""
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


def analyze_medoids(df, model, dist_matrix):
    """Prints the actual representative records (medoids)."""
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


def create_sample_data(n_samples=300):
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
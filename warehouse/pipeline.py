import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

# ===== STEP 1: Prepare Data for K-Prototypes =====
def prepare_data_for_kprototypes(df):
   """
   Prepare mixed data for K-Prototypes clustering.
   K-Prototypes keeps categorical features as-is and only scales numerical features.
   """
   # Separate numerical and categorical columns
   numerical_cols = df.select_dtypes(exclude='object').columns.tolist()
   categorical_cols = df.select_dtypes(include='object').columns.tolist()

   print(f"Numerical features: {len(numerical_cols)}")
   print(f"Categorical features: {len(categorical_cols)}")

   # Scale numerical features only
   scaler = StandardScaler()
   df_scaled = df.copy()
   df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])

   # Get indices of categorical columns (K-Prototypes needs this)
   categorical_indices = [df_scaled.columns.get_loc(col) for col in categorical_cols]

   return df_scaled, categorical_indices, numerical_cols, categorical_cols, scaler


# ===== STEP 2: Find Optimal Number of Clusters =====
def find_optimal_clusters(data, categorical_indices, k_range=range(2, 11), n_init=10):
   """
   Use elbow method (cost) and silhouette score to find optimal k.
   """
   costs = []
   silhouettes = []

   for k in k_range:
       print(f"Testing k={k}...")

       kproto = KPrototypes(n_clusters=k, init='Huang', n_init=n_init, verbose=0, random_state=42)
       clusters = kproto.fit_predict(data, categorical=categorical_indices)

       costs.append(kproto.cost_)

       # Calculate silhouette score (need to convert data for this metric)
       # Use Gower distance approximation or just numerical features
       numerical_data = data.iloc[:, [i for i in range(data.shape[1]) if i not in categorical_indices]]
       if len(numerical_data.columns) > 0:
           sil_score = silhouette_score(numerical_data, clusters)
           silhouettes.append(sil_score)
       else:
           silhouettes.append(None)

   # Plot results
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

   # Elbow plot
   ax1.plot(k_range, costs, 'bo-', linewidth=2, markersize=8)
   ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
   ax1.set_ylabel('Cost (Within-Cluster Sum of Distances)', fontsize=12)
   ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
   ax1.grid(True, alpha=0.3)

   # Silhouette plot
   if None not in silhouettes:
       ax2.plot(k_range, silhouettes, 'ro-', linewidth=2, markersize=8)
       ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
       ax2.set_ylabel('Silhouette Score', fontsize=12)
       ax2.set_title('Silhouette Score by k', fontsize=14, fontweight='bold')
       ax2.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

   return costs, silhouettes


# ===== STEP 3: Fit K-Prototypes with Optimal k =====
def fit_kprototypes(data, categorical_indices, n_clusters=5, n_init=10):
   """
   Fit K-Prototypes model with specified number of clusters.
   """
   kproto = KPrototypes(n_clusters=n_clusters, init='Huang', n_init=n_init,
                       verbose=1, random_state=42)

   clusters = kproto.fit_predict(data, categorical=categorical_indices)

   return kproto, clusters


# ===== STEP 4: Evaluate Cluster Quality =====
def evaluate_clusters(data, clusters, categorical_indices, numerical_cols):
   """
   Comprehensive cluster quality evaluation.
   """
   print("\n" + "="*60)
   print("CLUSTER QUALITY METRICS")
   print("="*60)

   # Basic cluster statistics
   unique, counts = np.unique(clusters, return_counts=True)
   print(f"\nCluster sizes:")
   for cluster_id, count in zip(unique, counts):
       print(f"  Cluster {cluster_id}: {count} samples ({count/len(clusters)*100:.1f}%)")

   # Extract numerical data for sklearn metrics
   numerical_data = data.iloc[:, [i for i in range(data.shape[1]) if i not in categorical_indices]]

   if len(numerical_data.columns) > 1:
       # Silhouette Score (higher is better, range: -1 to 1)
       sil_score = silhouette_score(numerical_data, clusters)
       print(f"\nSilhouette Score: {sil_score:.4f}")
       print("  → Measures how similar objects are to their own cluster vs other clusters")
       print("  → Range: -1 (poor) to +1 (excellent), >0.5 is good")

       # Davies-Bouldin Index (lower is better)
       db_score = davies_bouldin_score(numerical_data, clusters)
       print(f"\nDavies-Bouldin Index: {db_score:.4f}")
       print("  → Average similarity ratio of each cluster with its most similar cluster")
       print("  → Lower is better, values closer to 0 indicate better separation")

       # Calinski-Harabasz Index (higher is better)
       ch_score = calinski_harabasz_score(numerical_data, clusters)
       print(f"\nCalinski-Harabasz Index: {ch_score:.2f}")
       print("  → Ratio of between-cluster to within-cluster dispersion")
       print("  → Higher is better, indicates dense and well-separated clusters")

   print("\n" + "="*60)

   return {
       'silhouette': sil_score if len(numerical_data.columns) > 1 else None,
       'davies_bouldin': db_score if len(numerical_data.columns) > 1 else None,
       'calinski_harabasz': ch_score if len(numerical_data.columns) > 1 else None,
       'cluster_sizes': dict(zip(unique, counts))
   }


# ===== STEP 5: Analyze Cluster Characteristics =====
def analyze_clusters(data, clusters, numerical_cols, categorical_cols):
   """
   Provide detailed cluster profiling.
   """
   df_with_clusters = data.copy()
   df_with_clusters['Cluster'] = clusters

   print("\n" + "="*60)
   print("CLUSTER CHARACTERISTICS")
   print("="*60)

   # Numerical features by cluster
   if len(numerical_cols) > 0:
       print("\nNumerical Features by Cluster (mean values):")
       print(df_with_clusters.groupby('Cluster')[numerical_cols].mean().round(3))

   # Categorical features by cluster
   if len(categorical_cols) > 0:
       print("\n\nCategorical Features by Cluster (mode values):")
       for col in categorical_cols:
           print(f"\n{col}:")
           mode_per_cluster = df_with_clusters.groupby('Cluster')[col].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else None)
           print(mode_per_cluster)

   return df_with_clusters


# ===== STEP 6: Visualize Clusters =====
def visualize_clusters(data, clusters, categorical_indices, method='umap'):
   """
   Visualize clusters using dimensionality reduction.
   """
   from sklearn.decomposition import PCA

   # Use only numerical features for visualization
   numerical_data = data.iloc[:, [i for i in range(data.shape[1]) if i not in categorical_indices]]

   if len(numerical_data.columns) < 2:
       print("Not enough numerical features for visualization")
       return

   # Reduce to 2D
   if method == 'pca' or len(numerical_data.columns) > 50:
       reducer = PCA(n_components=2, random_state=42)
       embedding = reducer.fit_transform(numerical_data)
       explained_var = sum(reducer.explained_variance_ratio_) * 100
       title_suffix = f"(PCA - {explained_var:.1f}% variance explained)"
   else:
       # Use UMAP if available
       try:
           import umap
           reducer = umap.UMAP(n_components=2, random_state=42)
           embedding = reducer.fit_transform(numerical_data)
           title_suffix = "(UMAP)"
       except ImportError:
           reducer = PCA(n_components=2, random_state=42)
           embedding = reducer.fit_transform(numerical_data)
           explained_var = sum(reducer.explained_variance_ratio_) * 100
           title_suffix = f"(PCA - {explained_var:.1f}% variance explained)"

   # Create visualization
   plt.figure(figsize=(12, 8))
   scatter = plt.scatter(embedding[:, 0], embedding[:, 1],
                        c=clusters, cmap='tab10',
                        s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
   plt.colorbar(scatter, label='Cluster')
   plt.xlabel('Dimension 1', fontsize=12)
   plt.ylabel('Dimension 2', fontsize=12)
   plt.title(f'K-Prototypes Clusters Visualization {title_suffix}',
             fontsize=14, fontweight='bold')
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.show()


# ===== EXAMPLE USAGE =====
if __name__ == "__main__":
   # Example with synthetic data
   print("Creating example dataset...")

   # Create sample data with mixed types
   np.random.seed(42)
   n_samples = 1000

   sample_data = pd.DataFrame({
       'revenue': np.random.exponential(100, n_samples),
       'visits': np.random.poisson(10, n_samples),
       'avg_session_duration': np.random.gamma(2, 30, n_samples),
       'bounce_rate': np.random.beta(2, 5, n_samples),
       'device': np.random.choice(['mobile', 'desktop', 'tablet'], n_samples),
       'channel': np.random.choice(['organic', 'paid', 'social', 'direct'], n_samples),
       'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples)
   })

   print("\n1. Preparing data...")
   data_scaled, cat_indices, num_cols, cat_cols, scaler = prepare_data_for_kprototypes(sample_data)

   print("\n2. Finding optimal number of clusters...")
   costs, silhouettes = find_optimal_clusters(data_scaled, cat_indices, k_range=range(2, 9))

   print("\n3. Fitting K-Prototypes with k=4...")
   model, clusters = fit_kprototypes(data_scaled, cat_indices, n_clusters=4)

   print("\n4. Evaluating cluster quality...")
   metrics = evaluate_clusters(data_scaled, clusters, cat_indices, num_cols)

   print("\n5. Analyzing cluster characteristics...")
   df_clustered = analyze_clusters(sample_data, clusters, num_cols, cat_cols)

   print("\n6. Visualizing clusters...")
   visualize_clusters(data_scaled, clusters, cat_indices, method='pca')

   print("\n✓ Clustering complete! Check the plots and metrics above.")
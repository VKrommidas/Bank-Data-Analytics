import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Loading the dataset
file_path = 'bank-full.csv'
data = pd.read_csv(file_path, sep=';')

# Replacing 'unknown' values ​​with NaN and removing these rows
#data.replace('unknown', np.nan, inplace=True)
#data.dropna(inplace=True)
data = data.dropna()

# Σelection of the features that will be used for clustering
features_for_clustering = ['age', 'balance', 'duration']

# Creation of the feature table
X = data[features_for_clustering]

# Data normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# Part 1: K-Means Clustering
# -------------------------

# Selection of the number of clusters (k) for K-means
k = 3

# Creation of the K-means model
kmeans = KMeans(n_clusters=k, random_state=42,n_init=10)
kmeans.fit(X_scaled)

# Prediction of clusters
clusters_kmeans = kmeans.predict(X_scaled)

# Calculation of the Silhouette Score for K-means
#silhouette_avg_kmeans = silhouette_score(X_scaled, clusters_kmeans)
silhouette_avg_kmeans = silhouette_score(
    X_scaled,
    clusters_kmeans,
    sample_size=5000,
    random_state=42
)
print(f"K-means Silhouette Score: {silhouette_avg_kmeans:.4f}")

# Visualization of K-means results in the feature space (scatter plot)
plt.figure(figsize=(10, 6))
plt.scatter(X['age'], X['balance'], c=clusters_kmeans, cmap='viridis',alpha=0.6)
plt.xlabel('Age')
plt.ylabel('Balance')
plt.title('K-means Clustering')
plt.colorbar(label='Cluster')
plt.show()

# -------------------------
# Part 2: DBSCAN Clustering
# -------------------------

# Selection of parameters for the DBSCAN algorithm
epsilon = 0.5
min_samples = 5

# Creating the DBSCAN model
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
clusters_dbscan = dbscan.fit_predict(X_scaled)

# Evaluation of DBSCAN, excluding noise points
non_noise_mask = clusters_dbscan != -1
dbscan_labels = clusters_dbscan[non_noise_mask]
unique_clusters = np.unique(dbscan_labels)

print("Number of DBSCAN clusters:", len(unique_clusters))
print("Number of noise points:", np.sum(~non_noise_mask))

if len(unique_clusters) >= 2 and len(unique_clusters) < len(dbscan_labels):
    silhouette_avg_dbscan = silhouette_score(
        X_scaled[non_noise_mask],
        dbscan_labels,
        sample_size=min(5000, len(dbscan_labels)),
        random_state=42
    )
    print(f"DBSCAN Silhouette Score (excluding noise): {silhouette_avg_dbscan:.4f}")
else:
    print("DBSCAN Silhouette Score cannot be calculated reliably.")

# Visualization of DBSCAN results in the feature space (scatter plot)
plt.figure(figsize=(10, 6))
plt.scatter(X['age'], X['balance'], c=clusters_dbscan, cmap='viridis',alpha=0.6)
plt.xlabel('Age')
plt.ylabel('Balance')
plt.title('DBSCAN Clustering')
plt.colorbar(label='Cluster')
plt.show()
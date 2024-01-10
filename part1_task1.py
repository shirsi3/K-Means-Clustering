import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate synthetic data
n_samples = 300
n_features = 2
n_clusters = 4
random_state = 42

X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=random_state)

# Print some information about the input data
print(f"Number of samples: {n_samples}")
print(f"Number of features: {n_features}")
print(f"Number of clusters: {n_clusters}")

# Perform K-Means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)  # Explicitly set n_init
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
centers = kmeans.cluster_centers_

# Print cluster assignments
print("Cluster Assignments:")
for i in range(n_samples):
    print(f"Sample {i}: Cluster {y_kmeans[i]}")

# Plot the data points and cluster centroids
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=50)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.legend()
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

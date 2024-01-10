import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


file_path = "userknowledge.csv"
data = pd.read_csv(file_path)

X = data[['STG', 'SCG', 'STR', 'LPR', 'PEG']]

# Perform K-Means clustering for different values of k
k_values = range(2, 11)  # Set the range of k values to try
wcss_scores = []  # To store WCSS values
silhouette_scores = []  # To store silhouette scores

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # Set n_init explicitly
    kmeans.fit(X)
    
    # Calculate WCSS (inertia) and silhouette score
    wcss = kmeans.inertia_
    silhouette_avg = silhouette_score(X, kmeans.labels_)
    
    wcss_scores.append(wcss)
    silhouette_scores.append(silhouette_avg)

# Plot the WCSS and silhouette scores for different values of k
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(k_values, wcss_scores, marker='o')
plt.title('WCSS vs. Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')

plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Score vs. Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')

plt.tight_layout()
plt.show()

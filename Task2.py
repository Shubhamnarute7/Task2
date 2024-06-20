# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Example customer purchase history data (replace with your dataset)
data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'TotalSpent': [1000, 500, 1200, 800, 1500, 300, 100, 600, 2000, 400],
    'NumPurchases': [5, 2, 6, 4, 7, 1, 0, 3, 8, 2]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Select features for clustering
X = df[['TotalSpent', 'NumPurchases']]

# Standardize the features (important for K-means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the number of clusters (K)
k = 3

# Apply K-means clustering
kmeans = KMeans(n_clusters=k, random_state=0)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original DataFrame
df['Cluster'] = clusters

# Visualize the clusters
plt.figure(figsize=(8, 6))
plt.scatter(df['TotalSpent'], df['NumPurchases'], c=df['Cluster'], cmap='viridis', s=50, alpha=0.7)
plt.xlabel('Total Spending')
plt.ylabel('Number of Purchases')
plt.title('K-means Clustering of Customers')
plt.colorbar(label='Cluster')
plt.show()

# Print cluster centers (centroid of each cluster)
print('Cluster Centers:')
print(pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=['TotalSpent', 'NumPurchases']))

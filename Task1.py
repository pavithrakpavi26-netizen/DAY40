# Step 1: Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 2: Load dataset
# Make sure Mall_Customers.csv is in the same folder
df = pd.read_csv("Mall_Customers.csv")

# Step 3: Display first few rows
print("Dataset Preview:")
print(df.head())

# Step 4: Select required features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 5: Elbow Method to find optimal K
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure()
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.show()

# Step 6: Apply K-Means (choose K based on elbow graph, usually 5)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Step 7: Add cluster labels to dataset
df['Cluster'] = y_kmeans

print("\nClustered Data:")
print(df.head())

# Step 8: Visualize clusters
plt.figure()

# Plot each cluster with different colors
plt.scatter(X[y_kmeans == 0]['Annual Income (k$)'],
            X[y_kmeans == 0]['Spending Score (1-100)'],
            label='Cluster 1')

plt.scatter(X[y_kmeans == 1]['Annual Income (k$)'],
            X[y_kmeans == 1]['Spending Score (1-100)'],
            label='Cluster 2')

plt.scatter(X[y_kmeans == 2]['Annual Income (k$)'],
            X[y_kmeans == 2]['Spending Score (1-100)'],
            label='Cluster 3')

plt.scatter(X[y_kmeans == 3]['Annual Income (k$)'],
            X[y_kmeans == 3]['Spending Score (1-100)'],
            label='Cluster 4')

plt.scatter(X[y_kmeans == 4]['Annual Income (k$)'],
            X[y_kmeans == 4]['Spending Score (1-100)'],
            label='Cluster 5')

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=200, label='Centroids')

plt.title('Customer Segmentation')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
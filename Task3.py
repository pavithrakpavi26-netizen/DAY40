# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("students.csv")

# Select features
X = df[['Math Score', 'Reading Score', 'Writing Score']]


# Step A: Elbow Method

wcss = []

for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure()
plt.plot(range(1, 10), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()


#  Step B: Apply K-Means (Try K=3)

kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Add cluster labels
df['Cluster'] = y_kmeans

print("\nClustered Data:")
print(df)

# Step C: Visualize Clusters (2D)

plt.figure()

plt.scatter(X[y_kmeans == 0]['Math Score'],
            X[y_kmeans == 0]['Reading Score'],
            label='Cluster 1')

plt.scatter(X[y_kmeans == 1]['Math Score'],
            X[y_kmeans == 1]['Reading Score'],
            label='Cluster 2')

plt.scatter(X[y_kmeans == 2]['Math Score'],
            X[y_kmeans == 2]['Reading Score'],
            label='Cluster 3')

# Plot centroids
centroids = kmeans.cluster_centers_

plt.scatter(centroids[:, 0],
            centroids[:, 1],
            s=200,
            label='Centroids')

plt.xlabel("Math Score")
plt.ylabel("Reading Score")
plt.title("Student Clusters")
plt.legend()
plt.show()

# ================================
# 📌 Step D: Show Centroids (Average Scores)
# ================================
centroid_df = pd.DataFrame(centroids,
                          columns=['Math Avg', 'Reading Avg', 'Writing Avg'])

print("\nCluster Centroids (Average Student Scores):")
print(centroid_df)
# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 2: Load dataset
df = pd.read_csv("credit_card_dataset.csv")

# Step 3: Select features
# PURCHASES = large values
# PURCHASES_FREQUENCY = small values (0–1)
X = df[['PURCHASES', 'PURCHASES_FREQUENCY']]


# STEP A: WITHOUT SCALING

kmeans_no_scale = KMeans(n_clusters=3, random_state=42)
y_no_scale = kmeans_no_scale.fit_predict(X)

# Plot WITHOUT scaling
plt.figure()
plt.scatter(X['PURCHASES'], X['PURCHASES_FREQUENCY'], c=y_no_scale)
plt.scatter(kmeans_no_scale.cluster_centers_[:, 0],
            kmeans_no_scale.cluster_centers_[:, 1],
            s=200)
plt.title("K-Means WITHOUT Scaling")
plt.xlabel("PURCHASES")
plt.ylabel("PURCHASES_FREQUENCY")
plt.show()



# STEP B: WITH STANDARD SCALING

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans_scaled = KMeans(n_clusters=3, random_state=42)
y_scaled = kmeans_scaled.fit_predict(X_scaled)

# Plot WITH scaling
plt.figure()
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_scaled)
plt.scatter(kmeans_scaled.cluster_centers_[:, 0],
            kmeans_scaled.cluster_centers_[:, 1],
            s=200)
plt.title("K-Means WITH Scaling")
plt.xlabel("PURCHASES (Scaled)")
plt.ylabel("PURCHASES_FREQUENCY (Scaled)")
plt.show()
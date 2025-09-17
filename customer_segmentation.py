import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("customers.csv")

# Selecting features (AnnualIncome & SpendingScore)
X = data[["AnnualIncome", "SpendingScore"]]

# KMeans clustering (3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
data["Cluster"] = kmeans.fit_predict(X)

# Save clustered data
data.to_csv("segmented_customers.csv", index=False)

# Plot clusters
plt.figure(figsize=(8,6))
plt.scatter(X["AnnualIncome"], X["SpendingScore"],
            c=data["Cluster"], cmap="viridis", s=50)
plt.scatter(kmeans.cluster_centers_[:,0],
            kmeans.cluster_centers_[:,1],
            s=200, c="red", marker="X", label="Centroids")
plt.title("Customer Segmentation (KMeans)")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()
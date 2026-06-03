import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("Mall_Customers.csv")

# Select features
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# Create K-Means model
kmeans = KMeans(n_clusters=5, random_state=42)

# Fit model and predict clusters
data["Cluster"] = kmeans.fit_predict(X)

# Display first few rows
print(data.head())

# Visualize clusters
plt.scatter(
    X["Annual Income (k$)"],
    X["Spending Score (1-100)"],
    c=data["Cluster"]
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")

plt.savefig("customer_clusters.png")

plt.show()
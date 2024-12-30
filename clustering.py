import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

def preprocess_data(data):
    """Preprocess the dataset: handle non-numeric columns with one-hot encoding."""
    # One-hot encode non-numeric columns
    non_numeric_cols = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=non_numeric_cols, drop_first=True)
    
    # Optionally, handle missing values (e.g., by filling with mean or dropping)
    data = data.fillna(data.mean())  # Filling NaNs with column mean
    
    return data

def apply_kmeans_clustering(data, n_clusters):
    """Apply K-Means clustering and add cluster labels to the dataset."""
    # Standardize the data for better clustering
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data_scaled)
    
    # Add the cluster labels to a copy of the original data
    clustered_data = data.copy()
    clustered_data['Cluster'] = kmeans.labels_
    
    return clustered_data, kmeans

def plot_cluster_centers(data, kmeans, title):
    """Plot the cluster centers for K-Means clustering."""
    # Ensure the 'Cluster' column exists
    if 'Cluster' not in data.columns:
        raise ValueError("The 'Cluster' column is missing in the data.")
    
    # Reduce data to 2 dimensions using PCA for better visualization
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data.drop('Cluster', axis=1))  # Exclude 'Cluster' column for PCA
    
    # Scatter plot of 2D projection
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data_2d[:, 0], y=data_2d[:, 1], hue=data['Cluster'], palette='Set1', s=100)
    
    # Plot the cluster centers
    cluster_centers_2d = pca.transform(kmeans.cluster_centers_)
    plt.scatter(cluster_centers_2d[:, 0], cluster_centers_2d[:, 1], marker='X', s=200, c='black')
    
    plt.title(title, fontsize=16)
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)
    plt.legend(title='Cluster')
    plt.show()

# Load datasets
orbit_df = pd.read_csv(r'C:\Users\ankit\Downloads\final_orbit.csv')
impact_df = pd.read_csv(r'C:\Users\ankit\Downloads\final_impact.csv')

# Preprocess datasets
orbit_df = preprocess_data(orbit_df)
impact_df = preprocess_data(impact_df)

# Apply K-Means clustering for both datasets
n_clusters = 3  # You can adjust the number of clusters as needed

print("Applying K-Means Clustering for Orbit Dataset...")
orbit_clustered, orbit_kmeans = apply_kmeans_clustering(orbit_df, n_clusters)

print("Applying K-Means Clustering for Impact Dataset...")
impact_clustered, impact_kmeans = apply_kmeans_clustering(impact_df, n_clusters)

# Visualize clusters
print("Visualizing Clusters for Orbit Dataset...")
plot_cluster_centers(orbit_clustered, orbit_kmeans, title="K-Means Clustering for Orbit Dataset")

print("Visualizing Clusters for Impact Dataset...")
plot_cluster_centers(impact_clustered, impact_kmeans, title="K-Means Clustering for Impact Dataset")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Streamlit app
st.title("Asteroid Orbit and Impact Analysis")

# File upload section
st.sidebar.header("Upload Datasets")
impact_file = st.sidebar.file_uploader("Upload impacts.csv", type=["csv"])
orbit_file = st.sidebar.file_uploader("Upload orbits.csv", type=["csv"])

# Function to preprocess data
def preprocess_data(data):
    """Preprocess the dataset: handle non-numeric columns with one-hot encoding."""
    non_numeric_cols = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=non_numeric_cols, drop_first=True)
    return data

# Function to apply K-Means clustering
def apply_kmeans_clustering(data, n_clusters):
    """Apply K-Means clustering and add cluster labels to the dataset."""
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data_scaled)
    data['Cluster'] = kmeans.labels_
    return data, kmeans

# Function to plot clusters
def plot_cluster_centers(data, kmeans, title):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1], hue='Cluster', palette='Set1', s=100)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', s=200, c='black')
    plt.title(title, fontsize=16)
    plt.xlabel(data.columns[0], fontsize=14)
    plt.ylabel(data.columns[1], fontsize=14)
    st.pyplot(plt)

# Processing datasets
if impact_file and orbit_file:
    # Load datasets
    impact_df = pd.read_csv(impact_file)
    orbit_df = pd.read_csv(orbit_file)

    # Display initial data
    st.subheader("Uploaded Datasets")
    st.write("**Impact Dataset**")
    st.dataframe(impact_df.head())
    st.write("**Orbit Dataset**")
    st.dataframe(orbit_df.head())

    # Check missing values
    st.subheader("Missing Values")
    st.write("**Impact Dataset**", impact_df.isnull().sum())
    st.write("**Orbit Dataset**", orbit_df.isnull().sum())

    # Handle missing values
    orbit_df.dropna(inplace=True)

    # Summary statistics
    st.subheader("Summary Statistics")
    st.write("**Impact Dataset**")
    st.dataframe(impact_df.describe())
    st.write("**Orbit Dataset**")
    st.dataframe(orbit_df.describe())

    # Clustering
    st.sidebar.subheader("Clustering Parameters")
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)

    st.subheader("K-Means Clustering")
    if st.button("Cluster and Visualize"):
        # Preprocess and cluster
        st.write("Processing Orbit Dataset...")
        orbit_df_processed = preprocess_data(orbit_df)
        orbit_clustered, orbit_kmeans = apply_kmeans_clustering(orbit_df_processed, n_clusters)

        st.write("Processing Impact Dataset...")
        impact_df_processed = preprocess_data(impact_df)
        impact_clustered, impact_kmeans = apply_kmeans_clustering(impact_df_processed, n_clusters)

        # Plot clusters
        st.subheader("Orbit Dataset Clusters")
        plot_cluster_centers(orbit_df_processed, orbit_kmeans, title="K-Means Clustering for Orbit Dataset")
        
        st.subheader("Impact Dataset Clusters")
        plot_cluster_centers(impact_df_processed, impact_kmeans, title="K-Means Clustering for Impact Dataset")

        # Save processed files
        orbit_csv = orbit_clustered.to_csv(index=False).encode('utf-8')
        impact_csv = impact_clustered.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download Processed Orbit Dataset",
            data=orbit_csv,
            file_name="processed_orbit.csv",
            mime="text/csv"
        )

        st.download_button(
            label="Download Processed Impact Dataset",
            data=impact_csv,
            file_name="processed_impact.csv",
            mime="text/csv"
        )
else:
    st.write("Upload both datasets to start the analysis.")

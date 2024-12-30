import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from clustering import apply_kmeans_clustering, preprocess_data, plot_cluster_centers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load datasets
impact_df = pd.read_csv('data/final_impact.csv')  # Dataset for asteroid impact information
orbit_df = pd.read_csv('data/final_orbit.csv')    # Dataset for asteroid orbital data

# Preprocess datasets
impact_df = preprocess_data(impact_df)
orbit_df = preprocess_data(orbit_df)

# Load the pre-trained model
with open('models/random_forest_model.pkl', 'rb') as model_file:
    random_forest_model = pickle.load(model_file)

# Initialize StandardScaler (assuming model expects scaled data)
scaler = StandardScaler()

# Streamlit Page Configuration
st.set_page_config(page_title="Asteroid Prediction System", layout="wide")

# Page Title
st.title("Asteroid Prediction System")

# Sidebar: Upload Dataset (Optional for testing)
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv"])

# Main form for Asteroid Prediction
with st.form(key="prediction-form"):
    period_start = st.number_input("Period Start", value=0)
    period_end = st.number_input("Period End", value=0)
    impact_probability = st.number_input("Cumulative Impact Probability", value=0.0, step=0.01)
    velocity = st.number_input("Asteroid Velocity (km/s)", value=0.0, step=0.1)
    magnitude = st.number_input("Magnitude", value=0.0, step=0.1)
    diameter = st.number_input("Asteroid Diameter (km)", value=0.0, step=0.1)
    cumulative_palermo = st.number_input("Cumulative Palermo Scale", value=0.0, step=0.1)
    max_palermo = st.number_input("Maximum Palermo Scale", value=0.0, step=0.1)
    max_torino = st.number_input("Maximum Torino Scale", value=0.0, step=0.1)

    submit_button = st.form_submit_button(label="Predict")

# Handle prediction form submission
if submit_button:
    input_data = {
        "Period Start": period_start,
        "Period End": period_end,
        "Cumulative Impact Probability": impact_probability,
        "Asteroid Velocity": velocity,
        "Asteroid Magnitude": magnitude,
        "Asteroid Diameter (km)": diameter,
        "Cumulative Palermo Scale": cumulative_palermo,
        "Maximum Palermo Scale": max_palermo,
        "Maximum Torino Scale": max_torino
    }

    # Preprocess and scale data
    input_df = pd.DataFrame([input_data])

    # Check and reorder columns to match model's expected order
    expected_columns = [
        "Period Start", "Period End", "Cumulative Impact Probability", 
        "Asteroid Velocity", "Asteroid Magnitude", "Asteroid Diameter (km)", 
        "Cumulative Palermo Scale", "Maximum Palermo Scale", "Maximum Torino Scale"
    ]
    input_df = input_df[expected_columns]

    # Scale the input data using the pre-trained scaler
    input_df_scaled = scaler.fit_transform(input_df)

    # Run prediction using the pre-trained model
    prediction = random_forest_model.predict(input_df_scaled)

    st.write(f"Prediction: {prediction[0]}")

# Clustering Function

st.header("Clustering of Impact DF")
n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)
cluster_button = st.button("Generate Clusters")

if cluster_button:
    # Generate Dummy Data
    np.random.seed(42)  # For reproducibility
    dummy_data = np.random.rand(600, 7)  # 600 samples with 7 features

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(dummy_data)

    # Reduce dimensions using PCA for visualization
    pca = PCA(n_components=2)
    dummy_data_pca = pca.fit_transform(dummy_data)

    # Visualize clusters
    st.write("Cluster Visualization:")
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for cluster in range(n_clusters):
        cluster_points = dummy_data_pca[cluster_labels == cluster]
        ax.scatter(
            cluster_points[:, 0], 
            cluster_points[:, 1], 
            label=f"Cluster {cluster + 1}", 
            alpha=0.8, 
            edgecolor='k', 
            s=50  # Adjust marker size for better visibility
        )

    # Add cluster centroids
    centroids_pca = pca.transform(kmeans.cluster_centers_)
    ax.scatter(
        centroids_pca[:, 0], 
        centroids_pca[:, 1], 
        color='red', 
        marker='X', 
        s=200, 
        label='Centroids'
    )




# Histogram Section: Identifying Harmful Asteroids
st.header("Asteroid Risk Analysis: Harmful or Not")

# Define thresholds for harmful asteroid identification
impact_probability_threshold = 0.01  # Example threshold for Cumulative Impact Probability
diameter_threshold = 1.0  # Example threshold for Diameter (in km)

# Add a button to generate the histogram
histogram_button = st.button("Generate Risk Histogram")

if histogram_button:
    # Identify harmful asteroids based on thresholds
    impact_df['Is_Harmful'] = (impact_df['Cumulative Impact Probability'] > impact_probability_threshold) & \
                              (impact_df['Asteroid Diameter (km)'] > diameter_threshold)
    
    # Count harmful and non-harmful asteroids
    harmful_count = impact_df['Is_Harmful'].sum()
    non_harmful_count = len(impact_df) - harmful_count

    # Create a histogram
    st.write(f"Total Harmful Asteroids: {harmful_count}")
    st.write(f"Total Non-Harmful Asteroids: {non_harmful_count}")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(["Non-Harmful", "Harmful"], [non_harmful_count, harmful_count], color=['green', 'red'])
    ax.set_title("Asteroid Risk Distribution", fontsize=16)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_xlabel("Asteroid Classification", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Display the plot
    st.pyplot(fig)

    # Enhance aesthetics
    ax.set_title(" Clustering Visualization of Impact Dataframe", fontsize=16)
    ax.set_xlabel("Principal Component 1", fontsize=12)
    ax.set_ylabel("Principal Component 2", fontsize=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, linestyle='--', alpha=0.6)

    st.pyplot(fig)
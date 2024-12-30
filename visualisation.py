import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(data, dataset_name):
    """Preprocess the dataset: handle non-numeric columns with one-hot encoding."""
    print(f"--- Preprocessing {dataset_name} ---")
    
    # Describe the dataset
    print(f"Shape: {data.shape}")
    print("\nColumns and Data Types:")
    print(data.dtypes)
    print("\nMissing Values:")
    print(data.isnull().sum())
    print("\nPreview of Dataset:")
    print(data.head())
    
    # One-hot encode non-numeric columns
    non_numeric_cols = data.select_dtypes(include=['object']).columns
    print(f"\nNon-numeric Columns in {dataset_name}: {non_numeric_cols.tolist()}")
    data = pd.get_dummies(data, columns=non_numeric_cols, drop_first=True)
    
    print(f"\nShape after One-Hot Encoding: {data.shape}")
    print(f"\nColumns after Encoding: {data.columns.tolist()}")
    return data

def plot_correlation_heatmap(data, title):
    """Plots a heatmap of correlations for the dataset."""
    numeric_data = data.select_dtypes(include=["number"])  # Select only numeric columns
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(title, fontsize=16)
    plt.show()

def plot_distribution(data, feature, title):
    """Plots a distribution plot for a specific feature."""
    plt.figure(figsize=(10, 6))
    sns.histplot(data[feature], kde=True, color="blue", bins=30)
    plt.title(f"Distribution of {title}", fontsize=16)
    plt.xlabel(feature, fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.show()

def plot_boxenplot(data, feature, title):
    """Plots a boxenplot for outlier detection."""
    plt.figure(figsize=(10, 6))
    sns.boxenplot(x=data[feature], color="green")
    plt.title(f"Boxenplot of {title}", fontsize=16)
    plt.xlabel(feature, fontsize=14)
    plt.show()

# Load datasets
orbit_df = pd.read_csv(r'C:\Users\ankit\Downloads\final_orbit.csv')
impact_df = pd.read_csv(r'C:\Users\ankit\Downloads\final_impact.csv')

# Preprocess datasets
orbit_df = preprocess_data(orbit_df, dataset_name="Orbit Dataset")
impact_df = preprocess_data(impact_df, dataset_name="Impact Dataset")

# Correlation heatmaps
print("Visualizing Correlation Heatmap for Orbit Dataset...")
plot_correlation_heatmap(orbit_df, title="Orbit Dataset Correlation Heatmap")

print("Visualizing Correlation Heatmap for Impact Dataset...")
plot_correlation_heatmap(impact_df, title="Impact Dataset Correlation Heatmap")

# Distribution plots
print("Visualizing Distribution of Asteroid Magnitude (Orbit Dataset)...")
if 'Asteroid Magnitude' in orbit_df.columns:
    plot_distribution(orbit_df, feature='Asteroid Magnitude', title="Asteroid Magnitude (Orbit Dataset)")

print("Visualizing Distribution of Asteroid Diameter (Impact Dataset)...")
if 'Asteroid Diameter (km)' in impact_df.columns:
    plot_distribution(impact_df, feature='Asteroid Diameter (km)', title="Asteroid Diameter (Impact Dataset)")

# Boxenplots
print("Visualizing Boxenplot for Asteroid Magnitude (Orbit Dataset)...")
if 'Asteroid Magnitude' in orbit_df.columns:
    plot_boxenplot(orbit_df, feature='Asteroid Magnitude', title="Asteroid Magnitude (Orbit Dataset)")

print("Visualizing Boxenplot for Cumulative Palermo Scale (Impact Dataset)...")
if 'Cumulative Palermo Scale' in impact_df.columns:
    plot_boxenplot(impact_df, feature='Cumulative Palermo Scale', title="Cumulative Palermo Scale (Impact Dataset)")

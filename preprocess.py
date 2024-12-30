import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
impact_df = pd.read_csv(r'C:\Users\ankit\Downloads\impacts.csv')
orbit_df = pd.read_csv(r'C:\Users\ankit\Downloads\orbits.csv')

# Display the first few rows
print("Impact DataFrame Head:")
print(impact_df.head())
print("\nOrbit DataFrame Head:")
print(orbit_df.head())

# Check dataset sizes
print("\nSize of impact_df:", impact_df.shape)
print("Size of orbit_df:", orbit_df.shape)

# Check for missing values
print("\nMissing values in impact_df:\n", impact_df.isnull().sum())
print("\nMissing values in orbit_df:\n", orbit_df.isnull().sum())

# Handle missing values in orbit_df
orbit_df.dropna(inplace=True)

# Summary statistics
print("\nOrbit Dataset Summary:")
print(orbit_df.describe())
print("\nImpact Dataset Summary:")
print(impact_df.describe())

# Dataset info
print("\nOrbit Dataset Info:")
orbit_df.info()
print("\nImpact Dataset Info:")
impact_df.info()

# Export cleaned datasets
orbit_df.to_csv(r'C:\Users\ankit\Downloads/final_orbit.csv', index=False)
impact_df.to_csv(r'C:\Users\ankit\Downloads/final_impact.csv', index=False)

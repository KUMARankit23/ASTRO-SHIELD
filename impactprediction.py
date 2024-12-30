import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA

# Load and preprocess data
df = pd.read_csv(r'C:\Users\ankit\Downloads\final_impact.csv')

# Drop 'Object Name' column as it is not needed for prediction
df.drop(columns=['Object Name'], inplace=True)

# Handle missing or special characters (like '(*)') by replacing them with NaN
df.replace(r'\(.*\)', np.nan, regex=True, inplace=True)

# Convert numeric columns to appropriate types (if any are read as strings)
df = df.apply(pd.to_numeric, errors='ignore')

# Handle missing values by filling with column-specific methods
# For numerical columns, use the mean of the column
df.fillna(df.mean(), inplace=True)

# For categorical columns (if any), fill with the mode (most frequent value)
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# One-hot encode categorical columns if any
encoder = OneHotEncoder(drop='first', sparse_output=False)

df_encoded = pd.DataFrame(encoder.fit_transform(df[categorical_columns]))
df_encoded.columns = encoder.get_feature_names_out(categorical_columns)

# Drop the original categorical columns and concatenate the encoded columns
df = df.drop(columns=categorical_columns)
df = pd.concat([df, df_encoded], axis=1)

# Split into X (features) and y (target)
X = df.drop(columns='Possible Impacts')
y = df['Possible Impacts']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering to check possible impact groups
kmeans = KMeans(n_clusters=3)  # You can tune the number of clusters
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Plot clustering results
plt.figure(figsize=(8, 8))
sb.scatterplot(x=df['Asteroid Velocity'], y=df['Asteroid Magnitude'], hue=df['Cluster'], palette='viridis')
plt.title('Clustering of Asteroids Based on Velocity and Magnitude')
plt.show()

# Dimensionality reduction (Optional, if needed for large feature set)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)

# Models for impact prediction
rf = RandomForestRegressor()
lr = LinearRegression()
elasticNet = ElasticNet(alpha=0.05, l1_ratio=1)

# Train RandomForest
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Train LinearRegression
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Train ElasticNet
elasticNet.fit(X_train, y_train)
y_pred_en = elasticNet.predict(X_test)

# Model Evaluation
print("RandomForest MAE:", mean_absolute_error(y_test, y_pred_rf))
print("RandomForest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("RandomForest R2:", r2_score(y_test, y_pred_rf))

print("LinearRegression MAE:", mean_absolute_error(y_test, y_pred_lr))
print("LinearRegression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))
print("LinearRegression R2:", r2_score(y_test, y_pred_lr))

print("ElasticNet MAE:", mean_absolute_error(y_test, y_pred_en))
print("ElasticNet RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_en)))
print("ElasticNet R2:", r2_score(y_test, y_pred_en))

# Scatter plot to visualize predictions
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_rf, y_test, label='RandomForest', alpha=0.7)
plt.scatter(y_pred_lr, y_test, label='LinearRegression', alpha=0.7)
plt.scatter(y_pred_en, y_test, label='ElasticNet', alpha=0.7)
plt.plot([0, max(y_test)], [0, max(y_test)], color='black', linestyle='--')
plt.legend()
plt.title('Predicted vs Actual Impacts')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

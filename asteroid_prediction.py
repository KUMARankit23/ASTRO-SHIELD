import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import pickle

class AsteroidPrediction:
    def __init__(self, data_path):
        # Load the dataset
        self.df = pd.read_csv(data_path)
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.rf = RandomForestClassifier()
        self.dt = DecisionTreeClassifier()

    def preprocess_data(self):
        # Display the columns in the dataset
        print("Columns in the dataset:")
        print(self.df.columns)

        # Check for missing values
        print("\nMissing values in the dataset:")
        print(self.df.isnull().sum())

        # Handle missing values: Fill numerical columns with mean and categorical columns with mode
        for column in self.df.columns:
            if self.df[column].dtype == 'object':  # Categorical columns
                self.df[column].fillna(self.df[column].mode()[0], inplace=True)
            else:  # Numerical columns
                self.df[column].fillna(self.df[column].mean(), inplace=True)

        # Check again for missing values after imputation
        print("\nMissing values after imputation:")
        print(self.df.isnull().sum())

        # Drop 'Object Name' as it's an identifier and won't be useful for the model
        self.df = self.df.drop(columns=['Object Name'])

        # One-Hot Encoding for 'Maximum Torino Scale'
        if 'Maximum Torino Scale' in self.df.columns:
            self.df = pd.get_dummies(self.df, columns=['Maximum Torino Scale'], drop_first=True)

        # Define X (features) and y (target variable)
        self.X = self.df.drop(columns=['Possible Impacts'])
        self.y = self.df['Possible Impacts']

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=42)

    def train_models(self):
        # Train the models (Random Forest and Decision Tree)
        self.rf.fit(self.X_train, self.y_train)
        self.dt.fit(self.X_train, self.y_train)

        # Save the Random Forest model to a pickle file
        with open('random_forest_model.pkl', 'wb') as file:
            pickle.dump(self.rf, file)
        print("\nRandom Forest model saved to 'random_forest_model.pkl'.")

    def evaluate_models(self):
        # Make predictions using both models
        y_pred_rf = self.rf.predict(self.X_test)
        y_pred_dt = self.dt.predict(self.X_test)

        # Evaluate and display accuracy of both models
        print("\nAccuracy of Random Forest Classifier:", accuracy_score(self.y_test, y_pred_rf))
        print("Accuracy of Decision Tree Classifier:", accuracy_score(self.y_test, y_pred_dt))

        # Plot confusion matrices for both models
        cm_rf = confusion_matrix(self.y_test, y_pred_rf)
        cm_dt = confusion_matrix(self.y_test, y_pred_dt)

        disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
        disp_rf.plot(cmap=plt.cm.Blues)
        plt.title("Random Forest Classifier Confusion Matrix")
        plt.show()

        disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt)
        disp_dt.plot(cmap=plt.cm.Blues)
        plt.title("Decision Tree Classifier Confusion Matrix")
        plt.show()

        # Conclusion based on the model performances
        if accuracy_score(self.y_test, y_pred_rf) > accuracy_score(self.y_test, y_pred_dt):
            print("Random Forest Classifier performed better.")
        else:
            print("Decision Tree Classifier performed better.")

def classify_asteroid(self, input_data):
    """
    Classify a single asteroid based on input data using the saved Random Forest model.
    """
    # Ensure the input data matches the trained model's feature set
    required_features = ['Period Start', 'Period End', 'Cumulative Impact Probability',
                         'Asteroid Velocity', 'Asteroid Magnitude', 'Asteroid Diameter (km)',
                         'Cumulative Palermo Scale', 'Maximum Palermo Scale',
                         'Maximum Torino Scale_1']

    # Convert input data to match the order of the required features
    data = [input_data[feature] for feature in required_features]

    # Load the saved model
    with open('random_forest_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    # Predict the class
    prediction = loaded_model.predict([data])[0]
    prediction = int(prediction)  # Convert numpy.int64 to Python int
    print(f"\nPrediction: {prediction}")
    return prediction




# Example usage of the class
if __name__ == "__main__":
    asteroid_predictor = AsteroidPrediction('C:/Users/ankit/Downloads/final_impact.csv')

    # Preprocess the data
    asteroid_predictor.preprocess_data()

    # Train the models
    asteroid_predictor.train_models()

    # Evaluate the models
    asteroid_predictor.evaluate_models()

    # Example input for classification
    input_data_example = {
        'Period Start': 2055,
        'Period End': 2069,
        'Cumulative Impact Probability': 0.1,
        'Asteroid Velocity': 25.5,
        'Asteroid Magnitude': 22.3,
        'Asteroid Diameter (km)': 1.2,
        'Cumulative Palermo Scale': -3.5,
        'Maximum Palermo Scale': -3.8,
        'Maximum Torino Scale_1': 0,
        'Maximum Torino Scale_2': 1  # Example for one-hot encoded columns
    }

    # Classify the input data
    prediction = asteroid_predictor.classify_asteroid(input_data_example)
    print("\nPredicted Class:", prediction)

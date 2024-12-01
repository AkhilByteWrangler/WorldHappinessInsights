# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np

def train_non_dl_model(file_path, target_column, exclude_columns):
    """
    Trains a non-deep learning model (Random Forest) on the dataset with hyperparameter tuning.

    Args:
        file_path (str): Path to the feature-engineered dataset.
        target_column (str): Name of the target column.
        exclude_columns (list): List of columns to exclude from the features.

    Returns:
        dict: Best model and its evaluation metrics.
    """
    # Load the dataset
    print("Loading feature-engineered data...")
    df = pd.read_csv(file_path)

    # Exclude specified columns and separate features and target
    X = df.drop(columns=exclude_columns)
    y = pd.read_csv("data/processed/y.csv")[target_column]

    # Handle categorical features 
    for col in X.select_dtypes(include=['object']).columns:
        print(f"Encoding categorical column: {col}")
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Split the data into training and testing sets
    print("Splitting the data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the Random Forest model
    rf = RandomForestRegressor(random_state=42)

    # Hyperparameter distribution for tuning
    param_distributions = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }

    # Perform RandomizedSearchCV for hyperparameter tuning
    print("Starting hyperparameter tuning...")
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=50,  # Number of parameter combinations to try
        scoring='neg_mean_squared_error',
        cv=3,
        random_state=42,
        verbose=2,
        n_jobs=-1  # Use all available cores
    )
    random_search.fit(X_train, y_train)

    # Get the best model and its metrics
    best_model = random_search.best_estimator_
    print(f"Best parameters: {random_search.best_params_}")

    # Evaluate the model on the test set
    print("Evaluating the best model...")
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test Mean Squared Error (MSE): {mse}")

    return {
        'model': best_model,
        'mse': mse,
        'best_params': random_search.best_params_
    }

# Script execution
if __name__ == "__main__":
    file_path = "data/processed/X_engineered.csv"
    target_column = "Life Ladder"
    exclude_columns = ['Country name', 'year']  # Exclude non-numeric columns

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Please check the path.")
    else:
        print("Training non-deep learning model...")
        results = train_non_dl_model(file_path, target_column, exclude_columns)

        # Save the best model 
        import joblib
        os.makedirs("models", exist_ok=True)
        joblib.dump(results['model'], "models/random_forest_model.pkl")
        print("Random Forest model saved to 'models/random_forest_model.pkl'.")

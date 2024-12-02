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
    """
    # Load the feature-engineered dataset.
    print("Loading feature-engineered data...")
    df = pd.read_csv(file_path)

    # Exclude unnecessary columns and separate features (X) from the target (y).
    X = df.drop(columns=exclude_columns)
    y = pd.read_csv("data/processed/y.csv")[target_column]

    # Encode categorical features, ensuring they’re represented as numbers.
    # This is necessary because models like our dear Random Forest can’t directly work with strings or object types.
    for col in X.select_dtypes(include=['object']).columns:
        print(f"Encoding categorical column: {col}")
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # Split the dataset into training and testing sets.
    print("Splitting the data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the Random Forest Regressor.
    # Random Forest is a powerful ensemble learning method that combines multiple decision trees.
    # Each tree makes a prediction, and the forest aggregates them for a final, robust result.
    # Fun Fact: Unlike XGBoost (Which conveniently just input NaNs for missing values), Random Forest can actually handle missing values implictly! 
    # How? Decision trees in Random Forest can split on available values, bypassing NaNs during the training phase. 
    # This makes it naturally resilient to datasets with gaps.
    
    # Another Fun Fact: Random Forest is non-parametric, meaning it doesn’t make assumptions about the underlying data distribution.
    # So whether your data relationships are linear, non-linear, or completely quirky, Random Forest can handle it.

    # Bonus Fun Fact: Random Forest is immune to multicollinearity.
    # Even if some features are highly correlated, the trees in the forest can independently decide how to split, reducing bias.

    rf = RandomForestRegressor(random_state=42)

    # Why Random Forest? 
    # - It’s resilient to overfitting due to averaging across many trees. 
    # - Handles non-linear relationships effortlessly.
    # - Works well with tabular data, making it ideal for our structured dataset.
    # - Provides feature importance, helping us identify key drivers of happiness (like GDP or social support).

    # Define hyperparameter distributions for RandomizedSearchCV.
    param_distributions = {
        'n_estimators': [50, 100, 200, 300],  # Number of trees in the forest
        'max_depth': [10, 20, 30, None],     # Depth of each tree
        'min_samples_split': [2, 5, 10, 15], # Minimum samples needed to split a node
        'min_samples_leaf': [1, 2, 4],       # Minimum samples required at a leaf node
        'max_features': ['sqrt', 'log2', None]  # Features to consider when splitting
    }

    # Perform hyperparameter tuning using RandomizedSearchCV.
    # RandomizedSearchCV is a smart way to find the best hyperparameters for our model.
    
    # Fun Fact: RandomizedSearchCV is faster and more efficient than Grid Search, especially when we have many hyperparameters or large datasets.
    # Why? Grid Search tries every possible combination, which can grow exponentially with more parameters, turning into an AIPI class (time sink).
    # RandomizedSearchCV, on the other hand, focuses on randomly chosen samples, giving us a good balance between performance and computational cost.
    
    # Why use RandomizedSearchCV for Random Forest?
    # - Random Forest has several hyperparameters, and the possible combinations can quickly become overwhelming.
    # - RandomizedSearchCV ensures we explore the parameter space effectively without spending days (hell) waiting for results.

    print("Starting hyperparameter tuning...")
    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=50,  # Number of parameter combinations to try
        scoring='neg_mean_squared_error',  # Optimize for Mean Squared Error
        cv=3,  # 3-fold cross-validation for reliable performance estimates
        random_state=42,
        verbose=2,
        n_jobs=-1 
    )
    random_search.fit(X_train, y_train)

    # Retrieve the best model
    best_model = random_search.best_estimator_
    print(f"Best parameters: {random_search.best_params_}")

    # Evaluate the best model on the test set.
    print("Evaluating the best model...")
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test Mean Squared Error (MSE): {mse}")

    # Return the trained model and metrics.
    return {
        'model': best_model,
        'mse': mse,
        'best_params': random_search.best_params_
    }

if __name__ == "__main__":
    # Define paths
    file_path = "data/processed/X_engineered.csv"
    target_column = "Life Ladder"
    exclude_columns = ['Country name', 'year']  # Exclude non-predictive columns

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Please check the path.")
    else:
        print("Training non-deep learning model...")
        results = train_non_dl_model(file_path, target_column, exclude_columns)

        # Save the trained model for future use.
        import joblib
        os.makedirs("models", exist_ok=True)
        joblib.dump(results['model'], "models/random_forest_model.pkl")
        print("Random Forest model saved to 'models/random_forest_model.pkl'.")

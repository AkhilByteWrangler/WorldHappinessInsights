import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# Define the TCN model
def build_tcn(input_shape, filters=32, kernel_size=3, dilation_rate=2, dropout_rate=0.2, learning_rate=0.001):
    """
    Builds a Temporal Convolutional Network (TCN) model.

    Args:
        input_shape (tuple): Shape of the input data (timesteps, features).
        filters (int): Number of filters for the Conv1D layers.
        kernel_size (int): Size of the convolution kernel.
        dilation_rate (int): Dilation rate for the convolution.
        dropout_rate (float): Dropout rate for regularization.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        tf.keras.Model: Compiled TCN model.
    """
    model = Sequential([
        Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate,
               activation='relu', padding='causal', input_shape=input_shape),
        Dropout(dropout_rate),
        Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate,
               activation='relu', padding='causal'),
        Dropout(dropout_rate),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='linear')  # Regression output
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

# Custom Keras wrapper for compatibility with RandomizedSearchCV
class TCNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_shape, filters=32, kernel_size=3, dilation_rate=2, dropout_rate=0.2, learning_rate=0.001, epochs=50, batch_size=32):
        self.input_shape = input_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def fit(self, X, y):
        self.model = build_tcn(input_shape=self.input_shape, filters=self.filters, kernel_size=self.kernel_size,
                               dilation_rate=self.dilation_rate, dropout_rate=self.dropout_rate, learning_rate=self.learning_rate)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, X):
        return self.model.predict(X).flatten()

    def score(self, X, y):
        y_pred = self.predict(X)
        return -mean_squared_error(y, y_pred)  # Negative MSE for compatibility with scikit-learn scoring

# Prepare data for TCN
def prepare_tcn_data(X, y):
    """
    Prepares data for TCN by reshaping features to include a time dimension.

    Args:
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target data.

    Returns:
        tuple: Reshaped X and y.
    """
    X_reshaped = X.values.reshape(X.shape[0], X.shape[1], 1)  # Add time dimension
    y = y.values
    return X_reshaped, y

# Train TCN model
def train_tcn_model(file_path, target_column, exclude_columns):
    """
    Trains a TCN model using RandomizedSearchCV.

    Args:
        file_path (str): Path to the feature-engineered dataset.
        target_column (str): Name of the target column.
        exclude_columns (list): List of columns to exclude from the features.

    Returns:
        dict: Best model and its evaluation metrics.
    """
    # Load dataset
    print("Loading feature-engineered data...")
    df = pd.read_csv(file_path)

    # Exclude specified columns
    X = df.drop(columns=exclude_columns)
    y = pd.read_csv("data/processed/y.csv")[target_column]

    # Prepare data for TCN
    print("Preparing data for TCN...")
    X, y = prepare_tcn_data(X, y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the input shape
    input_shape = (X_train.shape[1], X_train.shape[2])

    # Define the hyperparameter grid
    param_distributions = {
        'filters': [16, 32, 64],
        'kernel_size': [2, 3, 5],
        'dilation_rate': [1, 2, 4],
        'dropout_rate': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.01],
        'epochs': [50],
        'batch_size': [16, 32]
    }

    # RandomizedSearchCV for hyperparameter tuning
    print("Starting hyperparameter tuning...")
    tcn = TCNRegressor(input_shape=input_shape)
    random_search = RandomizedSearchCV(estimator=tcn, param_distributions=param_distributions,
                                       n_iter=10, scoring='neg_mean_squared_error', cv=3, random_state=42)
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
    exclude_columns = ['Country name', 'year']

    print("Training TCN model...")
    results = train_tcn_model(file_path, target_column, exclude_columns)

    # Save the best model
    os.makedirs("models", exist_ok=True)
    results['model'].model.save("models/tcn_model.h5")
    print("TCN model saved to 'models/tcn_model.h5'.")

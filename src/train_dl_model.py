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
    """
    # Imagine TCNs as time-traveling AIPI Sourcing Data students for your data.
    # They organize past events (like GDP trends and life expectancy changes), 
    # highlight important patterns (via convolutional layers), and connect the dots over years (dilated filters).
    
    # But here’s the twist—they never cheat :)! Thanks to causal padding, they refuse to peek into the future 
    # (even if it’s super tempting). No spoilers for what’s coming next (we don't like movie recaps!).

    # They also use shortcuts to revisit earlier insights, like flipping back to earlier chapters of a book 
    # when they need context. No detail gets lost in their timeline journey!

    # Why do we love TCNs? For our happiness dataset, they’re like detectives with a strict code of ethics:
    # only learning from the past to predict the future while uncovering long-term, time-sensitive trends. 
    # They’re perfect for capturing the sequential nature of our data.

    model = Sequential([
        Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate,
               activation='relu', padding='causal', input_shape=input_shape),  # First convolutional layer
        Dropout(dropout_rate),  # Regularization to prevent overfitting
        Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate,
               activation='relu', padding='causal'),  # Second convolutional layer for deeper patterns
        Dropout(dropout_rate),
        Flatten(),  
        Dense(64, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='linear')  
    ])
    # Adam optimizer is chosen for its adaptive learning rate and efficiency in training deep networks.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model

# Custom Keras wrapper
class TCNRegressor(BaseEstimator, RegressorMixin):
    """
    Wrapper to make TCN compatible with scikit-learn's RandomizedSearchCV.
    """
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
        # Fit the model using the TCN architecture
        self.model = build_tcn(input_shape=self.input_shape, filters=self.filters, kernel_size=self.kernel_size,
                               dilation_rate=self.dilation_rate, dropout_rate=self.dropout_rate, learning_rate=self.learning_rate)
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, X):
        return self.model.predict(X).flatten()

    def score(self, X, y):
        y_pred = self.predict(X)
        return -mean_squared_error(y, y_pred)

# Prepare data for TCN
def prepare_tcn_data(X, y):
    """
    Prepares data for TCN by reshaping features to include a time dimension.
    """
    # Adding a time dimension because TCNs expect data with a sequential format
    X_reshaped = X.values.reshape(X.shape[0], X.shape[1], 1) 
    y = y.values 
    return X_reshaped, y

# Train TCN model
def train_tcn_model(file_path, target_column, exclude_columns):
    """
    Trains a TCN model using RandomizedSearchCV.
    """
    # Load the dataset
    print("Loading feature-engineered data...")
    df = pd.read_csv(file_path)

    # Exclude columns like 'Country name' and 'year' which don't need direct encoding.
    X = df.drop(columns=exclude_columns)
    y = pd.read_csv("data/processed/y.csv")[target_column]

    # Prepare data by adding a time dimension
    print("Preparing data for TCN...")
    X, y = prepare_tcn_data(X, y)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_shape = (X_train.shape[1], X_train.shape[2])

    # Define hyperparameter grid for tuning
    param_distributions = {
        'filters': [16, 32, 64],  # Varying depth for pattern detection
        'kernel_size': [2, 3, 5],  # Wider kernels capture larger temporal patterns
        'dilation_rate': [1, 2, 4],  
        'dropout_rate': [0.1, 0.2, 0.3],  # Regularization to prevent overfitting
        'learning_rate': [0.001, 0.01],  
        'epochs': [50],  
        'batch_size': [16, 32]  
    }

    # Hyperparameter tuning with RandomizedSearchCV again
    print("Starting hyperparameter tuning...")
    tcn = TCNRegressor(input_shape=input_shape)
    random_search = RandomizedSearchCV(estimator=tcn, param_distributions=param_distributions,
                                       n_iter=10, scoring='neg_mean_squared_error', cv=3, random_state=42)
    random_search.fit(X_train, y_train)

    # Retrieve the best model and parameters
    best_model = random_search.best_estimator_
    print(f"Best parameters: {random_search.best_params_}")

    # Evaluate the model on the test set
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test Mean Squared Error (MSE): {mse}")

    return {
        'model': best_model,
        'mse': mse,
        'best_params': random_search.best_params_
    }

if __name__ == "__main__":
    file_path = "data/processed/X_engineered.csv"
    target_column = "Life Ladder"
    exclude_columns = ['Country name', 'year']

    print("Training TCN model...")
    results = train_tcn_model(file_path, target_column, exclude_columns)

    # Save the trained model
    os.makedirs("models", exist_ok=True)
    results['model'].model.save("models/tcn_model.h5")
    print("TCN model saved to 'models/tcn_model.h5'.")

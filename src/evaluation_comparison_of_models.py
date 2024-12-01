import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             explained_variance_score, max_error, mean_absolute_percentage_error)
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import joblib
import os


def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluates a model using multiple metrics and returns the results.

    Args:
        y_true (array): True target values.
        y_pred (array): Predicted target values.
        model_name (str): Name of the model being evaluated.

    Returns:
        dict: Evaluation metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    max_err = max_error(y_true, y_pred)
    
    metrics = {
        'model': model_name,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'explained_variance': evs,
        'mape': mape,
        'max_error': max_err
    }
    
    print(f"\n{model_name} Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key.replace('_', ' ').title()}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    return metrics


def save_plot(fig, folder, filename):
    """
    Saves a plot to the specified folder with the given filename.

    Args:
        fig (Figure): Matplotlib figure object.
        folder (str): Directory to save the plot.
        filename (str): Name of the file (with extension).
    """
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, filename))
    plt.close(fig)


def plot_calibration(y_true, y_pred, model_name, folder):
    """
    Plots predicted vs. actual values to assess calibration and saves the plot.

    Args:
        y_true (array): True target values.
        y_pred (array): Predicted target values.
        model_name (str): Name of the model being evaluated.
        folder (str): Directory to save the plot.
    """
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, label='Predictions')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', label='Perfect Calibration')
    plt.title(f"Calibration Plot: {model_name}")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    save_plot(fig, folder, f"{model_name}_calibration_plot.png")


def plot_residuals(y_true, y_pred, model_name, folder):
    """
    Plots residuals for a model to visualize prediction errors and saves the plot.

    Args:
        y_true (array): True target values.
        y_pred (array): Predicted target values.
        model_name (str): Name of the model being evaluated.
        folder (str): Directory to save the plot.
    """
    residuals = y_true - y_pred
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, label='Residuals')
    plt.axhline(y=0, color='red', linestyle='--', label='Zero Error Line')
    plt.title(f"Residual Plot: {model_name}")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.legend()
    save_plot(fig, folder, f"{model_name}_residual_plot.png")


def plot_error_distribution(y_true, y_pred, model_name, folder):
    """
    Plots the distribution of residuals (errors) and saves the plot.

    Args:
        y_true (array): True target values.
        y_pred (array): Predicted target values.
        model_name (str): Name of the model being evaluated.
        folder (str): Directory to save the plot.
    """
    residuals = y_true - y_pred
    fig = plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30, alpha=0.7, color='blue', edgecolor='k')
    plt.title(f"Error Distribution: {model_name}")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    save_plot(fig, folder, f"{model_name}_error_distribution.png")


def feature_importance_plot(model, feature_names, folder):
    """
    Plots the feature importance for tree-based models and saves the plot.

    Args:
        model: Trained tree-based model (e.g., Random Forest).
        feature_names (list): List of feature names.
        folder (str): Directory to save the plot.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        fig = plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.title("Feature Importance")
        plt.xlabel("Feature")
        plt.ylabel("Importance Score")
        plt.tight_layout()
        save_plot(fig, folder, "Random_Forest_Feature_Importance.png")
    else:
        print("Feature importance is not available for this model.")


# Main evaluation script
if __name__ == "__main__":
    # Paths
    folder = "plots"
    os.makedirs(folder, exist_ok=True)

    # Load test data
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    # Load Random Forest Model
    print("\nEvaluating Random Forest Model...")
    rf_model = joblib.load("models/random_forest_model.pkl")
    rf_pred = rf_model.predict(X_test)
    rf_metrics = evaluate_model(y_test, rf_pred, model_name="Random Forest")
    plot_calibration(y_test, rf_pred, "Random Forest", folder)
    plot_residuals(y_test, rf_pred, "Random Forest", folder)
    plot_error_distribution(y_test, rf_pred, "Random Forest", folder)
    feature_importance_plot(rf_model, X_test.columns, folder)

    # Load TCN Model
    print("\nEvaluating TCN Model...")
    tcn_model = load_model("models/tcn_model.h5", compile=False)
    tcn_model.compile(loss=MeanSquaredError(), optimizer="adam", metrics=["mae"])
    X_test_tcn = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
    tcn_pred = tcn_model.predict(X_test_tcn).flatten()
    tcn_metrics = evaluate_model(y_test, tcn_pred, model_name="TCN")
    plot_calibration(y_test, tcn_pred, "TCN", folder)
    plot_residuals(y_test, tcn_pred, "TCN", folder)
    plot_error_distribution(y_test, tcn_pred, "TCN", folder)

    # Compare Metrics
    print("\nComparison of Model Metrics:")
    comparison_df = pd.DataFrame([rf_metrics, tcn_metrics])
    print(comparison_df)

    # Save metrics to file
    comparison_df.to_csv("models/model_comparison_metrics.csv", index=False)
    print("\nModel comparison metrics saved to 'models/model_comparison_metrics.csv'.")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from tensorflow.keras.models import load_model
import joblib
from lime import lime_tabular
from mpl_toolkits.mplot3d import Axes3D
import time
import os
import warnings
warnings.filterwarnings("ignore")

# Set up page config
st.set_page_config(page_title="World Happiness Dashboard 🌍", layout="wide")

# Helper functions
@st.cache_data
def load_data():
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")["Life Ladder"]
    y_test = pd.read_csv("data/processed/y_test.csv")["Life Ladder"]
    return X_train, X_test, y_train, y_test

@st.cache_resource
def load_models():
    rf_model = joblib.load("models/random_forest_model.pkl")
    tcn_model = load_model("models/tcn_model.h5", compile=False)
    tcn_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return rf_model, tcn_model

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2

def plot_3d_feature_importance(features, importances):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    xs = np.arange(len(features))
    ys = importances
    zs = np.zeros_like(xs)
    ax.bar(xs, ys, zs, zdir='y', alpha=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.set_ylabel("Features")
    ax.set_zlabel("Importance")
    ax.set_title("3D Feature Importance")
    return fig

def plot_animated_graph(y_test, rf_pred, tcn_pred):
    fig, ax = plt.subplots()
    x = range(len(y_test))
    line1, = ax.plot([], [], label="Random Forest", color="blue")
    line2, = ax.plot([], [], label="TCN", color="green")
    ax.set_xlim(0, len(y_test))
    ax.set_ylim(min(y_test) - 1, max(y_test) + 1)
    ax.legend()

    for i in range(len(x)):
        line1.set_data(x[:i], rf_pred[:i])
        line2.set_data(x[:i], tcn_pred[:i])
        plt.pause(0.05)
    return fig

def explain_model_with_lime(model, X, y, explainer_type):
    st.markdown(f"### Understanding the {explainer_type} Model with LIME")
    explainer = lime_tabular.LimeTabularExplainer(
        X.values, feature_names=X.columns, class_names=["Life Ladder"], mode="regression"
    )
    i = st.slider("Select an instance to explain", 0, len(X) - 1, 0)
    exp = explainer.explain_instance(X.values[i], model.predict)
    st.write(f"Explanation for instance {i}:")
    st.pyplot(exp.as_pyplot_figure())
    return exp

# Load everything
X_train, X_test, y_train, y_test = load_data()
rf_model, tcn_model = load_models()

# Navigation
st.sidebar.title("Navigate the Dashboard")
page = st.sidebar.radio("Go to:", ["Introduction", "Data Visualization", "Model Training", "Model Evaluation", "Interactive Insights"])

# Pages
if page == "Introduction":
    st.title("Welcome to the World Happiness Dashboard! 🌍😊")
    st.write("""
    This dashboard is all about exploring happiness trends across the world.
    Whether you're here to see some cool 3D graphs, animated predictions, or just get insights, we’ve got you covered. 
    Let’s dive in and see what makes the world smile!
    """)

elif page == "Data Visualization":
    st.title("Visualizing Happiness Data 📊")
    st.write("Let’s get a feel for our data before diving into models.")

    if st.checkbox("Show Feature Distribution"):
        feature = st.selectbox("Select a feature to visualize", X_train.columns)
        fig, ax = plt.subplots()
        ax.hist(X_train[feature], bins=30, alpha=0.7, color="blue")
        ax.set_title(f"Distribution of {feature}")
        st.pyplot(fig)

    if st.checkbox("3D Feature Importance (Random Forest)"):
        rf_importances = rf_model.feature_importances_
        fig = plot_3d_feature_importance(X_train.columns, rf_importances)
        st.pyplot(fig)

elif page == "Model Training":
    st.title("Model Training 🛠️")
    st.write("Let’s explore the training process for our models.")

    model_type = st.radio("Which model are you interested in?", ["Random Forest", "TCN"])
    if model_type == "Random Forest":
        rf_pred = rf_model.predict(X_test)
        mse, mae, r2 = calculate_metrics(y_test, rf_pred)
        st.success(f"Random Forest Metrics: MSE = {mse:.3f}, MAE = {mae:.3f}, R² = {r2:.3f}")
    else:
        X_test_tcn = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
        tcn_pred = tcn_model.predict(X_test_tcn).flatten()
        mse, mae, r2 = calculate_metrics(y_test, tcn_pred)
        st.success(f"TCN Metrics: MSE = {mse:.3f}, MAE = {mae:.3f}, R² = {r2:.3f}")

elif page == "Model Evaluation":
    st.title("Evaluating Our Models 📈")
    st.write("Time to compare and understand how our models performed!")

    # Generate predictions
    rf_pred = rf_model.predict(X_test)
    X_test_tcn = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
    tcn_pred = tcn_model.predict(X_test_tcn).flatten()

    # Display evaluation metrics
    metrics_df = pd.DataFrame({
        "Model": ["Random Forest", "TCN"],
        "MSE": [mean_squared_error(y_test, rf_pred), mean_squared_error(y_test, tcn_pred)],
        "MAE": [mean_absolute_error(y_test, rf_pred), mean_absolute_error(y_test, tcn_pred)],
        "R²": [r2_score(y_test, rf_pred), r2_score(y_test, tcn_pred)],
    })
    st.table(metrics_df)

    animated_plot_path = "plots/animated_prediction_graph.jpg"

    if st.checkbox("Show Prediction Graph"):
        if not os.path.exists(animated_plot_path):
            os.makedirs("plots", exist_ok=True) 

            fig = plot_animated_graph(y_test, rf_pred, tcn_pred)

            fig.savefig(animated_plot_path)

        st.write("### Prediction Graph")
        st.image(animated_plot_path, caption="Prediction Comparison (Random Forest vs. TCN)")


elif page == "Interactive Insights":
    st.title("Interactive Model Insights 🔍")
    st.write("Let’s dive deeper with LIME to see why our models make certain predictions.")

    lime_model = st.radio("Which model do you want to explain?", ["Random Forest", "TCN"])
    if lime_model == "Random Forest":
        explain_model_with_lime(rf_model, X_test, y_test, "Random Forest")
    else:
        st.write("LIME currently supports tabular data models like Random Forest. For TCNs, custom explainers are required!")

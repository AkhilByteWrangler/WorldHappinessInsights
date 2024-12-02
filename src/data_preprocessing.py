import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os

def main(file_path, target_column):
    """
    This function is where 
    We'll load, clean, and transform our dataset to get it ready for modeling. 
    No more messy data.
    """

    # (1) Load the data
    print(" (1) Load the data (fingers crossed it’s in good shape)")
    try:
        # Reading the data from an Excel file because that’s what we’ve got.
        df = pd.read_excel(file_path)
        print(f"Data loaded successfully! Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    except Exception as e:
        print(f"Uh-oh! Something went wrong while loading the data. Error: {e}")
        raise

    # (2) Clean the data
    print("\n (2) Clean the data (bye-bye duplicates and NaNs)")
    # Remove duplicate rows because they don’t contribute anything new to the model.
    # Duplicates can inflate the importance of certain trends, so we eliminate them.
    df = df.drop_duplicates()

    # Handle missing values with mean imputation. It’s a safe choice because it preserves the overall distribution
    # of the data while filling gaps. Essential for numeric features!
    imputer = SimpleImputer(strategy='mean')
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = imputer.fit_transform(df[[col]])

    print("Data cleaned! This dataset is looking much better already.")

    # (3) Preprocess the features
    print("\n (3) Preprocess the features (time to give our data some love)")
    # Separate the features (X) from the target variable (y).
    # The target column is what we’re trying to predict—everything else is input for the model.
    columns_to_exclude = ['Country name', 'year']  # These columns are contextually important (Name and Temporal) but don’t need scaling.
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Identify numeric columns that need scaling or transformations.
    # We exclude anything categorical or context-related (like country and year).
    numeric_features = [
        col for col in X.select_dtypes(include=['float64', 'int64']).columns
        if col not in columns_to_exclude
    ]

    # Handle outliers using the Inter Quartile Range (IQR) method to clip extreme values. 
    # Outliers can distort the model’s understanding of patterns, so we bring them within a reasonable range.
    print("Handling outliers...")
    for col in numeric_features:
        q1 = X[col].quantile(0.25)
        q3 = X[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        X[col] = X[col].clip(lower=lower, upper=upper)

    # Numerical features are scaled to ensure no single feature dominates the model because of its range (Distance Based Models would be HAPPY!).
    # StandardScaler centers everything (mean=0, std=1) to give each feature equal weight.
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Combine transformations into a pipeline.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)  
        ],
        remainder='passthrough'  # Pass Country name and year as-is—they’re still useful for context.
    )

    # Apply transformations. 
    X_transformed = preprocessor.fit_transform(X)

    transformed_numeric_features = preprocessor.named_transformers_['num'].get_feature_names_out(numeric_features)
    passthrough_columns = [col for col in columns_to_exclude if col in X.columns]
    column_names = list(transformed_numeric_features) + passthrough_columns

    X_df = pd.DataFrame(X_transformed, columns=column_names)

    # Scale the target variable (y). Although a lot of models don’t require it, scaling helps neural networks converge better.
    print("Scaling the target variable...")
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    print("\nPreprocessing complete! Our data is now shiny and ready to roll.")
    return X_df, y, column_names

if __name__ == "__main__":
    file_path = "data/rawdata.xls"
    target_column = "Life Ladder"

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Please double-check the path.")
    else:
        # Execute the main preprocessing workflow.
        X, y, column_names = main(file_path, target_column)

        print("\nSaving preprocessed X and y...")
        os.makedirs("data/processed", exist_ok=True)

        # Save features (X), target (y), and column names so we can reuse them in modeling.
        X.to_csv("data/processed/X.csv", index=False)
        pd.DataFrame(y, columns=[target_column]).to_csv("data/processed/y.csv", index=False)
        pd.DataFrame(column_names, columns=["Feature Names"]).to_csv("data/processed/feature_names.csv", index=False)

        print("Preprocessed X, y, and feature names saved in 'data/processed/' directory. Time to model!")

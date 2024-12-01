# Import libraries to wrangle, scale, and preprocess our data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os

def main(file_path, target_column):
    """
    This function is where the magic happens.
    We'll load, clean, and transform our dataset to get it ready for modeling. 
    No more messy data—just pristine, model-ready glory.

    Args:
        file_path (str): Path to the dataset (Excel file).
        target_column (str): The column we're trying to predict.

    Returns:
        tuple: Preprocessed features (X), target (y), and transformed column names.
    """

    # Load the data
    print(" (1) Load the data (fingers crossed it’s in good shape)")
    try:
        # Let's load the Excel file and take a peek
        df = pd.read_excel(file_path)
        print(f"Data loaded successfully! Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    except Exception as e:
        print(f"Uh-oh! Something went wrong while loading the data. Error: {e}")
        raise

    # Clean the data
    print("\n (2) Clean the data (bye-bye duplicates and NaNs)")
    # First, let's drop any duplicates—because who needs fakes?
    df = df.drop_duplicates()

    # Missing values? We'll handle them with mean imputation (simple and effective).
    imputer = SimpleImputer(strategy='mean')
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = imputer.fit_transform(df[[col]])

    print("Data cleaned! This dataset is looking much better already.")

    # Preprocess the features
    print("\n (3) Preprocess the features (time to give our data some love)")
    # Separate the features from the target column
    columns_to_exclude = ['Country name', 'year']  # No touchy these columns
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Identify numerical columns that actually need scaling (we’re skipping the ones we don’t touch)
    numeric_features = [
        col for col in X.select_dtypes(include=['float64', 'int64']).columns
        if col not in columns_to_exclude
    ]

    # Outlier management: Let’s keep things within a reasonable range (using IQR method).
    print("Handling outliers...")
    for col in numeric_features:
        q1 = X[col].quantile(0.25)
        q3 = X[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        X[col] = X[col].clip(lower=lower, upper=upper)

    # Time to scale numerical features so no column hogs the spotlight.
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Combine everything into a mega-preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)  # Scale the numeric features
        ],
        remainder='passthrough'  # Pass through Country name and year as they are
    )

    # Apply our transformations (here’s where the data glow-up happens)
    X_transformed = preprocessor.fit_transform(X)

    # Merge back column names: scaled columns + untouched columns
    transformed_numeric_features = preprocessor.named_transformers_['num'].get_feature_names_out(numeric_features)
    passthrough_columns = [col for col in columns_to_exclude if col in X.columns]
    column_names = list(transformed_numeric_features) + passthrough_columns

    # Convert to a DataFrame so we can save it later with proper headers
    X_df = pd.DataFrame(X_transformed, columns=column_names)

    # Scale the target 
    y_scaler = None
    print("Scaling the target variable...")
    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    print("\nPreprocessing complete! Our data is now shiny and ready to roll.")
    return X_df, y, column_names

# Only run the script when we call it directly
if __name__ == "__main__":
    file_path = "data/rawdata.xls"
    target_column = "Life Ladder"

    # Let’s make sure the file exists before diving in
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Please double-check the path.")
    else:
        X, y, column_names = main(file_path, target_column)

        print("\nSaving preprocessed X and y...")
        os.makedirs("data/processed", exist_ok=True)

        # Save our preprocessed features and target variable
        X.to_csv("data/processed/X.csv", index=False)
        pd.DataFrame(y, columns=[target_column]).to_csv("data/processed/y.csv", index=False)

        # Save column names for reference 
        pd.DataFrame(column_names, columns=["Feature Names"]).to_csv("data/processed/feature_names.csv", index=False)

        print("Preprocessed X, y, and feature names saved in 'data/processed/' directory. Time to model!")

import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_and_save_processed_data(X_file, y_file, target_column, exclude_columns=['Country name', 'year'], test_size=0.2, random_state=42):
    """
    Splits feature-engineered data into training and testing sets and saves the splits.

    Args:
        X_file (str): Path to the processed features file (CSV).
        y_file (str): Path to the processed target file (CSV).
        target_column (str): Name of the target column.
        exclude_columns (list): List of columns to exclude from features.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.
    """
    print("Loading feature-engineered data...")
    try:
        # Load the feature-engineered data
        X = pd.read_csv(X_file)
        y = pd.read_csv(y_file)[target_column]

        # Exclude specified columns
        if exclude_columns:
            print(f"Excluding columns: {exclude_columns}")
            X = X.drop(columns=exclude_columns)

        # Split the data
        print(f"Splitting the data (test_size={test_size}, random_state={random_state})...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        output_dir = "data/processed"
        os.makedirs(output_dir, exist_ok=True)

        # Save datasets
        print("Saving the datasets...")
        X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
        pd.DataFrame(y_train, columns=[target_column]).to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
        pd.DataFrame(y_test, columns=[target_column]).to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

        print(f"Datasets saved in '{output_dir}'!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Paths to the feature-engineered and target files
    X_file = "data/processed/X_engineered.csv" 
    y_file = "data/processed/y.csv"          
    target_column = "Life Ladder"             
    exclude_columns = ['Country name', 'year'] 

    # Run the splitting and saving process
    print("Starting the splitting and saving process...")
    split_and_save_processed_data(X_file, y_file, target_column, exclude_columns)

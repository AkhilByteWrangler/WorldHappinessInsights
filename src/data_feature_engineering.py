import pandas as pd

def feature_engineering(df):
    """
    Performs feature engineering on the dataset to create logically derived and model-relevant features.
    
    Args:
        df (pd.DataFrame): The original dataset.

    Returns:
        pd.DataFrame: The dataset with additional engineered features.
    """
    # Let's make a copy to keep the original data safe—our golden rule.
    df = df.copy()

    # Feature 1: Economic and Social Index
    # Hypothesis: Happiness is strongly tied to economic strength (GDP) and social support.
    # We're weighting GDP higher than social support, as it has a more direct impact on life quality.
    print("Adding Economic and Social Index...")
    df['Economic_Social_Index'] = (
        df['Log GDP per capita'] * 0.7 + df['Social support'] * 0.3
    )

    # Feature 2: Emotional Stability Score
    # Hypothesis: People with more positive emotions and fewer negative ones are generally happier.
    # This feature captures the balance between the two.
    print("Adding Emotional Stability Score...")
    df['Emotional_Stability'] = df['Positive affect'] - df['Negative affect']

    # Feature 3: Freedom-Corruption Ratio
    # Hypothesis: A balance of personal freedom and lower corruption correlates with happiness.
    # If freedom outweighs corruption, people perceive life more positively.
    print("Adding Freedom-Corruption Ratio...")
    df['Freedom_Corruption_Ratio'] = (
        df['Freedom to make life choices'] / (df['Perceptions of corruption'] + 1e-6)
    )  # Add a small constant to avoid division by zero.

    # Feature 4: Generosity Adjusted for Corruption
    # Hypothesis: Generosity might have a stronger impact when corruption is low. 
    # This adjustment accounts for the undermining effect corruption can have on generosity.
    print("Adding Generosity-Corruption Adjustment...")
    df['Generosity_Corruption_Adjustment'] = (
        df['Generosity'] - df['Perceptions of corruption']
    )

    # Feature 5: Happiness Trend (Year-over-Year GDP Change)
    # Hypothesis: Changes in GDP over time affect happiness trends within countries.
    # A positive trend might indicate improving conditions, boosting happiness.
    print("Adding Happiness Trend (Lag Feature)...")
    df['Happiness_Trend'] = df.groupby('Country name')['Log GDP per capita'].diff().fillna(0)

    # Feature 6: Perceived Well-Being Index
    # Hypothesis: Health, freedom, and social support collectively influence how people perceive their well-being.
    # We're combining these factors into one index.
    print("Adding Perceived Well-Being Index...")
    df['Well_Being_Index'] = (
        df['Healthy life expectancy at birth'] * 0.5 +
        df['Freedom to make life choices'] * 0.25 +
        df['Social support'] * 0.25
    )

    # Feature 7: Normalized Generosity
    # Hypothesis: Standardizing generosity allows the model to compare it fairly across different scales.
    # This helps reduce the influence of outliers.
    print("Normalizing Generosity...")
    df['Normalized_Generosity'] = (
        df['Generosity'] - df['Generosity'].mean()
    ) / df['Generosity'].std()

    # Let’s do a quick sanity check to avoid any duplicate or unintended columns.
    df = df.loc[:, ~df.columns.duplicated()]
    
    print("Feature engineering complete. Here's the list of new features added:")
    print([col for col in df.columns if col not in ['Country name', 'year', 'Life Ladder']])

    return df

if __name__ == "__main__":
    # Load preprocessed data
    file_path = "data/processed/X.csv"
    print("Loading preprocessed data...")
    df = pd.read_csv(file_path)

    # Apply feature engineering
    print("Starting feature engineering...")
    df_fe = feature_engineering(df)

    # Save the updated dataset
    print("Saving the dataset with engineered features...")
    output_path = "data/processed/X_engineered.csv"
    df_fe.to_csv(output_path, index=False)
    print(f"Feature-engineered dataset saved at '{output_path}'.")

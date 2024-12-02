import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def feature_engineering(df):
    """
    Performs feature engineering on our dataset to create logically derived (research backed) and model-relevant features.
    """
    # We start with a clean copy of the data. 
    df = df.copy()

    # Feature 1: Economic and Social Index
    # Hypothesis: Happiness is influenced by both economic stability (GDP) and social connections (social support).
    # We’re weighting GDP higher because economic strength tends to have a more measurable impact on life quality.
    print("Adding Economic and Social Index...")
    df['Economic_Social_Index'] = (
        df['Log GDP per capita'] * 0.7 + df['Social support'] * 0.3
    )

    # Feature 2: Emotional Stability Score
    # Hypothesis: People with more positive emotions and fewer negative ones are generally happier (duh).
    # This feature subtracts negative affect from positive affect to measure the emotional balance.
    print("Adding Emotional Stability Score...")
    df['Emotional_Stability'] = df['Positive affect'] - df['Negative affect']

    # Feature 3: Freedom-Corruption Ratio
    # Hypothesis: People are happier when they have freedom and face less corruption.
    # This ratio shows how well freedom outweighs corruption in a given country.
    print("Adding Freedom-Corruption Ratio...")
    df['Freedom_Corruption_Ratio'] = (
        df['Freedom to make life choices'] / (df['Perceptions of corruption'] + 1e-6)
    )  # We add a small constant to avoid division by zero in case corruption is missing.

    # Feature 4: Generosity Adjusted for Corruption
    # Hypothesis: Generosity loses its perceived value when corruption is high.
    # This adjustment factors in the dampening effect corruption might have on the benefits of generosity.
    print("Adding Generosity-Corruption Adjustment...")
    df['Generosity_Corruption_Adjustment'] = (
        df['Generosity'] - df['Perceptions of corruption']
    )

    # Feature 5: Happiness Trend (Year-over-Year GDP Change)
    # Hypothesis: A steady improvement in GDP over time might boost happiness, as it signals economic progress.
    # We calculate the year-over-year change in GDP to capture this trend.
    print("Adding Happiness Trend (Lag Feature)...")
    df['Happiness_Trend'] = df.groupby('Country name')['Log GDP per capita'].diff().fillna(0)

    # Feature 6: Perceived Well-Being Index
    # Hypothesis: Health, freedom, and social support collectively shape how people perceive their quality of life.
    # We assign weights to each factor to create a composite index.
    print("Adding Perceived Well-Being Index...")
    df['Well_Being_Index'] = (
        df['Healthy life expectancy at birth'] * 0.5 +
        df['Freedom to make life choices'] * 0.25 +
        df['Social support'] * 0.25
    )

    # Feature 7: Normalized Generosity
    # Hypothesis: Normalizing generosity ensures a fair comparison across countries with different scales.
    # This standardization reduces the influence of outliers and brings the feature to a consistent scale.
    print("Normalizing Generosity...")
    df['Normalized_Generosity'] = (
        df['Generosity'] - df['Generosity'].mean()
    ) / df['Generosity'].std()
    
    df = df.loc[:, ~df.columns.duplicated()]
    
    print("Feature engineering complete. Here's the list of new features added:")
    print([col for col in df.columns if col not in ['Country name', 'year', 'Life Ladder']])

    return df

if __name__ == "__main__":
    # Load the preprocessed data.
    print("Loading preprocessed data...")
    file_path = "data/processed/X.csv"
    df = pd.read_csv(file_path)

    # Apply feature engineering to enhance the dataset.
    print("Starting feature engineering...")
    df_fe = feature_engineering(df)

    # Save the newly engineered dataset for future use.
    print("Saving the dataset with engineered features...")
    output_path = "data/processed/X_engineered.csv"
    df_fe.to_csv(output_path, index=False)
    print(f"Feature-engineered dataset saved at '{output_path}'.")

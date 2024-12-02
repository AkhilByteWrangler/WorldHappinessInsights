import os
import subprocess

def run_command(command, description):
    """
    Runs a shell command and logs its output.
    """
    print(f"\nStarting: {description}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ {description} completed successfully!")
        print(result.stdout)
    else:
        print(f"❌ {description} failed. Error:\n{result.stderr}")
        exit(1)

if __name__ == "__main__":
    print("Welcome to the World Happiness Insights Automation Script!")
    
    # Preprocess Data
    run_command(
        "python src/data_preprocessing.py",
        "Data Preprocessing"
    )
    
    # Feature Engineering
    run_command(
        "python src/data_feature_engineering.py",
        "Feature Engineering"
    )
    
    # Split Data into Training and Testing Sets
    run_command(
        "python src/data_splitting.py",
        "Data Splitting"
    )
    
    # Train Random Forest Model
    run_command(
        "python src/train_non_dl_model.py",
        "Random Forest Model Training"
    )
    
    # Train TCN Model
    run_command(
        "python src/train_dl_model.py",
        "Temporal Convolutional Network (TCN) Training"
    )
    
    # Evaluate Models
    run_command(
        "python src/evaluation_comparison_of_models.py",
        "Model Evaluation and Comparison"
    )
    
    # Launch Streamlit Dashboard
    print("\nAll steps completed! Launching the Streamlit Dashboard for visualization...")
    run_command(
        "streamlit run dashboard.py",
        "Streamlit Dashboard"
    )

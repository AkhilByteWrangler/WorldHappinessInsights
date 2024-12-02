# World Happiness Insights 🌍😊

This project explores the World Happiness dataset to model happiness across countries. Using both traditional machine learning and deep learning approaches, we uncover trends and make predictions about happiness based on various socio-economic and health-related factors.

---

**Visit the dashboard directly:**

[World Happiness Insights Dashboard](https://worldhappinessinsights.streamlit.app/)


**Watch the Video Presentation:**

[Video Presentation on Youtube](https://www.youtube.com/watch?v=YXku-uC-lgo)

---

## **Dataset**

The dataset contains various metrics that influence happiness across countries, including:
- **Log GDP per capita**: Economic strength.
- **Social support**: Interpersonal relationships.
- **Healthy life expectancy**: Health-related metrics.
- **Freedom to make life choices**: Personal freedoms.
- **Generosity**: Philanthropy levels.
- **Perceptions of corruption**: Institutional trust.
- **Positive/Negative affect**: Emotional balance.

**Target Variable**: `Life Ladder` (Happiness Index).

**Source**: This dataset is derived from the **World Happiness Report** and can be accessed at [World Happiness Report](https://worldhappiness.report/). The report is an authoritative source on the global state of happiness, providing data-driven insights into what drives well-being across the globe.

---

## **Data Pipeline**

### **1. Data Preprocessing**
- **Handling Missing Data**: Missing values are imputed using mean strategy (SimpleImputer).
- **Outlier Management**: Outliers are clipped using the Inter Quartile Range (IQR) method.
- **Scaling**: Numerical features are scaled using `StandardScaler`.

---

### **2. Feature Engineering**
New features are derived to enhance model performance:

| **Feature Name**                | **Formula**                                                                                                                                      | **Hypothesis**                                                                                      |
|----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| **Economic_Social_Index**        | `Economic_Social_Index = (Log GDP per capita * 0.7) + (Social support * 0.3)`                                                                    | GDP and social support are the strongest drivers of happiness, with GDP having a more direct impact. |
| **Emotional_Stability**          | `Emotional_Stability = Positive affect - Negative affect`                                                                                       | Positive emotions and fewer negative emotions improve happiness.                                   |
| **Freedom_Corruption_Ratio**     | `Freedom_Corruption_Ratio = Freedom to make life choices / (Perceptions of corruption + 1e-6)`                                                   | A balance of freedom and lower corruption correlates with higher happiness.                        |
| **Generosity_Corruption_Adjustment** | `Generosity_Corruption_Adjustment = Generosity - Perceptions of corruption`                                                                     | Generosity’s impact is stronger when corruption is low, as corruption undermines perceived benefits. |
| **Happiness_Trend**              | `Happiness_Trend = Log GDP per capita (current year) - Log GDP per capita (previous year)`                                                       | Year-over-year GDP growth reflects improving conditions, boosting happiness.                        |
| **Well_Being_Index**             | `Well_Being_Index = (Healthy life expectancy at birth * 0.5) + (Freedom to make life choices * 0.25) + (Social support * 0.25)`                  | Health, freedom, and social support collectively influence perceived well-being, with health as the strongest factor. |
| **Normalized_Generosity**        | `Normalized_Generosity = (Generosity - mean(Generosity)) / std(Generosity)`                                                                      | Standardizing generosity reduces bias from extreme values and enables fairer comparison.            |

---

### **3. Data Splitting**
Data is split into training and testing datasets (80-20 ratio) to ensure fair evaluation. And also we conducted 3-fold cross validation. 

---

## **Models**

### **1. Random Forest (Non-Deep Learning Model)**
- A robust ensemble-based decision tree model.
- Tuned using RandomizedSearchCV for hyperparameter optimization.
- Handles non-linear relationships and provides feature importance.

### **2. Temporal Convolutional Network (TCN)**
- A deep learning architecture specialized for sequential data.
- Captures time-dependent patterns like GDP trends over years.
- Uses causal padding to respect the temporal order of data.

---

## **Evaluation Metrics & Relevance of Metrics to the Dataset and Task**

| **Metric**            | **Why Relevant to Our Dataset and Task**                                                                                  | **Specific Insights**                                                                                              |
|------------------------|---------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| **Mean Squared Error (MSE)**  | Captures large deviations in predictions, penalizing extreme errors (e.g., very high or low happiness scores).                  | Ensures the model doesn’t make significant errors for countries with unusual happiness trends or scores.          |
| **Mean Absolute Error (MAE)** | Provides an intuitive measure of average prediction error in the same unit as the happiness score (`Life Ladder`).            | Helps policymakers understand how far off predictions are, on average, for each country’s happiness score.        |
| **R² (Coefficient of Determination)** | Measures how well the model explains variance in happiness scores, highlighting key drivers like GDP or social support.     | Indicates whether the model captures the primary factors influencing happiness across diverse countries.          |
| **Explained Variance** | Validates R² by ensuring robustness against outliers or skewed happiness scores in the dataset.                                      | Confirms that the model explains happiness variability consistently, even for extreme cases (e.g., war-torn areas). |
| **Mean Absolute Percentage Error (MAPE)** | Shows relative prediction errors, enabling easy comparison of errors across countries with differing happiness scales.         | Highlights model performance across regions with varying happiness levels (e.g., Scandinavian vs. African nations). |
| **Max Error**          | Identifies the worst-case prediction error, which is crucial for ensuring robustness in real-world applications.                     | Ensures that no country receives a highly inaccurate happiness prediction, which could lead to misleading insights. |

---

## **How to Run the Project**

### 1. Install Dependencies
To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

### 2. Run the Automated Pipeline Script
Execute the pipeline script by running:

```bash
python run_pipeline.py
```

### 3. Quick Access to the Dashboard
If you'd like to skip the setup and explore the insights immediately, visit the dashboard directly:

[World Happiness Insights Dashboard](https://worldhappinessinsights.streamlit.app/)



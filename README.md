# Customer Churn Prediction and Lifetime Value Retention Strategy

## Executive Summary
This project provides an end-to-end machine learning pipeline designed to predict customer churn within a telecommunications subscription model. By transitioning from baseline predictive accuracy to explainable artificial intelligence (XAI), the framework empowers non-technical stakeholders to implement targeted retention strategies, thereby safeguarding Customer Lifetime Value (LTV).

## Business Objectives
1. Identify customers at high risk of canceling their subscriptions.
2. Deconstruct the underlying drivers of churn using game-theoretic explainability (SHAP).
3. Deploy an interactive dashboard for customer service and marketing teams to evaluate real-time risk profiles.

## Architecture and Methodology

### 1. Data Engineering Pipeline
- **Modular Processing:** Raw data is processed through a reproducible Python script (`src/data_prep.py`) ensuring a production-ready environment.
- **Data Integrity:** Missing values are handled systematically, and categorical variables are transformed using one-hot encoding to prevent multicollinearity.
- **Stratified Splitting:** The training and testing matrices are stratified to maintain the natural class imbalance of the baseline churn rate (approximately 26.5%).

### 2. Machine Learning Model
- **Algorithm:** XGBoost Classifier.
- **Optimization Strategy:** The model natively addresses class imbalance via the `scale_pos_weight` hyperparameter. The evaluation metric prioritizes Recall, ensuring the business correctly identifies the maximum number of true churners, rather than optimizing for misleading global accuracy.

### 3. Explainable AI (SHAP)
- **Macro-Level Analysis:** Global feature importance analysis confirms that low tenure, month-to-month contracts, and high monthly charges are the primary catalysts for customer attrition.
- **Micro-Level Analysis:** Waterfall plots generate granular, per-customer insights, explaining the specific variables influencing an individual's predicted risk score.

## Strategic Recommendations
Based on the SHAP analysis, the following retention protocols are recommended for immediate business implementation:
- **The First-90-Days Protocol:** Deploy aggressive onboarding support and incentives during the initial three months of the customer lifecycle.
- **Contract Incentivization:** Offer strategic discounts to transition users from volatile month-to-month agreements to secure annual contracts.
- **Targeted Price Relief:** Initiate automated loyalty discounts for high-risk customers exhibiting high price sensitivity.

## Interactive Dashboard
The predictive model and SHAP visualizations are deployed via a local Streamlit web application. This interface allows business stakeholders to simulate database queries and view customer risk profiles in real time without interacting with the underlying Python codebase.

## Repository Structure

```text
├── app/
│   └── app.py                            # Streamlit interactive dashboard
├── data/
│   ├── processed/                        # Cleaned matrices (Ignored by Git)
│   └── raw/                              # Original datasets (Ignored by Git)
├── models/
│   └── churn_xgboost_model.pkl           # Serialized XGBoost model (Ignored by Git)
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   └── 03_explainable_ai.ipynb
├── src/
│   ├── data_prep.py                      # Automated feature engineering pipeline
│   └── train.py                          # Model training and evaluation script
├── .gitignore
└── README.md
```

--- 

## Execution Instructions

### 1. Environment Setup
Ensure you have an active Python virtual environment, then install the required dependencies:

``` Bash
pip install pandas numpy scikit-learn xgboost shap streamlit matplotlib joblib
```

### 2. Execute the Data Pipeline
Process the raw data and generate the training matrices:

``` Bash
python src/data_prep.py
```

### 3. Train the Model
Train the XGBoost classifier and generate the serialized .pkl file:

``` Bash
python src/train.py
```

### 4. Launch the Dashboard
Initialize the local Streamlit server to view the interactive application:

``` Bash
streamlit run app/app.py
```

# 📊 Customer Churn Prediction & Lifetime Value (LTV) Strategy

## 🚀 Project Overview
[cite_start]While anyone can train a classification model, the goal of this project is to translate machine learning metrics into actionable business decisions[cite: 102, 103]. This repository contains an end-to-end data science pipeline that predicts whether a customer will cancel their subscription and forecasts their future monetary value. 

[cite_start]By identifying at-risk customers, this model allows stakeholders to determine exactly which customer segments are worth a targeted marketing retention budget and which are a lost cause[cite: 103, 147].

## 🛠️ Tech Stack
[cite_start]This project utilizes a modern machine learning stack and follows software engineering best practices by being clean, commented, and version-controlled[cite: 94, 138]:
* **Data Processing & EDA:** Pandas, Scikit-Learn
* [cite_start]**Machine Learning:** XGBoost / LightGBM [cite: 101, 143]
* [cite_start]**Explainable AI:** SHAP (SHapley Additive exPlanations) to visualize which specific features drive customer churn [cite: 101, 145]
* [cite_start]**Deployment:** Streamlit (Interactive Web Application) 

## 📂 Repository Structure
```text
churn-ltv-prediction/
│
├── data/                   # Contains raw and processed datasets (Ignored in git)
├── notebooks/              # Jupyter notebooks for EDA and initial modeling
├── src/                    # Modular Python scripts (data_prep.py, train.py, predict.py)
├── models/                 # Serialized XGBoost/LightGBM model files
├── app/                    # Streamlit web application script
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
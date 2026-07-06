import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Loads the raw dataset."""
    return pd.read_csv(filepath)

def clean_data(df):
    """Handles missing values, incorrect data types, and drops useless columns."""
    # Fix TotalCharges: Convert to numeric, coercing spaces to NaN, then fill with 0
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    # Drop customerID as it holds no predictive value and introduces noise
    df = df.drop('customerID', axis=1)
    
    # Convert the Target variable (Churn) into binary format (1 = Yes, 0 = No)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    return df

def engineer_features(df):
    """Encodes categorical variables into numerical format."""
    # Separate features (X) and target (y)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Identify categorical columns
    cat_cols = X.select_dtypes(include=['object']).columns
    
    # One-Hot Encode categorical variables (drop_first=True avoids multicollinearity)
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    return X_encoded, y

def split_and_scale(X, y):
    """Splits data and scales numerical features for optimal model performance."""
    # Train-test split using Stratify
    # Stratification ensures both train and test sets have the exact same 26.5% churn rate
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    # ONLY transform the test set to prevent data leakage from the future
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    return X_train, X_test, y_train, y_test

def main():
    print("🚀 Starting data preparation pipeline...")
    
    # 1. Load Data (Adjust path if running from a different directory)
    df = load_data('data/raw/telco_churn.csv')
    
    # 2. Clean Data
    df_clean = clean_data(df)
    
    # 3. Engineer Features
    X, y = engineer_features(df_clean)
    
    # 4. Split and Scale
    X_train, X_test, y_train, y_test = split_and_scale(X, y)
    
    # Ensure the processed data directory exists
    os.makedirs('data/processed', exist_ok=True)
    
    # Save the processed, model-ready data
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    print("✅ Data preparation complete. Processed files saved to data/processed/")

if __name__ == "__main__":
    main()
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import os

def load_processed_data():
    """Load the processed training and testing datasets."""
    # Note: Adjust paths if running from a different directory
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    
    # .values.ravel() flattens the target data into a 1D array required by XGBoost
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Initializes and trains the XGBoost classifier."""
    print("Training XGBoost model...")
    
    # scale_pos_weight = (Number of Negative Cases) / (Number of Positive Cases)
    # Roughly 73% retained / 26.5% churned = ~2.7
    # This heavily penalizes the model for missing a churner.
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        scale_pos_weight=2.7, 
        random_state=42,
        learning_rate=0.05,
        n_estimators=100,
        max_depth=4
    )
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and prints metrics formatted for business stakeholders."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] # Get probabilities for the 'Churn' class
    
    print("\nModel Evaluation Metrics")
    print("-" * 40)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nBusiness Context:")
    print("- Focus on 'Recall' for Class 1 (Churn).")
    print("- A high recall means we are successfully catching the majority of customers who are about to leave.")
    print("- A False Positive (guessing someone will churn when they won't) just means we send them a nice discount coupon. A False Negative (missing a churner) means we lose their lifetime value forever.")

def save_model(model, model_path='models/churn_xgboost_model.pkl'):
    """Serializes and saves the trained model for future deployment."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nModel successfully saved to {model_path}")

def main():
    print("Starting model training pipeline...")
    
    # 1. Load Data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # 2. Train Model
    model = train_model(X_train, y_train)
    
    # 3. Evaluate Model
    evaluate_model(model, X_test, y_test)
    
    # 4. Save Model
    save_model(model)

if __name__ == "__main__":
    main()
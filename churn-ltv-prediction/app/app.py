import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# --- 1. Page Configuration ---
st.set_page_config(page_title="Churn Prediction Dashboard", page_icon="📊", layout="wide")

st.title("📊 Customer Retention Dashboard")
st.write("Identify at-risk customers and understand the specific drivers behind their churn probability.")

# --- 2. Load Assets (Cached for performance) ---
@st.cache_resource
def load_model_and_data():
    """Loads the XGBoost model and the processed testing data."""
    model = joblib.load('models/churn_xgboost_model.pkl')
    # We load the test data to simulate a database of current customers
    X_test = pd.read_csv('data/processed/X_test.csv')
    return model, X_test

model, X_test = load_model_and_data()

# --- 3. Sidebar UI: Customer Selection ---
st.sidebar.header("Customer Search")
st.sidebar.write("Simulate pulling a customer record from the database.")

# Let the user pick a customer index from the test set
max_index = len(X_test) - 1
customer_index = st.sidebar.number_input(
    f"Enter Customer ID (Index 0 to {max_index})", 
    min_value=0, 
    max_value=max_index, 
    value=4 # Defaulting to customer #4 as they are high-risk
)

# Extract that specific customer's data
customer_data = X_test.iloc[[customer_index]]

# --- 4. Main Dashboard Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Customer Risk Profile")
    
    # Generate prediction and probability
    probability = model.predict_proba(customer_data)[0][1]
    
    # Business Logic: Threshold set at 50%
    if probability > 0.5:
        st.error(f"**Status: HIGH RISK OF CHURN**")
    else:
        st.success(f"**Status: LOW RISK (Healthy)**")
        
    st.markdown(f"**Predicted Churn Probability:** {probability * 100:.2f}%")
    
    st.write("---")
    st.write("**Customer Attributes (Scaled Matrix):**")
    st.dataframe(customer_data.T)

with col2:
    st.subheader("Explainable AI: Why is this happening?")
    st.write("The waterfall chart below deconstructs the prediction. **Red bars** push the customer toward cancellation, while **blue bars** push them toward retention.")
    
    # Generate SHAP explanation for this specific customer
    explainer = shap.Explainer(model)
    shap_values = explainer(customer_data)
    
    # Plot using Matplotlib and pass to Streamlit
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    
    # Tight layout ensures the graph fits nicely in the web app
    plt.tight_layout()
    st.pyplot(fig)
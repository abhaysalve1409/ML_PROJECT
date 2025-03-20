import streamlit as st
import os
import joblib
import pandas as pd
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.getcwd(), "fraudulent_transactions_Model.pkl")
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    return joblib.load("fraudulent_transactions_Model.pkl")  # Ensure your model file is named correctly

model = load_model()

st.title("💳 Fraudulent Transactions Prediction App")

# Get feature names from the model
expected_features = model.feature_names_in_  # Retrieves the exact feature order

# Collect user input
def user_input():
    st.sidebar.header("Enter Transaction Details")
    transaction_amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, max_value=10000.0, value=500.0)
    transaction_type = st.sidebar.selectbox("Transaction Type", ["Online", "In-Person", "ATM Withdrawal"])
    account_balance = st.sidebar.number_input("Account Balance", min_value=0.0, max_value=100000.0, value=1000.0)
    customer_age = st.sidebar.number_input("Customer Age", min_value=18, max_value=100, value=30)
    location = st.sidebar.selectbox("Transaction Location", ["Urban", "Suburban", "Rural"])
    previous_fraud = st.sidebar.selectbox("Previous Fraudulent Activity", ["Yes", "No"])
    time_of_transaction = st.sidebar.slider("Time of Transaction (Hour)", 0, 23, 12)
    
    # Encode categorical variables
    transaction_type_encoded = 1 if transaction_type == "Online" else (2 if transaction_type == "In-Person" else 3)
    location_encoded = 1 if location == "Urban" else (2 if location == "Suburban" else 3)
    previous_fraud_encoded = 1 if previous_fraud == "Yes" else 0
    
    data = pd.DataFrame([[transaction_amount, transaction_type_encoded, account_balance, customer_age, location_encoded, previous_fraud_encoded, time_of_transaction]], 
                         columns=["Transaction_Amount", "Transaction_Type", "Account_Balance", "Customer_Age", "Location", "Previous_Fraudulent_Activity", "Time_of_Transaction"])
    
    # Reorder input data to match model training order
    data = data[expected_features]
    return data

input_data = user_input()

if st.button("Predict"):  # Run prediction when button is clicked
    prediction = model.predict(input_data)
    result = "Fraudulent" if prediction[0] == 1 else "Not Fraudulent"
    st.write(f"### Prediction: {result}")

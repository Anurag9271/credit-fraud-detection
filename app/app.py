import streamlit as st
import joblib
import pandas as pd

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "fraud_model_reduced.joblib")

model = joblib.load(model_path)

# model = joblib.load("../models/fraud_model_reduced.joblib")

st.title("💳 Credit Card Fraud Detection")

transaction_amount = st.number_input("Transaction Amount")

num_transactions_24h = st.number_input("Transactions in last 24h")

num_transactions_7d = st.number_input("Transactions in last 7 days")

previous_fraud_count = st.number_input("Previous Fraud Count")

avg_transaction_amount = st.number_input("Average Transaction Amount")

input_data = pd.DataFrame({
    "transaction_amount":[transaction_amount],
    "num_transactions_24h":[num_transactions_24h],
    "num_transactions_7d":[num_transactions_7d],
    "previous_fraud_count":[previous_fraud_count],
    "avg_transaction_amount":[avg_transaction_amount]
})

if st.button("Predict Fraud"):

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction")
    else:
        st.success("✅ Legitimate Transaction")
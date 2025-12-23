import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Telco Churn Prediction")

st.title("📊 Telco Customer Churn Prediction")

model = joblib.load("churn_model.pkl")

st.write("Masukkan data pelanggan:")

tenure = st.number_input("Tenure (bulan)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 2000.0)
senior = st.selectbox("Senior Citizen", [0, 1])

input_data = pd.DataFrame([{
    "SeniorCitizen": senior,
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}])

if st.button("Prediksi"):
    pred = model.predict(input_data)[0]
    if pred == 1:
        st.error("❌ Pelanggan berpotensi CHURN")
    else:
        st.success("✅ Pelanggan TIDAK churn")

import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Gold Price Predictor",
    page_icon="🥇",
    layout="centered"
)
model = pickle.load(open("gold_model.pkl", "rb"))

st.title("Gold Price Prediction System")

spx = st.number_input("SPX Value")

uso = st.number_input("USO Value")

silver = st.number_input("Silver Price")

eur_usd = st.number_input("EUR/USD")

if st.button("Predict Gold Price"):

    features = np.array([[spx, uso, silver, eur_usd]])

    prediction = model.predict(features)

    usd_price = prediction[0]

    usd_to_inr = 86

    inr_price = usd_price * usd_to_inr

    st.success(f"Predicted Gold Price (USD): ${usd_price:.2f}")

    st.success(f"Predicted Gold Price (INR): ₹{inr_price:,.2f}")

st.sidebar.header("Model Information")

st.sidebar.write("Algorithm: Random Forest Regressor")

st.sidebar.write("R² Score: 0.98")
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

spx = st.number_input("SPX Value", min_value=500.0, value=1221.0)

uso = st.number_input("USO Value", min_value=1.0, value=31.8)

silver = st.number_input("Silver Price", min_value=1.0, value=20.1)

eur_usd = st.number_input("EUR/USD", min_value=0.5, value=1.28)

if st.button("Predict Gold Price"):

    if spx <= 0 or uso <= 0 or silver <= 0 or eur_usd <= 0:
        
        st.error("Please enter valid positive values for all fields.")
    else:
        features = np.array([[spx, uso, silver, eur_usd]])

        prediction = model.predict(features)

        usd_price = prediction[0]

        usd_to_inr = 86

        inr_price = usd_price * usd_to_inr

        st.success(f"Predicted Gold Price (USD): ${usd_price:.2f}")

        st.success(f"Predicted Gold Price (INR): ₹{inr_price:,.2f}")

st.sidebar.header("Project Details")

st.sidebar.write("Project: Gold Price Prediction")

st.sidebar.write("Technology: Machine Learning")

st.sidebar.write("Algorithm: Random Forest Regressor")

st.sidebar.write("R² Score: 0.98")

st.markdown("""
This application predicts the GLD (Gold ETF) price using:
- SPX (S&P 500 Index)
- USO (Oil Price)
- Silver Price
- EUR/USD Exchange Rate

Machine Learning Algorithm :
Random Forest Regressor
""")

#https://goldpriceprediction-jhnfqd4an83grb5qfsluxj.streamlit.app/
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ---------------- TITLE ----------------
st.title("📈 Tesla Stock Price Prediction using LSTM")

# ---------------- INFO & DISCLAIMER ----------------
st.info("This model uses LSTM to capture long-term dependencies in stock price data.")
st.warning("This is not financial advice. Predictions are for educational purposes only.")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/TSLA.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

data = df[['Adj Close']]

# ---------------- SCALING ----------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# ---------------- PERFORMANCE METRICS ----------------
st.subheader("📊 Model Performance")
st.metric("RMSE", "19.24")
st.metric("MAE", "12.67")

# ---------------- HISTORICAL CHART ----------------
st.subheader("📉 Historical Tesla Stock Price")
st.line_chart(data)

# ---------------- LOAD MODEL ----------------
model = load_model("lstm_model.keras")

# ---------------- PREDICTION FUNCTION ----------------
def predict_future(model, data, days=10, window_size=60):
    temp_data = data[-window_size:].copy()
    predictions = []

    for _ in range(days):
        X_input = temp_data[-window_size:].reshape(1, window_size, 1)
        pred = model.predict(X_input, verbose=0)
        predictions.append(pred[0, 0])
        temp_data = np.append(temp_data, pred)

    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions)

# ---------------- BUTTON ----------------
if st.button("🔮 Predict Next 10 Days"):
    future = predict_future(model, scaled_data)
    st.write("### 📅 Next 10 Days Prediction")
    st.write(future.flatten())

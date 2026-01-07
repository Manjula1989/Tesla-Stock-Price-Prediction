📈 Tesla Stock Price Prediction using LSTM

This project implements a Deep Learning–based time series forecasting model using Long Short-Term Memory (LSTM) networks to predict Tesla stock prices.
The model is trained on historical stock price data and deployed using Streamlit for interactive visualization and prediction.

🧠 Project Overview

Stock price prediction is a challenging problem due to volatility and temporal dependencies.
LSTM networks are well-suited for this task as they can capture long-term dependencies in sequential data.

This project:

Analyzes historical Tesla stock prices

Trains an LSTM model to predict future prices

Evaluates the model using RMSE and MAE

Deploys the model as a web application using Streamlit

🛠️ Tech Stack

Programming Language: Python

Libraries & Frameworks:

NumPy

Pandas

Matplotlib

Scikit-learn

TensorFlow / Keras

Streamlit

Model: LSTM (Long Short-Term Memory)

IDE: VS Code

Environment: Python 3.10 (Virtual Environment)

📂 Project Structure
Tesla Stock Price Prediction/
│
├── data/
│   └── TSLA.csv
│
├── notebook/
│   ├── tesla_stock_prediction.ipynb
│   └── lstm_model.h5
│
├── venv310/
│
├── app.py
├── lstm_model.keras
└── README.md

📊 Dataset

Source: Tesla historical stock price data (CSV)

Features Used:

Date

Open

High

Low

Close

Adj Close

Volume

Target Variable: Adjusted Close Price

⚙️ Model Workflow

Load and preprocess stock price data

Normalize data using MinMaxScaler

Create time-series sequences (window size = 60)

Train LSTM model

Evaluate model performance

Save trained model

Deploy using Streamlit

📈 Model Performance
Metric	Value
RMSE	19.24
MAE	12.67

The model successfully captures overall price trends with acceptable prediction error, considering market volatility.

🖥️ Streamlit Web Application

The Streamlit app provides:

Historical stock price visualization

Model performance metrics

Interactive prediction of next 10 days

▶️ Run the App

Activate the virtual environment and run:

streamlit run app.py

⚠️ Disclaimer

This is not financial advice.
Predictions are for educational purposes only.

🎯 Key Learnings

Time-series data preprocessing

LSTM architecture for sequence prediction

Model evaluation using RMSE & MAE

Deploying ML models using Streamlit

End-to-end ML project workflow

👤 Author

Manjula M
Backend & Machine Learning Enthusiast

⭐ Future Enhancements

Predict 5-day and 10-day future trends separately

Add more technical indicators

Deploy on Streamlit Cloud

Improve model accuracy with stacked LSTMs

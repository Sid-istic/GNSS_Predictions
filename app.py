import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import joblib

# -----------------------
# CONFIG
# -----------------------
LOOKBACK = 16
FEATURES = 4

# -----------------------
# LOAD MODEL + SCALER
# -----------------------
@st.cache_resource
def load_artifacts():

    # Load LSTM from .pkl
    with open("lstm_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Load scaler
    scaler = joblib.load("scaler.pkl")

    return model, scaler

model, scaler = load_artifacts()

# -----------------------
# UI
# -----------------------
st.title("📡 GNSS Satellite Error Prediction (LSTM)")

st.write("Enter past 16 timesteps (x, y, z, clock error)")

# Input table
input_df = pd.DataFrame(
    np.zeros((LOOKBACK, FEATURES)),
    columns=["x_error", "y_error", "z_error", "satclock"]
)

edited_df = st.data_editor(input_df)

# -----------------------
# PREDICT
# -----------------------
if st.button("Predict"):

    data = edited_df.values

    # Scale input
    data_scaled = scaler.transform(data)

    # Reshape for LSTM
    data_scaled = data_scaled.reshape(1, LOOKBACK, FEATURES)

    # Predict
    preds = model.predict(data_scaled)[0]

    # Display
    st.subheader("📈 Predictions")

    for i, val in enumerate(preds):
        st.write(f"Horizon {i+1}: {val:.4f}")

    # Plot
    fig, ax = plt.subplots()

    ax.plot(range(LOOKBACK), edited_df["satclock"], label="Past", marker="o")
    ax.plot(range(LOOKBACK, LOOKBACK + len(preds)), preds, label="Predicted", marker="x")

    ax.legend()
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Clock Error")

    st.pyplot(fig)

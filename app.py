import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import joblib
import tensorflow as tf  # IMPORTANT: load TF before pickle

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
    
    # Load model.pkl
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    # Load scaler
    scaler = joblib.load("scaler_model.pkl")

    return model, scaler

model, scaler = load_artifacts()

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="GNSS LSTM Predictor", layout="centered")

st.title("📡 GNSS Satellite Clock Error Prediction")
st.write("Enter past 16 timesteps (x, y, z, satclock)")

# -----------------------
# INPUT TABLE
# -----------------------
input_df = pd.DataFrame(
    np.zeros((LOOKBACK, FEATURES)),
    columns=["x_error", "y_error", "z_error", "satclock"]
)

edited_df = st.data_editor(input_df, num_rows="fixed")

# -----------------------
# PREDICTION
# -----------------------
if st.button("Predict"):

    data = edited_df.values

    try:
        # Scale input
        data_scaled = scaler.transform(data)

        # Reshape for LSTM
        data_scaled = data_scaled.reshape(1, LOOKBACK, FEATURES)

        # Predict
        preds = model.predict(data_scaled)[0]

        # -----------------------
        # DISPLAY RESULTS
        # -----------------------
        st.subheader("📈 Predicted Future Clock Error")

        for i, val in enumerate(preds):
            st.write(f"Horizon {i+1}: {val:.4f}")

        # -----------------------
        # PLOT
        # -----------------------
        fig, ax = plt.subplots()

        ax.plot(range(LOOKBACK), edited_df["satclock"], label="Past", marker="o")
        ax.plot(range(LOOKBACK, LOOKBACK + len(preds)), preds, label="Predicted", marker="x")

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Clock Error")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error during prediction: {e}")

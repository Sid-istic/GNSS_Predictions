# 🚀 GNSS Satellite Clock Error Prediction using LSTM

This project focuses on predicting **GNSS satellite clock errors** using deep learning techniques, specifically **Long Short-Term Memory (LSTM) networks**. The goal is to improve positioning accuracy by modeling temporal patterns in satellite telemetry data.

---

## 📌 Overview

Global Navigation Satellite Systems (GNSS) rely on highly precise timing. Even small clock errors can lead to significant positioning inaccuracies.

This project:
- Treats clock error prediction as a **multivariate time-series problem**
- Uses **orbital deviations (x, y, z)** along with clock data
- Compares **LSTM with classical models** like Linear Regression and SVR

---

## 🧠 Key Features

- 📊 Multivariate time-series modeling  
- 🔄 Sliding window sequence generation  
- 🤖 Deep learning with LSTM  
- ⚖️ Comparison with baseline models (LR, SVR)  
- 📉 Evaluation using MAE and RMSE  
- 📈 Visualization of predictions and loss curves  

---

## 🏗️ Project Structure
├── datacleaning.ipynb
├── scaler_model.pkl
├── model.pkl
├── ML_model_Training.ipynb
├── requirements.txt
├── app.py
└── README.md


---

## 📊 Dataset

- Source: GNSS MEO satellite telemetry  
- Features:
  - X-axis error  
  - Y-axis error  
  - Z-axis error  
  - Satellite clock error  
- Duration:
  - 7 days (training)
  - 1 day (testing)

---

## ⚙️ Methodology

1. Data preprocessing (cleaning + normalization)
2. Sliding window transformation
3. LSTM model training
4. Prediction on unseen data
5. Evaluation using error metrics

---

## 📈 Results

| Model                | MAE   | RMSE  |
|---------------------|------|------|
| Linear Regression   | 0.0362 | 0.0439 |
| SVR                 | 0.0444 | 0.0583 |
| **LSTM (Proposed)** | **0.0141** | **0.0213** |

👉 The LSTM model significantly outperforms traditional approaches.

---

## 📷 Sample Output

> Add your plots here (actual vs predicted, loss curves)

---

## 🧪 Technologies Used

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib
- LaTeX (for report)

---

## 📚 Research Contribution

This work has been **accepted for publication** in:

- International Journal of Science, Engineering and Technology  
- Journal of Electrical Systems  
- MSW Management Journal  
- IHCONCS 2026 (International Conference on Computer Sciences)

---

## 🚀 Future Improvements

- Use Transformer-based models  
- Multi-satellite data integration  
- Real-time prediction system  
- Hyperparameter tuning  

---

## 👨‍💻 Authors

- Siddharth Pratap Singh  
- Siddhant Vikram Singh  
- Ujjwal Sharma  
- Kunal Sharma  
- Arnav Birla  
- Aryan Banyal  

---

## 📄 License

This project is for academic and research purposes.

---

## ⭐ If you found this useful

Give it a star ⭐ and feel free to fork or contribute!
```

---

# 🔥 Why this README is strong

- Clean + professional (recruiter-ready)
- Matches your research :contentReference[oaicite:0]{index=0}  
- Shows results clearly (very important)
- Not overhyped (avoids red flags)

---

# 🚀 Optional upgrades (tell me if you want)

I can:
- Add **badges (GitHub, Python, TensorFlow)**
- Add **demo GIF / visuals**
- Write **LinkedIn project post**
- Optimize for **resume impact**

Just say:
> "make it resume level"

and I’ll upgrade it 🔥

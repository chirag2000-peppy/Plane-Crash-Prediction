# ✈️ Plane Crash Risk Prediction

This project analyzes historical plane crash data and builds a Machine Learning model to predict the probability of a plane crash based on flight details like operator, route, aircraft type, and location.

---

## 📌 Features
- **Data Cleaning & Preprocessing**: Handles missing values, encodes categorical data.
- **Data Visualization**: Identifies trends in crashes over time, airlines, and locations.
- **Machine Learning Models**: Implements Logistic Regression and Random Forest for crash probability prediction.
- **Prediction on New Flights**: Uses trained models to estimate the likelihood of a crash.

---

## 📂 Dataset
- Source: Plane Crash Data
- File: `planecrashinfo_20181121001952.csv`
- Includes information on **date, location, operator, aircraft type, route, fatalities, and ground impact**.

---

## 🚀 Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/plane-crash-prediction.git
cd plane-crash-prediction
```

### 2️⃣ Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 3️⃣ Run the Analysis
```bash
python plane_crash_analysis.py
```

---

## 📊 Data Visualization
### **Crashes Over Time**
Plots the number of crashes per year to identify trends.

### **Most Dangerous Airlines**
Identifies the airlines with the highest number of crashes.

### **Most Crash-Prone Locations**
Displays locations with the highest number of incidents.

---

## 🤖 Machine Learning Models
### **1️⃣ Logistic Regression**
- A simple baseline model for crash probability estimation.
- Predicts if a flight will have **fatalities or not**.

### **2️⃣ Random Forest Classifier**
- A more advanced model with higher accuracy.
- Provides probability-based predictions for crash likelihood.

---

## 🔮 Predicting Crash Probability
Run the following script to predict the probability of a crash for a new flight:
```python
import numpy as np

# Example: Predict a flight in 2025 with encoded values for location, operator, route, aircraft type
new_flight = np.array([[2025, 10, 5, 8, 12]])  # Replace with valid encodings
probability = rf_model.predict_proba(new_flight)[0][1]
print(f"Predicted Probability of a Crash: {probability:.4f}")
```

---

## 🛠 Future Enhancements
- Improve feature selection & engineering.
- Implement Deep Learning models (Neural Networks).
- Add real-time flight tracking for live risk estimation.

---

## 📜 License
This project is open-source under the MIT License.

---

## 🤝 Contributing
Pull requests are welcome! Feel free to contribute improvements and new features.

---

## 🔗 Connect
- **Author**: Chirag
- **GitHub**: [yourgithub](https://github.com/yourusername)

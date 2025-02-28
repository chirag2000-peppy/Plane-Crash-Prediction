import pandas as pd

# Load the dataset
file_path = "planecrashinfo_20181121001952.csv"  # Ensure the file is in the same folder
df = pd.read_csv(file_path)

# Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Extract year for analysis
df['year'] = df['date'].dt.year

# Extract total fatalities
df['fatalities'] = df['fatalities'].str.extract(r'(\d+)').astype(float)

# Convert 'ground' fatalities to numeric
df['ground'] = pd.to_numeric(df['ground'], errors='coerce')

# Fill NaN fatalities with 0
df['fatalities'].fillna(0, inplace=True)

df = df[['year', 'location', 'operator', 'route', 'ac_type', 'fatalities', 'ground']]

from sklearn.preprocessing import LabelEncoder

# Encode categorical columns
encoder = LabelEncoder()
for col in ['location', 'operator', 'route', 'ac_type']:
    df[col] = encoder.fit_transform(df[col])

import matplotlib.pyplot as plt
import seaborn as sns

# Count crashes per year
crashes_per_year = df['year'].value_counts().sort_index()

# Features (X) - Input Variables
X = df[['year', 'operator', 'route', 'ac_type', 'location']]

# Target (y) - Crash Severity (we'll use 'fatalities' as a proxy)
y = df['fatalities']

# Convert to binary classification (1 = Fatal crash, 0 = Non-fatal crash)
y = (y > 0).astype(int)  # If fatalities > 0, it's a "severe" crash

from sklearn.model_selection import train_test_split

# Split dataset: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
#print("Accuracy:", accuracy_score(y_test, y_pred))
#print("Classification Report:\n", classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluation
#print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
#print("Classification Report:\n", classification_report(y_test, y_pred_rf))

import numpy as np

# Example: Predicting a flight in 2025 from an encoded location/operator
new_flight = np.array([[2025, 10, 5, 8, 12]])  # Replace with valid encodings

# Predict probability of crash
probability = rf_model.predict_proba(new_flight)[0][1]  # Probability of crash
print(f"Predicted Probability of a Crash: {probability:.4f}")
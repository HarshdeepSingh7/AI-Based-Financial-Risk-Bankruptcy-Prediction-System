import joblib
import numpy as np
import pandas as pd
import os

# Get current file directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go to project root
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Define model path correctly
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "xgb_model.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
FEATURE_PATH = os.path.join(PROJECT_ROOT, "models", "feature_columns.pkl")

# Load artifacts
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURE_PATH)

def calculate_risk(prob):
    score = round(prob * 100, 2)

    if score < 30:
        category = "Low Risk"
    elif score < 60:
        category = "Medium Risk"
    else:
        category = "High Risk"

    return score, category


def predict_company_risk(input_dict):
    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Ensure correct column order
    input_df = input_df[feature_columns]

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict probability
    prob = float(model.predict_proba(input_scaled)[:, 1][0])

    # Calculate risk score
    score, category = calculate_risk(prob)

    return {
        "Probability": round(prob, 4),
        "Risk Score": score,
        "Risk Category": category
    }
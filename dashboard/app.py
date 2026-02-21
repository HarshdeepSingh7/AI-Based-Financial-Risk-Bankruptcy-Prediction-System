import sys
import os
import random

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
from src.risk_engine import predict_company_risk

st.set_page_config(page_title="Financial Risk AI", layout="wide")

st.title("AI-Based Financial Risk & Bankruptcy Prediction System")

st.markdown("""
This system predicts corporate bankruptcy risk using XGBoost,
trained on financial ratio data with severe class imbalance handling.
Includes SHAP-based explainability for transparent financial insights.
""")

# ------------------------------------
# LOAD MODEL ARTIFACTS
# ------------------------------------

model = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# ------------------------------------
# RISK GAUGE FUNCTION
# ------------------------------------

def risk_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Risk Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "red"},
            ],
        }
    ))
    return fig


# ------------------------------------
# LOAD DATA WITH REALISTIC NAMES
# ------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("data/bankruptcy_data.csv")
    df.columns = df.columns.str.strip()
    df = df.drop(columns=['Net Income Flag'])

    prefixes = ["Taiwan", "Formosa", "Evergreen", "Cathay", "Sino", "Global", "United", "Pacific"]
    industries = ["Semiconductor", "Electronics", "Manufacturing", "Financial", "Technology", "Industrial"]
    suffixes = ["Co., Ltd.", "Corporation", "Holdings Ltd.", "Group", "Industrial Co."]

    names = []
    for i in range(len(df)):
        name = f"{random.choice(prefixes)} {random.choice(industries)} {random.choice(suffixes)} ({i})"
        names.append(name)

    df["Company Name"] = names
    return df

df_full = load_data()

st.markdown("---")

# ------------------------------------
# COMPANY SELECTION + SHAP
# ------------------------------------

st.subheader("Company Risk & SHAP Explanation")

selected_company = st.selectbox(
    "Select Company",
    df_full["Company Name"].values
)

if st.button("Analyze Selected Company"):

    selected_row = df_full[df_full["Company Name"] == selected_company]

    # Prepare input
    input_df = selected_row.drop(columns=["Bankrupt?", "Company Name"])

    # Ensure feature order
    input_df = input_df[feature_columns]

    # Predict
    result = predict_company_risk(input_df.iloc[0].to_dict())

    st.write("### Risk Prediction")

    col1, col2, col3 = st.columns(3)
    col1.metric("Probability", result["Probability"])
    col2.metric("Risk Score", result["Risk Score"])
    col3.metric("Risk Category", result["Risk Category"])

    st.plotly_chart(risk_gauge(result["Risk Score"]))

    # ------------------------------------
    # SHAP EXPLANATION
    # ------------------------------------

    st.markdown("### SHAP Explanation (Why This Risk?)")

    # Scale
    input_scaled = scaler.transform(input_df)

    # Convert back to DataFrame to preserve feature names
    input_scaled_df = pd.DataFrame(input_scaled, columns=feature_columns)

    # Explain
    explainer = shap.Explainer(model)
    shap_values = explainer(input_scaled_df)

    # Plot waterfall
    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(plt.gcf())
    plt.close()

    st.write("### Selected Company Financial Data")
    st.dataframe(selected_row)

st.markdown("---")

# ------------------------------------
# BATCH CSV PREDICTION
# ------------------------------------

st.subheader("Batch Risk Prediction (Upload Financial CSV)")

uploaded_file = st.file_uploader("Upload Financial CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)

        results = []

        for _, row in input_df.iterrows():
            result = predict_company_risk(row.to_dict())
            results.append(result)

        input_df["Probability"] = [r["Probability"] for r in results]
        input_df["Risk Score"] = [r["Risk Score"] for r in results]
        input_df["Risk Category"] = [r["Risk Category"] for r in results]

        st.write("### Batch Prediction Results")
        st.dataframe(input_df)

        csv = input_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name="risk_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")
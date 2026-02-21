from src.risk_engine import predict_company_risk
import joblib

# Load one sample from test data
feature_columns = joblib.load("models/feature_columns.pkl")

import pandas as pd
df = pd.read_csv("data/bankruptcy_data.csv")
df.columns = df.columns.str.strip()

df = df.drop(columns=['Net Income Flag'])
X = df.drop(columns=['Bankrupt?'])

sample = X.iloc[0].to_dict()

result = predict_company_risk(sample)
print(result)
from fastapi import FastAPI
from src.risk_engine import predict_company_risk
import joblib
import pandas as pd

app = FastAPI(title="Financial Risk Prediction API")

# Load sample data once for testing
df = pd.read_csv("data/bankruptcy_data.csv")
df.columns = df.columns.str.strip()
df = df.drop(columns=['Net Income Flag'])
X = df.drop(columns=['Bankrupt?'])

@app.get("/")
def home():
    return {"message": "Financial Risk Prediction API is running"}

@app.get("/predict-sample")
def predict_sample():
    sample = X.iloc[0].to_dict()
    result = predict_company_risk(sample)
    return result


from fastapi import UploadFile, File
import pandas as pd
import io

@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    
    results = []
    
    for _, row in df.iterrows():
        result = predict_company_risk(row.to_dict())
        results.append(result)
    
    df["Probability"] = [r["Probability"] for r in results]
    df["Risk Score"] = [r["Risk Score"] for r in results]
    df["Risk Category"] = [r["Risk Category"] for r in results]
    
    return df.to_dict(orient="records")
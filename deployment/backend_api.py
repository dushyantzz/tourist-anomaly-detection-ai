"""
Python FastAPI backend for serving ML model predictions
Run locally or deploy on cloud for mobile inference connectivity.
"""
from fastapi import FastAPI, Request
import joblib
import numpy as np
from pydantic import BaseModel
import uvicorn

app = FastAPI()
lgbm = joblib.load('./deployment/model_lgbm.pkl')
xgb = joblib.load('./deployment/model_xgb.pkl')

class InputData(BaseModel):
    features: list # List of floats

@app.post('/predict/lgbm')
def predict_lgbm(data: InputData):
    arr = np.array(data.features).reshape(1, -1)
    pred = lgbm.predict(arr)
    return {'prediction': int(pred[0])}

@app.post('/predict/xgb')
def predict_xgb(data: InputData):
    arr = np.array(data.features).reshape(1, -1)
    pred = xgb.predict(arr)
    return {'prediction': int(pred[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

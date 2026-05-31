from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

from src.features.build_features import engineer_features

# ---------- Startup ----------
app = FastAPI(title="Credit Default Predictor", version="1.0.0")

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# Load assets on startup
model = None
scaler = None
model_card = None

@app.on_event("startup")
def load_assets():
    global model, scaler, model_card
    model_path = MODEL_DIR / "final_xgboost_model.joblib"
    scaler_path = MODEL_DIR / "scaler.joblib"
    card_path = MODEL_DIR / "model_card.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    if card_path.exists():
        with open(card_path) as f:
            model_card = json.load(f)
    else:
        model_card = {"model": "XGBoost", "version": "1.0.0"}

# ---------- Input Schema ----------
class CreditApplication(BaseModel):
    credit_limit: float = Field(..., ge=10000, description="Credit limit in NTD")
    sex: int = Field(..., ge=0, le=1, description="0 = male, 1 = female")
    education: int = Field(..., ge=1, le=4, description="1=graduate, 2=university, 3=high school, 4=others")
    marital_status: int = Field(..., ge=1, le=3, description="1=married, 2=single, 3=others")
    age: int = Field(..., ge=21, le=100)
    pay_status_sept: int
    pay_status_aug: int
    pay_status_jul: int
    pay_status_jun: int
    pay_status_may: int
    pay_status_apr: int
    bill_amt_sept: float
    bill_amt_aug: float
    bill_amt_jul: float
    bill_amt_jun: float
    bill_amt_may: float
    bill_amt_apr: float
    pay_amt_sept: float
    pay_amt_aug: float
    pay_amt_jul: float
    pay_amt_jun: float
    pay_amt_may: float
    pay_amt_apr: float

# ---------- Preprocessing ----------
def preprocess_input(app_input: CreditApplication) -> np.ndarray:
    """Convert validated input into scaled feature array for the model."""
    # Convert to DataFrame
    df = pd.DataFrame([app_input.model_dump()])

    # Engineer new features (same logic as in build_features.py)
    df_feat, _ = engineer_features(df, None)  # y is not needed

    # Ensure column order matches the scaler's fitting order
    expected_cols = scaler.feature_names_in_  # from sklearn 1.0+
    df_feat = df_feat[expected_cols]

    # Scale
    scaled = scaler.transform(df_feat)
    return scaled

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": model_card.get("model", "unknown"),
        "version": model_card.get("version", "unknown"),
        "roc_auc": model_card.get("test_metrics", {}).get("roc_auc", None)
    }

@app.post("/predict")
def predict(app: CreditApplication):
    try:
        features = preprocess_input(app)
        proba = model.predict_proba(features)[0, 1]
        pred = int(proba >= 0.5)  # default threshold
        return {
            "default_probability": round(float(proba), 4),
            "prediction": pred,
            "prediction_label": "Default" if pred == 1 else "No default"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
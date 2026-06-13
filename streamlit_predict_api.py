"""
streamlit_predict_api.py
========================
FastAPI microservice that serves the hypoglycemia prediction model.
Deploy on Render.com / Railway.app.

Run locally:
    pip install fastapi uvicorn scikit-learn pandas numpy
    uvicorn streamlit_predict_api:app --host 0.0.0.0 --port 8000

Render start command (must match your service):
    uvicorn streamlit_predict_api:app --host 0.0.0.0 --port $PORT
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle, os, numpy as np, pandas as pd

# ── Load model ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(_HERE, "final_model.pkl"))

MODEL_LOADED = False
model = scaler = feature_columns = None
threshold = 0.5

try:
    with open(MODEL_PATH, "rb") as f:
        model_package = pickle.load(f)
    model           = model_package["model"]
    scaler          = model_package["scaler"]
    feature_columns = model_package["feature_columns"]
    threshold       = model_package.get("recommended_threshold", model_package.get("threshold", 0.5))
    MODEL_LOADED    = True
    print(f"[INFO] Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"[WARN] Model not loaded from {MODEL_PATH}: {e}")
    MODEL_LOADED = False

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="HutanoSense Prediction API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your domain in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

class FeaturePayload(BaseModel):
    features: dict


# Render (and many uptime pingers) send HEAD requests to check liveness.
# GET-only routes return 405 on HEAD, which just clutters the logs.
@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"status": "ok", "service": "HutanoSense Prediction API", "model_loaded": MODEL_LOADED}


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok", "model_loaded": MODEL_LOADED}


@app.post("/predict")
def predict(payload: FeaturePayload):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not available")

    raw = payload.features

    # Build a DataFrame with the correct column order
    row = {}
    for col in feature_columns:
        row[col] = raw.get(col, 0)

    df = pd.DataFrame([row])

    try:
        X_scaled    = scaler.transform(df)
        probability = float(model.predict_proba(X_scaled)[0, 1])
        prediction  = 1 if probability >= threshold else 0
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if probability < 0.3:
        risk_level = "LOW"
    elif probability < 0.7:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    return {
        "prediction":  prediction,
        "probability": round(probability, 4),
        "risk_level":  risk_level,
    }

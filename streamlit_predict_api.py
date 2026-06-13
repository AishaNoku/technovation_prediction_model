"""
streamlit_predict_api.py
========================
Add this as a separate file in your Streamlit project repo (same folder as
streamlit_cgm_simulator.py). Deploy it alongside your Streamlit app using a
lightweight ASGI server, OR integrate the /predict route directly into
Streamlit via the approach below.

OPTION A – Separate FastAPI microservice (recommended for production)
---------------------------------------------------------------------
Deploy this on Render.com free tier or Railway.app free tier.
Update the $streamlitUrl in predict.php to point here.

Run locally:
    pip install fastapi uvicorn scikit-learn pandas numpy
    uvicorn streamlit_predict_api:app --host 0.0.0.0 --port 8000

OPTION B – Streamlit built-in (simpler, same Streamlit Cloud deployment)
-------------------------------------------------------------------------
Streamlit doesn't natively expose REST endpoints, so for the Streamlit Cloud
deployment at technovationpredictionmodel.streamlit.app you need Option A,
OR you can use Streamlit's query-param trick (see bottom of this file).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle, os, numpy as np, pandas as pd

# ── Load model (same pickle your Streamlit app loads) ─────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "hypoglycemia_model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model_package = pickle.load(f)
    model           = model_package["model"]
    scaler          = model_package["scaler"]
    feature_columns = model_package["feature_columns"]
    threshold       = model_package.get("threshold", 0.5)
    MODEL_LOADED    = True
except Exception as e:
    print(f"[WARN] Model not loaded: {e}")
    MODEL_LOADED = False

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="HutanoSense Prediction API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your InfinityFree domain in prod
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

class FeaturePayload(BaseModel):
    features: dict

@app.get("/health")
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


# ── OPTION B helper: Streamlit query-param trick ──────────────────────────────
# If you MUST keep everything inside one Streamlit app, add this block to
# the TOP of streamlit_cgm_simulator.py (before st.title):
#
#   import streamlit as st
#   from urllib.parse import parse_qs
#
#   params = st.query_params
#   if params.get("api") == ["predict"]:
#       import json, sys
#       payload_str = params.get("payload", ["{}"])[0]
#       features_raw = json.loads(payload_str)
#       # ... run the model exactly as in /predict above ...
#       st.write(json.dumps({"prediction": prediction,
#                            "probability": probability,
#                            "risk_level": risk_level}))
#       st.stop()
#
# Then in predict.php change the URL to:
#   $streamlitUrl = 'https://technovationpredictionmodel.streamlit.app/?api=predict&payload=' . urlencode(json_encode(['features' => $features]));
# NOTE: Streamlit Cloud adds latency and this approach is fragile; Option A is
# strongly preferred for any production use.

"""
fastapi.py
Multi-city AQI API. Exposes:
 - GET /health
 - GET /cities
 - GET /anomalies?city=<city>&min_votes=<int>&start_date=&end_date=
 - GET /predict?city=<city>&date=YYYY-MM-DD
"""

import os, glob
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import numpy as np

ART_DIR = "artifacts"
app = FastAPI(title="AQI Multi-city API")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def list_cities():
    files = glob.glob(os.path.join(ART_DIR, "ts_prepared_*.csv"))
    cities = []
    for f in files:
        base = os.path.basename(f)
        name = base.replace("ts_prepared_","").replace(".csv","")
        cities.append(name.replace("_"," ").title())
    return sorted(cities)

def load_city(city: str):
    safe = city.strip().lower().replace(" ","_")
    ts_file = os.path.join(ART_DIR, f"ts_prepared_{safe}.csv")
    model_file = os.path.join(ART_DIR, f"model_{safe}.joblib")
    scaler_file = os.path.join(ART_DIR, f"scaler_{safe}.joblib")
    if not os.path.exists(ts_file):
        raise HTTPException(404, f"No artifacts for {city}. Run model.py first.")
    df = pd.read_csv(ts_file)
    # normalize timestamp column
    ts_col = next((c for c in df.columns if c.lower() in ("timestamp","date","datetime")), None)
    if ts_col is None:
        raise HTTPException(500, "No timestamp column in CSV")
    df = df.rename(columns={ts_col:"timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.set_index("timestamp").sort_index()
    model = joblib.load(model_file) if os.path.exists(model_file) else None
    scaler = joblib.load(scaler_file) if os.path.exists(scaler_file) else None
    return df, model, scaler

@app.get("/health")
def health():
    return {"status":"ok", "available_cities": list_cities()}

@app.get("/cities")
def cities():
    return {"cities": list_cities()}

@app.get("/anomalies")
def anomalies(city: str = Query(...), min_votes: int = Query(2), start_date: str = None, end_date: str = None):
    df, model, scaler = load_city(city)
    d = df.copy()
    if start_date:
        d = d[d.index >= pd.to_datetime(start_date)]
    if end_date:
        d = d[d.index <= pd.to_datetime(end_date)]
    out = d[d["anomaly_votes"] >= min_votes][["value","anomaly_z","anomaly_iqr","anomaly_iso","anomaly_votes"]].reset_index()
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")

@app.get("/predict")
def predict(city: str = Query(...), date: str = Query(...)):
    df, model, scaler = load_city(city)
    try:
        dt = pd.to_datetime(date).normalize()
    except:
        raise HTTPException(400, "Invalid date (use YYYY-MM-DD)")

    tmp = df.copy()
    if dt not in tmp.index:
        tmp = tmp.reindex(tmp.index.union([dt])).sort_index()
        tmp["value"] = tmp["value"].interpolate(method="time", limit_direction="both")

    row = tmp.loc[dt]
    features = [
        row["value"],
        tmp["value"].rolling(7, min_periods=1).mean().loc[dt],
        tmp["value"].rolling(7, min_periods=1).std().fillna(0).loc[dt],
        dt.dayofyear,
        dt.dayofweek
    ]

    if model is None or scaler is None:
        raise HTTPException(500, "Model or scaler missing for this city")

    Xs = scaler.transform([features])
    iso_pred = bool(model.predict(Xs)[0] == -1)
    score = float(model.decision_function(Xs)[0])

    # recompute zscore/iqr for that date
    roll_mean = tmp["value"].rolling(30, min_periods=7).mean()
    roll_std  = tmp["value"].rolling(30, min_periods=7).std().replace(0, np.nan)
    z = (row["value"] - roll_mean.loc[dt]) / roll_std.loc[dt] if pd.notnull(roll_std.loc[dt]) else 0
    anomaly_z = abs(z) > 3

    roll = tmp["value"].rolling(30, min_periods=7)
    Q1 = roll.quantile(0.25)
    Q3 = roll.quantile(0.75)
    IQR = Q3 - Q1
    anomaly_iqr = (row["value"] < (Q1.loc[dt] - 1.5 * IQR.loc[dt])) or (row["value"] > (Q3.loc[dt] + 1.5 * IQR.loc[dt]))

    votes = int(anomaly_z) + int(anomaly_iqr) + int(iso_pred)
    return {
        "city": city,
        "date": dt.strftime("%Y-%m-%d"),
        "value": float(row["value"]),
        "anomaly_z": bool(anomaly_z),
        "anomaly_iqr": bool(anomaly_iqr),
        "anomaly_iso": bool(iso_pred),
        "iso_score": score,
        "anomaly_votes": votes,
        "anomaly_any": votes >= 2
    }

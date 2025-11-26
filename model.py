"""
model.py
Multi-city training script.

Reads: city_day.csv (must contain City and a date + AQI column)
Outputs per-city artifacts into ./artifacts:
 - ts_prepared_<safe_city>.csv
 - model_<safe_city>.joblib
 - scaler_<safe_city>.joblib
 - meta_<safe_city>.json
"""

import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# ---------- CONFIG ----------
DATA_PATH = "city_day.csv"          # your CSV (put in project folder)
ART_DIR = "artifacts"
MIN_DAYS = 30                       # skip city if less than this many days
CONTAMINATION = 0.03                # IsolationForest contamination param
os.makedirs(ART_DIR, exist_ok=True)

# ---------- Helpers ----------
def safe_name(s: str) -> str:
    return s.strip().lower().replace(" ", "_")

def detect_columns(df: pd.DataFrame):
    # find a date and an AQI column under common names
    date_col = next((c for c in df.columns if c.lower() in ("date","datetime","timestamp")), None)
    aqi_col = next((c for c in df.columns if c.lower() in ("aqi","value","pm2.5","pm25","aqi_value")), None)
    if date_col is None or aqi_col is None:
        raise RuntimeError(f"Could not detect date/AQI columns. Found columns: {df.columns.tolist()}")
    return date_col, aqi_col

# ---------- Load data ----------
print("Loading:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]

date_col, aqi_col = detect_columns(df)
print("Detected date col:", date_col, " AQI col:", aqi_col)

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df = df.dropna(subset=[date_col])
df["City"] = df["City"].astype(str).str.strip()

cities = sorted(df["City"].unique())
print("Found cities:", cities)

# ---------- Per-city processing ----------
for city in cities:
    print("\n=== Processing:", city, "===")
    dfc = df[df["City"] == city].copy().sort_values(date_col)

    # keep numeric AQI rows only
    dfc[aqi_col] = pd.to_numeric(dfc[aqi_col], errors="coerce")
    dfc = dfc.dropna(subset=[aqi_col])

    if len(dfc) < MIN_DAYS:
        print(f"Skipping {city}: only {len(dfc)} valid rows (need >= {MIN_DAYS})")
        continue

    # daily timeseries (reindex to daily freq & interpolate)
    dfc = dfc.set_index(date_col)[[aqi_col]].rename(columns={aqi_col: "value"})
    dfc = dfc[~dfc.index.duplicated(keep="first")]
    dfc = dfc.asfreq("D")
    dfc["value"] = dfc["value"].interpolate(method="time", limit_direction="both")
    dfc["value"] = dfc["value"].fillna(method="ffill").fillna(method="bfill")

    # Features
    ts = dfc.copy()
    ts["rolling_7d_mean"] = ts["value"].rolling(7, min_periods=1).mean()
    ts["rolling_7d_std"] = ts["value"].rolling(7, min_periods=1).std().fillna(0)
    ts["dayofyear"] = ts.index.dayofyear
    ts["dayofweek"] = ts.index.dayofweek

    # Z-score (rolling 30-day)
    roll_mean = ts["value"].rolling(30, min_periods=7).mean()
    roll_std  = ts["value"].rolling(30, min_periods=7).std().replace(0, np.nan)
    ts["zscore"] = (ts["value"] - roll_mean) / roll_std
    ts["anomaly_z"] = ts["zscore"].abs() > 3

    # Rolling IQR (30-day)
    roll = ts["value"].rolling(30, min_periods=7)
    q1 = roll.quantile(0.25)
    q3 = roll.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    ts["anomaly_iqr"] = (ts["value"] < lower) | (ts["value"] > upper)

    # ML model features & train
    features = ["value","rolling_7d_mean","rolling_7d_std","dayofyear","dayofweek"]
    X = ts[features].fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    iso = IsolationForest(n_estimators=200, contamination=CONTAMINATION, random_state=42)
    iso.fit(Xs)

    pred = iso.predict(Xs)         # -1 anomaly, 1 normal
    scores = iso.decision_function(Xs)

    ts["anomaly_iso"] = (pred == -1).astype(int)
    ts["iso_score"] = scores

    # Votes & flag
    ts["anomaly_votes"] = ts[["anomaly_z","anomaly_iqr","anomaly_iso"]].sum(axis=1).astype(int)
    ts["anomaly_any"] = (ts["anomaly_votes"] >= 2).astype(int)

    # Export artifacts
    safe = safe_name(city)
    ts_file = os.path.join(ART_DIR, f"ts_prepared_{safe}.csv")
    model_file = os.path.join(ART_DIR, f"model_{safe}.joblib")
    scaler_file = os.path.join(ART_DIR, f"scaler_{safe}.joblib")
    meta_file = os.path.join(ART_DIR, f"meta_{safe}.json")

    ts_reset = ts.reset_index().rename(columns={ts.index.name or "index":"timestamp"})
    ts_reset.to_csv(ts_file, index=False)
    joblib.dump(iso, model_file)
    joblib.dump(scaler, scaler_file)

    meta = {
        "city": city,
        "n_days": len(ts),
        "features": features,
        "ts_file": ts_file,
        "model_file": model_file,
        "scaler_file": scaler_file
    }
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved artifacts for {city}:")
    print(" ", ts_file)
    print(" ", model_file)
    print(" ", scaler_file)
    print(" ", meta_file)

print("\nTraining complete.")

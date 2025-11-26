"""
streamlit.py
Multi-city dashboard that uses FastAPI to fetch anomalies and predictions.
It also reads local artifacts/ts_prepared_<city>.csv for plotting (faster).
"""

import streamlit as st
import pandas as pd
import requests
import os
import plotly.graph_objects as go

API_BASE = st.sidebar.text_input("FastAPI URL", "http://127.0.0.1:8000")
st.set_page_config(layout="wide", page_title="Multi-City AQI Dashboard")

st.title("🌍 Multi-City AQI Monitoring Dashboard")

# get cities from API (fallback to artifacts folder)
cities = []
try:
    r = requests.get(f"{API_BASE}/cities", timeout=4)
    r.raise_for_status()
    cities = r.json().get("cities", [])
except Exception:
    # fallback: find artifact CSVs
    files = [f for f in os.listdir("artifacts") if f.startswith("ts_prepared_") and f.endswith(".csv")]
    cities = [f.replace("ts_prepared_","").replace(".csv","").replace("_"," ").title() for f in files]

if not cities:
    st.error("No cities available. Run model.py first to generate artifacts.")
    st.stop()

city = st.selectbox("Select a City", cities)

col1, col2 = st.columns([3,1])
with col2:
    min_votes = st.slider("Min anomaly votes", 1, 3, 2)
    start = st.date_input("Start date")
    end = st.date_input("End date")
    check_date = st.date_input("Single-date check")
    if st.button("Check date"):
        try:
            resp = requests.get(f"{API_BASE}/predict", params={"city": city, "date": str(check_date)}, timeout=6)
            resp.raise_for_status()
            st.json(resp.json())
        except Exception as e:
            st.error("Predict call failed: " + str(e))

with col1:
    st.subheader(f"AQI — {city}")

# load local csv (artifacts)
safe = city.lower().replace(" ", "_")
csv_local = f"artifacts/ts_prepared_{safe}.csv"
if os.path.exists(csv_local):
    df_full = pd.read_csv(csv_local)
    ts_col = next((c for c in df_full.columns if c.lower() in ("timestamp","date","datetime")), None)
    df_full = df_full.rename(columns={ts_col:"timestamp"}) if ts_col else df_full
    df_full["timestamp"] = pd.to_datetime(df_full["timestamp"], errors="coerce")
else:
    df_full = pd.DataFrame()
    st.warning(f"No local CSV for {city}. Ensure model.py created artifacts.")

# fetch anomalies via API
anoms = pd.DataFrame()
try:
    r = requests.get(f"{API_BASE}/anomalies", params={"city": city, "min_votes": min_votes, "start_date": str(start), "end_date": str(end)}, timeout=6)
    r.raise_for_status()
    anoms = pd.DataFrame(r.json())
    if not anoms.empty:
        anoms["timestamp"] = pd.to_datetime(anoms["timestamp"])
except Exception:
    st.warning("Could not fetch anomalies from API (make sure FastAPI is running).")

# Plot
fig = go.Figure()
if not df_full.empty:
    fig.add_trace(go.Scatter(x=df_full["timestamp"], y=df_full["value"], mode="lines", name="AQI"))
if not anoms.empty:
    fig.add_trace(go.Scatter(x=anoms["timestamp"], y=anoms["value"], mode="markers", name="Anomalies", marker=dict(color="red", size=8)))
fig.update_layout(height=450, xaxis_title="Date", yaxis_title="AQI", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Anomaly table")
if anoms.empty:
    st.info("No anomalies found.")
else:
    st.dataframe(anoms)

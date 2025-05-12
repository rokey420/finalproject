import streamlit as st
import pandas as pd

@st.cache_data(ttl=3600)
def load_metrics():
    return pd.read_csv("metrics.csv")

st.title("📊 Model Monitoring Dashboard")

try:
    metrics = load_metrics()
    station = st.selectbox("Filter by station", metrics["station"].unique())
    filtered = metrics[metrics["station"] == station]
    st.write("### Evaluation Metrics")
    st.dataframe(filtered)

    best = filtered.sort_values("MAE").iloc[0]
    st.success(f"✅ Best Model: {best['model']} (MAE: {best['MAE']:.2f}, MAPE: {best['MAPE']:.2f}%)")
except Exception:
    st.warning("⚠️ No metrics.csv found. Please add model evaluation logging.")

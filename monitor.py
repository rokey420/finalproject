import streamlit as st
import pandas as pd

@st.cache_data(ttl=3600)
def load_metrics():
    return pd.read_csv("metrics.csv")  # Ensure this file exists with columns: model, location_id, MAE, MAPE

metrics = load_metrics()

st.title("📊 Citi Bike Model Monitoring")
location = st.selectbox("Select Location", metrics["location_id"].unique())
filtered = metrics[metrics["location_id"] == location]

st.write("### Model Evaluation")
st.dataframe(filtered)

best_model = filtered.sort_values("MAE").iloc[0]
st.success(f"✅ Best Model: {best_model['model']} with MAE: {best_model['MAE']:.2f}, MAPE: {best_model['MAPE']:.2f}%")

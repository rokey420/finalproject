import streamlit as st
import pandas as pd
import plotly.express as px
import os
import glob

@st.cache_data(ttl=3600)
def load_predictions():
    # Automatically find latest forecast file
    files = glob.glob("predictions/citibike_forecast_*.csv")
    if not files:
        st.error("❌ No prediction files found.")
        return pd.DataFrame()
    latest = max(files, key=os.path.getctime)
    df = pd.read_csv(latest)
    return df

st.title("🚲 CitiBike Trip Forecast Viewer")

df = load_predictions()
if not df.empty:
    location = st.selectbox("Select a station", sorted(df["station"].unique()))
    forecast = df[df["station"] == location]

    fig = px.line(forecast, x="date", y="predicted_trips", title=f"📈 Predicted Trips for {location}")
    st.plotly_chart(fig)
else:
    st.warning("Upload a prediction file or re-run the inference pipeline.")

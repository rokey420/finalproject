import streamlit as st
import pandas as pd
import plotly.express as px

# Load prediction results
@st.cache_data(ttl=3600)
def load_predictions():
    return pd.read_csv("predictions.csv")  # Ensure this file exists with columns: timestamp, location_id, y_pred, y_true

df = load_predictions()

st.title("🚲 Citi Bike Trip Predictions")
location = st.selectbox("Select Location", df["location_id"].unique())

filtered = df[df["location_id"] == location]

# Line plot: actual vs predicted
fig = px.line(
    filtered,
    x="timestamp",
    y=["y_true", "y_pred"],
    labels={"value": "Trip Count", "timestamp": "Time"},
    title=f"Actual vs Predicted Trips - Location {location}",
)
st.plotly_chart(fig)

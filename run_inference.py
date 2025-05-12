#!/usr/bin/env python

import pandas as pd
import numpy as np
import hopsworks
import joblib
import os
import logging
import argparse
import subprocess
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Parse CLI
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--forecast-days', type=int, default=7)
    return parser.parse_args()

# Connect to Hopsworks
def connect_to_hopsworks():
    try:
        project = hopsworks.login(
            project="tejeshk1",
            api_key_value=os.environ["HOPSWORKS_API_KEY"]
        )
        logger.info(f"Connected to Hopsworks project: {project.name}")
        return project
    except Exception as e:
        logger.error(f"Hopsworks connection failed: {e}")
        return None

# Load data
def load_latest_data(project):
    try:
        fs = project.get_feature_store()
        fg = fs.get_feature_group(name="citibike_ts_features", version=1)
        df = fg.select_all().read()
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

# Load model
def load_model_from_registry(project):
    try:
        mr = project.get_model_registry()
        model = mr.get_model(name="citibike_trip_predictor", version=1)
        model_dir = model.download()
        predictor = joblib.load(os.path.join(model_dir, "model.pkl"))
        feature_info = joblib.load(os.path.join(model_dir, "feature_info.joblib"))
        return predictor, feature_info
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None

# Prepare features
def prepare_latest_features(df):
    latest_features = {}
    stations = df['start_station_name'].unique()
    for station in stations:
        station_data = df[df['start_station_name'] == station].sort_values('date')
        latest_features[station] = station_data.iloc[-1].to_dict()
    return latest_features

# Forecast dates
def generate_forecast_dates(days):
    today = datetime.now().date()
    return [today + timedelta(days=i) for i in range(1, days + 1)]

# Predict
def make_predictions(predictor, latest_features, forecast_dates, feature_info=None):
    predictions = {}
    for station, latest in latest_features.items():
        station_preds = []
        for i, date in enumerate(forecast_dates):
            features = {}
            for j in range(1, 8):
                features[f'trip_count_lag_{j}'] = latest.get(f'trip_count_lag_{j}', 0)
                features[f'avg_duration_lag_{j}'] = latest.get(f'avg_duration_lag_{j}', 0)
            for window in [3, 7, 14]:
                features[f'trip_count_rolling_{window}'] = latest.get(f'trip_count_rolling_{window}', 0)
                features[f'avg_duration_rolling_{window}'] = latest.get(f'avg_duration_rolling_{window}', 0)
            features['day_of_week'] = date.weekday()
            features['is_weekend'] = 1 if date.weekday() >= 5 else 0
            features['month'] = date.month
            df = pd.DataFrame([features])
            if feature_info and 'features' in feature_info:
                for col in feature_info['features']:
                    if col not in df:
                        df[col] = 0
                df = df[feature_info['features']]
            y_pred = predictor.predict(df)[0]
            station_preds.append({'date': date, 'prediction': max(0, round(y_pred))})
        predictions[station] = station_preds
    return predictions

# Save CSV + Plot
def save_predictions(predictions, forecast_days):
    os.makedirs("predictions", exist_ok=True)
    rows = []
    for station, preds in predictions.items():
        for entry in preds:
            rows.append({
                "station": station,
                "date": entry["date"],
                "predicted_trips": entry["prediction"]
            })
    df = pd.DataFrame(rows)
    date_str = datetime.now().strftime('%Y%m%d')
    csv_path = f"predictions/citibike_forecast_{date_str}_{forecast_days}days.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")
    return df, csv_path

# GitHub Commit
def commit_predictions_to_github(filepath):
    try:
        subprocess.run(["git", "config", "--global", "user.name", "github-actions"], check=True)
        subprocess.run(["git", "config", "--global", "user.email", "actions@github.com"], check=True)
        subprocess.run(["git", "add", filepath], check=True)
        subprocess.run(["git", "commit", "-m", f"Add prediction file {filepath}"], check=True)
        subprocess.run(["git", "push"], check=True)
        logger.info(f"Pushed {filepath} to GitHub.")
    except Exception as e:
        logger.error(f"GitHub push failed: {e}")

# Main
def main():
    args = parse_arguments()
    forecast_days = args.forecast_days
    project = connect_to_hopsworks()
    if not project: return
    df = load_latest_data(project)
    if df is None: return
    model, feature_info = load_model_from_registry(project)
    if model is None: return
    features = prepare_latest_features(df)
    forecast_dates = generate_forecast_dates(forecast_days)
    predictions = make_predictions(model, features, forecast_dates, feature_info)
    pred_df, csv_path = save_predictions(predictions, forecast_days)
    commit_predictions_to_github(csv_path)

if __name__ == "__main__":
    main()

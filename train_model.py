#!/usr/bin/env python
"""
CitiBike Model Training + Hopsworks Registration + Save to GitHub
"""

import os
import logging
import pandas as pd
import numpy as np
import joblib
import hopsworks
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting CitiBike model training and registration...")

    try:
        # Step 1: Simulate training data (replace with actual feature store fetch in future)
        X = pd.DataFrame(np.random.rand(100, 10), columns=[f"f{i}" for i in range(10)])
        y = np.random.rand(100)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 2: Train a RandomForest model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        logger.info(f"Model trained. MAE = {mae:.4f}")

        # Step 3: Save model and metadata locally
        os.makedirs("model_artifacts", exist_ok=True)
        model_path = "model_artifacts/model.pkl"
        features_path = "model_artifacts/feature_info.joblib"

        joblib.dump(model, model_path)
        joblib.dump({"features": list(X_train.columns)}, features_path)

        logger.info(f"Model saved to {model_path}")
        logger.info(f"Feature info saved to {features_path}")

        # Step 4: Register model in Hopsworks
        logger.info("Connecting to Hopsworks...")
        project = hopsworks.login(project="tejeshk1", api_key_value=os.environ["HOPSWORKS_API_KEY"])
        mr = project.get_model_registry()

        logger.info("Registering model in Hopsworks...")
        model_instance = mr.python.create_model(
            name="citibike_trip_predictor",
            model_dir="model_artifacts",
            description="RandomForest model to predict CitiBike demand",
            input_example=X_train.head(1),
            model_type="sklearn",
            metrics={"mae": mae}
        )

        logger.info("Model successfully registered in Hopsworks")

    except Exception as e:
        logger.exception(f" Error during model training or registration: {e}")

if __name__ == "__main__":
    main()

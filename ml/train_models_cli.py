"""
Climate AI - Training and Dataset Generation CLI
Uses pure-Python Simple ML Engine to train:
1. Environmental Condition Predictor (Ridge Regression with interaction terms)
2. 7-Day Time-Series Weather Forecaster
Generates historical 30-day datasets for global locations.
"""

import os
import json
import csv
from datetime import datetime
from simple_ml import (
    RidgeRegressionModel,
    TimeSeriesWeatherForecaster,
    WeatherDatasetGenerator
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ML_DIR = os.path.join(BASE_DIR, 'ml')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ML_DIR, exist_ok=True)

CITIES = [
    {"id": "delhi", "name": "New Delhi, India", "base_temp": 28.0, "climate": "Subtropical Continental"},
    {"id": "mumbai", "name": "Mumbai, India", "base_temp": 30.0, "climate": "Tropical Coastal"},
    {"id": "london", "name": "London, UK", "base_temp": 14.0, "climate": "Temperate Oceanic"},
    {"id": "newyork", "name": "New York, USA", "base_temp": 18.0, "climate": "Humid Continental"},
    {"id": "tokyo", "name": "Tokyo, Japan", "base_temp": 20.0, "climate": "Humid Subtropical"},
    {"id": "bengaluru", "name": "Bengaluru, India", "base_temp": 24.0, "climate": "Tropical Savanna"}
]

def run():
    print("==================================================")
    print("🌿 Climate AI - Machine Learning Pipeline Training")
    print("==================================================")

    # 1. Generate multi-year training dataset (1000 days)
    print("\n[1/4] Generating 1,000-day meteorological training observations...")
    training_data = WeatherDatasetGenerator.generate_daily_series(days=1000, base_temp=25.0, seed=101)

    # Save to CSV
    csv_path = os.path.join(DATA_DIR, 'weather_dataset.csv')
    fieldnames = list(training_data[0].keys())
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(training_data)
    print(f" Saved full dataset to: {csv_path}")

    # 2. Train Environmental Condition Model
    print("\n[2/4] Training Environmental Temperature Predictor (Ridge Regularized)...")
    # Features: temp, humidity, wind_speed, rainfall, pressure, plus physical interaction terms
    X_env = []
    y_env = []
    feature_names = [
        "Base Temperature",
        "Relative Humidity",
        "Wind Speed",
        "Rainfall",
        "Atmospheric Pressure",
        "Temp-Humidity Interaction",
        "Pressure Anomaly"
    ]

    for row in training_data:
        t = row["temp_avg"]
        h = row["humidity"]
        w = row["wind_speed"]
        r = row["rainfall"]
        p = row["pressure"]
        # Interaction feature: higher humidity amplifies perceived temperature
        th_inter = (t * h) / 100.0
        # Pressure anomaly from sea level standard
        p_anom = p - 1013.25

        X_env.append([t, h, w, r, p, th_inter, p_anom])
        y_env.append(row["apparent_temp"])

    env_model = RidgeRegressionModel(alpha=0.8)
    env_model.fit(X_env, y_env, feature_names)
    env_eval = env_model.evaluate(X_env, y_env)

    print(f"  ✓ Model R² Accuracy: {env_eval['r2_score'] * 100:.2f}%")
    print(f"  ✓ Mean Absolute Error (MAE): {env_eval['mae']:.3f} °C")
    print(f"  ✓ Root Mean Squared Error (RMSE): {env_eval['rmse']:.3f} °C")
    print("  ✓ Feature Importances:")
    for feat in env_eval["feature_importance"]:
        print(f"    - {feat['feature']:<26}: {feat['importance']:>5.1f}% (weight: {feat['weight']:+.4f})")

    env_model_path = os.path.join(ML_DIR, 'environmental_model.json')
    with open(env_model_path, 'w', encoding='utf-8') as f:
        json.dump(env_model.to_dict(), f, indent=2)
    print(f" Saved model to: {env_model_path}")

    # 3. Train 7-Day Time Series Forecaster
    print("\n[3/4] Training 7-Day Time-Series Weather Forecaster...")
    forecaster = TimeSeriesWeatherForecaster(horizon=7)
    ts_eval = forecaster.fit(training_data)

    print(f"  ✓ Multi-horizon Average R²: {ts_eval['average_r2'] * 100:.2f}%")
    print(f"  ✓ Forecast MAE:            {ts_eval['average_mae']:.3f} °C")
    print(f"  ✓ Forecast RMSE:           {ts_eval['average_rmse']:.3f} °C")

    forecaster_path = os.path.join(ML_DIR, 'forecaster_model.json')
    with open(forecaster_path, 'w', encoding='utf-8') as f:
        json.dump(forecaster.to_dict(), f, indent=2)
    print(f" Saved forecaster to: {forecaster_path}")

    # 4. Generate 30-day historical trends for each city
    print("\n[4/4] Generating 30-day realistic historical weather trends for all stations...")
    city_trends = {}
    for city in CITIES:
        city_history = WeatherDatasetGenerator.generate_daily_series(
            days=30,
            city=city["name"],
            base_temp=city["base_temp"],
            seed=sum(ord(c) for c in city["id"])
        )
        city_trends[city["id"]] = city_history

    trends_path = os.path.join(DATA_DIR, 'weather_history_cities.json')
    with open(trends_path, 'w', encoding='utf-8') as f:
        json.dump(city_trends, f, indent=2)

    # Also save default 30-day trend
    default_trend_path = os.path.join(DATA_DIR, 'weather_history_30days.json')
    with open(default_trend_path, 'w', encoding='utf-8') as f:
        json.dump(city_trends["delhi"], f, indent=2)
    print(f" Saved 30-day trends to: {trends_path}")

    # Save metadata & metrics
    metrics_summary = {
        "project": "Climate AI - AI-Based Temperature Prediction for Environmental Monitoring",
        "type": "Environmental Studies (EVS) Machine Learning Project",
        "trained_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_samples": len(training_data),
        "environmental_model": {
            "name": "Atmospheric Ridge Regression with Interaction Terms",
            "r2_score": env_eval["r2_score"],
            "accuracy_percent": round(env_eval["r2_score"] * 100, 2),
            "mae": env_eval["mae"],
            "rmse": env_eval["rmse"],
            "feature_importance": env_eval["feature_importance"]
        },
        "timeseries_forecaster": {
            "name": "7-Day Multi-Horizon Autoregressive Harmonic Forecaster",
            "horizon_days": 7,
            "average_r2": ts_eval["average_r2"],
            "accuracy_percent": round(ts_eval["average_r2"] * 100, 2),
            "mae": ts_eval["average_mae"],
            "rmse": ts_eval["average_rmse"],
            "day_metrics": ts_eval["day_metrics"]
        },
        "cities": CITIES
    }

    metrics_path = os.path.join(ML_DIR, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f" Saved comprehensive metrics to: {metrics_path}")

    print("\n All Models Trained and Saved Successfully!")

if __name__ == "__main__":
    run()

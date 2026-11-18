"""
Climate AI - Machine Learning Model Training Script
Trains:
1. Environmental Condition Temperature Predictor (Random Forest Regressor)
2. 7-Day Time-Series Weather Forecaster using Lag Features & Trend Dynamics
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ML_DIR = os.path.join(BASE_DIR, 'ml')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ML_DIR, exist_ok=True)

def generate_synthetic_climate_data(n_days=1095):
    """
    Generate 3 years of daily realistic meteorological data
    with seasonal variation, physical correlations, and realistic noise.
    """
    np.random.seed(42)
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_days)]

    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    
    # Seasonal temperature oscillation (Base 24°C, amplitude 10°C, peak around day 170 / June)
    seasonal_temp = 24.0 + 9.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    
    # Generate persistent temperature series (AR-1 process)
    temps = np.zeros(n_days)
    current_temp = seasonal_temp[0]
    for i in range(n_days):
        noise = np.random.normal(0, 1.5)
        # Reversion toward seasonal mean
        current_temp = 0.85 * current_temp + 0.15 * seasonal_temp[i] + noise
        temps[i] = current_temp

    # Atmospheric Pressure (typically 1005 - 1020 hPa, lower during hot stormy weather)
    pressure = 1013.25 - 0.25 * (temps - 24.0) + np.random.normal(0, 3.0, n_days)

    # Relative Humidity (%) - inversely correlated with temp, higher in monsoon/winter
    base_humidity = 60.0 - 0.8 * (temps - 24.0) + 15.0 * np.cos(2 * np.pi * (day_of_year - 200) / 365.25)
    humidity = np.clip(base_humidity + np.random.normal(0, 8.0, n_days), 20.0, 98.0)

    # Rainfall (mm) - higher probability when humidity > 70 and pressure < 1010
    rain_prob = 1.0 / (1.0 + np.exp(-0.15 * (humidity - 70) + 0.1 * (pressure - 1010)))
    is_rain = np.random.rand(n_days) < rain_prob
    rainfall = np.where(is_rain, np.random.exponential(8.0, n_days) * (humidity / 50.0), 0.0)
    rainfall = np.round(np.clip(rainfall, 0.0, 120.0), 1)

    # Wind speed (km/h)
    wind_speed = np.abs(np.random.normal(12.0, 5.0, n_days) + 0.1 * np.abs(pressure - 1013))
    wind_speed = np.round(np.clip(wind_speed, 2.0, 65.0), 1)

    # Min and Max daily temperatures derived from daily average
    diurnal_range = 8.0 + 4.0 * (1 - humidity / 100.0) + np.random.normal(0, 1.0, n_days)
    diurnal_range = np.clip(diurnal_range, 3.0, 15.0)
    temp_max = np.round(temps + diurnal_range * 0.55, 1)
    temp_min = np.round(temps - diurnal_range * 0.45, 1)
    temp_avg = np.round(temps, 1)

    # Heat Index / Apparent feel temperature
    # Rothfusz heat index approximation
    heat_index = (
        -8.78469475556
        + 1.61139411 * temp_avg
        + 2.33854883889 * (humidity / 100.0 * 20.0)
        - 0.14611605 * temp_avg * (humidity / 100.0 * 20.0)
        - 0.012308094 * (temp_avg ** 2)
        - 0.0164248277778 * ((humidity / 100.0 * 20.0) ** 2)
        + 0.002211732 * (temp_avg ** 2) * (humidity / 100.0 * 20.0)
        + 0.00072546 * temp_avg * ((humidity / 100.0 * 20.0) ** 2)
        - 0.000003582 * (temp_avg ** 2) * ((humidity / 100.0 * 20.0) ** 2)
    )
    # Wind chill adjustment
    wind_factor = 0.15 * (wind_speed - 10.0)
    target_apparent_temp = np.round(0.7 * temp_avg + 0.3 * heat_index - wind_factor, 1)

    df = pd.DataFrame({
        'date': [d.strftime('%Y-%m-%d') for d in dates],
        'temp_avg': temp_avg,
        'temp_min': temp_min,
        'temp_max': temp_max,
        'humidity': np.round(humidity, 1),
        'pressure': np.round(pressure, 1),
        'wind_speed': wind_speed,
        'rainfall': rainfall,
        'apparent_temp': target_apparent_temp
    })

    return df

def train_models():
    print("Generating comprehensive environmental dataset...")
    df = generate_synthetic_climate_data(n_days=1095)
    csv_path = os.path.join(DATA_DIR, 'weather_dataset.csv')
    df.to_csv(csv_path, index=False)
    print(f"Dataset saved to {csv_path} ({len(df)} records).")

    # ==========================================
    # 1. Environmental Prediction Model
    # Predicts expected apparent/adjusted temperature
    # Features: temperature, humidity, windSpeed, rainfall, pressure
    # ==========================================
    print("\nTraining Environmental Prediction Model (Random Forest Regressor)...")
    X_env = df[['temp_avg', 'humidity', 'wind_speed', 'rainfall', 'pressure']]
    y_env = df['apparent_temp']

    X_train, X_test, y_train, y_test = train_test_split(X_env, y_env, test_size=0.2, random_state=42)

    env_model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    env_model.fit(X_train, y_train)

    y_pred_env = env_model.predict(X_test)
    env_r2 = float(r2_score(y_test, y_pred_env))
    env_mae = float(mean_absolute_error(y_test, y_pred_env))
    env_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_env)))

    feature_names = ['Temperature', 'Humidity', 'Wind Speed', 'Rainfall', 'Pressure']
    importances = [float(x) for x in env_model.feature_importances_]
    feature_importance = [
        {"feature": name, "importance": round(imp * 100, 2)}
        for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    ]

    print(f"Environmental Model Metrics:")
    print(f"  R² Score: {env_r2:.4f}")
    print(f"  MAE:      {env_mae:.4f} °C")
    print(f"  RMSE:     {env_rmse:.4f} °C")

    # Save Environmental Model
    env_model_path = os.path.join(ML_DIR, 'temperature_model.joblib')
    joblib.dump(env_model, env_model_path)
    print(f"Saved environmental model to {env_model_path}")

    # ==========================================
    # 2. 7-Day Time-Series Weather Forecaster
    # Uses 7 lag days of temp, humidity, pressure + rolling features
    # to predict temperature trajectories for the next 7 days.
    # ==========================================
    print("\nTraining 7-Day Time-Series Forecaster...")
    
    # Create lag features
    ts_df = df.copy()
    for lag in range(1, 8):
        ts_df[f'temp_lag_{lag}'] = ts_df['temp_avg'].shift(lag)
        ts_df[f'humidity_lag_{lag}'] = ts_df['humidity'].shift(lag)
        ts_df[f'pressure_lag_{lag}'] = ts_df['pressure'].shift(lag)
    
    # Rolling averages over the past 7 days
    ts_df['temp_rolling_mean_7d'] = ts_df['temp_avg'].shift(1).rolling(7).mean()
    ts_df['temp_rolling_std_7d'] = ts_df['temp_avg'].shift(1).rolling(7).std().fillna(1.0)
    
    # Seasonal harmonics
    doy = pd.to_datetime(ts_df['date']).dt.dayofyear
    ts_df['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
    ts_df['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)

    # 7-day forward targets (t+1 to t+7)
    for lead in range(1, 8):
        ts_df[f'temp_lead_{lead}'] = ts_df['temp_avg'].shift(-lead)

    ts_df = ts_df.dropna().reset_index(drop=True)

    feature_cols = [
        f'temp_lag_{i}' for i in range(1, 8)
    ] + [
        f'humidity_lag_{i}' for i in range(1, 4)
    ] + [
        f'pressure_lag_{i}' for i in range(1, 4)
    ] + ['temp_rolling_mean_7d', 'temp_rolling_std_7d', 'doy_sin', 'doy_cos']

    target_cols = [f'temp_lead_{i}' for i in range(1, 8)]

    X_ts = ts_df[feature_cols]
    y_ts = ts_df[target_cols]

    X_ts_train, X_ts_test, y_ts_train, y_ts_test = train_test_split(X_ts, y_ts, test_size=0.2, random_state=42)

    forecaster = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    forecaster.fit(X_ts_train, y_ts_train)

    y_ts_pred = forecaster.predict(X_ts_test)
    ts_r2 = float(r2_score(y_ts_test, y_ts_pred))
    ts_mae = float(mean_absolute_error(y_ts_test, y_ts_pred))
    ts_rmse = float(np.sqrt(mean_squared_error(y_ts_test, y_ts_pred)))

    print(f"7-Day Forecaster Metrics:")
    print(f"  Multi-Step R² Score: {ts_r2:.4f}")
    print(f"  MAE:                 {ts_mae:.4f} °C")
    print(f"  RMSE:                {ts_rmse:.4f} °C")

    forecaster_path = os.path.join(ML_DIR, 'forecaster_model.joblib')
    joblib.dump({
        'model': forecaster,
        'feature_cols': feature_cols,
        'target_cols': target_cols
    }, forecaster_path)
    print(f"Saved 7-day forecaster to {forecaster_path}")

    # ==========================================
    # 3. Save 30-Day Recent Trend Sample Data
    # For instant zero-latency loading and historical graphs
    # ==========================================
    recent_30 = df.tail(30).to_dict(orient='records')
    # Determine realistic weather conditions
    for row in recent_30:
        if row['rainfall'] > 5.0:
            row['condition'] = 'Rainy'
            row['icon'] = 'cloud-rain'
        elif row['rainfall'] > 0.0:
            row['condition'] = 'Drizzle'
            row['icon'] = 'cloud-drizzle'
        elif row['humidity'] > 75:
            row['condition'] = 'Cloudy'
            row['icon'] = 'cloud'
        elif row['wind_speed'] > 25:
            row['condition'] = 'Windy'
            row['icon'] = 'wind'
        elif row['humidity'] > 50:
            row['condition'] = 'Partly Cloudy'
            row['icon'] = 'cloud-sun'
        else:
            row['condition'] = 'Sunny'
            row['icon'] = 'sun'

    trend_path = os.path.join(DATA_DIR, 'weather_history_30days.json')
    with open(trend_path, 'w', encoding='utf-8') as f:
        json.dump(recent_30, f, indent=2)
    print(f"Saved recent 30-day historical trend to {trend_path}")

    # Save summary metrics
    metrics_data = {
        'model_name': 'Climate AI - Multi-Stage Random Forest Regressor',
        'algorithm': 'Random Forest Ensemble with Time-Series Lag & Meteorological Feature Engineering',
        'trained_at': datetime.now().isoformat(),
        'dataset_size': len(df),
        'environmental_model': {
            'r2_score': round(env_r2, 4),
            'mae': round(env_mae, 4),
            'rmse': round(env_rmse, 4),
            'feature_importance': feature_importance
        },
        'forecaster_model': {
            'r2_score': round(ts_r2, 4),
            'mae': round(ts_mae, 4),
            'rmse': round(ts_rmse, 4),
            'horizon_days': 7
        }
    }

    metrics_path = os.path.join(ML_DIR, 'metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved metrics to {metrics_path}")
    print("\n Model Training Completed Successfully!")

if __name__ == '__main__':
    train_models()

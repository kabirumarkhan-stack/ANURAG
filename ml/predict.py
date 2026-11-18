"""
Climate AI - Prediction Service
Can be executed via CLI or imported into API services.
"""

import os
import sys
import json
import argparse
from simple_ml import RidgeRegressionModel, TimeSeriesWeatherForecaster

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DIR = os.path.join(BASE_DIR, 'ml')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Cache loaded models
_ENV_MODEL = None
_FORECASTER_MODEL = None
_CITIES_DATA = None

def get_env_model():
    global _ENV_MODEL
    if _ENV_MODEL is None:
        path = os.path.join(ML_DIR, 'environmental_model.json')
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        _ENV_MODEL = RidgeRegressionModel.from_dict(data)
    return _ENV_MODEL

def get_forecaster_model():
    global _FORECASTER_MODEL
    if _FORECASTER_MODEL is None:
        path = os.path.join(ML_DIR, 'forecaster_model.json')
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        _FORECASTER_MODEL = TimeSeriesWeatherForecaster.from_dict(data)
    return _FORECASTER_MODEL

def get_city_trends():
    global _CITIES_DATA
    if _CITIES_DATA is None:
        path = os.path.join(DATA_DIR, 'weather_history_cities.json')
        with open(path, 'r', encoding='utf-8') as f:
            _CITIES_DATA = json.load(f)
    return _CITIES_DATA

def predict_temperature(temperature, humidity, wind_speed, rainfall, pressure):
    """
    Predicts perceived / adjusted temperature and provides environmental insights.
    """
    model = get_env_model()
    t = float(temperature)
    h = float(humidity)
    w = float(wind_speed)
    r = float(rainfall)
    p = float(pressure)

    th_inter = (t * h) / 100.0
    p_anom = p - 1013.25
    features = [t, h, w, r, p, th_inter, p_anom]

    predicted_temp = round(model.predict_one(features), 1)

    # Thermal Comfort Category
    if predicted_temp < 10.0:
        comfort = "Cold (Risk of Hypothermia/Discomfort)"
        comfort_level = "Cold"
        comfort_badge = "cold"
    elif predicted_temp < 18.0:
        comfort = "Cool (Crisp & Mild)"
        comfort_level = "Cool"
        comfort_badge = "cool"
    elif predicted_temp <= 26.0:
        comfort = "Optimal / Pleasant Comfort Zone"
        comfort_level = "Pleasant"
        comfort_badge = "optimal"
    elif predicted_temp <= 32.0:
        comfort = "Warm (Elevated Transpiration)"
        comfort_level = "Warm"
        comfort_badge = "warm"
    elif predicted_temp <= 39.0:
        comfort = "Hot (Heat Caution - Stay Hydrated)"
        comfort_level = "Hot"
        comfort_badge = "hot"
    else:
        comfort = "Extreme Danger / Severe Heat Alert"
        comfort_level = "Extreme"
        comfort_badge = "extreme"

    # Environmental / EVS Insights
    insights = []
    if h > 75 and t > 28:
        insights.append("High humidity restricts evaporative cooling; high heat index caution.")
    if p < 1005:
        insights.append("Low atmospheric barometric pressure indicates storm or cyclonic activity.")
    elif p > 1020:
        insights.append("High pressure system suggests stable, clear atmospheric conditions.")
    if w > 30:
        insights.append("Gale force winds will increase soil erosion and heat dissipation.")
    if r > 15:
        insights.append("Substantial precipitation recorded; surface runoff and aquifer recharge active.")

    if not insights:
        insights.append("Atmospheric indicators are within stable baseline seasonal expectations.")

    return {
        "status": "success",
        "input": {
            "temperature": t,
            "humidity": h,
            "windSpeed": w,
            "rainfall": r,
            "pressure": p
        },
        "predictedTemperature": predicted_temp,
        "feelsLikeDelta": round(predicted_temp - t, 1),
        "thermalComfort": comfort,
        "thermalComfortLevel": comfort_level,
        "comfortBadge": comfort_badge,
        "environmentalInsights": insights,
        "modelUsed": "Climate AI - Ridge Atmospheric Ensemble v1.0"
    }

def predict_7day_forecast(city_id="delhi"):
    """
    Generates 7-day future predictions using the trained time-series forecaster.
    """
    forecaster = get_forecaster_model()
    cities = get_city_trends()
    history = cities.get(city_id) or cities.get("delhi")

    forecast = forecaster.forecast_7days(history)
    return {
        "city": city_id,
        "forecast_days": len(forecast),
        "generated_at": history[-1]["date"],
        "forecast": forecast
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Climate AI Temperature Prediction CLI")
    parser.add_argument("--temperature", type=float, default=28.0, help="Temperature in °C")
    parser.add_argument("--humidity", type=float, default=65.0, help="Humidity percentage (0-100)")
    parser.add_argument("--wind", type=float, default=12.0, help="Wind speed in km/h")
    parser.add_argument("--rain", type=float, default=0.0, help="Rainfall in mm")
    parser.add_argument("--pressure", type=float, default=1013.0, help="Atmospheric pressure in hPa")
    parser.add_argument("--forecast", action="store_true", help="Generate 7-day forecast")
    parser.add_argument("--city", type=str, default="delhi", help="City code for 7-day forecast")

    args = parser.parse_args()

    if args.forecast:
        result = predict_7day_forecast(args.city)
        print(json.dumps(result, indent=2))
    else:
        result = predict_temperature(args.temperature, args.humidity, args.wind, args.rain, args.pressure)
        print(json.dumps(result, indent=2))

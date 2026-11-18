"""
Climate AI - Live Weather API Service
Supports:
1. Open-Meteo API (100% Free, No API Key Required)
2. OpenWeatherMap API (Uses OPENWEATHER_API_KEY if configured in environment)
3. Automatic graceful fallback to local dataset and trained ML forecaster
"""

import os
import json
import urllib.request
import urllib.error
from datetime import datetime

CITY_COORDINATES = {
    "delhi": {"lat": 28.6139, "lon": 77.2090, "name": "New Delhi, India"},
    "mumbai": {"lat": 19.0760, "lon": 72.8777, "name": "Mumbai, India"},
    "bengaluru": {"lat": 12.9716, "lon": 77.5946, "name": "Bengaluru, India"},
    "london": {"lat": 51.5074, "lon": -0.1278, "name": "London, UK"},
    "newyork": {"lat": 40.7128, "lon": -74.0060, "name": "New York, USA"},
    "tokyo": {"lat": 35.6762, "lon": 139.6503, "name": "Tokyo, Japan"}
}

def fetch_live_open_meteo(city_id="delhi"):
    """
    Fetches real-time 30-day historical trend and 7-day forecast from Open-Meteo.
    Requires ZERO API Key.
    """
    city_info = CITY_COORDINATES.get(city_id.lower(), CITY_COORDINATES["delhi"])
    lat = city_info["lat"]
    lon = city_info["lon"]

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&"
        f"daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
        f"precipitation_sum,relative_humidity_2m_mean,surface_pressure_mean,wind_speed_10m_max&"
        f"past_days=30&forecast_days=7&timezone=auto"
    )

    req = urllib.request.Request(url, headers={"User-Agent": "ClimateAI/1.0"})
    with urllib.request.urlopen(req, timeout=5) as response:
        if response.status == 200:
            return json.loads(response.read().decode('utf-8'))
    return None

def fetch_live_openweathermap(city_id="delhi"):
    """
    Fetches current weather from OpenWeatherMap if OPENWEATHER_API_KEY is set.
    """
    api_key = os.environ.get("OPENWEATHER_API_KEY", "").strip()
    if not api_key:
        return None

    city_name = CITY_COORDINATES.get(city_id.lower(), {}).get("name", city_id).split(",")[0]
    encoded_city = urllib.parse.quote(city_name)
    url = f"https://api.openweathermap.org/data/2.5/weather?q={encoded_city}&appid={api_key}&units=metric"

    req = urllib.request.Request(url, headers={"User-Agent": "ClimateAI/1.0"})
    with urllib.request.urlopen(req, timeout=5) as response:
        if response.status == 200:
            return json.loads(response.read().decode('utf-8'))
    return None

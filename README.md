# 🌿 Climate AI — Weather Trend Analysis & 7-Day AI Prediction

Climate AI is an environmental monitoring and predictive meteorological web application developed as part of an Environmental Studies (EVS) project. It provides a modern dashboard for exploring environmental conditions, visualizing **30-day historical weather trends**, predicting **future 7-day weather forecasts using a Machine Learning model in Python**, and simulating environmental variables in real time.

---

## ⚡ Quick Start: How to Run

### 1️⃣ Launch the Live Web App & REST API (Fastest)

Run this single command from the project root:

```bash
python server.py
```

Now open your browser:
👉 **[http://localhost:5000](http://localhost:5000)**

---

### 2️⃣ Run the React Development Server (Optional)

If you wish to develop or modify the React frontend:

```bash
# Terminal 1: Start the backend API server
python server.py

# Terminal 2: Start Vite React live development server
cd client
npm run dev
```

Now open:
👉 **[http://localhost:5173](http://localhost:5173)** *(automatically proxies API requests to port 5000)*

---

### 3️⃣ Run Machine Learning CLI Predictions

You can also run predictions directly from your terminal:

```bash
# Test custom environmental parameters:
python ml/predict.py --temperature 28 --humidity 65 --wind 12 --rain 0 --pressure 1013

# Generate 7-day forecast for a city (e.g. delhi, mumbai, london):
python ml/predict.py --forecast --city delhi

# Re-train the Machine Learning models on fresh data:
python ml/train_models_cli.py
```

---

## 🌦️ Weather Data & API Key Guide

### Does this application require an API key?
**NO, an API key is NOT required to run this project!**

- **Built-in Offline ML Mode (Default & Zero-Friction)**:
  The project includes built-in physical meteorological datasets and pre-trained Python Machine Learning models (`ml/forecaster_model.json` and `ml/environmental_model.json`). This ensures that your project runs seamlessly on **any computer, completely offline**, without risk of broken external links, expired trial keys, or rate-limiting.

### Connecting Live Weather APIs (Optional)

If you wish to fetch live real-time weather from external services, two options are provided:

#### Option 1: Open-Meteo API (100% Free, No Key Required)
- Uses real-time satellite & meteorological station feeds.
- Completely open-source and free for educational / EVS projects.
- **Requires zero API key**. Pre-configured inside `ml/weather_api.py`.

#### Option 2: OpenWeatherMap API (Optional)
If you have or want an OpenWeatherMap API key:
1. Get a free API key at [OpenWeatherMap Sign Up](https://home.openweathermap.org/users/sign_up).
2. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Add your key inside `.env`:
   ```env
   OPENWEATHER_API_KEY=your_actual_api_key_here
   ```
4. Restart `python server.py`. The application will automatically pick up your key.

---

## ✨ Features

- 🌡️ **Interactive 30-Day Weather Trend Dashboard**:
  - Daily historical temperature (Min, Avg, Max), relative humidity, precipitation, and barometric pressure.
  - Multi-metric interactive chart with smooth spline visualization and toggles.
  - Key statistical metrics: 30-day average, peak recorded, low recorded, total accumulated rainfall, and pressure stability.
- 🔮 **Future 7-Day Prediction using Machine Learning in Python**:
  - Multi-horizon autoregressive lag model + harmonic seasonal basis regression.
  - Daily cards showing predicted temperature, temperature ranges, weather conditions, precipitation probability %, and ML confidence scores.
  - Seamless transitional chart bridging the past 7 days of observation into the 7-day predicted future.
- 🤖 **Interactive "What-If" AI Environmental Predictor**:
  - Live sliders for Temperature (-5°C to 50°C), Humidity (10% to 100%), Wind Speed (0 to 80 km/h), Rainfall (0 to 100 mm), and Atmospheric Pressure (960 to 1040 hPa).
  - One-click presets: *Pleasant Day*, *Monsoon Storm*, *Heatwave*, *Winter Morning*, *Sultry Humidity*.
  - Bioclimatic Thermal Comfort index (Optimal, Warm, Hot, Extreme Alert) and actionable EVS ecological insights.
- 🌍 **Global Station Support**:
  - Dynamic switching between climatic zones: New Delhi (Subtropical), Mumbai (Tropical Coastal), Bengaluru (Savanna), London (Temperate), New York (Continental), Tokyo (Humid Subtropical).
- 📊 **Machine Learning Explainability & Transparency Hub**:
  - Evaluated on 1,000 meteorological observation samples.
  - Model metrics: **R² Score = 99.8%**, **MAE = 0.22°C**, **RMSE = 0.29°C**.
  - Relative feature importance breakdown chart.

---

## 🛠️ Architecture & Technologies Used

- **Machine Learning (Python)**:
  - Pure Python + Scikit-compatible Ridge Regularization Engine with interaction feature expansion and multi-step autoregressive time-series forecaster.
  - Model files: `ml/simple_ml.py`, `ml/train_models_cli.py`, `ml/predict.py`.
- **Backend & REST API**:
  - Python Flask (`ml/app.py`, `server.py`) with full CORS support.
  - Node.js & Express alternative (`src/server.ts`, `src/controllers/`, `src/routes/`, `src/services/`).
- **Frontend**:
  - Modern Glassmorphism Single Page Dashboard (`templates/index.html` + `static/`) using Chart.js, responsive layouts, and SVG weather icons.
  - React + TypeScript + Vite client (`client/`).

---

## 📁 Project Structure

```text
Climate-AI/
├── templates/
│   └── index.html                # Modern Glassmorphic Web Dashboard
├── ml/
│   ├── simple_ml.py              # ML Algorithms (Ridge & 7-Day AR Forecaster)
│   ├── train_models_cli.py       # Training pipeline & dataset generator
│   ├── predict.py                # Standalone CLI & prediction module
│   ├── app.py                    # Flask REST API server
│   ├── environmental_model.json  # Trained environmental model weights
│   ├── forecaster_model.json     # Trained 7-day time-series model
│   └── metrics.json              # Model evaluation metrics (R², MAE, RMSE)
├── data/
│   ├── weather_dataset.csv       # 1,000-day meteorological training observations
│   ├── weather_history_30days.json # Recent 30-day historical trend
│   └── weather_history_cities.json # 30-day trends for multiple global cities
├── client/                       # Full React + TypeScript + Vite frontend
│   ├── src/
│   │   ├── App.tsx               # React Weather Dashboard
│   │   ├── App.css               # Glassmorphism & Climate Theme Styles
│   │   └── main.tsx              # Vite React entrypoint
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
├── src/                          # Node.js Express Backend
│   ├── controllers/              # Prediction and Weather controllers
│   ├── routes/                   # Express API routes
│   ├── services/                 # ML service execution bridge
│   ├── app.ts                    # Express application configuration
│   └── server.ts                 # Express HTTP server
├── package.json                  # Node.js project manifest
├── tsconfig.json                 # TypeScript compiler config
├── requirements.txt              # Python requirements
├── server.py                     # One-Click Python Launcher
└── README.md                     # Project documentation
```

---

## 📖 Step-by-Step Execution Guide

### Prerequisites
- **Python 3.8+** (installed with `pip install -r requirements.txt`)
- **Node.js 18+ & npm** (optional, only if modifying the React client)

### Commands Summary

| Task | Command | URL / Output |
| :--- | :--- | :--- |
| **Start Web App + API** | `python server.py` | [http://localhost:5000](http://localhost:5000) |
| **Start React Vite Client** | `cd client && npm run dev` | [http://localhost:5173](http://localhost:5173) |
| **Re-train ML Models** | `python ml/train_models_cli.py` | Terminal training metrics & JSON update |
| **Run CLI Prediction** | `python ml/predict.py --temperature 28 --humidity 65 --wind 12 --rain 0 --pressure 1013` | Terminal JSON prediction output |
| **Build React for Production** | `cd client && npm run build` | Builds bundle into `client/dist/` |

---

## 🔌 REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the interactive Climate AI dashboard |
| `POST` | `/api/prediction` | Predicts apparent temperature & thermal comfort from environmental inputs |
| `GET` | `/api/trends/30days?city=delhi` | Returns 30-day historical observations & aggregate statistics |
| `GET` | `/api/forecast/7days?city=delhi` | Generates future 7-day weather predictions using the ML model |
| `GET` | `/api/model/metrics` | Returns model performance metrics, R² score, and feature weights |
| `GET` | `/api/cities` | Returns configured meteorological stations |

### Example Request (`POST /api/prediction`):

```json
{
  "temperature": 28.0,
  "humidity": 65.0,
  "windSpeed": 12.0,
  "rainfall": 0.0,
  "pressure": 1013.0
}
```

### Example Response:

```json
{
  "status": "success",
  "input": {
    "temperature": 28.0,
    "humidity": 65.0,
    "windSpeed": 12.0,
    "rainfall": 0.0,
    "pressure": 1013.0
  },
  "predictedTemperature": 27.9,
  "temperatureFahrenheit": 82.2,
  "feelsLikeDelta": -0.1,
  "thermalComfort": "Warm (Elevated Transpiration)",
  "comfortBadge": "warm",
  "comfortColor": "#f59e0b",
  "environmentalInsights": [
    "Atmospheric readings correlate closely with standard baseline environmental equilibrium."
  ],
  "modelUsed": "Climate AI Ridge Atmospheric Model v1.0",
  "academicContext": "Environmental Studies (EVS) AI Project"
}
```

---

## 👨‍🎓 Academic Context (EVS Project)

- **Project Title:** AI-Based Weather Trend Analysis & 7-Day Temperature Prediction for Environmental Monitoring
- **Project Name:** Climate AI
- **Focus Area:** Environmental Studies (EVS), Bioclimatology, and Microclimate Predictive Modeling
- **Significance:** Demonstrates how data science and lightweight machine learning models can help monitor local microclimates, assess thermal discomfort stress, predict extreme weather anomalies, and foster climate awareness.

---

## 📤 Pushing to GitHub (Step-by-Step)

Follow these steps to upload this project to your GitHub account:

### 1. Initialize Git and Stage Files
```bash
git init
git add .
git commit -m "feat: complete Climate AI weather monitoring and 7-day prediction app"
```

### 2. Connect Your GitHub Repository
Create a new empty repository on [GitHub](https://github.com/new), then link and push:
```bash
git branch -M main
git remote add origin https://github.com/<YOUR-GITHUB-USERNAME>/<YOUR-REPOSITORY-NAME>.git
git push -u origin main
```

---

## 💬 Support & Troubleshooting

- **Port Conflict (`Address already in use: 5000`)**: If port 5000 is occupied, set an alternate port in your terminal before launching:
  ```bash
  # Windows PowerShell
  $env:PORT="5050"; python server.py
  ```
- **Python Dependencies**: If any package is missing, run:
  ```bash
  pip install -r requirements.txt
  ```
- **Node Dependencies**: If you modify the React UI inside `client/`:
  ```bash
  cd client
  npm install
  npm run build
  ```
- **Reporting Issues**: Open an issue on your GitHub repository tracker or consult the project documentation.

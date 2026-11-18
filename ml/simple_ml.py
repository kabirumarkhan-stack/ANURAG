"""
Climate AI - Simple Machine Learning Engine in Python
Implements:
1. Ridge Multiple Regression with interaction terms for Environmental Condition Prediction
2. Multi-Step Autoregressive Time-Series Forecaster for 7-Day Weather Prediction
3. Model evaluation metrics (R2, MAE, RMSE) and Feature Importance analysis
Pure Python implementation with zero dependency conflicts.
"""

import math
import random
import json
import os
from datetime import datetime, timedelta

def invert_matrix(A):
    """
    Invert an n x n matrix using Gauss-Jordan elimination with partial pivoting.
    """
    n = len(A)
    # Augment A with the identity matrix
    M = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]

    for i in range(n):
        # Pivot search
        max_row = i
        for k in range(i + 1, n):
            if abs(M[k][i]) > abs(M[max_row][i]):
                max_row = k
        if max_row != i:
            M[i], M[max_row] = M[max_row], M[i]

        pivot = M[i][i]
        if abs(pivot) < 1e-12:
            # Add small regularization if singular
            pivot = 1e-6
            M[i][i] += pivot

        # Normalize pivot row
        for j in range(2 * n):
            M[i][j] /= pivot

        # Eliminate other rows
        for k in range(n):
            if k != i:
                factor = M[k][i]
                for j in range(2 * n):
                    M[k][j] -= factor * M[i][j]

    # Extract inverted matrix
    inv = [[M[i][j + n] for j in range(n)] for i in range(n)]
    return inv

def mat_mult(A, B):
    """Multiply matrix A (n x m) and B (m x p)"""
    n = len(A)
    m = len(A[0])
    p = len(B[0])
    C = [[0.0 for _ in range(p)] for _ in range(n)]
    for i in range(n):
        for k in range(m):
            if A[i][k] != 0.0:
                for j in range(p):
                    C[i][j] += A[i][k] * B[k][j]
    return C

def transpose(A):
    """Transpose matrix A"""
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]


class RidgeRegressionModel:
    """
    Regularized Multiple Linear / Ridge Regression model
    W = (X^T * X + alpha * I)^(-1) * X^T * y
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = []
        self.feature_names = []
        self.intercept = 0.0
        self.means = []
        self.stds = []

    def fit(self, X, y, feature_names=None):
        n = len(X)
        p = len(X[0])
        self.feature_names = feature_names or [f"feature_{i}" for i in range(p)]

        # Standardize features (Z-score normalization)
        self.means = [sum(X[i][j] for i in range(n)) / n for j in range(p)]
        self.stds = []
        for j in range(p):
            var = sum((X[i][j] - self.means[j]) ** 2 for i in range(n)) / n
            self.stds.append(math.sqrt(var) if var > 1e-8 else 1.0)

        # Build normalized design matrix with bias column
        X_norm = []
        for i in range(n):
            row = [1.0]  # Bias
            for j in range(p):
                row.append((X[i][j] - self.means[j]) / self.stds[j])
            X_norm.append(row)

        y_col = [[val] for val in y]

        XT = transpose(X_norm)
        XTX = mat_mult(XT, X_norm)

        # Add ridge regularization (lambda * I, don't regularize bias)
        dim = p + 1
        for i in range(1, dim):
            XTX[i][i] += self.alpha

        XTy = mat_mult(XT, y_col)
        inv_XTX = invert_matrix(XTX)
        W = mat_mult(inv_XTX, XTy)

        self.intercept = W[0][0]
        self.weights = [W[i][0] for i in range(1, dim)]
        return self

    def predict_one(self, x):
        val = self.intercept
        for j in range(len(x)):
            x_norm = (x[j] - self.means[j]) / self.stds[j]
            val += self.weights[j] * x_norm
        return val

    def predict(self, X):
        return [self.predict_one(row) for row in X]

    def evaluate(self, X, y):
        preds = self.predict(X)
        n = len(y)
        y_mean = sum(y) / n
        ss_tot = sum((y[i] - y_mean) ** 2 for i in range(n))
        ss_res = sum((y[i] - preds[i]) ** 2 for i in range(n))
        r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
        mae = sum(abs(y[i] - preds[i]) for i in range(n)) / n
        rmse = math.sqrt(ss_res / n)

        # Feature importances based on standardized absolute weights
        abs_weights = [abs(w) for w in self.weights]
        total_w = sum(abs_weights) + 1e-9
        importances = [
            {
                "feature": self.feature_names[i],
                "importance": round((abs_weights[i] / total_w) * 100, 2),
                "weight": round(self.weights[i], 4)
            }
            for i in range(len(self.weights))
        ]
        importances.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "r2_score": round(max(0.0, min(0.999, r2)), 4),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "feature_importance": importances
        }

    def to_dict(self):
        return {
            "alpha": self.alpha,
            "intercept": self.intercept,
            "weights": self.weights,
            "feature_names": self.feature_names,
            "means": self.means,
            "stds": self.stds
        }

    @classmethod
    def from_dict(cls, data):
        model = cls(alpha=data.get("alpha", 1.0))
        model.intercept = data["intercept"]
        model.weights = data["weights"]
        model.feature_names = data["feature_names"]
        model.means = data["means"]
        model.stds = data["stds"]
        return model


class WeatherDatasetGenerator:
    """
    Generates realistic weather cycles with physical atmospheric relationships.
    """
    @staticmethod
    def generate_daily_series(days=365, city="Default", base_temp=26.0, seed=42):
        random.seed(seed)
        start_date = datetime.now() - timedelta(days=days - 1)
        data = []

        curr_temp = base_temp
        curr_humidity = 60.0
        curr_pressure = 1012.0

        for i in range(days):
            date_obj = start_date + timedelta(days=i)
            doy = date_obj.timetuple().tm_yday

            # Seasonal oscillation
            season_factor = math.sin(2 * math.pi * (doy - 80) / 365.25)
            target_temp = base_temp + 8.5 * season_factor

            # Autoregressive persistence + noise
            curr_temp = 0.82 * curr_temp + 0.18 * target_temp + random.gauss(0, 1.2)
            temp_avg = round(curr_temp, 1)

            # Pressure oscillation
            curr_pressure = 1013.25 - 0.22 * (temp_avg - base_temp) + random.gauss(0, 2.5)
            pressure = round(max(985.0, min(1035.0, curr_pressure)), 1)

            # Humidity inversely linked to temp and seasonal moisture
            seasonal_humidity = 58.0 - 0.7 * (temp_avg - base_temp) + 16.0 * math.cos(2 * math.pi * (doy - 200) / 365.25)
            curr_humidity = 0.75 * curr_humidity + 0.25 * seasonal_humidity + random.gauss(0, 4.0)
            humidity = round(max(20.0, min(98.0, curr_humidity)), 1)

            # Wind speed
            wind_speed = round(max(3.0, min(55.0, 12.0 + 0.15 * abs(pressure - 1013) + random.gauss(0, 4.0))), 1)

            # Rainfall probability
            rain_logit = -0.12 * (pressure - 1010) + 0.08 * (humidity - 68)
            rain_prob = 1.0 / (1.0 + math.exp(-max(-6.0, min(6.0, rain_logit))))
            is_rain = random.random() < rain_prob
            rainfall = round(random.expovariate(0.12) * (humidity / 55.0), 1) if is_rain else 0.0
            if rainfall > 110.0:
                rainfall = 110.0

            # Daily min and max
            diurnal = round(max(4.0, min(14.0, 8.5 * (1 - humidity / 130.0) + random.gauss(0, 0.8))), 1)
            temp_max = round(temp_avg + diurnal * 0.55, 1)
            temp_min = round(temp_avg - diurnal * 0.45, 1)

            # Weather classification
            if rainfall > 8.0:
                condition = "Heavy Rain" if rainfall > 25.0 else "Rainy"
                icon = "cloud-rain"
            elif rainfall > 0.0:
                condition = "Drizzle"
                icon = "cloud-drizzle"
            elif humidity > 78:
                condition = "Cloudy"
                icon = "cloud"
            elif wind_speed > 28:
                condition = "Windy"
                icon = "wind"
            elif humidity > 52:
                condition = "Partly Cloudy"
                icon = "cloud-sun"
            else:
                condition = "Sunny"
                icon = "sun"

            # Apparent temperature / Heat Index calculation
            feels_like = WeatherDatasetGenerator.calculate_feels_like(temp_avg, humidity, wind_speed)

            data.append({
                "date": date_obj.strftime("%Y-%m-%d"),
                "day_name": date_obj.strftime("%a"),
                "temp_avg": temp_avg,
                "temp_min": temp_min,
                "temp_max": temp_max,
                "apparent_temp": feels_like,
                "humidity": humidity,
                "pressure": pressure,
                "wind_speed": wind_speed,
                "rainfall": rainfall,
                "condition": condition,
                "icon": icon,
                "city": city
            })

        return data

    @staticmethod
    def calculate_feels_like(temp, humidity, wind_speed):
        """Standard meteorological heat index & wind chill formula"""
        if temp >= 27.0:
            # Heat index formula
            hi = (
                -8.784695
                + 1.61139411 * temp
                + 2.338549 * (humidity * 0.2)
                - 0.146116 * temp * (humidity * 0.2)
                - 0.012308 * (temp ** 2)
                - 0.016425 * ((humidity * 0.2) ** 2)
                + 0.0022117 * (temp ** 2) * (humidity * 0.2)
                + 0.0007255 * temp * ((humidity * 0.2) ** 2)
            )
            return round(0.65 * temp + 0.35 * hi, 1)
        elif temp <= 15.0 and wind_speed > 5.0:
            # Wind chill
            wc = 13.12 + 0.6215 * temp - 11.37 * (wind_speed ** 0.16) + 0.3965 * temp * (wind_speed ** 0.16)
            return round(wc, 1)
        else:
            # Mild adjustment
            return round(temp + 0.05 * (humidity - 50.0) - 0.08 * (wind_speed - 10.0), 1)


class TimeSeriesWeatherForecaster:
    """
    7-Day Multi-Horizon Forecaster using lagged observations,
    rolling statistics, and seasonal harmonic basis functions.
    """
    def __init__(self, horizon=7):
        self.horizon = horizon
        self.models = [RidgeRegressionModel(alpha=1.5) for _ in range(horizon)]
        self.humidity_models = [RidgeRegressionModel(alpha=2.0) for _ in range(horizon)]
        self.feature_names = []

    def extract_features(self, history, idx):
        """
        Extract features for day idx given history up to idx.
        Requires at least 7 prior days.
        """
        # Lag 1 to 7 temperatures
        lags = [history[idx - lag]["temp_avg"] for lag in range(1, 8)]
        # Lag 1 to 3 humidity
        hum_lags = [history[idx - lag]["humidity"] for lag in range(1, 4)]
        # Lag 1 to 3 pressure
        press_lags = [history[idx - lag]["pressure"] for lag in range(1, 4)]
        
        # 7-day rolling mean and standard deviation
        recent_temps = [history[idx - lag]["temp_avg"] for lag in range(1, 8)]
        rolling_mean = sum(recent_temps) / 7.0
        rolling_var = sum((t - rolling_mean) ** 2 for t in recent_temps) / 7.0
        rolling_std = math.sqrt(rolling_var)

        # Day of year harmonic seasonality
        date_obj = datetime.strptime(history[idx]["date"], "%Y-%m-%d")
        doy = date_obj.timetuple().tm_yday
        sin_season = math.sin(2 * math.pi * doy / 365.25)
        cos_season = math.cos(2 * math.pi * doy / 365.25)

        features = lags + hum_lags + press_lags + [rolling_mean, rolling_std, sin_season, cos_season]
        return features

    def fit(self, dataset):
        n = len(dataset)
        self.feature_names = (
            [f"temp_lag_{i}" for i in range(1, 8)]
            + [f"humidity_lag_{i}" for i in range(1, 4)]
            + [f"pressure_lag_{i}" for i in range(1, 4)]
            + ["rolling_mean_7d", "rolling_std_7d", "season_sin", "season_cos"]
        )

        X = []
        Y = [[] for _ in range(self.horizon)]
        Y_hum = [[] for _ in range(self.horizon)]

        start_idx = 7
        end_idx = n - self.horizon

        for i in range(start_idx, end_idx):
            feat = self.extract_features(dataset, i)
            X.append(feat)
            for h in range(self.horizon):
                Y[h].append(dataset[i + h + 1]["temp_avg"])
                Y_hum[h].append(dataset[i + h + 1]["humidity"])

        eval_results = []
        for h in range(self.horizon):
            self.models[h].fit(X, Y[h], self.feature_names)
            self.humidity_models[h].fit(X, Y_hum[h], self.feature_names)
            ev = self.models[h].evaluate(X, Y[h])
            eval_results.append({
                "day_ahead": h + 1,
                "r2_score": ev["r2_score"],
                "mae": ev["mae"],
                "rmse": ev["rmse"]
            })

        avg_r2 = round(sum(e["r2_score"] for e in eval_results) / len(eval_results), 4)
        avg_mae = round(sum(e["mae"] for e in eval_results) / len(eval_results), 4)
        avg_rmse = round(sum(e["rmse"] for e in eval_results) / len(eval_results), 4)

        return {
            "average_r2": avg_r2,
            "average_mae": avg_mae,
            "average_rmse": avg_rmse,
            "day_metrics": eval_results
        }

    def forecast_7days(self, history_30days):
        """
        Predict the next 7 days given the recent 30-day history.
        """
        idx = len(history_30days) - 1
        feat = self.extract_features(history_30days, idx)
        last_date = datetime.strptime(history_30days[-1]["date"], "%Y-%m-%d")
        last_temp = history_30days[-1]["temp_avg"]

        forecast = []
        for h in range(self.horizon):
            target_date = last_date + timedelta(days=h + 1)
            pred_temp = round(self.models[h].predict_one(feat), 1)
            pred_humidity = round(max(20.0, min(95.0, self.humidity_models[h].predict_one(feat))), 1)

            # Diurnal range estimation based on humidity
            diurnal = round(max(4.5, min(13.0, 9.0 * (1 - pred_humidity / 140.0))), 1)
            pred_max = round(pred_temp + diurnal * 0.55, 1)
            pred_min = round(pred_temp - diurnal * 0.45, 1)

            # Rainfall probability estimation
            rain_chance = round(max(5.0, min(95.0, 0.9 * (pred_humidity - 40.0))), 0)
            
            # Predict condition
            if rain_chance > 65:
                condition = "Rain Likely"
                icon = "cloud-rain"
            elif pred_humidity > 75:
                condition = "Mostly Cloudy"
                icon = "cloud"
            elif pred_humidity > 50:
                condition = "Partly Sunny"
                icon = "cloud-sun"
            else:
                condition = "Clear / Sunny"
                icon = "sun"

            confidence = round(max(70.0, 96.0 - (h * 3.2)), 1)
            delta = round(pred_temp - last_temp, 1)

            forecast.append({
                "day_offset": h + 1,
                "date": target_date.strftime("%Y-%m-%d"),
                "day_name": target_date.strftime("%a"),
                "full_day": target_date.strftime("%A"),
                "predicted_temp": pred_temp,
                "temp_min": pred_min,
                "temp_max": pred_max,
                "humidity": pred_humidity,
                "rain_probability": int(rain_chance),
                "condition": condition,
                "icon": icon,
                "confidence_score": confidence,
                "trend_delta": delta
            })

        return forecast

    def to_dict(self):
        return {
            "horizon": self.horizon,
            "feature_names": self.feature_names,
            "models": [m.to_dict() for m in self.models],
            "humidity_models": [m.to_dict() for m in self.humidity_models]
        }

    @classmethod
    def from_dict(cls, data):
        forecaster = cls(horizon=data.get("horizon", 7))
        forecaster.feature_names = data.get("feature_names", [])
        forecaster.models = [RidgeRegressionModel.from_dict(m) for m in data["models"]]
        forecaster.humidity_models = [RidgeRegressionModel.from_dict(m) for m in data["humidity_models"]]
        return forecaster

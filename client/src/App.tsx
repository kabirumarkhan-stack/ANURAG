import { useState, useEffect } from 'react';
import './App.css';

interface WeatherDay {
  date: string;
  temp_avg: number;
  temp_min: number;
  temp_max: number;
  humidity: number;
  pressure: number;
  wind_speed: number;
  rainfall: number;
  condition: string;
}

interface ForecastDay {
  day_offset: number;
  date: string;
  day_name: string;
  full_day: string;
  predicted_temp: number;
  temp_min: number;
  temp_max: number;
  humidity: number;
  rain_probability: number;
  condition: string;
  confidence_score: number;
  trend_delta: number;
}

export default function App() {
  const [city, setCity] = useState('delhi');
  const [trends, setTrends] = useState<WeatherDay[]>([]);
  const [summary, setSummary] = useState<any>(null);
  const [forecast, setForecast] = useState<ForecastDay[]>([]);
  
  // Predictor state
  const [temp, setTemp] = useState(28);
  const [humidity, setHumidity] = useState(65);
  const [wind, setWind] = useState(12);
  const [rain, setRain] = useState(0);
  const [pressure, setPressure] = useState(1013);
  const [predictionResult, setPredictionResult] = useState<any>(null);

  useEffect(() => {
    fetchStationData(city);
    runPrediction();
  }, [city]);

  const fetchStationData = async (selectedCity: string) => {
    try {
      const [trendRes, foreRes] = await Promise.all([
        fetch(`/api/trends/30days?city=${selectedCity}`).then(r => r.json()),
        fetch(`/api/forecast/7days?city=${selectedCity}`).then(r => r.json())
      ]);

      if (trendRes.status === 'success') {
        setTrends(trendRes.data);
        setSummary(trendRes.summary);
      }
      if (foreRes.status === 'success') {
        setForecast(foreRes.forecast);
      }
    } catch (err) {
      console.error('API Fetch error:', err);
    }
  };

  const runPrediction = async () => {
    try {
      const res = await fetch('/api/prediction', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          temperature: temp,
          humidity,
          windSpeed: wind,
          rainfall: rain,
          pressure
        })
      });
      const data = await res.json();
      if (data.status === 'success') {
        setPredictionResult(data);
      }
    } catch (err) {
      console.error('Prediction error:', err);
    }
  };

  return (
    <div className="container">
      {/* Header */}
      <header className="header">
        <div>
          <h1 className="brand-title">Climate AI</h1>
          <p style={{ color: '#94a3b8', fontSize: '0.85rem' }}>
            Environmental Studies (EVS) Weather Monitoring & 7-Day ML Forecaster
          </p>
        </div>

        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <select 
            value={city} 
            onChange={(e) => setCity(e.target.value)}
            style={{
              background: '#0f172a',
              color: '#f8fafc',
              border: '1px solid rgba(255,255,255,0.1)',
              padding: '8px 16px',
              borderRadius: '20px',
              cursor: 'pointer'
            }}
          >
            <option value="delhi">New Delhi, India</option>
            <option value="mumbai">Mumbai, India</option>
            <option value="bengaluru">Bengaluru, India</option>
            <option value="london">London, UK</option>
            <option value="newyork">New York, USA</option>
            <option value="tokyo">Tokyo, Japan</option>
          </select>
        </div>
      </header>

      {/* Summary Metrics */}
      {summary && (
        <div className="metrics-grid">
          <div className="metric-item">
            <div style={{ color: '#94a3b8', fontSize: '0.8rem', textTransform: 'uppercase' }}>30-Day Mean Temp</div>
            <div className="metric-val" style={{ color: '#10b981' }}>{summary.avgTemperature}°C</div>
            <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Min: {summary.minTemperature}°C | Max: {summary.maxTemperature}°C</div>
          </div>
          <div className="metric-item">
            <div style={{ color: '#94a3b8', fontSize: '0.8rem', textTransform: 'uppercase' }}>Avg Humidity</div>
            <div className="metric-val" style={{ color: '#06b6d4' }}>{summary.avgHumidity}%</div>
            <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Atmospheric Moisture Index</div>
          </div>
          <div className="metric-item">
            <div style={{ color: '#94a3b8', fontSize: '0.8rem', textTransform: 'uppercase' }}>Accumulated Rain</div>
            <div className="metric-val" style={{ color: '#3b82f6' }}>{summary.totalRainfall} mm</div>
            <div style={{ fontSize: '0.75rem', color: '#64748b' }}>{summary.rainyDays} rain events</div>
          </div>
          <div className="metric-item">
            <div style={{ color: '#94a3b8', fontSize: '0.8rem', textTransform: 'uppercase' }}>Avg Pressure</div>
            <div className="metric-val" style={{ color: '#f59e0b' }}>{summary.avgPressure} hPa</div>
            <div style={{ fontSize: '0.75rem', color: '#64748b' }}>Barometric Stability</div>
          </div>
        </div>
      )}

      {/* 30-Day Weather Trend Section */}
      <div className="card">
        <h2 style={{ fontSize: '1.2rem', marginBottom: '16px' }}>30-Day Historical Trend Analysis</h2>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.85rem' }}>
            <thead>
              <tr style={{ color: '#94a3b8', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                <th style={{ padding: '8px 12px' }}>Date</th>
                <th style={{ padding: '8px 12px' }}>Avg Temp</th>
                <th style={{ padding: '8px 12px' }}>Range (Min/Max)</th>
                <th style={{ padding: '8px 12px' }}>Humidity</th>
                <th style={{ padding: '8px 12px' }}>Pressure</th>
                <th style={{ padding: '8px 12px' }}>Rainfall</th>
                <th style={{ padding: '8px 12px' }}>Condition</th>
              </tr>
            </thead>
            <tbody>
              {trends.slice(-10).reverse().map((day) => (
                <tr key={day.date} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                  <td style={{ padding: '8px 12px', fontFamily: 'monospace' }}>{day.date}</td>
                  <td style={{ padding: '8px 12px', color: '#34d399', fontWeight: 'bold' }}>{day.temp_avg}°C</td>
                  <td style={{ padding: '8px 12px', color: '#94a3b8' }}>{day.temp_min}° / {day.temp_max}°</td>
                  <td style={{ padding: '8px 12px' }}>{day.humidity}%</td>
                  <td style={{ padding: '8px 12px' }}>{day.pressure} hPa</td>
                  <td style={{ padding: '8px 12px', color: day.rainfall > 0 ? '#38bdf8' : '#64748b' }}>{day.rainfall} mm</td>
                  <td style={{ padding: '8px 12px' }}>{day.condition}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* 7-Day ML Forecast Section */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap' }}>
          <h2 style={{ fontSize: '1.2rem' }}>Future 7-Day Weather Prediction (Machine Learning)</h2>
          <span style={{ fontSize: '0.8rem', color: '#10b981' }}>Trained Time-Series Autoregressive Model</span>
        </div>

        <div className="forecast-grid">
          {forecast.map((f) => (
            <div key={f.day_offset} className="forecast-card">
              <div style={{ fontSize: '0.7rem', color: '#10b981', fontWeight: 'bold', marginBottom: '4px' }}>
                {f.confidence_score}% ML Confidence
              </div>
              <div style={{ fontWeight: 'bold', fontSize: '0.95rem' }}>{f.full_day}</div>
              <div style={{ fontSize: '0.75rem', color: '#64748b', marginBottom: '8px' }}>{f.date}</div>
              <div style={{ fontSize: '1.6rem', fontWeight: 'bold', color: '#fff', margin: '6px 0' }}>
                {f.predicted_temp}°C
              </div>
              <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '8px' }}>
                {f.temp_min}° / {f.temp_max}°
              </div>
              <div style={{ fontSize: '0.75rem', padding: '2px 8px', borderRadius: '12px', background: 'rgba(255,255,255,0.08)', display: 'inline-block', marginBottom: '8px' }}>
                {f.condition}
              </div>
              <div style={{ fontSize: '0.75rem', color: '#94a3b8', borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: '6px' }}>
                Rain chance: {f.rain_probability}%
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Interactive Predictor */}
      <div className="card">
        <h2 style={{ fontSize: '1.2rem', marginBottom: '8px' }}>Interactive Environmental Temperature Predictor</h2>
        <p style={{ color: '#94a3b8', fontSize: '0.85rem', marginBottom: '20px' }}>
          Adjust temperature, humidity, wind, rainfall, and pressure to see live ML predictions and bioclimatic comfort metrics.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
          <div>
            <label style={{ fontSize: '0.85rem' }}>Ambient Temperature: {temp}°C</label>
            <input type="range" className="range-slider" min="-5" max="50" step="0.5" value={temp} onChange={(e) => setTemp(Number(e.target.value))} />
          </div>
          <div>
            <label style={{ fontSize: '0.85rem' }}>Humidity: {humidity}%</label>
            <input type="range" className="range-slider" min="10" max="100" value={humidity} onChange={(e) => setHumidity(Number(e.target.value))} />
          </div>
          <div>
            <label style={{ fontSize: '0.85rem' }}>Wind Speed: {wind} km/h</label>
            <input type="range" className="range-slider" min="0" max="80" value={wind} onChange={(e) => setWind(Number(e.target.value))} />
          </div>
          <div>
            <label style={{ fontSize: '0.85rem' }}>Rainfall: {rain} mm</label>
            <input type="range" className="range-slider" min="0" max="100" value={rain} onChange={(e) => setRain(Number(e.target.value))} />
          </div>
          <div>
            <label style={{ fontSize: '0.85rem' }}>Pressure: {pressure} hPa</label>
            <input type="range" className="range-slider" min="960" max="1040" value={pressure} onChange={(e) => setPressure(Number(e.target.value))} />
          </div>
        </div>

        <button className="predict-btn" onClick={runPrediction}>
          Run AI Prediction
        </button>

        {predictionResult && (
          <div style={{ marginTop: '20px', padding: '16px', borderRadius: '12px', background: 'rgba(15,23,42,0.8)', border: '1px solid rgba(16,185,129,0.3)', textAlign: 'center' }}>
            <div style={{ fontSize: '0.8rem', color: '#94a3b8' }}>PREDICTED APPARENT TEMPERATURE</div>
            <div style={{ fontSize: '2.8rem', fontWeight: 'bold', color: '#10b981' }}>{predictionResult.predictedTemperature}°C</div>
            <div style={{ fontSize: '0.9rem', color: predictionResult.comfortColor, fontWeight: 'bold' }}>{predictionResult.thermalComfort}</div>
            <div style={{ marginTop: '10px', fontSize: '0.8rem', color: '#94a3b8' }}>
              {predictionResult.environmentalInsights?.map((t: string, i: number) => (
                <div key={i}>• {t}</div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

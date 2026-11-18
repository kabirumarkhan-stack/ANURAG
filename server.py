"""
Climate AI - Main Server Launcher
Starts the Flask server hosting both the REST API and the Frontend Dashboard.
"""

import os
import sys

# Ensure ml directory is in python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml'))

from app import app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 60)
    print("🌿 Climate AI — Weather Monitoring & 7-Day ML Prediction")
    print("=" * 60)
    print(f" Web Dashboard & REST API active at: http://localhost:{port}")
    print(f" Endpoints:")
    print(f"   • POST /api/prediction     (Environmental condition prediction)")
    print(f"   • GET  /api/trends/30days  (30-day historical weather trend)")
    print(f"   • GET  /api/forecast/7days (Future 7-day ML model forecast)")
    print(f"   • GET  /api/model/metrics  (Model evaluation & feature weights)")
    print(f"   • GET  /api/cities         (Configured meteorological stations)")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=False)

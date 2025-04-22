import os
import json
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def dashboard():
    stats_path = os.path.join(os.path.dirname(__file__), 'dashboard_stats.json')

    # Fallback default stats
    default_stats = {
        "total_logs": 0,
        "anomalies": 0,
        "anomaly_percent": 0,
        "normal_logs": 0,
        "normal_avg": 0,
        "anomalous_avg": 0
    }

    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
    else:
        # Create it for now so Flask won't crash
        with open(stats_path, "w") as f:
            json.dump(default_stats, f, indent=2)
        stats = default_stats

    return render_template('dashboard.html', stats=stats)

if __name__ == '__main__':
    app.run(debug=True, port=5050)

"""
Real-Time Anomaly Detection Server
Uses synthetic data + Z-score anomaly detection
"""

import time
import threading
from datetime import datetime
import numpy as np
from flask import Flask, jsonify
from flask_cors import CORS

# ------------------ Flask App ------------------
app = Flask(__name__)
CORS(app)

# ------------------ Global Storage ------------------
DATA_STORE = {
    "CPU_Usage": [],
    "Financial_Data": [],
    "Network_Traffic": []
}

WINDOW_SIZE = 60
Z_THRESHOLD = 2.5

# ------------------ Utility Functions ------------------
def z_score_anomaly(series, value):
    if len(series) < 10:
        return False
    mean = np.mean(series)
    std = np.std(series)
    if std == 0:
        return False
    z = abs((value - mean) / std)
    return z > Z_THRESHOLD

def trim_window(source):
    if len(DATA_STORE[source]) > WINDOW_SIZE:
        DATA_STORE[source] = DATA_STORE[source][-WINDOW_SIZE:]

# ------------------ Synthetic Data Generators ------------------
def generate_cpu():
    return np.random.normal(45, 5) + np.random.choice([0, 30, -20], p=[0.9, 0.05, 0.05])

def generate_financial():
    return np.random.normal(100, 10) + np.random.choice([0, 50, -40], p=[0.9, 0.05, 0.05])

def generate_network():
    return np.random.normal(450, 40) + np.random.choice([0, 200, -150], p=[0.9, 0.05, 0.05])

# ------------------ Background Worker ------------------
def data_worker():
    while True:
        for source, generator in [
            ("CPU_Usage", generate_cpu),
            ("Financial_Data", generate_financial),
            ("Network_Traffic", generate_network),
        ]:
            value = round(float(generator()), 2)
            past_values = [d["value"] for d in DATA_STORE[source]]
            is_anomaly = z_score_anomaly(past_values, value)

            DATA_STORE[source].append({
                "value": value,
                "timestamp": datetime.now().isoformat(),
                "is_anomaly": is_anomaly
            })

            trim_window(source)

        time.sleep(2)

# ------------------ API Endpoints ------------------
@app.route("/health")
def health():
    return jsonify({"status": "healthy"})

@app.route("/api/data/<source>")
def get_data(source):
    if source not in DATA_STORE:
        return jsonify({"error": "Invalid source"}), 400

    return jsonify({
        "source": source,
        "data_points": DATA_STORE[source]
    })

@app.route("/api/stats")
def stats():
    total = sum(len(v) for v in DATA_STORE.values())
    anomalies = sum(
        1 for v in DATA_STORE.values() for d in v if d["is_anomaly"]
    )

    return jsonify({
        "total_data_points": total,
        "total_anomalies": anomalies,
        "anomaly_rate": round((anomalies / total) * 100, 2) if total else 0
    })

# ------------------ Main ------------------
if __name__ == "__main__":
    print("🚀 Starting Anomaly Detection Server on http://localhost:5000")

    thread = threading.Thread(target=data_worker, daemon=True)
    thread.start()

    app.run(host="0.0.0.0", port=5000, debug=False)

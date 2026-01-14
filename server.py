"""
Real-Time Anomaly Detection Server
Auto-fetches data, stores it, detects anomalies, and sends notifications
"""
import os
import sys
import json
import sqlite3
import threading
import time
from datetime import datetime
from collections import deque
import numpy as np
from flask import Flask, jsonify, request, Response
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processor import TimeSeriesProcessor
from models import StatisticalDetector, IsolationForestDetector
from data.generate_data import (
    generate_cpu_usage_data,
    generate_financial_data,
    generate_network_traffic_data
)
from data.real_world_data import (
    generate_fraud_detection_data,
    generate_electricity_usage_data,
    generate_machine_health_data,
    generate_server_traffic_data,
    generate_stock_risk_data,
    generate_iot_sensor_data
)

# Initialize Flask
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
CORS(app)  # Enable CORS for all routes

# Database setup
DB_FILE = 'anomaly_detection.db'

# In-memory event buffer for Server-Sent Events (SSE)
sse_events = deque(maxlen=1000)

def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Create tables
    c.execute('''CREATE TABLE IF NOT EXISTS data_points
                 (id INTEGER PRIMARY KEY, 
                  source TEXT, 
                  value REAL, 
                  timestamp DATETIME,
                  is_anomaly INTEGER,
                  score REAL)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS alerts
                 (id INTEGER PRIMARY KEY,
                  source TEXT,
                  message TEXT,
                  severity TEXT,
                  timestamp DATETIME,
                  data_value REAL)''')
    
    conn.commit()
    conn.close()

def store_data_point(source, value, is_anomaly, score):
    """Store data point in database"""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''INSERT INTO data_points (source, value, timestamp, is_anomaly, score)
                     VALUES (?, ?, ?, ?, ?)''',
                  (source, value, datetime.now(), is_anomaly, score))
        conn.commit()
        conn.close()
        return True
    except:
        return False

def create_alert(source, message, severity, data_value):
    """Create and store alert"""
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute('''INSERT INTO alerts (source, message, severity, timestamp, data_value)
                     VALUES (?, ?, ?, ?, ?)''',
                  (source, message, severity, datetime.now(), data_value))
        conn.commit()
        conn.close()
        
        # Print notification
        print(f"\n{'='*70}")
        print(f"ALERT [{severity}] - {source}")
        print(f"{'='*70}")
        print(f"Message: {message}")
        print(f"Value: {data_value:.2f}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")

        # Push alert into SSE buffer
        try:
            sse_events.append({
                'type': 'alert',
                'source': source,
                'message': message,
                'severity': severity,
                'value': float(data_value),
                'timestamp': datetime.now().isoformat()
            })
        except Exception:
            pass
        
        return True
    except:
        return False

class StreamingDetector:
    """Real-time streaming anomaly detector"""
    
    def __init__(self, source_name):
        self.source_name = source_name
        self.data_buffer = deque(maxlen=100)
        self.processor = TimeSeriesProcessor()
        self.detector = StatisticalDetector(threshold=2.5)
        self.detector_if = IsolationForestDetector(contamination=0.08)
        self.is_trained = False
        self.total_points = 0
        self.anomaly_count = 0
    
    def train(self, data):
        """Train on initial data"""
        normalized = self.processor.normalize(data, fit=True)
        self.detector.fit(normalized)
        self.detector_if.fit(normalized)
        self.is_trained = True
    
    def process(self, value):
        """Process new data point"""
        self.data_buffer.append(value)
        self.total_points += 1
        
        if not self.is_trained or len(self.data_buffer) < 10:
            return None
        
        # Detect
        data_array = np.array(list(self.data_buffer))
        normalized = self.processor.normalize(data_array, fit=False)
        
        pred_stat = self.detector.predict(normalized)[-1]
        pred_if = self.detector_if.predict(normalized)[-1]
        score = max(self.detector.score(normalized)[-1], 
                   self.detector_if.score(normalized)[-1])
        
        is_anomaly = 1 if (pred_stat + pred_if) >= 1 else 0
        
        if is_anomaly:
            self.anomaly_count += 1
        
        # Store in database
        store_data_point(self.source_name, value, is_anomaly, score)
        
        # Create alert if anomaly
        if is_anomaly:
            create_alert(self.source_name, 
                        f"Anomaly detected in {self.source_name}",
                        "HIGH",
                        value)
            # Push anomaly event to SSE buffer
            try:
                sse_events.append({
                    'type': 'anomaly',
                    'source': self.source_name,
                    'value': float(value),
                    'score': float(score),
                    'timestamp': datetime.now().isoformat()
                })
            except Exception:
                pass
        
        return {
            'value': value,
            'is_anomaly': is_anomaly,
            'score': float(score)
        }
    
    def get_stats(self):
        """Get current statistics"""
        return {
            'source': self.source_name,
            'total_points': self.total_points,
            'anomalies': self.anomaly_count,
            'anomaly_rate': self.anomaly_count / max(self.total_points, 1),
            'is_trained': self.is_trained,
            'buffer_size': len(self.data_buffer)
        }

# Global detectors
detectors = {
    'cpu': StreamingDetector('CPU_Usage'),
    'financial': StreamingDetector('Financial_Data'),
    'network': StreamingDetector('Network_Traffic'),
    'fraud': StreamingDetector('Fraud_Detection'),
    'electricity': StreamingDetector('Electricity_Usage'),
    'machine': StreamingDetector('Machine_Health'),
    'server': StreamingDetector('Server_Traffic'),
    'stock': StreamingDetector('Stock_Risk'),
    'iot': StreamingDetector('IoT_Sensors')
}

# Background data fetching
def auto_fetch_and_process():
    """Background thread: auto-fetch data and detect anomalies"""
    print("\nBackground worker started - auto-fetching data...")
    
    # Train initially
    print("Training detectors on initial data...")
    
    for key, source_name in [('cpu', 'cpu_usage'), 
                              ('financial', 'price'), 
                              ('network', 'traffic_mbps'),
                              ('fraud', 'transaction_amount'),
                              ('electricity', 'power_kwh'),
                              ('machine', 'temperature_celsius'),
                              ('server', 'requests_per_min'),
                              ('stock', 'stock_price'),
                              ('iot', 'humidity_percent')]:
        try:
            if key == 'cpu':
                df = generate_cpu_usage_data(n_samples=100)
                data = df['cpu_usage'].values
            elif key == 'financial':
                df = generate_financial_data(n_samples=100)
                data = df['price'].values
            elif key == 'network':
                df = generate_network_traffic_data(n_samples=100)
                data = df['traffic_mbps'].values
            elif key == 'fraud':
                df = generate_fraud_detection_data(n_samples=100)
                data = df['transaction_amount'].values
            elif key == 'electricity':
                df = generate_electricity_usage_data(n_samples=100)
                data = df['power_kwh'].values
            elif key == 'machine':
                df = generate_machine_health_data(n_samples=100)
                data = df['temperature_celsius'].values
            elif key == 'server':
                df = generate_server_traffic_data(n_samples=100)
                data = df['requests_per_min'].values
            elif key == 'stock':
                df = generate_stock_risk_data(n_samples=100)
                data = df['stock_price'].values
            else:
                df = generate_iot_sensor_data(n_samples=100)
                data = df['humidity_percent'].values
            
            detectors[key].train(data)
            print(f"  {key.upper()} detector trained")
        except Exception as e:
            print(f"  {key.upper()} training error: {e}")
    
    print("Starting continuous data processing...\n")
    
    # Generate stream data
    streams = {
        'cpu': generate_cpu_usage_data(n_samples=1000)['cpu_usage'].values,
        'financial': generate_financial_data(n_samples=1000)['price'].values,
        'network': generate_network_traffic_data(n_samples=1000)['traffic_mbps'].values,
        'fraud': generate_fraud_detection_data(n_samples=1000)['transaction_amount'].values,
        'electricity': generate_electricity_usage_data(n_samples=1000)['power_kwh'].values,
        'machine': generate_machine_health_data(n_samples=1000)['temperature_celsius'].values,
        'server': generate_server_traffic_data(n_samples=1000)['requests_per_min'].values,
        'stock': generate_stock_risk_data(n_samples=1000)['stock_price'].values,
        'iot': generate_iot_sensor_data(n_samples=1000)['humidity_percent'].values
    }
    
    stream_indices = {k: 0 for k in streams}
    
    # Continuous processing
    while True:
        try:
            for key in streams:
                if stream_indices[key] < len(streams[key]):
                    value = streams[key][stream_indices[key]]
                    result = detectors[key].process(value)
                    stream_indices[key] += 1
            
            time.sleep(0.5)
        except Exception as e:
            print(f"Error in background worker: {e}")
            time.sleep(1)

# API Endpoints

@app.route('/', methods=['GET'])
def index():
    """Home endpoint"""
    return jsonify({
        'service': 'Anomaly Detection Server',
        'status': 'RUNNING',
        'endpoints': {
            'status': 'GET /api/status',
            'stats': 'GET /api/stats',
            'data': 'GET /api/data/<source>',
            'alerts': 'GET /api/alerts',
            'recent_anomalies': 'GET /api/anomalies',
            'stream_anomalies': 'GET /api/stream/anomalies',
            'submit_data': 'POST /api/submit',
            'configure': 'POST /api/configure'
        }
    })

@app.route('/api/stream/anomalies', methods=['GET'])
def stream_anomalies():
    """Server-Sent Events stream of anomalies and alerts.
    Optional query param 'source' filters by source name.
    """
    source_filter = request.args.get('source')

    def generate():
        idx = 0
        while True:
            try:
                if idx < len(sse_events):
                    event = sse_events[idx]
                    idx += 1
                    if source_filter and event.get('source') != source_filter:
                        continue
                    yield f"data: {json.dumps(event)}\n\n"
                else:
                    time.sleep(0.5)
            except Exception:
                time.sleep(0.5)

    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get server status"""
    stats = {}
    for key, detector in detectors.items():
        stats[key] = detector.get_stats()
    
    return jsonify({
        'service': 'Anomaly Detection Server',
        'status': 'ACTIVE',
        'timestamp': datetime.now().isoformat(),
        'detectors': stats,
        'total_anomalies': sum(d.get_stats()['anomalies'] for d in detectors.values())
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get detailed statistics"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Total stats
    c.execute('SELECT COUNT(*) FROM data_points')
    total_points = c.fetchone()[0]
    
    c.execute('SELECT COUNT(*) FROM data_points WHERE is_anomaly=1')
    total_anomalies = c.fetchone()[0]
    
    # Per-source stats
    c.execute('SELECT source, COUNT(*), SUM(is_anomaly) FROM data_points GROUP BY source')
    source_stats = c.fetchall()
    
    conn.close()
    
    return jsonify({
        'total_data_points': total_points,
        'total_anomalies': total_anomalies,
        'anomaly_rate': total_anomalies / max(total_points, 1),
        'sources': [
            {
                'name': s[0],
                'data_points': s[1],
                'anomalies': s[2] or 0,
                'anomaly_rate': (s[2] or 0) / s[1] if s[1] > 0 else 0
            }
            for s in source_stats
        ]
    })

@app.route('/api/data/<source>', methods=['GET'])
def get_data(source):
    """Get recent data for a source"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    limit = request.args.get('limit', 100, type=int)
    c.execute('''SELECT value, timestamp, is_anomaly, score 
                 FROM data_points 
                 WHERE source=? 
                 ORDER BY timestamp DESC 
                 LIMIT ?''', (source, limit))
    
    results = c.fetchall()
    conn.close()
    
    return jsonify({
        'source': source,
        'data_points': [
            {
                'value': r[0],
                'timestamp': r[1],
                'is_anomaly': r[2],
                'score': r[3]
            }
            for r in reversed(results)
        ]
    })

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get recent alerts"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    limit = request.args.get('limit', 50, type=int)
    c.execute('''SELECT source, message, severity, timestamp, data_value 
                 FROM alerts 
                 ORDER BY timestamp DESC 
                 LIMIT ?''', (limit,))
    
    results = c.fetchall()
    conn.close()
    
    return jsonify({
        'alerts': [
            {
                'source': r[0],
                'message': r[1],
                'severity': r[2],
                'timestamp': r[3],
                'value': r[4]
            }
            for r in reversed(results)
        ]
    })

@app.route('/api/anomalies', methods=['GET'])
def get_anomalies():
    """Get recent anomalies"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    limit = request.args.get('limit', 50, type=int)
    c.execute('''SELECT source, value, timestamp, score 
                 FROM data_points 
                 WHERE is_anomaly=1 
                 ORDER BY timestamp DESC 
                 LIMIT ?''', (limit,))
    
    results = c.fetchall()
    conn.close()
    
    return jsonify({
        'anomalies': [
            {
                'source': r[0],
                'value': r[1],
                'timestamp': r[2],
                'score': r[3]
            }
            for r in reversed(results)
        ]
    })

@app.route('/api/data/all', methods=['GET'])
def get_all_data():
    """Get all data points"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    limit = request.args.get('limit', 500, type=int)
    c.execute('''SELECT source, value, timestamp, is_anomaly, score 
                 FROM data_points 
                 ORDER BY timestamp DESC 
                 LIMIT ?''', (limit,))
    
    results = c.fetchall()
    conn.close()
    
    return jsonify({
        'data_points': [
            {
                'source': r[0],
                'value': r[1],
                'timestamp': r[2],
                'is_anomaly': r[3],
                'score': r[4]
            }
            for r in reversed(results)
        ]
    })

@app.route('/api/submit', methods=['POST'])
def submit_data():
    """Submit manual data point"""
    try:
        data = request.json
        source = data.get('source')
        value = float(data.get('value'))
        
        if not source or source not in detectors:
            return jsonify({'error': 'Invalid source'}), 400
        
        result = detectors[source].process(value)
        
        return jsonify({
            'success': True,
            'result': result,
            'detector_status': detectors[source].get_stats()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/configure', methods=['POST'])
def configure():
    """Configure detector parameters"""
    try:
        data = request.json
        source = data.get('source')
        threshold = float(data.get('threshold', 2.5))
        
        if source not in detectors:
            return jsonify({'error': 'Invalid source'}), 400
        
        detectors[source].detector.set_threshold(threshold)
        
        return jsonify({
            'success': True,
            'message': f'Configured {source} with threshold {threshold}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/dashboard')
def dashboard():
    """Serve the dashboard HTML with real-time charts"""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-Time Anomaly Detection Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --primary: #667eea;
            --secondary: #764ba2;
            --danger: #ff6b6b;
            --warning: #ffd93d;
            --success: #6bcf7f;
            --bg-light: #f8f9fa;
            --bg-dark: #1a1a2e;
            --text-light: #333;
            --text-dark: #f0f0f0;
            --card-light: #ffffff;
            --card-dark: #16213e;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            min-height: 100vh;
            padding: 20px;
            color: var(--text-light);
            transition: all 0.3s ease;
        }
        
        body.dark-theme {
            background: var(--bg-dark);
            color: var(--text-dark);
        }
        
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            flex-wrap: wrap;
            gap: 20px;
        }
        
        .header-content h1 {
            color: white;
            font-size: 2.2em;
            margin-bottom: 5px;
        }
        
        .header-content .subtitle {
            color: rgba(255,255,255,0.9);
            font-size: 1em;
        }
        
        body.dark-theme .header-content h1,
        body.dark-theme .header-content .subtitle {
            color: var(--text-dark);
        }
        
        .controls {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        button {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            background: white;
            color: var(--primary);
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .container { max-width: 1600px; margin: 0 auto; }
        
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .status-box {
            background: var(--card-light);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            text-align: center;
            border-top: 4px solid var(--primary);
            transition: all 0.3s ease;
        }
        
        body.dark-theme .status-box {
            background: var(--card-dark);
            color: var(--text-dark);
        }
        
        .status-box:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        }
        
        .status-box.danger { border-top-color: var(--danger); }
        .status-box.warning { border-top-color: var(--warning); }
        .status-box.success { border-top-color: var(--success); }
        
        .status-label {
            color: #999;
            font-size: 0.9em;
            margin-bottom: 10px;
        }
        
        body.dark-theme .status-label { color: #aaa; }
        
        .status-value {
            font-size: 2.5em;
            font-weight: bold;
            color: var(--primary);
        }
        
        .status-value.danger { color: var(--danger); }
        .status-value.warning { color: var(--warning); }
        .status-value.success { color: var(--success); }
        
        .status-change {
            font-size: 0.8em;
            margin-top: 5px;
            color: #666;
        }
        
        body.dark-theme .status-change { color: #aaa; }
        
        .filters {
            background: var(--card-light);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        body.dark-theme .filters {
            background: var(--card-dark);
        }
        
        .filters select, .filters input {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 0.95em;
        }
        
        body.dark-theme .filters select,
        body.dark-theme .filters input {
            background: #333;
            color: var(--text-dark);
            border-color: #555;
        }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .chart-card {
            background: var(--card-light);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            position: relative;
        }
        
        body.dark-theme .chart-card {
            background: var(--card-dark);
        }
        
        .chart-card h3 {
            color: var(--primary);
            margin-bottom: 15px;
            font-size: 1.2em;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        body.dark-theme .chart-card h3 {
            color: #6bcf7f;
        }
        
        .chart-container {
            position: relative;
            height: 300px;
        }
        
        .list-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }
        
        .list-card {
            background: var(--card-light);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            max-height: 500px;
            overflow-y: auto;
        }
        
        body.dark-theme .list-card {
            background: var(--card-dark);
        }
        
        .list-card h3 {
            color: var(--primary);
            margin-bottom: 15px;
            border-bottom: 2px solid var(--primary);
            padding-bottom: 10px;
        }
        
        body.dark-theme .list-card h3 {
            color: #6bcf7f;
            border-color: #6bcf7f;
        }
        
        .item {
            padding: 12px;
            margin-bottom: 8px;
            border-radius: 5px;
            border-left: 4px solid var(--primary);
            background: #f0f4ff;
            transition: all 0.3s ease;
        }
        
        body.dark-theme .item {
            background: #2a2a4e;
            border-left-color: #6bcf7f;
        }
        
        .item:hover {
            transform: translateX(5px);
            box-shadow: 0 5px 10px rgba(0,0,0,0.1);
        }
        
        .item.anomaly {
            border-left-color: var(--danger);
            background: #ffe5e5;
        }
        
        body.dark-theme .item.anomaly {
            background: #4e2a2a;
            border-left-color: #ff6b6b;
        }
        
        .item.warning {
            border-left-color: var(--warning);
            background: #fff3cd;
        }
        
        .item-value {
            color: var(--danger);
            font-weight: bold;
            font-size: 0.95em;
        }
        
        body.dark-theme .item-value {
            color: #ff8a8a;
        }
        
        .item-time {
            font-size: 0.85em;
            color: #999;
            margin-top: 5px;
        }
        
        body.dark-theme .item-time {
            color: #aaa;
        }
        
        .item-source {
            font-size: 0.8em;
            color: #667eea;
            font-weight: bold;
            margin-top: 3px;
        }
        
        body.dark-theme .item-source {
            color: #6bcf7f;
        }
        
        .empty {
            text-align: center;
            color: #999;
            padding: 20px;
            font-style: italic;
        }
        
        body.dark-theme .empty {
            color: #aaa;
        }
        
        .stats-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #eee;
        }
        
        body.dark-theme .stats-row {
            border-top-color: #444;
        }
        
        .stat-mini {
            font-size: 0.85em;
        }
        
        .stat-mini-label {
            color: #999;
            font-size: 0.75em;
        }
        
        body.dark-theme .stat-mini-label {
            color: #aaa;
        }
        
        .stat-mini-value {
            font-weight: bold;
            color: var(--primary);
            margin-top: 3px;
        }
        
        body.dark-theme .stat-mini-value {
            color: #6bcf7f;
        }
        
        .refresh-badge {
            display: inline-block;
            background: var(--success);
            color: white;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.7em;
            margin-left: 10px;
        }
        
        @media (max-width: 1024px) {
            .charts-grid { grid-template-columns: 1fr; }
            .list-grid { grid-template-columns: 1fr; }
            .header { flex-direction: column; align-items: flex-start; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-content">
                <h1>🚀 Anomaly Detection Dashboard</h1>
                <p class="subtitle">Real-Time Monitoring & Alert System</p>
            </div>
            <div class="controls">
                <button onclick="toggleTheme()">🌙 Dark Mode</button>
                <button onclick="exportData()">📥 Export</button>
                <button onclick="refreshNow()">🔄 Refresh</button>
            </div>
        </div>
        
        <div class="status-grid">
            <div class="status-box success">
                <div class="status-label">Server Status</div>
                <div class="status-value">🟢 ACTIVE</div>
            </div>
            <div class="status-box">
                <div class="status-label">Total Data Points</div>
                <div class="status-value" id="dataPoints">0</div>
            </div>
            <div class="status-box danger">
                <div class="status-label">Anomalies Detected</div>
                <div class="status-value anomaly" id="anomalyCount">0</div>
            </div>
            <div class="status-box warning">
                <div class="status-label">Anomaly Rate</div>
                <div class="status-value" id="anomalyRate">0%</div>
            </div>
        </div>
        
        <div class="charts-grid">
            <div class="chart-card">
                <h3>📊 CPU Usage <span class="refresh-badge" id="cpuTime"></span></h3>
                <div class="chart-container"><canvas id="cpuChart"></canvas></div>
            </div>
            <div class="chart-card">
                <h3>💰 Financial Data <span class="refresh-badge" id="finTime"></span></h3>
                <div class="chart-container"><canvas id="financialChart"></canvas></div>
            </div>
            <div class="chart-card">
                <h3>🌐 Network Traffic <span class="refresh-badge" id="netTime"></span></h3>
                <div class="chart-container"><canvas id="networkChart"></canvas></div>
            </div>
            <div class="chart-card">
                <h3>🚨 Fraud Detection <span class="refresh-badge" id="fraudTime"></span></h3>
                <div class="chart-container"><canvas id="fraudChart"></canvas></div>
            </div>
            <div class="chart-card">
                <h3>⚡ Electricity Usage <span class="refresh-badge" id="elecTime"></span></h3>
                <div class="chart-container"><canvas id="electricityChart"></canvas></div>
            </div>
            <div class="chart-card">
                <h3>🔧 Machine Health <span class="refresh-badge" id="machTime"></span></h3>
                <div class="chart-container"><canvas id="machineChart"></canvas></div>
            </div>
            <div class="chart-card">
                <h3>🖥️ Server Traffic <span class="refresh-badge" id="servTime"></span></h3>
                <div class="chart-container"><canvas id="serverChart"></canvas></div>
            </div>
            <div class="chart-card">
                <h3>📈 Stock Risk <span class="refresh-badge" id="stockTime"></span></h3>
                <div class="chart-container"><canvas id="stockChart"></canvas></div>
            </div>
            <div class="chart-card">
                <h3>🌡️ IoT Sensors <span class="refresh-badge" id="iotTime"></span></h3>
                <div class="chart-container"><canvas id="iotChart"></canvas></div>
            </div>
        </div>
        
        <div class="list-grid">
            <div class="list-card">
                <h3>🔔 Recent Anomalies (Top 15)</h3>
                <div id="anomaliesList" class="empty">Loading...</div>
            </div>
            
            <div class="list-card">
                <h3>⚠️ Recent Alerts (Top 15)</h3>
                <div id="alertsList" class="empty">Loading...</div>
            </div>
        </div>
    </div>
    
    <script>
        const charts = {};
        const maxDataPoints = 60;
        let currentTheme = localStorage.getItem('theme') || 'light';
        
        function toggleTheme() {
            currentTheme = currentTheme === 'light' ? 'dark' : 'light';
            document.body.classList.toggle('dark-theme');
            localStorage.setItem('theme', currentTheme);
        }
        
        function refreshNow() {
            updateDashboard();
        }
        
        function exportData() {
            alert('Export feature coming soon! Data export will generate CSV files.');
        }
        
        if (currentTheme === 'dark') {
            document.body.classList.add('dark-theme');
        }
        
        async function updateDashboard() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                let totalPoints = 0, totalAnomalies = 0;
                const detectors = data.detectors || {};
                
                for (const source in detectors) {
                    totalPoints += detectors[source].total_points || 0;
                    totalAnomalies += detectors[source].anomalies || 0;
                }
                
                const rate = totalPoints > 0 ? ((totalAnomalies / totalPoints) * 100).toFixed(2) : 0;
                document.getElementById('dataPoints').textContent = totalPoints.toLocaleString();
                document.getElementById('anomalyCount').textContent = totalAnomalies.toLocaleString();
                document.getElementById('anomalyRate').textContent = rate + '%';
                
                await updateCharts();
                await fetchAnomalies();
                await fetchAlerts();
            } catch (e) {
                console.error('Error:', e);
            }
        }
        
        async function updateCharts() {
            try {
                const response = await fetch('/api/data/all');
                let data = { data_points: [] };
                
                try {
                    data = await response.json();
                } catch (e) {
                    const sources = ['CPU_Usage', 'Financial_Data', 'Network_Traffic'];
                    for (const source of sources) {
                        const res = await fetch(`/api/data/${source}`);
                        const sourceData = await res.json();
                        if (sourceData.data_points) {
                            data.data_points.push(...sourceData.data_points);
                        }
                    }
                }
                
                const dataPoints = data.data_points || [];
                
                const sources = {
                    'CPU_Usage': { color: '#FF6384', label: 'CPU Usage', canvasId: 'cpuChart', badgeId: 'cpuTime' },
                    'Financial_Data': { color: '#36A2EB', label: 'Financial', canvasId: 'financialChart', badgeId: 'finTime' },
                    'Network_Traffic': { color: '#FFCE56', label: 'Network', canvasId: 'networkChart', badgeId: 'netTime' },
                    'Fraud_Detection': { color: '#4BC0C0', label: 'Fraud Detection', canvasId: 'fraudChart', badgeId: 'fraudTime' },
                    'Electricity_Usage': { color: '#9966FF', label: 'Electricity', canvasId: 'electricityChart', badgeId: 'elecTime' },
                    'Machine_Health': { color: '#FF9F40', label: 'Machine Health', canvasId: 'machineChart', badgeId: 'machTime' },
                    'Server_Traffic': { color: '#66BB6A', label: 'Server Traffic', canvasId: 'serverChart', badgeId: 'servTime' },
                    'Stock_Risk': { color: '#EF5350', label: 'Stock Risk', canvasId: 'stockChart', badgeId: 'stockTime' },
                    'IoT_Sensors': { color: '#AB47BC', label: 'IoT Sensors', canvasId: 'iotChart', badgeId: 'iotTime' }
                };
                
                for (const [source, config] of Object.entries(sources)) {
                    const sourceData = dataPoints
                        .filter(d => d.source === source)
                        .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp))
                        .slice(-maxDataPoints);
                    
                    if (sourceData.length === 0) continue;
                    
                    const labels = sourceData.map(d => new Date(d.timestamp).toLocaleTimeString());
                    const values = sourceData.map(d => d.value);
                    const backgroundColor = sourceData.map(d => d.is_anomaly ? 'rgba(255, 107, 107, 0.3)' : 'rgba(102, 126, 234, 0.1)');
                    const anomalyCount = sourceData.filter(d => d.is_anomaly).length;
                    
                    updateOrCreateChart(
                        config.canvasId,
                        labels,
                        values,
                        config.label,
                        config.color,
                        backgroundColor,
                        'line'
                    );
                    
                    if (config.badgeId) {
                        document.getElementById(config.badgeId).textContent = anomalyCount > 0 ? `${anomalyCount} anomalies` : 'OK';
                    }
                }
            } catch (e) {
                console.error('Chart update error:', e);
            }
        }
        
        function updateOrCreateChart(canvasId, labels, data, label, color, backgroundColor, type) {
            const ctx = document.getElementById(canvasId);
            if (!ctx) return;
            
            if (charts[canvasId]) {
                charts[canvasId].data.labels = labels;
                charts[canvasId].data.datasets[0].data = data;
                if (type === 'line' && Array.isArray(backgroundColor)) {
                    charts[canvasId].data.datasets[0].backgroundColor = backgroundColor;
                }
                charts[canvasId].update('none');
            } else {
                charts[canvasId] = new Chart(ctx, {
                    type: type,
                    data: {
                        labels: labels,
                        datasets: [{
                            label: label,
                            data: data,
                            borderColor: color,
                            backgroundColor: backgroundColor || 'rgba(102, 126, 234, 0.1)',
                            borderWidth: 2,
                            tension: 0.4,
                            fill: type === 'line',
                            pointRadius: 3,
                            pointBackgroundColor: color
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { display: true },
                            filler: { propagate: true }
                        },
                        scales: {
                            y: { beginAtZero: false }
                        }
                    }
                });
            }
        }
        
        async function fetchAnomalies() {
            try {
                const response = await fetch('/api/anomalies?limit=15');
                const data = await response.json();
                const anomalies = data.anomalies || [];
                
                const list = document.getElementById('anomaliesList');
                if (anomalies.length > 0) {
                    list.innerHTML = anomalies.map(a => `
                        <div class="item anomaly">
                            <div class="item-value">⚠️ ${a.source}: ${a.value.toFixed(2)}</div>
                            <div class="item-source">Score: ${(a.score * 100).toFixed(1)}%</div>
                            <div class="item-time">${new Date(a.timestamp).toLocaleString()}</div>
                        </div>
                    `).join('');
                } else {
                    list.innerHTML = '<div class="empty">No anomalies detected</div>';
                }
            } catch (e) {
                console.error('Error fetching anomalies:', e);
            }
        }
        
        async function fetchAlerts() {
            try {
                const response = await fetch('/api/alerts?limit=15');
                const data = await response.json();
                const alerts = data.alerts || [];
                
                const list = document.getElementById('alertsList');
                if (alerts.length > 0) {
                    list.innerHTML = alerts.map(a => `
                        <div class="item warning">
                            <div class="item-value">🔔 ${a.source}</div>
                            <div class="item-source">Severity: ${a.severity}</div>
                            <div class="item-time">${new Date(a.timestamp).toLocaleString()}</div>
                        </div>
                    `).join('');
                } else {
                    list.innerHTML = '<div class="empty">No active alerts</div>';
                }
            } catch (e) {
                console.error('Error fetching alerts:', e);
            }
        }
        
        updateDashboard();
        setInterval(updateDashboard, 2000);
    </script>
</body>
</html>"""
    return html

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({'status': 'healthy', 'service': 'Anomaly Detection Server'})

@app.route('/api/debug', methods=['GET'])
def debug_endpoint():
    """Debug endpoint to check data"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('SELECT source, COUNT(*) as count FROM data_points GROUP BY source')
    source_counts = dict(c.fetchall())
    
    c.execute('SELECT * FROM data_points ORDER BY timestamp DESC LIMIT 5')
    recent = c.fetchall()
    
    conn.close()
    
    return jsonify({
        'source_counts': source_counts,
        'recent_data': recent,
        'detectors': [d.get_stats() for d in detectors.values()]
    })

if __name__ == '__main__':
    print("\n" + "="*70)
    print("  ANOMALY DETECTION SERVER".center(70))
    print("="*70)
    
    # Initialize database
    init_database()
    print("\nDatabase initialized")
    
    # Start background worker
    worker_thread = threading.Thread(target=auto_fetch_and_process, daemon=True)
    worker_thread.start()
    
    print("\n" + "="*70)
    print("API ENDPOINTS:")
    print("="*70)
    print("GET  /                   - Home")
    print("GET  /api/status         - Server status")
    print("GET  /api/stats          - Statistics")
    print("GET  /api/data/<source>  - Get data for source")
    print("GET  /api/alerts         - Get alerts")
    print("GET  /api/anomalies      - Get anomalies")
    print("GET  /api/stream/anomalies - SSE anomalies/alerts stream")
    print("POST /api/submit         - Submit data point")
    print("POST /api/configure      - Configure detector")
    print("GET  /health             - Health check")
    print("="*70)
    
    # Run Flask server
    print("\nStarting Flask server on http://0.0.0.0:5001\n")
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)

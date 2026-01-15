"""
Flask Web Application for Anomaly Detection System
Simple web-based anomaly detection with synthetic time-series data
"""
import os
import sys
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify
from datetime import datetime
import plotly.graph_objects as go
import plotly.io as pio
import json
from collections import defaultdict
import random
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processor import TimeSeriesProcessor
from models import StatisticalDetector
from data.generate_data import (
    generate_cpu_usage_data, 
    generate_financial_data, 
    generate_network_traffic_data, 
    generate_sensor_data
)

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Store global data for streaming
streaming_data = {
    'CPU Usage (Server Monitoring)': {'data': [], 'anomalies': []},
    'Financial Data': {'data': [], 'anomalies': []},
    'Network Traffic (Packets/sec)': {'data': [], 'anomalies': []},
    'Temperature Sensor (°C)': {'data': [], 'anomalies': []},
    'Stock Price ($)': {'data': [], 'anomalies': []}
}

# Store anomaly history with timestamps
anomaly_history = defaultdict(list)

# Real-time CPU data generator state
cpu_time_index = 0
cpu_base_value = 50.0

# Real-time data generator state for all charts
realtime_state = {
    'CPU Usage (Server Monitoring)': {'index': 0, 'base_value': 50.0},
    'Financial Data': {'index': 0, 'base_value': 100.0},
    'Network Traffic (Packets/sec)': {'index': 0, 'base_value': 500.0},
    'Temperature Sensor (°C)': {'index': 0, 'base_value': 25.0},
    'Stock Price ($)': {'index': 0, 'base_value': 150.0}
}

# Create necessary folders
os.makedirs('logs', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

@app.route('/')
def index():
    """Main page - Shows modern dashboard with real-time anomaly detection"""
    print("\n" + "="*70)
    print("INITIALIZING MODERN ANOMALY DETECTION DASHBOARD...")
    print("="*70)
    
    # Initialize datasets
    datasets = {
        'CPU Usage (Server Monitoring)': generate_cpu_usage_data(n_samples=500),
        'Financial Data': generate_financial_data(n_samples=500),
        'Network Traffic (Packets/sec)': generate_network_traffic_data(n_samples=500),
        'Temperature Sensor (°C)': generate_sensor_data(n_samples=500),
        'Stock Price ($)': generate_financial_data(n_samples=500)
    }
    
    # Store initial data globally for streaming
    processor = TimeSeriesProcessor()
    detector = StatisticalDetector(threshold=2.5)
    
    for name, df in datasets.items():
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        data = df[numeric_cols[-1]].values.tolist()
        
        train_data, test_data = processor.split_data(np.array(data), train_ratio=0.7)
        train_normalized = processor.normalize(train_data, fit=True)
        test_normalized = processor.normalize(test_data, fit=False)
        
        detector.fit(train_normalized)
        result = detector.predict_with_scores(test_normalized)
        
        anomaly_indices = [i for i, pred in enumerate(result['predictions']) if pred == 1]
        
        streaming_data[name]['data'] = test_data.tolist()
        streaming_data[name]['anomalies'] = anomaly_indices
        
        # Store anomaly history with timestamps and values
        for idx in anomaly_indices:
            anomaly_history[name].append({
                'timestamp': datetime.now().isoformat(),
                'index': idx,
                'value': float(test_data[idx]),
                'detected_at': datetime.now().isoformat()
            })
        
        print(f"✓ {name}: {len(test_data)} points, {len(anomaly_indices)} anomalies")
    
    print("✅ Dashboard initialized - LIVE and ready!")
    print("="*70 + "\n")
    
    return render_template('modern_dashboard.html')


@app.route('/api/get-streaming-data')
def get_streaming_data():
    """API endpoint that returns current streaming data"""
    response_data = {}
    total_points = 0
    total_anomalies = 0
    
    for name in streaming_data.keys():
        response_data[name] = {
            'values': streaming_data[name]['data'],
            'anomalies': streaming_data[name]['anomalies'],
            'count': len(streaming_data[name]['data'])
        }
        total_points += len(streaming_data[name]['data'])
        total_anomalies += len(streaming_data[name]['anomalies'])
    
    response_data['summary'] = {
        'total_points': total_points,
        'total_anomalies': total_anomalies,
        'anomaly_rate': (total_anomalies / total_points * 100) if total_points > 0 else 0,
        'server_status': 'ACTIVE'
    }
    
    return jsonify(response_data)


@app.route('/api/get-anomaly-history')
def get_anomaly_history():
    """API endpoint that returns historical anomalies with timestamps"""
    history = {}
    
    for dataset_name, anomalies in anomaly_history.items():
        # Return last 50 anomalies for each dataset, most recent first
        history[dataset_name] = sorted(anomalies, key=lambda x: x['detected_at'], reverse=True)[:50]
    
    return jsonify({
        'history': history,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/get-realtime-cpu')
def get_realtime_cpu():
    """API endpoint that returns a single new CPU data point for real-time sliding chart"""
    return get_realtime_data('CPU Usage (Server Monitoring)', 20, 80, 2, 40)


@app.route('/api/get-realtime-financial')
def get_realtime_financial():
    """API endpoint for real-time Financial data"""
    return get_realtime_data('Financial Data', 50, 200, 5, 60)


@app.route('/api/get-realtime-network')
def get_realtime_network():
    """API endpoint for real-time Network Traffic data"""
    return get_realtime_data('Network Traffic (Packets/sec)', 200, 1000, 20, 200)


@app.route('/api/get-realtime-temperature')
def get_realtime_temperature():
    """API endpoint for real-time Temperature data"""
    return get_realtime_data('Temperature Sensor (°C)', 15, 35, 1, 8)


@app.route('/api/get-realtime-stock')
def get_realtime_stock():
    """API endpoint for real-time Stock Price data"""
    return get_realtime_data('Stock Price ($)', 100, 200, 3, 40)


def get_realtime_data(dataset_name, min_val, max_val, drift, anomaly_range):
    """Generic function to generate real-time data for any chart"""
    state = realtime_state[dataset_name]
    
    # Generate realistic pattern with drift
    state['base_value'] += random.uniform(-drift, drift)
    state['base_value'] = max(min_val, min(max_val, state['base_value']))
    
    # Add noise
    noise = random.gauss(0, drift)
    value = state['base_value'] + noise
    
    # Occasionally create anomalies (10% chance)
    is_anomaly = False
    if random.random() < 0.10:
        anomaly_type = random.choice(['spike', 'drop'])
        if anomaly_type == 'spike':
            value = state['base_value'] + random.uniform(anomaly_range * 0.6, anomaly_range)
        else:
            value = state['base_value'] - random.uniform(anomaly_range * 0.6, anomaly_range)
        is_anomaly = True
    
    # Clamp to range
    value = max(min_val * 0.8, min(max_val * 1.2, value))
    
    state['index'] += 1
    current_time = datetime.now().strftime('%H:%M:%S')
    
    # Store anomaly in history if detected
    if is_anomaly:
        anomaly_history[dataset_name].append({
            'timestamp': current_time,
            'index': state['index'],
            'value': float(value),
            'detected_at': datetime.now().isoformat()
        })
    
    return jsonify({
        'index': state['index'],
        'value': round(value, 2),
        'timestamp': current_time,
        'is_anomaly': is_anomaly,
        'dataset': dataset_name
    })


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  ANOMALY DETECTION WEB APPLICATION")
    print("=" * 70)
    print("\nStarting Flask server...")
    print("Open your browser and go to: http://localhost:5000")
    print("\nThis web application will:")
    print("  1. Automatically generate synthetic time-series data")
    print("  2. Detect anomalies using Statistical Z-Score method")
    print("  3. Display 4 interactive charts with anomalies highlighted")
    print("\n" + "=" * 70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

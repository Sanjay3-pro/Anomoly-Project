"""
Flask Web Application for Anomaly Detection System
Simple web-based anomaly detection with synthetic time-series data
"""
import os
import sys
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify
import plotly.graph_objects as go
import plotly.io as pio
import json

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

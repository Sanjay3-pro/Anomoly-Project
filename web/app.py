"""
Flask Web Application for Anomaly Detection System
"""
import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processor import TimeSeriesProcessor
from models import StatisticalDetector, IsolationForestDetector, LOFDetector
from data.generate_data import generate_cpu_usage_data, generate_financial_data, generate_network_traffic_data, generate_sensor_data

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store detector results in memory
detection_results = {}

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """Get list of available sample datasets"""
    datasets = {
        'cpu_usage': 'CPU Usage Monitoring',
        'network_traffic': 'Network Traffic Analysis',
        'sensor_data': 'Temperature Sensor Data',
        'financial_data': 'Financial Data (Stock Prices)'
    }
    return jsonify(datasets)

@app.route('/api/load-sample', methods=['POST'])
def load_sample():
    """Load a sample dataset"""
    try:
        dataset_name = request.json.get('dataset')
        
        # Load appropriate dataset
        if dataset_name == 'cpu_usage':
            df = generate_cpu_usage_data(n_samples=500)
        elif dataset_name == 'network_traffic':
            df = generate_network_traffic_data(n_samples=500)
        elif dataset_name == 'sensor_data':
            df = generate_sensor_data(n_samples=500)
        elif dataset_name == 'financial_data':
            df = generate_financial_data(n_samples=500)
        else:
            return jsonify({'error': 'Unknown dataset'}), 400
        
        # Get the value column (last numeric column)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        value_col = numeric_cols[-1]
        data = df[value_col].values.tolist()
        
        return jsonify({
            'success': True,
            'data': data,
            'dataset': dataset_name,
            'size': len(data)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload and process a CSV file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load data
        df = pd.read_csv(filepath)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return jsonify({'error': 'No numeric columns found'}), 400
        
        # Use last numeric column as data
        data = df[numeric_cols[-1]].values.tolist()
        
        return jsonify({
            'success': True,
            'data': data,
            'filename': filename,
            'size': len(data),
            'columns': df.columns.tolist()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detect', methods=['POST'])
def detect_anomalies():
    """Detect anomalies using selected methods"""
    try:
        data = np.array(request.json.get('data', []))
        methods = request.json.get('methods', ['statistical'])
        threshold = float(request.json.get('threshold', 2.5))
        train_ratio = float(request.json.get('train_ratio', 0.7))
        
        if len(data) < 10:
            return jsonify({'error': 'Need at least 10 data points'}), 400
        
        # Preprocess data
        processor = TimeSeriesProcessor()
        train_data, test_data = processor.split_data(data, train_ratio=train_ratio)
        train_normalized = processor.normalize(train_data, fit=True)
        test_normalized = processor.normalize(test_data, fit=False)
        
        results = {}
        
        # Statistical detector
        if 'statistical' in methods:
            detector = StatisticalDetector(threshold=threshold)
            detector.fit(train_normalized)
            result = detector.predict_with_scores(test_normalized)
            results['statistical'] = {
                'predictions': result['predictions'].tolist(),
                'scores': result['scores'].tolist(),
                'anomalies': int(result['anomaly_count']),
                'rate': float(result['anomaly_rate']),
                'threshold': threshold
            }
        
        # Isolation Forest
        if 'isolation_forest' in methods:
            detector = IsolationForestDetector(contamination=0.08)
            detector.fit(train_normalized)
            result = detector.predict_with_scores(test_normalized)
            results['isolation_forest'] = {
                'predictions': result['predictions'].tolist(),
                'scores': result['scores'].tolist(),
                'anomalies': int(result['anomaly_count']),
                'rate': float(result['anomaly_rate']),
                'threshold': result['threshold']
            }
        
        # LOF
        if 'lof' in methods:
            try:
                detector = LOFDetector(contamination=0.08)
                detector.fit(train_normalized)
                result = detector.predict_with_scores(test_normalized)
                results['lof'] = {
                    'predictions': result['predictions'].tolist(),
                    'scores': result['scores'].tolist(),
                    'anomalies': int(result['anomaly_count']),
                    'rate': float(result['anomaly_rate']),
                    'threshold': result['threshold']
                }
            except:
                pass  # Skip if LOF fails
        
        # Store for later use
        detection_results['last'] = {
            'data': test_data.tolist(),
            'results': results
        }
        
        return jsonify({
            'success': True,
            'test_size': len(test_data),
            'train_size': len(train_data),
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def compare_methods():
    """Compare different detection methods"""
    try:
        data = np.array(request.json.get('data', []))
        
        if len(data) < 20:
            return jsonify({'error': 'Need at least 20 data points'}), 400
        
        processor = TimeSeriesProcessor()
        train_data, test_data = processor.split_data(data, train_ratio=0.7)
        train_normalized = processor.normalize(train_data, fit=True)
        test_normalized = processor.normalize(test_data, fit=False)
        
        comparison = {}
        
        # Statistical
        detector = StatisticalDetector(threshold=2.5)
        detector.fit(train_normalized)
        result = detector.predict_with_scores(test_normalized)
        comparison['statistical'] = {
            'method': 'Statistical (Z-Score)',
            'anomalies': int(result['anomaly_count']),
            'rate': float(result['anomaly_rate']),
            'mean_score': float(result['scores'].mean()),
            'max_score': float(result['scores'].max())
        }
        
        # Isolation Forest
        detector = IsolationForestDetector(contamination=0.08)
        detector.fit(train_normalized)
        result = detector.predict_with_scores(test_normalized)
        comparison['isolation_forest'] = {
            'method': 'Isolation Forest',
            'anomalies': int(result['anomaly_count']),
            'rate': float(result['anomaly_rate']),
            'mean_score': float(result['scores'].mean()),
            'max_score': float(result['scores'].max())
        }
        
        return jsonify({
            'success': True,
            'comparison': comparison
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard')
def dashboard():
    """Real-time dashboard"""
    return render_template('dashboard.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'Anomaly Detection API'})

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  ANOMALY DETECTION WEB APPLICATION")
    print("=" * 70)
    print("\nStarting Flask server...")
    print("Open your browser and go to: http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /                    - Home page")
    print("  GET  /dashboard           - Real-time dashboard")
    print("  GET  /api/datasets        - List available datasets")
    print("  POST /api/load-sample     - Load sample data")
    print("  POST /api/upload          - Upload CSV file")
    print("  POST /api/detect          - Detect anomalies")
    print("  POST /api/compare         - Compare methods")
    print("  GET  /health              - Health check")
    print("\n" + "=" * 70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

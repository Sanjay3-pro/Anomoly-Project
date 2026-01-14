"""
Real-Time Anomaly Detection System
Continuous streaming data monitoring with live anomaly detection
"""
import numpy as np
import os
import sys
import time
from collections import deque
from datetime import datetime, timedelta
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processor import TimeSeriesProcessor
from models import StatisticalDetector, IsolationForestDetector
from data.generate_data import (
    generate_cpu_usage_data, 
    generate_financial_data, 
    generate_network_traffic_data
)

class RealtimeAnomalyDetector:
    """Real-time anomaly detection system"""
    
    def __init__(self, window_size=100, update_frequency=1):
        """
        Initialize real-time detector
        
        Args:
            window_size: Size of sliding window
            update_frequency: Update interval in seconds
        """
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.data_buffer = deque(maxlen=window_size)
        self.predictions_buffer = deque(maxlen=window_size)
        self.scores_buffer = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        
        self.processor = TimeSeriesProcessor()
        self.detector = StatisticalDetector(threshold=2.5)
        self.detector_if = IsolationForestDetector(contamination=0.08)
        
        self.is_trained = False
        self.anomaly_count = 0
        self.total_points = 0
        self.start_time = datetime.now()
        self.running = True
    
    def train(self, initial_data):
        """Train detectors on initial data"""
        normalized = self.processor.normalize(initial_data, fit=True)
        self.detector.fit(normalized)
        self.detector_if.fit(normalized)
        self.is_trained = True
    
    def add_point(self, value, timestamp=None):
        """Add new data point and detect anomaly"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.data_buffer.append(value)
        self.timestamps.append(timestamp)
        self.total_points += 1
        
        if len(self.data_buffer) >= 10 and self.is_trained:
            # Normalize new data
            data_array = np.array(list(self.data_buffer))
            normalized = self.processor.normalize(data_array, fit=False)
            
            # Detect anomaly
            pred_stat = self.detector.predict(normalized)[-1]
            pred_if = self.detector_if.predict(normalized)[-1]
            score_stat = self.detector.score(normalized)[-1]
            score_if = self.detector_if.score(normalized)[-1]
            
            # Ensemble voting
            ensemble_pred = 1 if (pred_stat + pred_if) >= 1 else 0
            
            self.predictions_buffer.append(ensemble_pred)
            self.scores_buffer.append(max(score_stat, score_if))
            
            if ensemble_pred == 1:
                self.anomaly_count += 1
            
            return {
                'value': value,
                'timestamp': timestamp,
                'is_anomaly': ensemble_pred,
                'score': max(score_stat, score_if),
                'score_stat': score_stat,
                'score_if': score_if
            }
        
        return None
    
    def get_stats(self):
        """Get current statistics"""
        if len(self.predictions_buffer) == 0:
            return None
        
        anomaly_rate = self.anomaly_count / max(self.total_points, 1)
        uptime = datetime.now() - self.start_time
        
        return {
            'total_points': self.total_points,
            'anomalies': self.anomaly_count,
            'anomaly_rate': anomaly_rate,
            'uptime': str(uptime).split('.')[0],
            'window_size': len(self.data_buffer),
            'avg_score': np.mean(list(self.scores_buffer)) if self.scores_buffer else 0,
            'max_score': np.max(list(self.scores_buffer)) if self.scores_buffer else 0,
            'min_value': np.min(list(self.data_buffer)),
            'max_value': np.max(list(self.data_buffer)),
            'mean_value': np.mean(list(self.data_buffer))
        }

def clear_screen():
    """Clear console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print header"""
    print("\n" + "=" * 80)
    print("  REAL-TIME ANOMALY DETECTION SYSTEM".center(80))
    print("=" * 80)

def print_live_metrics(detector, data_source):
    """Print live metrics in real-time"""
    print(f"\n[LIVE DATA SOURCE: {data_source}]")
    print("-" * 80)
    
    stats = detector.get_stats()
    if stats is None:
        print("Initializing...")
        return
    
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Uptime: {stats['uptime']}")
    print(f"Data Points Processed: {stats['total_points']}")
    print(f"Anomalies Detected: {stats['anomalies']} ({stats['anomaly_rate']*100:.2f}%)")
    print(f"Current Window Size: {stats['window_size']} / {detector.window_size}")
    print()
    print(f"Value Range: {stats['min_value']:.2f} - {stats['max_value']:.2f}")
    print(f"Mean Value: {stats['mean_value']:.2f}")
    print(f"Avg Anomaly Score: {stats['avg_score']:.4f}")
    print(f"Max Anomaly Score: {stats['max_score']:.4f}")

def print_visualization(detector):
    """Print ASCII visualization of data"""
    if len(detector.data_buffer) < 5:
        return
    
    data = list(detector.data_buffer)[-50:]  # Last 50 points
    preds = list(detector.predictions_buffer)[-50:]
    
    print("\n[LIVE DATA STREAM VISUALIZATION]")
    print("-" * 80)
    
    # Normalize for visualization
    min_val = min(data)
    max_val = max(data)
    range_val = max_val - min_val if max_val != min_val else 1
    
    for i, (val, pred) in enumerate(zip(data, preds)):
        # Create bar
        bar_height = int(((val - min_val) / range_val) * 30)
        bar = "█" * bar_height if bar_height > 0 else "▁"
        
        # Label
        status = "[ANOMALY!]" if pred == 1 else "[NORMAL  ]"
        
        print(f"  {i:3d} {status} {val:8.2f} {bar}")

def run_cpu_monitoring():
    """Real-time CPU usage monitoring"""
    print_header()
    print("\nMODE: CPU Usage Monitoring")
    print("Data source: Simulated CPU usage with anomalies")
    
    # Generate initial data for training
    print("\nTraining detector on initial 200 samples...")
    df = generate_cpu_usage_data(n_samples=200)
    initial_data = df['cpu_usage'].values
    
    detector = RealtimeAnomalyDetector(window_size=100, update_frequency=1)
    detector.train(initial_data)
    print("Detector trained!")
    
    # Stream new data
    print("\nStarting real-time stream...")
    print("Press Ctrl+C to stop\n")
    
    df_stream = generate_cpu_usage_data(n_samples=500)
    stream_data = df_stream['cpu_usage'].values
    
    try:
        for i, value in enumerate(stream_data):
            result = detector.add_point(value)
            
            if result and result['is_anomaly']:
                print(f"\n!!! ANOMALY DETECTED !!!")
                print(f"    Value: {result['value']:.2f}")
                print(f"    Score: {result['score']:.4f}")
                print(f"    Time: {result['timestamp'].strftime('%H:%M:%S')}")
            
            # Print status every 20 points
            if (i + 1) % 20 == 0:
                clear_screen()
                print_header()
                print("\nMODE: CPU Usage Monitoring (LIVE)")
                print_live_metrics(detector, "CPU Monitoring")
                print_visualization(detector)
            
            time.sleep(detector.update_frequency)
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
        return detector

def run_financial_monitoring():
    """Real-time financial data monitoring"""
    print_header()
    print("\nMODE: Financial Data Monitoring")
    print("Data source: Simulated stock price data")
    
    # Generate initial data
    print("\nTraining detector on initial 150 samples...")
    df = generate_financial_data(n_samples=150)
    initial_data = df['price'].values
    
    detector = RealtimeAnomalyDetector(window_size=100, update_frequency=0.5)
    detector.train(initial_data)
    print("Detector trained!")
    
    # Stream new data
    print("\nStarting real-time stream...")
    print("Press Ctrl+C to stop\n")
    
    df_stream = generate_financial_data(n_samples=500)
    stream_data = df_stream['price'].values
    
    try:
        for i, value in enumerate(stream_data):
            result = detector.add_point(value)
            
            if result and result['is_anomaly']:
                print(f"\n!!! PRICE ANOMALY DETECTED !!!")
                print(f"    Price: ${result['value']:.2f}")
                print(f"    Deviation Score: {result['score']:.4f}")
                print(f"    Alert Level: MEDIUM")
            
            # Print status every 15 points
            if (i + 1) % 15 == 0:
                clear_screen()
                print_header()
                print("\nMODE: Financial Data Monitoring (LIVE)")
                print_live_metrics(detector, "Stock Price Monitoring")
                print_visualization(detector)
            
            time.sleep(detector.update_frequency)
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
        return detector

def run_network_monitoring():
    """Real-time network traffic monitoring"""
    print_header()
    print("\nMODE: Network Traffic Monitoring")
    print("Data source: Simulated network bandwidth with anomalies")
    
    # Generate initial data
    print("\nTraining detector on initial 180 samples...")
    df = generate_network_traffic_data(n_samples=180)
    initial_data = df['traffic_mbps'].values
    
    detector = RealtimeAnomalyDetector(window_size=100, update_frequency=0.8)
    detector.train(initial_data)
    print("Detector trained!")
    
    # Stream new data
    print("\nStarting real-time stream...")
    print("Press Ctrl+C to stop\n")
    
    df_stream = generate_network_traffic_data(n_samples=500)
    stream_data = df_stream['traffic_mbps'].values
    
    try:
        for i, value in enumerate(stream_data):
            result = detector.add_point(value)
            
            if result and result['is_anomaly']:
                print(f"\n!!! NETWORK ANOMALY DETECTED !!!")
                print(f"    Bandwidth: {result['value']:.2f} Mbps")
                print(f"    Anomaly Score: {result['score']:.4f}")
                print(f"    Alert: Potential DDoS or traffic spike")
            
            # Print status every 18 points
            if (i + 1) % 18 == 0:
                clear_screen()
                print_header()
                print("\nMODE: Network Traffic Monitoring (LIVE)")
                print_live_metrics(detector, "Network Bandwidth")
                print_visualization(detector)
            
            time.sleep(detector.update_frequency)
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
        return detector

def print_final_report(detector):
    """Print final detection report"""
    stats = detector.get_stats()
    
    print("\n" + "=" * 80)
    print("  FINAL REPORT".center(80))
    print("=" * 80)
    print(f"\nTotal Runtime: {stats['uptime']}")
    print(f"Total Data Points: {stats['total_points']}")
    print(f"Anomalies Detected: {stats['anomalies']}")
    print(f"Anomaly Rate: {stats['anomaly_rate']*100:.2f}%")
    print(f"\nData Statistics:")
    print(f"  Min Value: {stats['min_value']:.2f}")
    print(f"  Max Value: {stats['max_value']:.2f}")
    print(f"  Mean Value: {stats['mean_value']:.2f}")
    print(f"  Avg Anomaly Score: {stats['avg_score']:.4f}")
    print(f"  Max Anomaly Score: {stats['max_score']:.4f}")
    print("\n" + "=" * 80)

def main():
    """Main menu"""
    while True:
        clear_screen()
        print("\n" + "=" * 80)
        print("  REAL-TIME ANOMALY DETECTION SYSTEM".center(80))
        print("=" * 80)
        print("\nSelect monitoring mode:")
        print("  1. CPU Usage Monitoring")
        print("  2. Financial Data Monitoring")
        print("  3. Network Traffic Monitoring")
        print("  4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        detector = None
        
        if choice == "1":
            detector = run_cpu_monitoring()
        elif choice == "2":
            detector = run_financial_monitoring()
        elif choice == "3":
            detector = run_network_monitoring()
        elif choice == "4":
            print("\nGoodbye!")
            break
        else:
            print("Invalid choice. Try again.")
            time.sleep(2)
            continue
        
        if detector:
            print_final_report(detector)
            input("\nPress Enter to return to menu...")

if __name__ == "__main__":
    main()

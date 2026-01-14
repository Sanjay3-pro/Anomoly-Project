"""
Quick start example: Simple anomaly detection with one detector
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_data import generate_financial_data
from utils.data_processor import TimeSeriesProcessor
from models import StatisticalDetector
from visualization.plotter import TimeSeriesPlotter

def quick_start():
    """Simple example to get started quickly"""
    
    print("🚀 Quick Start: Anomaly Detection in Financial Data\n")
    
    # Load data
    print("Loading financial data...")
    df = generate_financial_data(n_samples=300)
    data = df['price'].values
    
    # Process data
    print("Processing data...")
    processor = TimeSeriesProcessor()
    train_data, test_data = processor.split_data(data, train_ratio=0.7)
    train_norm = processor.normalize(train_data, fit=True)
    test_norm = processor.normalize(test_data, fit=False)
    
    # Detect anomalies
    print("Detecting anomalies...")
    detector = StatisticalDetector(threshold=2.5, method="zscore")
    detector.fit(train_norm)
    
    results = detector.predict_with_scores(test_norm)
    
    # Visualize
    print("Creating visualization...")
    os.makedirs("outputs", exist_ok=True)
    
    plotter = TimeSeriesPlotter()
    plotter.plot_anomalies_matplotlib(
        test_data,
        results['predictions'],
        results['scores'],
        title="Financial Data Anomaly Detection",
        savepath="outputs/quick_start_result.png"
    )
    
    # Summary
    print(f"\n✅ Detection Complete!")
    print(f"   Anomalies found: {results['anomaly_count']} out of {len(test_data)} points")
    print(f"   Anomaly rate: {results['anomaly_rate']:.2%}")
    print(f"   Visualization saved: outputs/quick_start_result.png")

if __name__ == "__main__":
    quick_start()

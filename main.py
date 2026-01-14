"""
Main example: Complete anomaly detection pipeline
"""
import numpy as np
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_data import generate_cpu_usage_data
from utils.data_processor import TimeSeriesProcessor
from models import StatisticalDetector, IsolationForestDetector, LOFDetector
from visualization.plotter import TimeSeriesPlotter
from config.config import config

def main():
    """Run complete anomaly detection example"""
    
    print("=" * 60)
    print("ANOMALY DETECTION SYSTEM - COMPLETE PIPELINE")
    print("=" * 60)
    
    # 1. Generate sample data
    print("\n[1/5] Generating sample data...")
    df = generate_cpu_usage_data(n_samples=500, anomaly_percentage=0.08)
    data = df['cpu_usage'].values
    print(f"✓ Generated {len(data)} data points")
    
    # 2. Data preprocessing
    print("\n[2/5] Preprocessing data...")
    processor = TimeSeriesProcessor(normalization_method="standard")
    
    # Split data
    train_data, test_data = processor.split_data(data, train_ratio=0.7)
    print(f"✓ Training samples: {len(train_data)}")
    print(f"✓ Testing samples: {len(test_data)}")
    
    # Normalize
    train_normalized = processor.normalize(train_data, fit=True)
    test_normalized = processor.normalize(test_data, fit=False)
    
    # 3. Train multiple detectors
    print("\n[3/5] Training detectors...")
    detectors = {
        "Statistical (Z-score)": StatisticalDetector(threshold=2.5, method="zscore"),
        "Statistical (IQR)": StatisticalDetector(threshold=2.5, method="iqr"),
        "Isolation Forest": IsolationForestDetector(contamination=0.08),
        "Local Outlier Factor": LOFDetector(n_neighbors=20, contamination=0.08)
    }
    
    for name, detector in detectors.items():
        detector.fit(train_normalized)
        print(f"✓ {name} trained")
    
    # 4. Detect anomalies in test data
    print("\n[4/5] Detecting anomalies...")
    results_dict = {}
    all_results = {}
    
    for name, detector in detectors.items():
        predictions = detector.predict(test_normalized)
        results = detector.predict_with_scores(test_normalized)
        results_dict[name] = predictions
        all_results[name] = results
        
        print(f"✓ {name}")
        print(f"  - Anomalies detected: {results['anomaly_count']} ({results['anomaly_rate']:.2%})")
    
    # 5. Visualization
    print("\n[5/5] Creating visualizations...")
    os.makedirs("outputs", exist_ok=True)
    
    plotter = TimeSeriesPlotter(figsize=(14, 8))
    
    # Plot each detector separately
    for name, detector in detectors.items():
        predictions = results_dict[name]
        scores = all_results[name]['scores']
        
        savepath = f"outputs/{name.replace('/', '_').replace(' ', '_')}.png"
        plotter.plot_anomalies_matplotlib(
            test_data,
            predictions,
            scores,
            title=f"CPU Usage Anomaly Detection - {name}",
            savepath=savepath
        )
    
    # Comparison plot
    savepath = "outputs/detector_comparison.png"
    plotter.plot_comparison(
        test_data,
        results_dict,
        title="Detector Comparison",
        savepath=savepath
    )
    
    # Statistics
    print("\n" + "=" * 60)
    print("DETECTION SUMMARY")
    print("=" * 60)
    for name, results in all_results.items():
        print(f"\n{name}:")
        print(f"  Anomalies: {results['anomaly_count']} ({results['anomaly_rate']:.2%})")
        print(f"  Min score: {results['scores'].min():.4f}")
        print(f"  Max score: {results['scores'].max():.4f}")
        print(f"  Mean score: {results['scores'].mean():.4f}")
    
    print("\n" + "=" * 60)
    print("✓ Pipeline completed successfully!")
    print(f"✓ Visualizations saved to: outputs/")
    print("=" * 60)

if __name__ == "__main__":
    main()

"""
Generate synthetic time-series data for demonstration
"""
import numpy as np
import pandas as pd
import os

def generate_cpu_usage_data(n_samples: int = 1000, anomaly_percentage: float = 0.05) -> pd.DataFrame:
    """
    Generate synthetic CPU usage data with anomalies
    
    Args:
        n_samples: Number of data points
        anomaly_percentage: Percentage of anomalies to inject
        
    Returns:
        DataFrame with timestamp and cpu_usage columns
    """
    # Normal pattern: cyclic behavior with daily peaks
    t = np.arange(n_samples)
    base = 30 + 20 * np.sin(2 * np.pi * t / 100) + np.random.normal(0, 2, n_samples)
    
    # Inject anomalies
    n_anomalies = int(n_samples * anomaly_percentage)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    base[anomaly_indices] += np.random.uniform(30, 50, n_anomalies)
    
    # Ensure values are positive and within reasonable range
    base = np.clip(base, 0, 100)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1min'),
        'cpu_usage': base
    })
    
    return df

def generate_network_traffic_data(n_samples: int = 1000, anomaly_percentage: float = 0.05) -> pd.DataFrame:
    """
    Generate synthetic network traffic data
    """
    t = np.arange(n_samples)
    # Weekly pattern
    base = 50 + 30 * np.sin(2 * np.pi * t / 168) + np.random.normal(0, 3, n_samples)
    
    # Inject anomalies (DDoS-like spikes)
    n_anomalies = int(n_samples * anomaly_percentage)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    base[anomaly_indices] += np.random.uniform(50, 100, n_anomalies)
    
    base = np.clip(base, 0, 150)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='h'),
        'traffic_mbps': base
    })
    
    return df

def generate_sensor_data(n_samples: int = 1000, anomaly_percentage: float = 0.05) -> pd.DataFrame:
    """
    Generate synthetic sensor data (temperature)
    """
    t = np.arange(n_samples)
    # Seasonal pattern
    base = 20 + 10 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 0.5, n_samples)
    
    # Inject anomalies
    n_anomalies = int(n_samples * anomaly_percentage)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    base[anomaly_indices] += np.random.choice([-15, 15], n_anomalies)
    
    base = np.clip(base, -10, 50)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1D'),
        'temperature': base
    })
    
    return df

def generate_financial_data(n_samples: int = 1000, anomaly_percentage: float = 0.05) -> pd.DataFrame:
    """
    Generate synthetic financial data (stock price)
    """
    t = np.arange(n_samples)
    # Random walk with drift
    returns = np.random.normal(0.001, 0.02, n_samples)
    price = 100 * np.exp(np.cumsum(returns))
    
    # Inject anomalies (market crashes/spikes)
    n_anomalies = int(n_samples * anomaly_percentage)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    price[anomaly_indices] *= np.random.choice([0.85, 1.15], n_anomalies)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='1D'),
        'price': price
    })
    
    return df

def save_all_datasets(data_dir: str = '.') -> None:
    """
    Generate and save all synthetic datasets
    
    Args:
        data_dir: Directory to save datasets
    """
    os.makedirs(data_dir, exist_ok=True)
    
    print("Generating datasets...")
    
    # Generate datasets
    datasets = {
        'cpu_usage.csv': generate_cpu_usage_data(),
        'network_traffic.csv': generate_network_traffic_data(),
        'sensor_data.csv': generate_sensor_data(),
        'financial_data.csv': generate_financial_data()
    }
    
    # Save datasets
    for filename, df in datasets.items():
        filepath = os.path.join(data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"[OK] Saved {filename} ({len(df)} samples)")
    
    print(f"\nAll datasets saved to {data_dir}")

if __name__ == "__main__":
    save_all_datasets()

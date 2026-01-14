"""
Real-world time-series data generators for various domains
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_fraud_detection_data(n_samples=500):
    """
    Bank/UPI Fraud Detection
    Normal: 100-5000 rupees, Regular patterns
    Anomaly: Sudden large transactions, unusual times
    """
    np.random.seed(42)
    data = []
    
    for i in range(n_samples):
        # 90% normal transactions
        if np.random.random() < 0.9:
            amount = np.random.normal(1500, 800)  # Normal distribution
            amount = max(100, min(5000, amount))
        else:
            # 10% anomalous transactions (fraud)
            amount = np.random.uniform(10000, 50000)
        
        data.append({
            'transaction_amount': amount,
            'timestamp': datetime.now() - timedelta(seconds=(n_samples-i)*10)
        })
    
    df = pd.DataFrame(data)
    df.to_csv('data/fraud_detection.csv', index=False)
    return df

def generate_electricity_usage_data(n_samples=500):
    """
    Electricity & Water Usage Monitoring
    Normal: 20-80 kWh daily, peaks during day
    Anomaly: Sudden spikes (equipment malfunction)
    """
    np.random.seed(42)
    data = []
    
    for i in range(n_samples):
        hour = (i // 20) % 24
        
        # Base consumption by hour
        if 6 <= hour <= 22:  # Daytime higher usage
            base = 60 + np.random.normal(0, 10)
        else:  # Night lower usage
            base = 30 + np.random.normal(0, 5)
        
        # 95% normal
        if np.random.random() < 0.95:
            consumption = base
        else:
            # 5% anomalous (sudden spike/drop)
            consumption = np.random.uniform(120, 200)
        
        data.append({
            'power_kwh': max(5, consumption),
            'timestamp': datetime.now() - timedelta(seconds=(n_samples-i)*30)
        })
    
    df = pd.DataFrame(data)
    df.to_csv('data/electricity_usage.csv', index=False)
    return df

def generate_machine_health_data(n_samples=500):
    """
    Machine Health Monitoring
    Normal: Temperature 60-80°C, Vibration 0.5-2.0 Hz
    Anomaly: Overheating or high vibration indicates failure
    """
    np.random.seed(42)
    data = []
    
    for i in range(n_samples):
        # 92% normal operation
        if np.random.random() < 0.92:
            temp = np.random.normal(70, 5)  # Normal temp
            vibration = np.random.normal(1.2, 0.3)  # Normal vibration
        else:
            # 8% anomalous (machine stress)
            temp = np.random.uniform(95, 120)  # Overheating
            vibration = np.random.uniform(4.0, 8.0)  # High vibration
        
        data.append({
            'temperature_celsius': max(50, temp),
            'vibration_hz': max(0.1, vibration),
            'timestamp': datetime.now() - timedelta(seconds=(n_samples-i)*5)
        })
    
    df = pd.DataFrame(data)
    df.to_csv('data/machine_health.csv', index=False)
    return df

def generate_server_traffic_data(n_samples=500):
    """
    Server & Network Traffic Monitoring
    Normal: 1000-5000 requests/min, steady
    Anomaly: DDoS attack (10000+) or server down (0-100)
    """
    np.random.seed(42)
    data = []
    
    for i in range(n_samples):
        # 88% normal traffic
        if np.random.random() < 0.88:
            requests = np.random.normal(2500, 800)
        elif np.random.random() < 0.5:
            # DDoS attack
            requests = np.random.uniform(15000, 50000)
        else:
            # Server down / connection lost
            requests = np.random.uniform(0, 100)
        
        data.append({
            'requests_per_min': max(0, requests),
            'response_time_ms': np.random.normal(150, 50),
            'timestamp': datetime.now() - timedelta(seconds=(n_samples-i)*20)
        })
    
    df = pd.DataFrame(data)
    df.to_csv('data/server_traffic.csv', index=False)
    return df

def generate_stock_risk_data(n_samples=500):
    """
    Stock & Financial Risk Monitoring
    Normal: Small daily fluctuations (±2%)
    Anomaly: Sudden price crashes or spikes (±10%+)
    """
    np.random.seed(42)
    data = []
    price = 1000
    
    for i in range(n_samples):
        # 85% normal market movement
        if np.random.random() < 0.85:
            change = np.random.normal(0, 1.5)  # Small changes
        else:
            # 15% anomalous (crash/spike)
            change = np.random.uniform(-15, 15)
        
        price = price * (1 + change/100)
        price = max(100, min(2000, price))  # Keep in bounds
        
        data.append({
            'stock_price': price,
            'volatility_percent': abs(change),
            'timestamp': datetime.now() - timedelta(seconds=(n_samples-i)*15)
        })
    
    df = pd.DataFrame(data)
    df.to_csv('data/stock_risk.csv', index=False)
    return df

def generate_iot_sensor_data(n_samples=500):
    """
    IoT Sensor Safety Monitoring
    Normal: Humidity 40-60%, Temp 18-28°C, Air quality good
    Anomaly: Sensor failure (extreme values) or environmental hazard
    """
    np.random.seed(42)
    data = []
    
    for i in range(n_samples):
        # 90% normal sensor readings
        if np.random.random() < 0.90:
            humidity = np.random.normal(50, 8)
            air_quality = np.random.normal(50, 15)  # 0-100 scale
            temperature = np.random.normal(23, 3)
        else:
            # 10% anomalous (sensor failure or hazard)
            humidity = np.random.choice([
                np.random.uniform(0, 10),      # Very dry
                np.random.uniform(90, 100)     # Very humid
            ])
            air_quality = np.random.uniform(150, 300)  # Hazardous
            temperature = np.random.choice([
                np.random.uniform(-10, 5),     # Too cold
                np.random.uniform(40, 50)      # Too hot
            ])
        
        data.append({
            'humidity_percent': max(0, min(100, humidity)),
            'air_quality_index': max(0, air_quality),
            'temperature_celsius': temperature,
            'timestamp': datetime.now() - timedelta(seconds=(n_samples-i)*8)
        })
    
    df = pd.DataFrame(data)
    df.to_csv('data/iot_sensors.csv', index=False)
    return df

# Generate all data when module is imported
if __name__ == '__main__':
    print("Generating real-world time-series datasets...")
    print("1. Fraud Detection: ", end="")
    generate_fraud_detection_data()
    print("✓")
    
    print("2. Electricity Usage: ", end="")
    generate_electricity_usage_data()
    print("✓")
    
    print("3. Machine Health: ", end="")
    generate_machine_health_data()
    print("✓")
    
    print("4. Server Traffic: ", end="")
    generate_server_traffic_data()
    print("✓")
    
    print("5. Stock Risk: ", end="")
    generate_stock_risk_data()
    print("✓")
    
    print("6. IoT Sensors: ", end="")
    generate_iot_sensor_data()
    print("✓")
    
    print("\nAll datasets generated successfully!")

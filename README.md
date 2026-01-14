# Anomaly Detection System for Time-Series Data

A comprehensive Python system for detecting anomalies in time-series data across various domains including banking transactions, electricity consumption, machine sensors, server traffic, and financial data.

## 🎯 Overview

This system learns normal behavior patterns from historical data and flags significant deviations as anomalies, enabling early detection of fraud, system faults, failures, and other risks.

## 📁 Project Structure

```
├── config/              # Configuration files
│   └── config.py       # Main configuration
├── data/               # Data handling
│   └── generate_data.py # Synthetic data generation
├── models/             # Anomaly detection models
│   ├── detector_base.py
│   ├── statistical_detector.py
│   ├── isolation_forest_detector.py
│   └── lof_detector.py
├── utils/              # Utility functions
│   └── data_processor.py
├── visualization/      # Plotting utilities
│   └── plotter.py
├── main.py             # Complete pipeline example
├── quick_start.py      # Simple quick-start example
└── requirements.txt    # Python dependencies
```

## 🚀 Getting Started

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Quick Start (2 minutes)

```bash
python quick_start.py
```

This runs a simple example detecting anomalies in synthetic financial data.

### Full Pipeline

```bash
python main.py
```

This demonstrates the complete workflow:
- Data generation and preprocessing
- Training multiple detectors
- Anomaly detection on test data
- Comparative visualization

## 🔬 Available Detection Methods

### 1. **Statistical Detector**
- Z-score method: Flags values deviating significantly from mean
- IQR method: Uses Interquartile Range for outlier detection
- Moving Average: Detects deviation from trend

```python
detector = StatisticalDetector(threshold=2.5, method="zscore")
```

### 2. **Isolation Forest**
- Ensemble-based method that isolates anomalies
- Effective for high-dimensional data
- Contamination parameter sets expected anomaly rate

```python
detector = IsolationForestDetector(contamination=0.05)
```

### 3. **Local Outlier Factor (LOF)**
- Density-based approach
- Identifies local density deviations
- Good for multivariate time-series

```python
detector = LOFDetector(n_neighbors=20, contamination=0.05)
```

## 📊 Data Processing

### TimeSeriesProcessor

```python
from utils.data_processor import TimeSeriesProcessor

processor = TimeSeriesProcessor(normalization_method="standard")

# Load data
df = processor.load_data("path/to/data.csv")

# Normalize
normalized = processor.normalize(data, fit=True)

# Split for training/testing
train, test = processor.split_data(data, train_ratio=0.8)

# Create sequences for LSTM
X, y = processor.create_sequences(data, lookback=10)
```

## 🎨 Visualization

### Matplotlib Plots (Static)
```python
from visualization.plotter import TimeSeriesPlotter

plotter = TimeSeriesPlotter()
plotter.plot_anomalies_matplotlib(
    data, 
    predictions, 
    scores,
    savepath="output.png"
)
```

### Plotly Charts (Interactive)
```python
plotter.plot_anomalies_plotly(data, predictions, savepath="output.html")
```

### Comparative Analysis
```python
plotter.plot_comparison({
    "Detector1": predictions1,
    "Detector2": predictions2
})
```

## ⚙️ Configuration

Edit `config/config.py`:
- Adjust anomaly thresholds
- Change normalization methods
- Set detection sensitivity
- Configure data paths

```python
from config.config import config

config.ANOMALY_THRESHOLD = 2.5
config.SENSITIVITY = "medium"
config.NORMALIZATION_METHOD = "standard"
```

## 📈 Supported Use Cases

| Domain | Example | Detection Focus |
|--------|---------|-----------------|
| **Banking** | Transaction fraud | Unusual spending patterns |
| **IoT/Sensors** | Temperature monitoring | Equipment malfunction |
| **Server Monitoring** | CPU/Memory usage | System resource anomalies |
| **Network** | Traffic analysis | DDoS/intrusion attempts |
| **Finance** | Stock prices | Market crashes/spikes |
| **Utilities** | Power consumption | Grid failures |

## 💡 Workflow

1. **Data Collection**: Load historical "normal" behavior data
2. **Preprocessing**: Normalize and clean data
3. **Training**: Fit detector on normal baseline
4. **Testing**: Apply detector to new incoming data
5. **Flagging**: Alert on detected anomalies
6. **Visualization**: Plot and analyze results

## 📝 Example Usage

```python
# Import modules
from data.generate_data import generate_cpu_usage_data
from utils.data_processor import TimeSeriesProcessor
from models import StatisticalDetector
from visualization.plotter import TimeSeriesPlotter

# Generate data
df = generate_cpu_usage_data(n_samples=1000)
data = df['cpu_usage'].values

# Process
processor = TimeSeriesProcessor()
train, test = processor.split_data(data, train_ratio=0.8)
train_norm = processor.normalize(train, fit=True)
test_norm = processor.normalize(test, fit=False)

# Detect
detector = StatisticalDetector()
detector.fit(train_norm)
results = detector.predict_with_scores(test_norm)

# Visualize
plotter = TimeSeriesPlotter()
plotter.plot_anomalies_matplotlib(
    test,
    results['predictions'],
    results['scores']
)

# Results
print(f"Anomalies found: {results['anomaly_count']}")
print(f"Anomaly rate: {results['anomaly_rate']:.2%}")
```

## 🔧 Advanced Features

### Custom Threshold Adjustment
```python
detector.set_threshold(3.0)  # Higher = fewer detections
```

### Multiple Detectors Ensemble
```python
detectors = [
    StatisticalDetector(),
    IsolationForestDetector(),
    LOFDetector()
]

# Combine predictions with voting
ensemble_predictions = np.mean([d.predict(data) for d in detectors], axis=0) > 0.5
```

### Sequence-based Detection
```python
# Create sequences for pattern detection
X, y = processor.create_sequences(data, lookback=10)

# Useful for LSTM-based approaches
```

## 📦 Dependencies

- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **scikit-learn**: ML algorithms and preprocessing
- **matplotlib**: Static visualization
- **plotly**: Interactive visualization
- **tensorflow**: Deep learning (for future LSTM extensions)
- **scipy**: Scientific computing

## 🎓 Learning Curve

**Beginner**: Run `quick_start.py` to see basic functionality

**Intermediate**: Modify `main.py` to experiment with different detectors

**Advanced**: Create custom detectors by extending `AnomalyDetector` base class

## 🤝 Extending the System

### Create Custom Detector

```python
from models.detector_base import AnomalyDetector

class CustomDetector(AnomalyDetector):
    def fit(self, data):
        # Your training logic
        pass
    
    def score(self, data):
        # Your scoring logic
        pass
    
    def predict(self, data):
        # Your prediction logic
        pass
```

### Add New Dataset

```python
from data.generate_data import save_all_datasets

# Add your custom generator function and call save_all_datasets()
```

## 📊 Performance Considerations

- **Computational Efficiency**: Statistical methods are fastest
- **Accuracy**: Ensemble approaches often perform best
- **Scalability**: Isolation Forest scales well to large datasets
- **Real-time**: Use Statistical detector for streaming applications

## 🎯 Key Advantages

✅ **Domain-agnostic**: Works across different data types
✅ **Multiple algorithms**: Compare different detection approaches
✅ **Easy to extend**: Modular architecture for custom detectors
✅ **Visualization**: Built-in plotting for analysis and reporting
✅ **Configuration**: Flexible settings for different sensitivity levels
✅ **Production-ready**: Clean code with error handling

## 📞 Support

For questions or issues, refer to:
- Main example: `main.py`
- Quick start: `quick_start.py`
- Configuration: `config/config.py`
- Model documentation: See docstrings in each model file

## 📄 License

This project is open-source and available for educational and research purposes.

---

**Happy Anomaly Hunting! 🔍**

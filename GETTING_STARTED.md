# Getting Started with Anomaly Detection System

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start (2 minutes)

Run the simple example:
```bash
python quick_start.py
```

## Complete Pipeline (5 minutes)

Run the full demonstration:
```bash
python main.py
```

This demonstrates:
- Data generation and preprocessing
- Training multiple anomaly detectors
- Comparing different detection algorithms
- Visualizing results

## Key Components

### Detection Methods
- **Statistical**: Z-score and IQR-based detection
- **Isolation Forest**: Ensemble method for anomaly detection
- **Local Outlier Factor**: Density-based approach

### Data Processing
- Normalization (StandardScaler, MinMaxScaler, RobustScaler)
- Train/test splitting
- Sequence generation for deep learning
- Outlier removal

### Visualization
- Static plots with Matplotlib
- Interactive charts with Plotly
- Comparison views
- Statistical summaries

## Example Usage

```python
from utils.data_processor import TimeSeriesProcessor
from models import StatisticalDetector
from visualization.plotter import TimeSeriesPlotter

# Load and process data
processor = TimeSeriesProcessor()
data = processor.load_data("data/cpu_usage.csv")
train, test = processor.split_data(data, train_ratio=0.8)

# Train detector
detector = StatisticalDetector(threshold=2.5)
detector.fit(processor.normalize(train, fit=True))

# Detect anomalies
results = detector.predict_with_scores(
    processor.normalize(test, fit=False)
)

# Visualize
plotter = TimeSeriesPlotter()
plotter.plot_anomalies_matplotlib(test, results['predictions'])
```

## Configuration

Edit `config/config.py` to customize:
- Anomaly thresholds
- Normalization methods
- Data paths
- Detection sensitivity

## Available Datasets

All datasets are in the `data/` folder:
- `cpu_usage.csv`: Server CPU monitoring
- `network_traffic.csv`: Network bandwidth data
- `sensor_data.csv`: Temperature sensor readings
- `financial_data.csv`: Stock price data

Each dataset has 1000 samples with ~8% anomalies injected.

## Next Steps

1. **Run quick start** to see basic functionality
2. **Explore detectors** by modifying threshold values
3. **Create custom detector** by extending `AnomalyDetector`
4. **Add your own data** by loading CSV files
5. **Tune parameters** in `config/config.py`

## Documentation

- Full README: See `README.md`
- API docs: See docstrings in each module
- Examples: See `main.py` and `quick_start.py`

---

For questions, refer to the README.md or check the example scripts!

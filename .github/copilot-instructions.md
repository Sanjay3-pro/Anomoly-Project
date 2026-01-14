  # Custom Instructions for Anomaly Detection Project

## Project Overview
Comprehensive anomaly detection system for time-series data that learns normal behavior patterns and flags deviations as anomalies across multiple domains.

## Technology Stack
- **Language**: Python 3.8+
- **Core Libraries**: NumPy, Pandas, Scikit-learn, Matplotlib, Plotly
- **ML Framework**: TensorFlow (for future deep learning extensions)
- **Data Processing**: Scikit-learn preprocessing utilities

## Key Features
1. Multiple detection algorithms (Statistical, Isolation Forest, LOF)
2. Data preprocessing and normalization utilities
3. Interactive and static visualization tools
4. Configurable thresholds and sensitivity levels
5. Synthetic data generation for testing

## Development Guidelines
- Maintain modular architecture with separation of concerns
- Use type hints for better code clarity
- Include comprehensive docstrings for all functions
- Follow PEP 8 style guidelines
- Test new features with synthetic data before production use

## Common Tasks
- **Run full pipeline**: `python main.py`
- **Quick test**: `python quick_start.py`
- **Generate data**: `python data/generate_data.py`
- **Install dependencies**: `pip install -r requirements.txt`

## Architecture Pattern
- **Base Classes**: `detector_base.py` defines interface
- **Implementations**: Specific detectors inherit from base
- **Data Pipeline**: `TimeSeriesProcessor` handles all preprocessing
- **Visualization**: `TimeSeriesPlotter` handles all visualizations
- **Config**: Centralized settings in `config/config.py`

## Extension Points
- Add new detectors by extending `AnomalyDetector`
- Add new datasets in `data/generate_data.py`
- Add new visualization types in `visualization/plotter.py`
- Customize thresholds in `config/config.py`

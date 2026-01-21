# Anomaly Detection System for Time-Series Data

Project: Real-Time Anomaly Detection System

This project implements a Python-based anomaly detection system for
time-series data. The system learns normal behavior from historical data
and detects abnormal patterns that deviate from it.

---

Overview

The main objective of this project is to identify anomalies in different
types of time-series data such as server metrics, sensor readings, and
financial values. Anomalies may indicate faults, risks, or unusual events.

The system is designed to be modular, configurable, and easy to extend.

---

Supported Use Cases

Banking  
Detection of unusual transaction patterns  

Server Monitoring  
CPU usage, memory usage, and network traffic anomalies  

IoT and Sensors  
Temperature and equipment health monitoring  

Finance  
Stock price spikes, drops, and abnormal movements  

Utilities  
Electricity consumption and power usage anomalies  

---

Project Structure

config/  
Contains centralized configuration and threshold settings  

data/  
Synthetic data generation and sample datasets  

models/  
All anomaly detection algorithms  

utils/  
Data preprocessing and normalization logic  

visualization/  
Plotting and visualization utilities  

main.py  
Complete end-to-end pipeline  

quick_start.py  
Simple demonstration script  

requirements.txt  
Python dependencies  

---

Getting Started

Installation

Install all required Python packages using:

pip install -r requirements.txt

---

Quick Start Execution

To quickly understand how the system works, run:

python quick_start.py

This script:
- Generates synthetic time-series data
- Trains a basic anomaly detector
- Displays anomaly detection results in the console

---

Complete Pipeline Execution

To run the full project pipeline, execute:

python main.py

This demonstrates:
- Data generation
- Data preprocessing
- Training multiple detectors
- Detecting anomalies on test data
- Visualizing results

---

Anomaly Detection Methods

Statistical Detector

Uses rule-based techniques such as:
- Z-score
- Interquartile Range (IQR)
- Moving average deviation

This method is fast and easy to interpret.

Isolation Forest

A tree-based ensemble method that isolates anomalies.
It performs well on large and complex datasets.

Local Outlier Factor (LOF)

A density-based detection method.
It detects anomalies by comparing local neighborhood density.

---

Data Processing Module

The TimeSeriesProcessor module is responsible for:
- Loading CSV time-series data
- Cleaning and preparing data
- Normalizing values using standard scaling methods
- Splitting data into training and testing sets
- Ensuring consistent scaling between train and test data

This ensures reliable input for anomaly detection.

---

Visualization Module

The visualization module provides:
- Static plots using Matplotlib
- Interactive charts using Plotly
- Highlighting of anomaly points
- Comparison between multiple detectors

These visualizations help in analysis and validation.

---

Configuration Handling

All configurable parameters are stored in:

config/config.py

This file allows:
- Adjusting anomaly thresholds
- Changing normalization methods
- Controlling detection sensitivity
- Managing data paths

No core code changes are required for tuning.

---

Datasets

All datasets are synthetic and stored in the data/ folder.

cpu_usage.csv  
Server CPU usage data  

network_traffic.csv  
Network traffic data  

sensor_data.csv  
Temperature sensor data  

financial_data.csv  
Stock and financial data  

Each dataset contains approximately 1000 samples with injected anomalies
for testing purposes.

---

Workflow Summary

1. Load historical time-series data  
2. Preprocess and normalize the data  
3. Train anomaly detectors on normal behavior  
4. Apply detection on new or test data  
5. Identify and flag anomalies  
6. Visualize results for analysis  

---

Project Status

Completed:
- Data preprocessing pipeline
- Anomaly detection logic
- Multiple detection algorithms
- Visualization utilities
- Configuration management

In Progress:
- Dashboard refinement
- Alert sensitivity tuning
- Real-time behavior improvements

---

Conclusion

This project successfully demonstrates a complete anomaly detection
workflow for time-series data. It is suitable for academic evaluation
and can be extended further for real-time monitoring systems.

---

Last Updated: January 21, 2026  
Status: In Progress

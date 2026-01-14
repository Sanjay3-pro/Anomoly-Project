"""
Data processing utilities for time-series data
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Tuple, Optional

class TimeSeriesProcessor:
    """Handles time-series data loading, preprocessing, and normalization"""
    
    def __init__(self, normalization_method: str = "standard"):
        """
        Initialize processor with normalization method
        
        Args:
            normalization_method: "standard", "minmax", or "robust"
        """
        self.normalization_method = normalization_method
        self.scaler = self._get_scaler()
        self.fitted = False
    
    def _get_scaler(self):
        """Get appropriate scaler based on method"""
        if self.normalization_method == "minmax":
            return MinMaxScaler()
        elif self.normalization_method == "robust":
            return RobustScaler()
        else:
            return StandardScaler()
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load time-series data from CSV
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with time-series data
        """
        df = pd.read_csv(filepath)
        # Ensure datetime column if present
        if "timestamp" in df.columns or "date" in df.columns:
            date_col = "timestamp" if "timestamp" in df.columns else "date"
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
        return df
    
    def normalize(self, data: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Normalize data using fitted scaler
        
        Args:
            data: Input data array
            fit: Whether to fit scaler on this data
            
        Returns:
            Normalized data array
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        if fit:
            normalized = self.scaler.fit_transform(data)
            self.fitted = True
        else:
            if not self.fitted:
                raise ValueError("Scaler must be fitted first")
            normalized = self.scaler.transform(data)
        
        return normalized.squeeze()
    
    def denormalize(self, data: np.ndarray) -> np.ndarray:
        """
        Reverse normalization
        
        Args:
            data: Normalized data array
            
        Returns:
            Original scale data
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        denormalized = self.scaler.inverse_transform(data)
        return denormalized.squeeze()
    
    def split_data(self, data: np.ndarray, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets
        
        Args:
            data: Input data
            train_ratio: Ratio for training set (0-1)
            
        Returns:
            Tuple of (train_data, test_data)
        """
        split_idx = int(len(data) * train_ratio)
        return data[:split_idx], data[split_idx:]
    
    def create_sequences(self, data: np.ndarray, lookback: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM/sequential models
        
        Args:
            data: Input time-series data
            lookback: Number of previous timesteps to use as input
            
        Returns:
            Tuple of (X, y) sequences
        """
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def remove_outliers(self, data: np.ndarray, method: str = "iqr", threshold: float = 1.5) -> np.ndarray:
        """
        Remove outliers from data before training
        
        Args:
            data: Input data
            method: "iqr" or "zscore"
            threshold: Sensitivity threshold
            
        Returns:
            Cleaned data
        """
        if method == "iqr":
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            mask = (data >= lower_bound) & (data <= upper_bound)
            return data[mask]
        
        elif method == "zscore":
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            mask = z_scores < threshold
            return data[mask]
        
        return data

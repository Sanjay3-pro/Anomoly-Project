"""
Statistical anomaly detector using Z-score and moving averages
"""
import numpy as np
from .detector_base import AnomalyDetector

class StatisticalDetector(AnomalyDetector):
    """Detects anomalies using statistical methods (Z-score, IQR, Moving Average)"""
    
    def __init__(self, threshold: float = 2.5, method: str = "zscore", window: int = 10):
        """
        Initialize statistical detector
        
        Args:
            threshold: Z-score threshold (default 2.5 ≈ 98.8% confidence)
            method: "zscore", "iqr", or "moving_average"
            window: Window size for moving average (if applicable)
        """
        super().__init__(threshold=threshold, name=f"StatisticalDetector({method})")
        self.method = method
        self.window = window
        self.mean = None
        self.std = None
        self.Q1 = None
        self.Q3 = None
    
    def fit(self, data: np.ndarray) -> None:
        """
        Learn statistics from normal data
        
        Args:
            data: Training data
        """
        self.mean = np.mean(data)
        self.std = np.std(data)
        self.Q1 = np.percentile(data, 25)
        self.Q3 = np.percentile(data, 75)
        self.is_fitted = True
        
        self.metadata = {
            "mean": float(self.mean),
            "std": float(self.std),
            "Q1": float(self.Q1),
            "Q3": float(self.Q3),
            "method": self.method
        }
    
    def score(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate anomaly scores
        
        Args:
            data: Input data
            
        Returns:
            Anomaly scores
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first")
        
        if self.method == "zscore":
            # Z-score: measures how many standard deviations away from mean
            scores = np.abs((data - self.mean) / (self.std + 1e-8))
        
        elif self.method == "iqr":
            # IQR method
            IQR = self.Q3 - self.Q1
            lower_bound = self.Q1 - 1.5 * IQR
            upper_bound = self.Q3 + 1.5 * IQR
            scores = np.where(
                (data < lower_bound) | (data > upper_bound),
                np.abs((data - self.mean) / (self.std + 1e-8)),
                0
            )
        
        elif self.method == "moving_average":
            # Deviation from moving average
            ma = self._moving_average(data, self.window)
            deviation = np.abs(data - ma)
            scores = deviation / (self.std + 1e-8)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return scores
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict anomalies
        
        Args:
            data: Input data
            
        Returns:
            Binary predictions (1 = anomaly, 0 = normal)
        """
        scores = self.score(data)
        return (scores > self.threshold).astype(int)
    
    def _moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate moving average"""
        ma = np.convolve(data, np.ones(window) / window, mode='same')
        # For the first few points, use mean of available data
        for i in range(window):
            ma[i] = np.mean(data[:i+1])
        return ma

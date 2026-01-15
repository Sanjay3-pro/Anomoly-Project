"""
Base class for anomaly detectors
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any

class AnomalyDetector(ABC):
    """Abstract base class for anomaly detection models"""
    
    def __init__(self, threshold: float = 2.5, name: str = "BaseDetector"):
        """
        Initialize detector
        
        Args:
            threshold: Anomaly threshold (sensitivity parameter)
            name: Detector name
        """
        self.threshold = threshold
        self.name = name
        self.is_fitted = False
        self.metadata = {}
    
    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """
        Train the detector on normal data
        
        Args:
            data: Training data (normal behavior)
        """
        pass
    
    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Detect anomalies in new data
        
        Args:
            data: Data to check for anomalies
            
        Returns:
            Binary array (1 = anomaly, 0 = normal)
        """
        pass
    
    @abstractmethod
    def score(self, data: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores for data
        
        Args:
            data: Data to score
            
        Returns:
            Scores array (higher = more anomalous)
        """
        pass
    
    def predict_with_scores(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Get predictions with detailed information
        
        Args:
            data: Data to analyze
            
        Returns:
            Dictionary with predictions, scores, and metadata
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before prediction")
        
        scores = self.score(data)
        predictions = self.predict(data)
        
        return {
            "predictions": predictions,
            "scores": scores,
            "threshold": self.threshold,
            "detector": self.name,
            "anomaly_count": np.sum(predictions),
            "anomaly_rate": np.sum(predictions) / len(predictions)
        }
    
    def set_threshold(self, threshold: float) -> None:
        """
        Update anomaly threshold
        
        Args:
            threshold: New threshold value
        """
        self.threshold = threshold
    
    def _check_fitted(self) -> None:
        """
        Check if the detector has been fitted
        
        Raises:
            ValueError: If detector is not fitted
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} must be fitted before prediction")

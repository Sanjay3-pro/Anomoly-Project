"""
Isolation Forest anomaly detector
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from .detector_base import AnomalyDetector

class IsolationForestDetector(AnomalyDetector):
    """Detects anomalies using Isolation Forest algorithm"""
    
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        """
        Initialize Isolation Forest detector
        
        Args:
            contamination: Expected proportion of anomalies (0-1)
            random_state: Random seed for reproducibility
        """
        super().__init__(name="IsolationForest")
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.threshold = 0  # Isolation Forest uses anomaly scores
    
    def fit(self, data: np.ndarray) -> None:
        """
        Train Isolation Forest on normal data
        
        Args:
            data: Training data (1D or 2D)
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        self.model.fit(data)
        self.is_fitted = True
        
        self.metadata = {
            "contamination": self.contamination,
            "n_features": data.shape[1],
            "n_samples": data.shape[0]
        }
    
    def score(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate anomaly scores (negative values are more anomalous)
        
        Args:
            data: Input data
            
        Returns:
            Anomaly scores
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first")
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Isolation Forest returns negative scores for anomalies
        # We invert to make higher values more anomalous
        scores = -self.model.score_samples(data)
        return scores
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict anomalies (1 = anomaly, -1 = normal)
        
        Args:
            data: Input data
            
        Returns:
            Predictions (0 = normal, 1 = anomaly)
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first")
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Convert -1, 1 to 0, 1
        predictions = self.model.predict(data)
        return np.where(predictions == -1, 1, 0)

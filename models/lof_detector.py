"""
Local Outlier Factor anomaly detector
"""
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from .detector_base import AnomalyDetector

class LOFDetector(AnomalyDetector):
    """Detects anomalies using Local Outlier Factor"""
    
    def __init__(self, n_neighbors: int = 20, contamination: float = 0.05):
        """
        Initialize LOF detector
        
        Args:
            n_neighbors: Number of neighbors to consider
            contamination: Expected proportion of anomalies
        """
        super().__init__(name="LocalOutlierFactor")
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True
        )
        self.threshold = 0
    
    def fit(self, data: np.ndarray) -> None:
        """
        Train LOF on normal data
        
        Args:
            data: Training data
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        self.model.fit(data)
        self.is_fitted = True
        
        self.metadata = {
            "n_neighbors": self.n_neighbors,
            "contamination": self.contamination,
            "n_features": data.shape[1]
        }
    
    def score(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate anomaly scores
        
        Args:
            data: Input data
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first")
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Get negative outlier factor (higher = more anomalous)
        # For novelty=True, we use negative_outlier_factor_ from training
        # and calculate scores based on prediction decisions
        scores = -self.model.score_samples(data)
        
        # Normalize to [0, 1] range for interpretability
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return scores
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict anomalies
        
        Args:
            data: Input data
            
        Returns:
            Predictions (0 = normal, 1 = anomaly)
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted first")
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # LOF returns -1 for anomalies, 1 for normal with novelty=True
        predictions = self.model.predict(data)
        return np.where(predictions == -1, 1, 0)

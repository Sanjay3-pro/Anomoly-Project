"""
Ensemble Anomaly Detector - Combines multiple detectors for higher accuracy
"""
import numpy as np
from typing import List
from .detector_base import AnomalyDetector
from .statistical_detector import StatisticalDetector
from .isolation_forest_detector import IsolationForestDetector
from .lof_detector import LOFDetector

class EnsembleDetector(AnomalyDetector):
    """
    Ensemble detector that combines multiple detection methods
    using voting or weighted averaging for improved accuracy
    """
    
    def __init__(self, voting: str = "majority", weights: List[float] = None):
        """
        Initialize ensemble detector
        
        Args:
            voting: "majority" for majority vote, "weighted" for weighted average
            weights: List of weights for each detector (if weighted voting)
        """
        super().__init__(name="EnsembleDetector")
        self.voting = voting
        self.weights = weights
        
        # Initialize multiple detectors with optimized parameters
        self.detectors = [
            StatisticalDetector(threshold=3.0, method="zscore"),
            IsolationForestDetector(contamination=0.05),
            LOFDetector(n_neighbors=20, contamination=0.05)
        ]
        
        if self.weights is None:
            # Default equal weights
            self.weights = [1.0] * len(self.detectors)
        
        # Normalize weights
        weight_sum = sum(self.weights)
        self.weights = [w / weight_sum for w in self.weights]
    
    def fit(self, data: np.ndarray) -> None:
        """
        Train all detectors in the ensemble
        
        Args:
            data: Training data
        """
        for detector in self.detectors:
            try:
                detector.fit(data)
            except Exception as e:
                print(f"Warning: {detector.name} failed to fit: {str(e)}")
        
        self.is_fitted = True
        
        self.metadata = {
            "voting": self.voting,
            "n_detectors": len(self.detectors),
            "detector_names": [d.name for d in self.detectors],
            "weights": self.weights
        }
    
    def score(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate ensemble anomaly scores
        
        Args:
            data: Test data
        
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if self.voting == "majority":
            # Get predictions from all detectors
            predictions = []
            for detector in self.detectors:
                try:
                    pred = detector.predict(data)
                    predictions.append(pred)
                except:
                    continue
            
            if not predictions:
                return np.zeros(len(data))
            
            # Majority vote - count how many detectors flagged as anomaly
            predictions = np.array(predictions)
            scores = np.sum(predictions, axis=0) / len(predictions)
            return scores
        
        else:  # weighted voting
            # Get normalized scores from all detectors
            weighted_scores = np.zeros(len(data))
            
            for detector, weight in zip(self.detectors, self.weights):
                try:
                    scores = detector.score(data)
                    # Normalize scores to [0, 1]
                    if scores.max() > scores.min():
                        normalized = (scores - scores.min()) / (scores.max() - scores.min())
                    else:
                        normalized = scores
                    weighted_scores += weight * normalized
                except:
                    continue
            
            return weighted_scores
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Predict anomalies using ensemble voting
        
        Args:
            data: Test data
        
        Returns:
            Binary predictions (1 = anomaly, 0 = normal)
        """
        scores = self.score(data)
        
        if self.voting == "majority":
            # If more than 50% of detectors agree, flag as anomaly
            return (scores > 0.5).astype(int)
        else:
            # Use threshold on weighted scores
            threshold = 0.6  # Adjustable
            return (scores > threshold).astype(int)
    
    def predict_with_scores(self, data: np.ndarray) -> dict:
        """
        Predict anomalies with detailed scores and confidence
        
        Args:
            data: Test data
        
        Returns:
            Dictionary with predictions, scores, and metadata
        """
        predictions = self.predict(data)
        scores = self.score(data)
        
        # Calculate confidence based on agreement level
        confidence = np.abs(scores - 0.5) * 2  # Range [0, 1]
        
        result = {
            "predictions": predictions,
            "scores": scores,
            "confidence": confidence,
            "anomaly_count": np.sum(predictions),
            "anomaly_rate": np.mean(predictions),
            "mean_score": np.mean(scores),
            "max_score": np.max(scores),
            "threshold": 0.5 if self.voting == "majority" else 0.6,
            "method": "Ensemble",
            "voting": self.voting
        }
        
        return result

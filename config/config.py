"""
Configuration module for anomaly detection system
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration"""
    PROJECT_NAME = "Anomaly Detection System"
    DEBUG = os.getenv("DEBUG", False)
    
    # Data paths
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved")
    OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
    
    # Model parameters
    ANOMALY_THRESHOLD = 2.5  # Z-score threshold
    SENSITIVITY = "medium"  # low, medium, high
    
    # Data parameters
    TRAIN_TEST_SPLIT = 0.8
    NORMALIZATION_METHOD = "standard"  # standard, minmax, robust
    
    # Visualization
    PLOT_STYLE = "seaborn-v0_8-darkgrid"
    FIGURE_SIZE = (14, 6)
    
    # Detection methods
    DETECTION_METHODS = ["statistical", "isolation_forest", "local_outlier_factor", "autoencode"]
    DEFAULT_METHOD = "statistical"

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    ANOMALY_THRESHOLD = 2.0

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    ANOMALY_THRESHOLD = 3.0

# Select config based on environment
env = os.getenv("ENV", "development")
if env == "production":
    config = ProductionConfig()
else:
    config = DevelopmentConfig()

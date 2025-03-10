# app/core/config.py
import os
from pydantic_settings import BaseSettings
from typing import Dict, Any

class Settings(BaseSettings):
    """Application settings for the Adaptive AI-based API Rate Limiter."""
    
    # API settings
    API_TITLE: str = "Adaptive AI-Based API Rate Limiter"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "A machine learning powered API rate limiter with dynamic adjustments"
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # Token bucket default configurations
    TOKEN_BUCKET_CONFIGS: Dict[str, Dict[str, Any]] = {
        "STD": {  # Standard tier users
            "capacity": 50,         # Maximum bucket capacity
            "refill_tokens": 10,    # Tokens added per refill
            "refill_duration": 60   # Refill every 60 seconds (10 tokens per minute)
        },
        "PRM": {  # Premium tier users
            "capacity": 100,        # Higher capacity for premium users
            "refill_tokens": 20,    # More tokens per refill
            "refill_duration": 60   # Refill every 60 seconds (20 tokens per minute)
        }
    }
    
    # Model paths
    MODEL_DIR: str = "app/models/trained"
    LSTM_MODEL_PATH: str = os.path.join(MODEL_DIR, "lstm_model.keras")
    LSTM_SCALER_PATH: str = os.path.join(MODEL_DIR, "lstm_scaler.pkl")
    ISOLATION_FOREST_PATH: str = os.path.join(MODEL_DIR, "isolation_forest.pkl")
    
    # Traffic monitoring settings
    MONITORING_INTERVAL: int = 60  # Check traffic every 60 seconds
    TRAFFIC_WINDOW: int = 1800     # Use 30 minutes of data for analysis
    
    # Traffic thresholds for different levels
    TRAFFIC_THRESHOLDS: Dict[str, int] = {
        "low": 50,      # Below this is low traffic
        "medium": 150,  # Between low and high is medium traffic
        "high": 300     # Above this is high traffic
    }
    
    # Rate limit modifiers for different traffic levels
    RATE_LIMIT_MODIFIERS: Dict[str, Dict[str, float]] = {
        "low": {
            "STD": 1.2,   # Increase standard user capacity by 20% during low traffic
            "PRM": 1.5    # Increase premium user capacity by 50% during low traffic
        },
        "medium": {
            "STD": 1.0,   # Standard capacity for standard users during medium traffic
            "PRM": 1.2    # Increase premium user capacity by 20% during medium traffic
        },
        "high": {
            "STD": 0.8,   # Reduce standard user capacity by 20% during high traffic
            "PRM": 1.0    # Keep standard capacity for premium users during high traffic
        }
    }
    
    # Data paths
    DATA_DIR: str = "data"
    SYNTHETIC_DATA_PATH: str = "data\\synthetic_api_traffic.csv"
    
    # Logging settings
    LOG_LEVEL: str = "DEBUG"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    class Config:
        env_prefix = "APP_"  # Environment variables with APP_ prefix can override these settings

# Create a global settings instance
settings = Settings()

#!/usr/bin/env python
# scripts/retrain_lstm.py
import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.lstm_model import TrafficPredictor
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=settings.LOG_FORMAT,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("retrain_lstm")

def load_and_preprocess_data(file_path):
    """
    Load the synthetic API traffic data and preprocess it for training.
    
    Args:
        file_path: Path to the synthetic data file
        
    Returns:
        DataFrame with time bins and request counts
    """
    logger.info(f"Loading data from {file_path}")
    
    # Check file extension and load accordingly
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Use .xlsx or .csv")
    
    logger.info(f"Loaded {len(df)} records")
    
    # Convert timestamp column to datetime
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='mixed')
    
    # Sort by timestamp
    df = df.sort_values('TIMESTAMP')
    
    # Aggregate by 30-second time bins
    df['time_bin'] = df['TIMESTAMP'].dt.floor('30s')
    
    # Calculate request counts per time bin
    traffic_counts = df.groupby('time_bin').size().reset_index(name='request_count')
    
    logger.info(f"Created {len(traffic_counts)} time bins")
    
    return traffic_counts

def main():
    """Main function to retrain the LSTM model with a smaller sequence length."""
    # Use absolute path for data file
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), settings.SYNTHETIC_DATA_PATH)
    
    # Load and preprocess data
    traffic_counts = load_and_preprocess_data(data_path)
    
    # Create model instance
    lstm_model = TrafficPredictor()
    
    # Set the reduced sequence length
    sequence_length = 5  # Reduced from 10 to 5
    
    # Params for training
    lstm_params = {
        'target_col': 'request_count',
        'time_col': 'time_bin',
        'sequence_length': sequence_length,
        'epochs': 50,
        'batch_size': 32,
        'test_size': 0.2
    }
    
    # Train the model
    logger.info(f"Training LSTM model with sequence_length={sequence_length}")
    training_result = lstm_model.train(
        data=traffic_counts,
        **lstm_params
    )
    
    logger.info(f"LSTM training complete. Test loss: {training_result['test_loss']:.4f}")
    
    # Save the model with a different name to avoid overwriting
    model_dir = settings.MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"lstm_model_seq{sequence_length}_{timestamp}.keras")
    scaler_path = os.path.join(model_dir, f"lstm_scaler_seq{sequence_length}_{timestamp}.pkl")
    
    lstm_model.save(model_path, scaler_path)
    logger.info(f"Saved model to {model_path}")
    logger.info(f"Saved scaler to {scaler_path}")
    
    # Also save to the default paths
    lstm_model.save(settings.LSTM_MODEL_PATH, settings.LSTM_SCALER_PATH)
    logger.info(f"Also saved to default paths for immediate use")
    
    logger.info("Retraining complete!")

if __name__ == "__main__":
    main()
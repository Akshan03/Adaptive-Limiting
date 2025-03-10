# scripts/train_models.py
import os
import pandas as pd
import numpy as np
import logging
import sys
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.lstm_model import TrafficPredictor
from app.models.isolation_forest import AnomalyDetector
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=settings.LOG_FORMAT,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("train_models")

def load_and_preprocess_data(file_path):
    """
    Load the synthetic API traffic data and preprocess it for training.
    
    Args:
        file_path: Path to the synthetic data file
        
    Returns:
        Tuple of (traffic_counts, feature_data) for LSTM and Isolation Forest models
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
    
    # Create time bins for aggregation (every minute)
    df['time_bin'] = df['TIMESTAMP'].dt.floor('1min')
    
    # Create traffic counts for LSTM model
    traffic_counts = df.groupby('time_bin').size().reset_index(name='request_count')
    
    # Create feature data for Isolation Forest
    feature_data = df.groupby('time_bin').agg({
        'response_code': ['mean', 'max', 'count'],
        'latency': ['mean', 'max'],
        'is_malicious': 'sum'
    }).reset_index()
    
    # Flatten multi-level columns
    feature_data.columns = ['time_bin', 'avg_resp_code', 'max_resp_code', 'request_count',
                          'avg_latency', 'max_latency', 'malicious_count']
    
    # Calculate additional features
    feature_data['error_rate'] = df.groupby('time_bin').apply(
        lambda x: (x['response_code'] >= 400).mean()
    ).values
    
    # Add user tier ratio (premium to standard)
    tier_ratios = df.groupby('time_bin').apply(
        lambda x: (x['USER_TIER'] == 'PRM').mean()
    ).values
    feature_data['premium_ratio'] = tier_ratios
    
    logger.info(f"Created {len(traffic_counts)} time bins for training")
    
    return traffic_counts, feature_data

def train_lstm_model(traffic_counts):
    """
    Train the LSTM model for traffic prediction.
    
    Args:
        traffic_counts: DataFrame with time_bin and request_count columns
        
    Returns:
        Trained TrafficPredictor instance
    """
    logger.info("Training LSTM model for traffic prediction")
    
    # Create and train LSTM model
    lstm_model = TrafficPredictor()
    
    # Params for training
    lstm_params = {
        'target_col': 'request_count',
        'time_col': 'time_bin',
        'sequence_length': 10,
        'epochs': 50,
        'batch_size': 32,
        'test_size': 0.2
    }
    
    # Train the model
    training_result = lstm_model.train(
        data=traffic_counts,
        **lstm_params
    )
    
    logger.info(f"LSTM training complete. Test loss: {training_result['test_loss']:.4f}")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(training_result['history']['loss'], label='Training Loss')
    plt.plot(training_result['history']['val_loss'], label='Validation Loss')
    plt.title('LSTM Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Create directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/lstm_training_loss.png')
    
    # Validate model with some predictions
    sequence_length = lstm_params['sequence_length']
    if len(traffic_counts) > sequence_length + 10:
        recent_data = traffic_counts['request_count'].values[-sequence_length-10:-10]
        actual_values = traffic_counts['request_count'].values[-10:]
        
        predictions = []
        for i in range(10):
            pred = lstm_model.predict(recent_data)
            predictions.append(pred)
            recent_data = np.append(recent_data[1:], pred)
        
        logger.info("LSTM Predictions vs Actual:")
        for i, (pred, actual) in enumerate(zip(predictions, actual_values)):
            logger.info(f"  Step {i+1}: Predicted={pred:.2f}, Actual={actual:.2f}, Error={abs(pred-actual)/actual:.2%}")
    
    return lstm_model

def train_isolation_forest(feature_data):
    """
    Train the Isolation Forest model for anomaly detection.
    
    Args:
        feature_data: DataFrame with features for anomaly detection
        
    Returns:
        Trained AnomalyDetector instance
    """
    logger.info("Training Isolation Forest model for anomaly detection")
    
    # Create and train Isolation Forest model
    anomaly_model = AnomalyDetector()
    
    # Features to use for anomaly detection
    feature_columns = [
        'request_count', 'avg_resp_code', 'max_resp_code',
        'avg_latency', 'max_latency', 'malicious_count', 'error_rate',
        'premium_ratio'
    ]
    
    # Train the model
    training_result = anomaly_model.train(
        data=feature_data,
        feature_columns=feature_columns,
        contamination=0.05  # Approximately 5% of data is anomalous
    )
    
    logger.info(f"Isolation Forest training complete:")
    logger.info(f"  Total samples: {training_result['num_samples']}")
    logger.info(f"  Detected anomalies: {training_result['num_anomalies']} ({training_result['anomaly_percentage']:.2f}%)")
    
    # Find anomalies in the training data
    predictions, scores = anomaly_model.detect_anomalies(feature_data[feature_columns])
    
    # Add results to feature data
    feature_data['anomaly'] = predictions == -1
    feature_data['anomaly_score'] = scores
    
    # Plot anomaly scores distribution
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, alpha=0.75)
    plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold')
    plt.title('Anomaly Score Distribution')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('plots/anomaly_score_distribution.png')
    
    # Plot anomalies over time
    plt.figure(figsize=(12, 6))
    plt.scatter(feature_data.index[~feature_data['anomaly']], feature_data.loc[~feature_data['anomaly'], 'request_count'], 
                color='blue', s=10, label='Normal')
    plt.scatter(feature_data.index[feature_data['anomaly']], feature_data.loc[feature_data['anomaly'], 'request_count'], 
                color='red', s=50, marker='x', label='Anomaly')
    plt.title('Detected Anomalies in Request Count')
    plt.xlabel('Time Index')
    plt.ylabel('Request Count')
    plt.legend()
    plt.savefig('plots/anomaly_detection_results.png')
    
    return anomaly_model

def main():
    """Main function to train and save models"""
    # Ensure models directory exists
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    
    # Load and preprocess data
    traffic_counts, feature_data = load_and_preprocess_data(settings.SYNTHETIC_DATA_PATH)
    
    # Train LSTM model
    lstm_model = train_lstm_model(traffic_counts)
    
    # Train Isolation Forest model
    anomaly_model = train_isolation_forest(feature_data)
    
    # Save the trained models
    lstm_model.save(settings.LSTM_MODEL_PATH, settings.LSTM_SCALER_PATH)
    anomaly_model.save(settings.ISOLATION_FOREST_PATH)
    
    logger.info(f"Models saved to {settings.MODEL_DIR}")
    logger.info("Training complete!")

if __name__ == "__main__":
    main()

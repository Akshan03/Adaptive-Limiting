# app/models/isolation_forest.py
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import pickle
import os
import logging
from typing import Tuple, List, Dict, Any, Union

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Isolation Forest model for detecting anomalous API usage patterns."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the anomaly detection model.
        
        Args:
            model_path: Path to saved model file
        """
        self.model = None
        self.feature_columns = None
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def load(self, model_path: str) -> None:
        """
        Load a pre-trained Isolation Forest model.
        
        Args:
            model_path: Path to the saved model
        """
        try:
            with open(model_path, 'rb') as f:
                saved_data = pickle.load(f)
                self.model = saved_data['model']
                self.feature_columns = saved_data.get('feature_columns')
            logger.info(f"Loaded Isolation Forest model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
    
    def train(self, data: pd.DataFrame, 
             feature_columns: List[str] = None,
             contamination: float = 0.05,
             random_state: int = 42) -> Dict[str, Any]:
        """
        Train the Isolation Forest model on traffic data.
        
        Args:
            data: DataFrame containing traffic data
            feature_columns: List of column names to use as features
            contamination: Expected proportion of anomalies in the data
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing training results
        """
        # If no feature columns provided, use all numeric columns
        if feature_columns is None:
            feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        self.feature_columns = feature_columns
        logger.info(f"Training Isolation Forest with features: {feature_columns}")
        
        # Extract features
        X = data[feature_columns].values
        
        # Initialize and train the model
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples='auto'
        )
        
        self.model.fit(X)
        
        # Get predictions and anomaly scores for training data
        predictions = self.model.predict(X)  # -1 for anomalies, 1 for normal
        anomaly_scores = -self.model.score_samples(X)  # Higher score = more anomalous
        
        # Count anomalies
        num_anomalies = sum(predictions == -1)
        
        logger.info(f"Training complete. Detected {num_anomalies} anomalies in training data.")
        
        return {
            "num_samples": len(X),
            "num_anomalies": num_anomalies,
            "anomaly_percentage": (num_anomalies / len(X)) * 100,
            "feature_columns": feature_columns
        }
    
    def detect_anomalies(self, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in the provided data.
        
        Args:
            data: DataFrame or numpy array containing feature data
            
        Returns:
            Tuple of (predictions, anomaly_scores)
                predictions: -1 for anomalies, 1 for normal instances
                anomaly_scores: Anomaly scores (higher values indicate more anomalous)
        """
        if self.model is None:
            logger.error("Model not initialized")
            return None, None
        
        # Convert DataFrame to array if needed
        if isinstance(data, pd.DataFrame):
            if self.feature_columns:
                # Use only the trained feature columns
                columns_to_use = [col for col in self.feature_columns if col in data.columns]
                if len(columns_to_use) != len(self.feature_columns):
                    logger.warning("Some feature columns are missing from the input data")
                X = data[columns_to_use].values
            else:
                X = data.values
        else:
            X = data
        
        # Get predictions and anomaly scores
        predictions = self.model.predict(X)
        anomaly_scores = -self.model.score_samples(X)
        
        return predictions, anomaly_scores
    
    def is_anomalous(self, data_point: Union[pd.DataFrame, np.ndarray, List]) -> Tuple[bool, float]:
        """
        Check if a single data point is anomalous.
        
        Args:
            data_point: A single data point
            
        Returns:
            Tuple of (is_anomaly, anomaly_score)
        """
        if self.model is None:
            logger.error("Model not initialized")
            return None, None
        
        # Convert to numpy array with shape (1, n_features)
        if isinstance(data_point, pd.DataFrame):
            if self.feature_columns:
                columns_to_use = [col for col in self.feature_columns if col in data_point.columns]
                X = data_point[columns_to_use].values
            else:
                X = data_point.values
        elif isinstance(data_point, list):
            X = np.array([data_point])
        else:
            X = data_point.reshape(1, -1)
        
        # Get prediction and score
        prediction = self.model.predict(X)[0]
        anomaly_score = -self.model.score_samples(X)[0]
        
        is_anomaly = prediction == -1  # -1 means anomaly
        
        return is_anomaly, anomaly_score
    
    def save(self, model_path: str) -> None:
        """
        Save the trained model.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model and feature columns
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'feature_columns': self.feature_columns
                }, f)
            logger.info(f"Model saved to {model_path}")

# app/models/lstm_model.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import logging
from typing import Tuple, List, Dict, Any, Union

logger = logging.getLogger(__name__)

class TrafficPredictor:
    """LSTM model for predicting API traffic patterns."""
    
    def __init__(self, model_path: str = None, scaler_path: str = None):
        """
        Initialize the traffic prediction model.
        
        Args:
            model_path: Path to saved Keras model file
            scaler_path: Path to saved scaler object
        """
        self.model = None
        self.scaler = None
        self.sequence_length = 5  # Reduced from 10 to 5
        self.historical_data = []  # Store historical data for accumulation
        
        # Load pre-trained model if paths provided
        if model_path and os.path.exists(model_path):
            try:
                self.model = load_model(model_path)
                logger.info(f"Loaded LSTM model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                
        if scaler_path and os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info(f"Loaded scaler from {scaler_path}")
            except Exception as e:
                logger.error(f"Error loading scaler: {str(e)}")
    
    def train(self, data: pd.DataFrame, target_col: str = 'request_count',
             time_col: str = 'timestamp', sequence_length: int = 10,
             test_size: float = 0.2, epochs: int = 50, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train the LSTM model on traffic data.
        
        Args:
            data: DataFrame containing traffic data
            target_col: Column name for the target variable (request count)
            time_col: Column name for the timestamp
            sequence_length: Number of time steps in each sequence
            test_size: Fraction of data to use for testing
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary containing training history and metrics
        """
        from sklearn.model_selection import train_test_split
        
        self.sequence_length = sequence_length
        
        # Ensure data is sorted by time
        data = data.sort_values(by=time_col)
        
        # Extract target variable
        values = data[target_col].values.reshape(-1, 1)
        
        # Initialize and fit scaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_values = self.scaler.fit_transform(values)
        
        # Create sequences for LSTM
        X, y = self._create_sequences(scaled_values, self.sequence_length)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Build the LSTM model
        self.model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        self.model.compile(optimizer='adam', loss='mse')
        
        # Train the model with early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate the model
        test_loss = self.model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"Test MSE: {test_loss}")
        
        return {
            "history": history.history,
            "test_loss": test_loss
        }
    
    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input sequences for the LSTM model.
        
        Args:
            data: Input array of shape (n_samples, n_features)
            sequence_length: Number of time steps in each sequence
            
        Returns:
            Tuple of (X, y) arrays for model training
        """
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)
    
    def predict(self, recent_data: Union[List[float], np.ndarray]) -> float:
        """
        Predict the next time step's traffic based on recent data.
        
        Args:
            recent_data: Array of recent traffic counts
            
        Returns:
            Predicted traffic count for the next time step
        """
        if self.model is None or self.scaler is None:
            logger.error("Model or scaler not initialized")
            return None
        
        # Add the new data points to our historical record
        if isinstance(recent_data, list):
            self.historical_data.extend(recent_data)
        else:
            self.historical_data.extend(recent_data.tolist())
        
        # Keep only the most recent data points
        self.historical_data = self.historical_data[-30:]  # Keep at most 30 points
        
        # Check if we have enough data points now
        if len(self.historical_data) < self.sequence_length:
            logger.warning(f"Not enough historical data for prediction. Have {len(self.historical_data)}, need {self.sequence_length}.")
            return None
        
        try:
            # Take the last sequence_length data points
            recent_data = np.array(self.historical_data[-self.sequence_length:]).reshape(-1, 1)
            
            # Scale the data
            scaled_data = self.scaler.transform(recent_data)
            
            # Reshape for LSTM [samples, time steps, features]
            X = scaled_data.reshape(1, self.sequence_length, 1)
            
            # Make prediction
            scaled_prediction = self.model.predict(X, verbose=0)
            
            # Inverse transform to get actual prediction
            prediction = self.scaler.inverse_transform(scaled_prediction)[0, 0]
            
            return prediction
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return None
    
    def save(self, model_path: str, scaler_path: str) -> None:
        """
        Save the trained model and scaler.
        
        Args:
            model_path: Path to save the model
            scaler_path: Path to save the scaler
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        
        if self.model is not None:
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")
        
        if self.scaler is not None:
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info(f"Scaler saved to {scaler_path}")

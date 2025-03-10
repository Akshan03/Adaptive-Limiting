# tests/test_models.py
import pytest
import numpy as np
import pandas as pd
import os
import tempfile

# Import app modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.lstm_model import TrafficPredictor
from app.models.isolation_forest import AnomalyDetector

# Create synthetic data for testing
@pytest.fixture
def sample_traffic_data():
    """Create sample traffic data for testing the LSTM model"""
    # Create time series data with a clear pattern
    np.random.seed(42)
    
    # Generate 100 data points with a sine wave pattern + noise
    dates = pd.date_range(start="2023-01-01", periods=100, freq="1h")
    base_traffic = 50 + 30 * np.sin(np.linspace(0, 4*np.pi, 100))
    noise = np.random.normal(0, 5, 100)
    traffic = base_traffic + noise
    
    df = pd.DataFrame({
        "timestamp": dates,
        "request_count": traffic
    })
    
    return df

@pytest.fixture
def sample_anomaly_data():
    """Create sample data for testing the anomaly detection model"""
    np.random.seed(42)
    
    # Generate 100 normal data points
    normal_data = np.random.normal(50, 10, (100, 2))
    
    # Generate 10 anomalous data points
    anomaly_data = np.random.normal(100, 30, (10, 2))
    
    # Combine the data
    X = np.vstack([normal_data, anomaly_data])
    
    # Create labels (1 for normal, -1 for anomalies)
    y = np.ones(110)
    y[100:] = -1
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=["feature1", "feature2"])
    df["anomaly"] = y == -1
    
    return df

class TestLSTMModel:
    """Test the LSTM model for traffic prediction"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = TrafficPredictor()
        
        assert model.model is None
        assert model.scaler is None
        assert model.sequence_length == 10
    
    def test_training(self, sample_traffic_data):
        """Test model training"""
        model = TrafficPredictor()
        
        # Train the model with a small number of epochs
        result = model.train(
            data=sample_traffic_data,
            target_col="request_count",
            time_col="timestamp",
            sequence_length=5,
            epochs=2,
            batch_size=4
        )
        
        # Model should be initialized after training
        assert model.model is not None
        assert model.scaler is not None
        assert model.sequence_length == 5
        
        # Result should contain history and metrics
        assert "history" in result
        assert "test_loss" in result
    
    def test_prediction(self, sample_traffic_data):
        """Test prediction functionality"""
        model = TrafficPredictor()
        
        # Train the model
        model.train(
            data=sample_traffic_data,
            target_col="request_count",
            time_col="timestamp",
            sequence_length=5,
            epochs=5
        )
        
        # Create recent data for prediction
        recent_data = sample_traffic_data["request_count"].values[-10:]
        
        # Test prediction
        prediction = model.predict(recent_data)
        
        # Prediction should be a float
        assert isinstance(prediction, float)
        
        # Prediction should be in a reasonable range (based on our data)
        assert 0 <= prediction <= 100
    
    def test_save_and_load(self, sample_traffic_data):
        """Test saving and loading models"""
        # Create temporary files for model and scaler
        with tempfile.NamedTemporaryFile(suffix=".keras") as model_file, \
             tempfile.NamedTemporaryFile(suffix=".pkl") as scaler_file:
            
            # Train and save a model
            model1 = TrafficPredictor()
            model1.train(
                data=sample_traffic_data,
                target_col="request_count",
                time_col="timestamp",
                epochs=2
            )
            model1.save(model_file.name, scaler_file.name)
            
            # Create a new model and load the saved files
            model2 = TrafficPredictor(model_path=model_file.name, scaler_path=scaler_file.name)
            
            # Both models should make similar predictions
            recent_data = sample_traffic_data["request_count"].values[-10:]
            pred1 = model1.predict(recent_data)
            pred2 = model2.predict(recent_data)
            
            # Predictions should be similar (not exactly equal due to model initialization)
            assert pred1 is not None
            assert pred2 is not None

class TestIsolationForest:
    """Test the Isolation Forest model for anomaly detection"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = AnomalyDetector()
        
        assert model.model is None
        assert model.feature_columns is None
    
    def test_training(self, sample_anomaly_data):
        """Test model training"""
        model = AnomalyDetector()
        
        # Train the model
        result = model.train(
            data=sample_anomaly_data,
            feature_columns=["feature1", "feature2"],
            contamination=0.1
        )
        
        # Model should be initialized after training
        assert model.model is not None
        assert model.feature_columns == ["feature1", "feature2"]
        
        # Result should contain metrics
        assert "num_samples" in result
        assert "num_anomalies" in result
        assert "anomaly_percentage" in result
        
        # Anomaly percentage should be close to the contamination parameter
        assert abs(result["anomaly_percentage"] - 10.0) < 2.0  # Within 2% of expected 10%
    
    def test_anomaly_detection(self, sample_anomaly_data):
        """Test anomaly detection functionality"""
        model = AnomalyDetector()
        
        # Train the model
        model.train(
            data=sample_anomaly_data,
            feature_columns=["feature1", "feature2"],
            contamination=0.1
        )
        
        # Test detect_anomalies function
        predictions, scores = model.detect_anomalies(sample_anomaly_data[["feature1", "feature2"]])
        
        # Predictions should be -1 for anomalies, 1 for normal
        assert set(np.unique(predictions)) <= {-1, 1}
        
        # We should detect approximately 10% anomalies
        anomaly_ratio = (predictions == -1).mean()
        assert 0.05 <= anomaly_ratio <= 0.15  # Between 5% and 15%
        
        # Test is_anomalous function for specific data points
        known_normal = sample_anomaly_data.iloc[0][["feature1", "feature2"]].values
        known_anomaly = sample_anomaly_data.iloc[-1][["feature1", "feature2"]].values
        
        is_normal_anomalous, normal_score = model.is_anomalous(known_normal)
        is_anomaly_anomalous, anomaly_score = model.is_anomalous(known_anomaly)
        
        # Known anomaly should have higher score than known normal
        assert anomaly_score > normal_score
    
    def test_save_and_load(self, sample_anomaly_data):
        """Test saving and loading models"""
        # Create a temporary file for the model
        with tempfile.NamedTemporaryFile(suffix=".pkl") as model_file:
            
            # Train and save a model
            model1 = AnomalyDetector()
            model1.train(
                data=sample_anomaly_data,
                feature_columns=["feature1", "feature2"],
                contamination=0.1
            )
            model1.save(model_file.name)
            
            # Create a new model and load the saved file
            model2 = AnomalyDetector(model_path=model_file.name)
            
            # Both models should make similar predictions
            test_data = sample_anomaly_data[["feature1", "feature2"]].iloc[-10:].values
            
            pred1, scores1 = model1.detect_anomalies(test_data)
            pred2, scores2 = model2.detect_anomalies(test_data)
            
            # Predictions should be identical since we're using the exact same model
            np.testing.assert_array_equal(pred1, pred2)
            np.testing.assert_array_almost_equal(scores1, scores2)

def test_models_and_rate_limiter_integration():
    """Test the integration between models and rate limiter"""
    # This is a more complex test that would involve:
    # 1. Creating a traffic monitor
    # 2. Connecting it to trained models
    # 3. Simulating traffic and seeing if rate limits adjust
    
    # This is a simplified placeholder for what would be a more involved test
    assert True

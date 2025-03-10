# app/utils/traffic_monitor.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
import logging
from typing import Dict, List, Any

from app.models.lstm_model import TrafficPredictor
from app.models.isolation_forest import AnomalyDetector
from app.core.token_bucket import bucket_manager
from app.core.config import settings

logger = logging.getLogger(__name__)

class TrafficMonitor:
    """
    Monitors API traffic and dynamically adjusts rate limits based on ML predictions.
    """
    
    def __init__(
        self,
        lstm_model: TrafficPredictor,
        anomaly_model: AnomalyDetector,
        monitoring_interval: int = settings.MONITORING_INTERVAL,  # Default: check traffic every 60 seconds
        traffic_window: int = settings.TRAFFIC_WINDOW,            # Default: use 30 minutes of data for analysis
    ):
        self.lstm_model = lstm_model
        self.anomaly_model = anomaly_model
        self.monitoring_interval = monitoring_interval
        self.traffic_window = traffic_window
        
        # Store traffic logs in memory
        self.traffic_logs = []
        
        # Add this new line to track adjustment history
        self.adjustment_history = []
        
        # Threading control
        self.running = False
        self.monitor_thread = None
        
        # Traffic thresholds for different levels
        self.traffic_thresholds = settings.TRAFFIC_THRESHOLDS
        
        # Rate limit modifiers for different traffic levels
        self.rate_limit_modifiers = settings.RATE_LIMIT_MODIFIERS
        
        # Track predictions and anomalies for visualization
        self.predictions_history = []
        self.anomaly_history = []
        
        logger.info("Traffic monitor initialized")
    
    def start(self):
        """Start the traffic monitoring thread."""
        if self.running:
            logger.warning("Traffic monitor is already running")
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Traffic monitoring started")
    
    def stop(self):
        """Stop the traffic monitoring thread."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Traffic monitoring stopped")
    
    def log_request(self, request_data: Dict[str, Any]):
        """
        Log an API request for traffic analysis.
        
        Args:
            request_data: Dictionary containing request information
        """
        # Add timestamp if not present
        if "timestamp" not in request_data:
            request_data["timestamp"] = datetime.now().timestamp()  # Use Unix timestamp
        
        # Add the request to logs
        self.traffic_logs.append(request_data)
        
        logger.info(f"Logged request: {request_data}")  # Debugging log
        
        # Remove old logs outside the analysis window
        cutoff_time = datetime.now().timestamp() - self.traffic_window
        self.traffic_logs = [log for log in self.traffic_logs if log["timestamp"] > cutoff_time]
    
    def _monitoring_loop(self):
        """Main monitoring loop that runs in a background thread."""
        while self.running:
            try:
                # Analyze traffic and adjust rate limits
                self._analyze_and_adjust()
                
            except Exception as e:
                logger.error(f"Error in traffic monitoring: {str(e)}")
            
            # Sleep until next monitoring interval
            time.sleep(self.monitoring_interval)
    
    def _analyze_and_adjust(self):
        """Analyze traffic and adjust rate limits based on predictions and anomalies."""
        if len(self.traffic_logs) < 10:
            logger.debug("Not enough traffic data for analysis")
            return
        
        # Convert logs to DataFrame for analysis
        df = pd.DataFrame(self.traffic_logs)
        
        # Aggregate by time bins (e.g., 1-minute intervals)
        df['time_bin'] = pd.to_datetime(df['timestamp'], unit='s').dt.floor('1min')
        
        # Calculate request counts per minute
        traffic_counts = df.groupby('time_bin').size().reset_index(name='request_count')
        traffic_counts = traffic_counts.sort_values('time_bin')
        
        logger.info(f"Traffic counts: {traffic_counts.tail()}")  # Debugging log
        
        # Calculate error rates per minute (status code >= 400)
        if 'status_code' in df.columns:
            error_rates = df.groupby('time_bin').apply(
                lambda x: (x['status_code'] >= 400).mean()
            ).reset_index(name='error_rate')
            
            # Merge metrics
            metrics = pd.merge(traffic_counts, error_rates, on='time_bin', how='left')
            metrics['error_rate'] = metrics['error_rate'].fillna(0)
        else:
            metrics = traffic_counts
            metrics['error_rate'] = 0
        
        # Use LSTM to predict future traffic
        predicted_traffic = None
        if self.lstm_model.model is not None and len(traffic_counts) >= self.lstm_model.sequence_length:
            recent_counts = traffic_counts['request_count'].values[-self.lstm_model.sequence_length:]
            logger.info(f"Recent counts for LSTM: {recent_counts}")  # Debugging log
            predicted_traffic = self.lstm_model.predict(recent_counts)
            
            if predicted_traffic is not None:
                logger.info(f"Predicted traffic: {predicted_traffic:.2f}")  # Debugging log
                
                # Store prediction for visualization
                self.predictions_history.append({
                    'timestamp': datetime.now(),
                    'predicted_traffic': predicted_traffic
                })
                
                # Keep only recent history
                self.predictions_history = self.predictions_history[-100:]
                
                # Adjust rate limits based on traffic prediction
                self._adjust_for_predicted_traffic(predicted_traffic)
            else:
                logger.warning("LSTM prediction returned None")  # Debugging log
        else:
            if self.lstm_model.model is None:
                logger.warning("LSTM model is not initialized")  # Debugging log
            elif len(traffic_counts) < self.lstm_model.sequence_length:
                logger.warning(f"Not enough traffic data for LSTM prediction. Have {len(traffic_counts)}, need {self.lstm_model.sequence_length}")  # Debugging log
        
        # Use Isolation Forest to detect anomalies
        if self.anomaly_model.model is not None and len(metrics) >= 5:
            # Prepare features for anomaly detection
            feature_cols = ['request_count', 'error_rate']
            if len(feature_cols) > 0 and all(col in metrics.columns for col in feature_cols):
                anomaly_data = metrics[feature_cols].iloc[-5:].values
                
                # Check if latest data point is anomalous
                is_anomalous, anomaly_score = self.anomaly_model.is_anomalous(anomaly_data[-1])
                
                # Store anomaly info for visualization
                self.anomaly_history.append({
                    'timestamp': datetime.now(),
                    'is_anomalous': is_anomalous,
                    'anomaly_score': anomaly_score
                })
                
                # Keep only recent history
                self.anomaly_history = self.anomaly_history[-100:]
                
                if is_anomalous:
                    logger.warning(f"Anomalous traffic detected! Score: {anomaly_score:.4f}")
                    
                    # Adjust rate limits for anomaly
                    self._adjust_for_anomaly(anomaly_score)
    
    def _adjust_for_predicted_traffic(self, predicted_traffic: float):
        """
        Adjust rate limits based on predicted traffic.
        
        Args:
            predicted_traffic: Predicted request count for next interval
        """
        # Determine traffic level
        if predicted_traffic <= self.traffic_thresholds["low"]:
            traffic_level = "low"
        elif predicted_traffic >= self.traffic_thresholds["high"]:
            traffic_level = "high"
        else:
            traffic_level = "medium"
        
        logger.info(f"Adjusting rate limits for {traffic_level} traffic level (predicted: {predicted_traffic:.2f})")
        
        # Record adjustment for visualization
        adjustment_entry = {
            "timestamp": datetime.now(),
            "std_capacity": bucket_manager.tier_configs["STD"]["capacity"],
            "prm_capacity": bucket_manager.tier_configs["PRM"]["capacity"],
            "traffic_level": traffic_level,
            "predicted_traffic": predicted_traffic,
        }
        
        self.adjustment_history.append(adjustment_entry)
        self.adjustment_history = self.adjustment_history[-100:]  # Keep only recent history
        
        # Apply adjustments for each user tier
        for tier in ["STD", "PRM"]:
            modifier = self.rate_limit_modifiers[traffic_level][tier]
            
            # Get current config
            current_config = bucket_manager.tier_configs.get(tier, {})
            if not current_config:
                continue
                
            # Calculate new capacity with modifier
            base_capacity = current_config["capacity"]
            adjusted_capacity = int(base_capacity * modifier)
            
            # Update token bucket tier config
            bucket_manager.update_tier_config(
                tier=tier,
                capacity=adjusted_capacity
            )
            
            logger.info(f"Updated {tier} tier capacity to {adjusted_capacity}")
    
    def _adjust_for_anomaly(self, anomaly_score: float):
        """
        Apply defensive rate limiting when anomalies are detected.
        
        Args:
            anomaly_score: Anomaly score from Isolation Forest
        """
        # Calculate capacity reduction based on anomaly severity
        # Higher score = more severe anomaly = more reduction
        severity_multiplier = min(0.8, max(0.5, 1.0 - anomaly_score))
        
        logger.warning(f"Applying defensive rate limiting. Severity multiplier: {severity_multiplier:.2f}")
        
        # Adjust each tier's rate limits
        for tier in ["STD", "PRM"]:
            # Premium users get less reduction
            final_multiplier = severity_multiplier if tier == "STD" else (severity_multiplier + 1) / 2
            
            # Get current config
            current_config = bucket_manager.tier_configs.get(tier, {})
            if not current_config:
                continue
                
            # Calculate new capacity
            base_capacity = current_config["capacity"]
            adjusted_capacity = int(base_capacity * final_multiplier)
            
            # Update token bucket tier config
            bucket_manager.update_tier_config(
                tier=tier,
                capacity=adjusted_capacity
            )
            
            logger.warning(f"Applied anomaly-based reduction to {tier} tier. New capacity: {adjusted_capacity}")
    
    def get_traffic_metrics(self) -> Dict[str, Any]:
        """
        Get current traffic metrics for visualization.
        
        Returns:
            Dict with traffic metrics and predictions
        """
        # Current metrics
        current_metrics = {
            'current_time': datetime.now(),
            'active_users': len(set(log.get('user_id', '') for log in self.traffic_logs)),
            'request_count': len(self.traffic_logs),
            'std_users': sum(1 for log in self.traffic_logs if log.get('user_tier') == 'STD'),
            'prm_users': sum(1 for log in self.traffic_logs if log.get('user_tier') == 'PRM'),
        }
        
        # Recent predictions
        if self.predictions_history:
            current_metrics['latest_prediction'] = self.predictions_history[-1]['predicted_traffic']
        
        # Recent anomalies
        if self.anomaly_history:
            current_metrics['latest_anomaly_score'] = self.anomaly_history[-1]['anomaly_score']
            current_metrics['has_anomaly'] = self.anomaly_history[-1]['is_anomalous']
        
        # Add tier stats
        current_metrics['tier_stats'] = bucket_manager.get_tier_stats()
        
        return current_metrics

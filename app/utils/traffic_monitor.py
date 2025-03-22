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
        
        # Add warm-up period for anomaly detection
        self.monitoring_cycles = 0
        self.min_anomaly_cycles = 3  # Reduced from 5 to 3 cycles
        self.min_request_threshold = 60  # Reduced from 100 to 60 requests
        
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
        if len(self.traffic_logs) < 5:
            logger.debug("Not enough traffic data for analysis")
            return
        
        # Convert logs to DataFrame for analysis
        df = pd.DataFrame(self.traffic_logs)
        
        # Aggregate by time bins (e.g., 30-second intervals)
        df['time_bin'] = pd.to_datetime(df['timestamp'], unit='s').dt.floor('30s')
        
        # Calculate request counts per time bin
        traffic_counts = df.groupby('time_bin').size().reset_index(name='request_count')
        traffic_counts = traffic_counts.sort_values('time_bin')
        
        logger.info(f"Traffic counts: {traffic_counts.tail()}")  # Debugging log
        
        # Add spike detection using rate of change
        is_spike = False
        spike_score = 0.0
        if len(traffic_counts) >= 3:  # Need at least 3 bins to detect a spike
            recent_counts = traffic_counts['request_count'].values[-3:]
            
            # Calculate percentage increase from previous to current
            if recent_counts[-2] > 0:  # Avoid division by zero
                pct_increase = (recent_counts[-1] - recent_counts[-2]) / recent_counts[-2]
                
                # Consider it a spike if traffic increases by more than 50%
                if pct_increase > 0.5:
                    is_spike = True
                    spike_score = min(1.0, pct_increase)  # Cap at 1.0
                    logger.warning(f"Traffic spike detected! Increase: {pct_increase:.2%}, Score: {spike_score:.2f}")
        
        # Check for malicious traffic patterns
        is_malicious_pattern = False
        malicious_score = 0.0
        
        # Calculate error rate for recent traffic
        if 'status_code' in df.columns and not df.empty:
            error_rate = (df['status_code'] >= 400).mean()
            
            # High error rate could indicate a DDoS or brute force attack
            if error_rate > 0.3:  # More than 30% of requests are errors
                is_malicious_pattern = True
                malicious_score = min(1.0, error_rate * 2)  # Scale up to 1.0
                logger.warning(f"High error rate detected: {error_rate:.2%}, Score: {malicious_score:.2f}")
                
        # Check for high frequency of requests from a single user or IP
        if 'user_id' in df.columns and not df.empty:
            # Count requests per user
            user_counts = df.groupby('user_id').size()
            if not user_counts.empty:
                max_user_requests = user_counts.max()
                total_requests = len(df)
                
                # If a single user accounts for more than 40% of traffic, it might be suspicious
                if total_requests > 20 and max_user_requests / total_requests > 0.4:
                    is_malicious_pattern = True
                    concentration_score = min(1.0, max_user_requests / total_requests * 2)
                    malicious_score = max(malicious_score, concentration_score)
                    logger.warning(f"High traffic concentration from a single user: {max_user_requests / total_requests:.2%}, Score: {concentration_score:.2f}")
        
        # Calculate additional metrics per time bin
        window_data = {
            'request_count': traffic_counts['request_count'].iloc[-1] if not traffic_counts.empty else 0
        }
        
        # Calculate response code metrics
        if 'status_code' in df.columns:
            status_metrics = df.groupby('time_bin').agg({
                'status_code': ['mean', 'max']
            }).reset_index()
            status_metrics.columns = ['time_bin', 'avg_resp_code', 'max_resp_code']
            window_data['avg_resp_code'] = status_metrics['avg_resp_code'].iloc[-1] if not status_metrics.empty else 0
            window_data['max_resp_code'] = status_metrics['max_resp_code'].iloc[-1] if not status_metrics.empty else 0
            window_data['error_rate'] = df[df['status_code'] >= 400].groupby('time_bin').size().divide(
                df.groupby('time_bin').size()).fillna(0).iloc[-1] if not df.empty else 0
        else:
            window_data['avg_resp_code'] = 0
            window_data['max_resp_code'] = 0
            window_data['error_rate'] = 0
        
        # Calculate latency metrics
        if 'latency' in df.columns:
            latency_metrics = df.groupby('time_bin').agg({
                'latency': ['mean', 'max']
            }).reset_index()
            latency_metrics.columns = ['time_bin', 'avg_latency', 'max_latency']
            window_data['avg_latency'] = latency_metrics['avg_latency'].iloc[-1] if not latency_metrics.empty else 0
            window_data['max_latency'] = latency_metrics['max_latency'].iloc[-1] if not latency_metrics.empty else 0
        else:
            window_data['avg_latency'] = 0
            window_data['max_latency'] = 0
        
        # Calculate malicious request count
        if 'is_malicious' in df.columns:
            window_data['malicious_count'] = df[df['is_malicious'] == True].groupby('time_bin').size().iloc[-1] if not df.empty else 0
        else:
            window_data['malicious_count'] = 0
        
        # Calculate premium user ratio
        if 'user_tier' in df.columns:
            premium_count = df[df['user_tier'] == 'PRM'].groupby('time_bin').size().iloc[-1] if not df.empty else 0
            total_count = window_data['request_count']
            window_data['premium_ratio'] = premium_count / total_count if total_count > 0 else 0
        else:
            window_data['premium_ratio'] = 0
        
        # Create feature vector for model input
        current_features = [
            window_data['request_count'],
            window_data['avg_resp_code'],
            window_data['max_resp_code'],
            window_data['avg_latency'],
            window_data['max_latency'],
            window_data['malicious_count'],
            window_data['error_rate'],
            window_data['premium_ratio']
        ]
        
        logger.debug(f"Feature vector: {current_features}")  # Check feature count
        
        # Validate feature dimensions
        EXPECTED_FEATURES = 8
        if len(current_features) != EXPECTED_FEATURES:
            logger.error(f"Feature dimension mismatch. Expected {EXPECTED_FEATURES}, got {len(current_features)}")
            return
        
        # Use LSTM to predict future traffic
        predicted_traffic = None
        if self.lstm_model.model is not None:
            # Even if we don't have enough time bins yet, still pass available data to the model
            # The model will accumulate history internally
            if not traffic_counts.empty:
                recent_counts = traffic_counts['request_count'].values
                logger.info(f"Available time bins for LSTM: {len(recent_counts)}")  # Log available time bins
                
                try:
                    # The LSTM model was trained only on request_count
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
                        logger.warning("LSTM prediction returned None - still accumulating history")  # Updated message
                except Exception as e:
                    logger.error(f"Error in prediction: {str(e)}")
            else:
                logger.warning("No traffic data available for prediction")
        else:
            logger.warning("LSTM model is not initialized")
        
        # Use Isolation Forest to detect anomalies
        if self.anomaly_model.model is not None:
            # Increment monitoring cycles counter
            self.monitoring_cycles += 1
            
            # Only run anomaly detection after warm-up period and if we have enough requests
            total_requests = len(self.traffic_logs)
            if self.monitoring_cycles < self.min_anomaly_cycles or total_requests < self.min_request_threshold:
                logger.info(f"Skipping anomaly detection: warm-up period ({self.monitoring_cycles}/{self.min_anomaly_cycles}) or insufficient requests ({total_requests}/{self.min_request_threshold})")
            else:
                # Prepare features for anomaly detection
                try:
                    # Make sure the feature vector has the correct dimensions
                    if len(current_features) == EXPECTED_FEATURES:
                        # Check if latest data point is anomalous
                        is_anomalous, anomaly_score = self.anomaly_model.is_anomalous(current_features)
                        
                        # Apply a higher threshold for anomaly score to reduce false positives
                        # Default threshold is typically around 0.5
                        anomaly_threshold = 0.58  # Reduced from 0.65 to 0.58 to be more sensitive
                        is_anomalous = is_anomalous and anomaly_score > anomaly_threshold
                        
                        # If we detected a spike, combine with anomaly detection
                        if is_spike:
                            # Increase the anomaly score if there's a spike
                            anomaly_score = max(anomaly_score, spike_score)
                            is_anomalous = True
                            logger.warning(f"Combining spike detection with anomaly detection. Combined score: {anomaly_score:.4f}")
                        
                        # If we detected a malicious pattern, combine with anomaly detection
                        if is_malicious_pattern:
                            # Increase the anomaly score for malicious patterns
                            anomaly_score = max(anomaly_score, malicious_score)
                            is_anomalous = True
                            logger.warning(f"Combining malicious pattern detection with anomaly detection. Combined score: {anomaly_score:.4f}")
                        
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
                        else:
                            logger.info(f"Normal traffic pattern. Anomaly score: {anomaly_score:.4f}")
                    else:
                        logger.error(f"Cannot perform anomaly detection: Expected {EXPECTED_FEATURES} features, got {len(current_features)}")
                except Exception as e:
                    logger.error(f"Error in anomaly detection: {str(e)}")
    
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
        # Make this moderately aggressive
        severity_multiplier = min(0.85, max(0.65, 1.0 - anomaly_score))  # Changed from 0.7-0.9 to 0.65-0.85
        
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
            
            # Make sure we don't reduce capacity too much by setting a minimum
            minimum_capacity = int(base_capacity * 0.6)  # Never reduce below 60% of base capacity
            adjusted_capacity = max(adjusted_capacity, minimum_capacity)
            
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

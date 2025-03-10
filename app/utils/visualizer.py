# app/utils/visualizer.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List
import os

from app.core.token_bucket import bucket_manager
from app.core.config import settings

logger = logging.getLogger(__name__)

# Create templates directory if it doesn't exist
os.makedirs("app/templates", exist_ok=True)

class DashboardManager:
    """Manages the visualization dashboard for the rate limiter."""
    
    def __init__(self, app: FastAPI, traffic_monitor = None):
        """
        Initialize the dashboard manager.
        
        Args:
            app: FastAPI application instance
            traffic_monitor: TrafficMonitor instance
        """
        self.app = app
        self.traffic_monitor = traffic_monitor
        
        # Set up templates
        self.templates = Jinja2Templates(directory="app/templates")
        
        # Create the dashboard HTML template
        self._create_dashboard_template()
        
        # Register routes
        self._register_routes()
        
        logger.info("Dashboard manager initialized")
    
    def _create_dashboard_template(self):
        """Create the HTML template for the dashboard."""
        dashboard_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Adaptive API Rate Limiter Dashboard</title>
            <meta http-equiv="refresh" content="5">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                .chart-container {
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                    padding: 15px;
                }
                .header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                }
                h1 {
                    color: #333;
                }
                .stats-container {
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 20px;
                }
                .stat-box {
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    padding: 15px;
                    flex: 1;
                    margin: 0 10px;
                    text-align: center;
                }
                .stat-box h3 {
                    margin-top: 0;
                    color: #555;
                }
                .stat-value {
                    font-size: 24px;
                    font-weight: bold;
                    color: #0066cc;
                }
                .anomaly-alert {
                    background-color: #ffebee;
                    color: #c62828;
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    display: none;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Adaptive API Rate Limiter Dashboard</h1>
                    <p>Last updated: <span id="update-time">{{ current_time }}</span></p>
                </div>
                
                <div id="anomaly-alert" class="anomaly-alert">
                    <strong>Alert:</strong> Anomalous traffic detected! Applying defensive rate limiting.
                </div>
                
                <div class="stats-container">
                    <div class="stat-box">
                        <h3>Active Users</h3>
                        <div class="stat-value">{{ stats.active_users }}</div>
                    </div>
                    <div class="stat-box">
                        <h3>Requests (Last 30m)</h3>
                        <div class="stat-value">{{ stats.request_count }}</div>
                    </div>
                    <div class="stat-box">
                        <h3>Predicted Traffic</h3>
                        <div class="stat-value">{{ stats.latest_prediction|default('N/A', true) }}</div>
                    </div>
                    <div class="stat-box">
                        <h3>Traffic Level</h3>
                        <div class="stat-value" id="traffic-level">Medium</div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h2>Token Bucket Status by User Tier</h2>
                    <div id="token-chart" style="height: 400px;"></div>
                </div>
                
                <div class="chart-container">
                    <h2>Traffic Prediction & Anomaly Detection</h2>
                    <div id="traffic-chart" style="height: 400px;"></div>
                </div>
                
                <div class="chart-container">
                    <h2>Rate Limit Configuration</h2>
                    <div id="config-chart" style="height: 300px;"></div>
                </div>
            </div>
            
            <script>
                // Token Bucket Chart
                var tokenData = {{ token_data|safe }};
                
                var tokenChart = Plotly.newPlot('token-chart', [{
                    type: 'bar',
                    x: tokenData.tiers,
                    y: tokenData.used_tokens,
                    name: 'Used Tokens',
                    marker: {color: '#ff7f0e'}
                }, {
                    type: 'bar',
                    x: tokenData.tiers,
                    y: tokenData.available_tokens,
                    name: 'Available Tokens',
                    marker: {color: '#1f77b4'}
                }], {
                    barmode: 'stack',
                    yaxis: {title: 'Tokens'},
                    xaxis: {title: 'User Tier'}
                });
                
                // Traffic Chart
                var trafficData = {{ traffic_data|safe }};
                
                var trafficChart = Plotly.newPlot('traffic-chart', [{
                    type: 'scatter',
                    x: trafficData.timestamps,
                    y: trafficData.requests,
                    name: 'Actual Requests',
                    mode: 'lines+markers'
                }, {
                    type: 'scatter',
                    x: trafficData.prediction_timestamps,
                    y: trafficData.predictions,
                    name: 'Predicted Traffic',
                    mode: 'lines',
                    line: {dash: 'dot', color: 'blue'}
                }, {
                    type: 'scatter',
                    x: trafficData.anomaly_timestamps,
                    y: trafficData.anomaly_points,
                    name: 'Anomalies',
                    mode: 'markers',
                    marker: {
                        size: 10,
                        color: 'red',
                        symbol: 'x'
                    }
                }], {
                    yaxis: {title: 'Request Count'},
                    xaxis: {title: 'Time'}
                });
                
                // Config Chart
                var configData = {{ config_data|safe }};
                
                var configChart = Plotly.newPlot('config-chart', [{
                    type: 'bar',
                    x: configData.tiers,
                    y: configData.capacity,
                    name: 'Bucket Capacity',
                    marker: {color: '#2ca02c'}
                }, {
                    type: 'bar',
                    x: configData.tiers,
                    y: configData.refill_rate,
                    name: 'Refill Rate (per min)',
                    marker: {color: '#9467bd'}
                }], {
                    yaxis: {title: 'Value'},
                    xaxis: {title: 'User Tier'},
                    barmode: 'group'
                });
                
                // Update traffic level
                var latestPrediction = {{ stats.latest_prediction|default(0, true) }};
                var trafficLevel = document.getElementById('traffic-level');
                
                if (latestPrediction < {{ settings.TRAFFIC_THRESHOLDS.low }}) {
                    trafficLevel.textContent = 'Low';
                    trafficLevel.style.color = '#2ca02c';
                } else if (latestPrediction > {{ settings.TRAFFIC_THRESHOLDS.high }}) {
                    trafficLevel.textContent = 'High';
                    trafficLevel.style.color = '#d62728';
                } else {
                    trafficLevel.textContent = 'Medium';
                    trafficLevel.style.color = '#ff7f0e';
                }
                
                // Show anomaly alert if needed
                var hasAnomaly = {{ 'true' if stats.has_anomaly else 'false' }};
                var anomalyAlert = document.getElementById('anomaly-alert');
                
                if (hasAnomaly) {
                    anomalyAlert.style.display = 'block';
                }
            </script>
        </body>
        </html>
        """
        
        # Write the template to a file
        with open("app/templates/dashboard.html", "w") as f:
            f.write(dashboard_html)
    
    def _register_routes(self):
        """Register dashboard routes with the FastAPI app."""
        
        @self.app.get("/dashboard", response_class=HTMLResponse)
        async def dashboard(request: Request):
            """Render the dashboard."""
            # Get traffic metrics
            stats = self._get_stats()
            
            # Prepare token data
            token_data = self._prepare_token_data()
            
            # Prepare traffic data
            traffic_data = self._prepare_traffic_data()
            
            # Prepare config data
            config_data = self._prepare_config_data()
            
            # Render template
            return self.templates.TemplateResponse(
                "dashboard.html", 
                {
                    "request": request,
                    "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "stats": stats,
                    "token_data": json.dumps(token_data),
                    "traffic_data": json.dumps(traffic_data),
                    "config_data": json.dumps(config_data),
                    "settings": settings
                }
            )
        
        @self.app.get("/api/dashboard/data")
        async def dashboard_data():
            """API endpoint to get dashboard data."""
            return {
                "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "stats": self._get_stats(),
                "token_data": self._prepare_token_data(),
                "traffic_data": self._prepare_traffic_data(),
                "config_data": self._prepare_config_data()
            }
    
    def _get_stats(self) -> Dict[str, Any]:
        """Get current statistics for the dashboard."""
        if self.traffic_monitor:
            # Use traffic monitor metrics if available
            return self.traffic_monitor.get_traffic_metrics()
        else:
            # Default stats if traffic monitor not available
            tier_stats = bucket_manager.get_tier_stats()
            return {
                "active_users": sum(stats["user_count"] for tier, stats in tier_stats.items()),
                "request_count": 0,
                "std_users": tier_stats.get("STD", {}).get("user_count", 0),
                "prm_users": tier_stats.get("PRM", {}).get("user_count", 0),
                "latest_prediction": None,
                "latest_anomaly_score": None,
                "has_anomaly": False,
                "tier_stats": tier_stats
            }
    
    def _prepare_token_data(self) -> Dict[str, List]:
        """Prepare token bucket data for visualization."""
        tier_stats = bucket_manager.get_tier_stats()
        
        tiers = list(tier_stats.keys())
        available_tokens = [stats.get("avg_available", 0) for stats in tier_stats.values()]
        used_tokens = [stats.get("avg_used", 0) for stats in tier_stats.values()]
        
        return {
            "tiers": tiers,
            "available_tokens": available_tokens,
            "used_tokens": used_tokens
        }
    
    def _prepare_traffic_data(self) -> Dict[str, List]:
        """Prepare traffic data for visualization."""
        # Default empty data
        data = {
            "timestamps": [],
            "requests": [],
            "prediction_timestamps": [],
            "predictions": [],
            "anomaly_timestamps": [],
            "anomaly_points": []
        }
        
        # If traffic monitor is available, use its data
        if self.traffic_monitor:
            # Recent traffic
            logs_df = None
            if self.traffic_monitor.traffic_logs:
                logs_df = pd.DataFrame(self.traffic_monitor.traffic_logs)
                logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp'])
                traffic_counts = logs_df.set_index('timestamp').resample('1min').size().reset_index()
                traffic_counts.columns = ['timestamp', 'count']
                
                data["timestamps"] = traffic_counts['timestamp'].dt.strftime('%H:%M:%S').tolist()
                data["requests"] = traffic_counts['count'].tolist()
            
            # Predictions
            if self.traffic_monitor.predictions_history:
                pred_df = pd.DataFrame(self.traffic_monitor.predictions_history)
                data["prediction_timestamps"] = pred_df['timestamp'].dt.strftime('%H:%M:%S').tolist()
                data["predictions"] = pred_df['predicted_traffic'].tolist()
            
            # Anomalies
            if self.traffic_monitor.anomaly_history:
                anom_df = pd.DataFrame(self.traffic_monitor.anomaly_history)
                # Filter to only show actual anomalies
                anom_df = anom_df[anom_df['is_anomalous'] == True]
                
                if not anom_df.empty:
                    # For each anomaly, find the corresponding request count
                    anomaly_points = []
                    anomaly_timestamps = []
                    
                    for _, row in anom_df.iterrows():
                        if logs_df is not None:
                            # Find the closest timestamp in the traffic data
                            closest_idx = abs(logs_df['timestamp'] - row['timestamp']).idxmin()
                            closest_timestamp = logs_df.loc[closest_idx, 'timestamp']
                            
                            # Get the count around this time
                            if closest_timestamp in traffic_counts['timestamp'].values:
                                count = traffic_counts.loc[
                                    traffic_counts['timestamp'] == closest_timestamp, 'count'
                                ].values[0]
                                
                                anomaly_timestamps.append(closest_timestamp.strftime('%H:%M:%S'))
                                anomaly_points.append(count)
                    
                    data["anomaly_timestamps"] = anomaly_timestamps
                    data["anomaly_points"] = anomaly_points
        
        return data
    
    def _prepare_config_data(self) -> Dict[str, List]:
        """Prepare configuration data for visualization."""
        tier_configs = bucket_manager.tier_configs
        
        tiers = list(tier_configs.keys())
        capacity = [config.get("capacity", 0) for config in tier_configs.values()]
        
        # Calculate refill rates in tokens per minute
        refill_rate = [
            config.get("refill_tokens", 0) * (60 / config.get("refill_duration", 60))
            for config in tier_configs.values()
        ]
        
        return {
            "tiers": tiers,
            "capacity": capacity,
            "refill_rate": refill_rate
        }

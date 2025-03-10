# dashboard/app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
import threading
import sys
import os
from datetime import datetime, timedelta
import json

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.core.token_bucket import bucket_manager

# Set page configuration
st.set_page_config(
    page_title="Adaptive API Rate Limiter Dashboard",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define API endpoint
API_URL = "http://localhost:8000"

# Function to get dashboard data from API
def get_dashboard_data():
    try:
        response = requests.get(f"{API_URL}/api/dashboard/data")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching dashboard data: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

# Function to get tier statistics
def get_tier_stats():
    try:
        response = requests.get(f"{API_URL}/api/admin/tier-stats")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception:
        return None

# Dashboard title and description
st.title("Adaptive AI-Based API Rate Limiter Dashboard")
st.markdown("""
This dashboard visualizes the real-time behavior of an AI-powered API rate limiter.
The system uses LSTM for traffic prediction and Isolation Forest for anomaly detection,
dynamically adjusting token bucket parameters based on traffic patterns.
""")

# Sidebar controls
st.sidebar.header("Dashboard Controls")

refresh_rate = st.sidebar.slider(
    "Refresh Rate (seconds)", 
    min_value=1, 
    max_value=60, 
    value=5
)

traffic_view = st.sidebar.selectbox(
    "Traffic View",
    ["Real-time", "Last Hour", "Last Day"]
)

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["Rate Limiting Status", "Traffic Analysis", "Configuration"])

# Get the latest data
dashboard_data = get_dashboard_data()
tier_stats = get_tier_stats()

# Tab 1: Rate Limiting Status
with tab1:
    if dashboard_data:
        # Create metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Users", dashboard_data["stats"]["active_users"])
        
        with col2:
            st.metric("Requests (Last 30m)", dashboard_data["stats"]["request_count"])
        
        with col3:
            pred_value = dashboard_data["stats"].get("latest_prediction", "N/A")
            pred_value = round(pred_value, 2) if isinstance(pred_value, (int, float)) else pred_value
            st.metric("Predicted Traffic", pred_value)
        
        with col4:
            has_anomaly = dashboard_data["stats"].get("has_anomaly", False)
            anomaly_status = "⚠️ Detected" if has_anomaly else "Normal"
            st.metric("Anomaly Status", anomaly_status)
        
        # Display token bucket visualization
        st.subheader("Token Bucket Status by User Tier")
        
        token_data = dashboard_data["token_data"]
        
        # Create token bucket figure
        token_fig = go.Figure()
        
        token_fig.add_trace(go.Bar(
            x=token_data["tiers"],
            y=token_data["used_tokens"],
            name="Used Tokens",
            marker_color="#ff7f0e"
        ))
        
        token_fig.add_trace(go.Bar(
            x=token_data["tiers"],
            y=token_data["available_tokens"],
            name="Available Tokens",
            marker_color="#1f77b4"
        ))
        
        token_fig.update_layout(
            barmode="stack",
            title="Token Bucket Status",
            xaxis_title="User Tier",
            yaxis_title="Tokens",
            height=400
        )
        
        st.plotly_chart(token_fig, use_container_width=True)
        
        # Anomalies Over Time Section
        st.subheader("Anomalies Detected Over Time")
        
        # Check if anomaly history exists in the dashboard data
        if "anomaly_history" in dashboard_data["stats"]:
            anomaly_history = dashboard_data["stats"]["anomaly_history"]
            
            # Convert anomaly history to a DataFrame for visualization
            anomaly_df = pd.DataFrame(anomaly_history)
            
            # Ensure timestamps are properly formatted
            anomaly_df["timestamp"] = pd.to_datetime(anomaly_df["timestamp"])
            
            # Create a scatter plot for anomaly scores over time
            fig = px.scatter(
                anomaly_df,
                x="timestamp",
                y="anomaly_score",
                title="Anomaly Scores Over Time",
                labels={"timestamp": "Time", "anomaly_score": "Anomaly Score"},
                color="is_anomalous",
                color_discrete_map={True: "red", False: "blue"},
                size="anomaly_score",  # Larger points for higher scores
                hover_data=["timestamp"]
            )
            
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Anomaly Score",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No anomaly data available yet.")
        
        # Display detailed stats for each tier
        if tier_stats:
            st.subheader("Detailed Tier Statistics")
            
            # Create columns for each tier
            tier_cols = st.columns(len(tier_stats))
            
            for i, (tier, stats) in enumerate(tier_stats.items()):
                with tier_cols[i]:
                    st.write(f"### {tier} Tier")
                    st.write(f"**Users:** {stats['user_count']}")
                    st.write(f"**Avg. Capacity:** {stats.get('avg_capacity', 0):.1f}")
                    st.write(f"**Avg. Available:** {stats.get('avg_available', 0):.1f}")
                    st.write(f"**Avg. Used:** {stats.get('avg_used', 0):.1f}")
                    st.write(f"**Tokens Consumed:** {stats.get('total_consumed', 0)}")
                    
                    # Display refill rate
                    config = stats.get('config', {})
                    refill_rate = config.get('refill_tokens', 0) / config.get('refill_duration', 60)
                    st.write(f"**Refill Rate:** {refill_rate:.2f} tokens/sec")
    else:
        st.warning("Unable to fetch real-time data. Make sure the API server is running.")

# Tab 2: Traffic Analysis
with tab2:
    if dashboard_data:
        st.subheader("API Traffic Patterns")
        
        traffic_data = dashboard_data.get("traffic_data", {})
        
        # Debug information
        st.write(f"Data available: {list(traffic_data.keys())}")
        
        # Create traffic figure
        traffic_fig = go.Figure()
        
        # Add actual traffic
        if traffic_data.get("timestamps") and traffic_data.get("requests"):
            traffic_fig.add_trace(go.Scatter(
                x=traffic_data["timestamps"],
                y=traffic_data["requests"],
                name="Actual Requests",
                mode="lines+markers",
                line=dict(color="#1f77b4")
            ))
            st.write(f"Actual traffic data points: {len(traffic_data['timestamps'])}")
        else:
            st.warning("No actual traffic data available")
        
        # Add predictions
        if traffic_data.get("prediction_timestamps") and traffic_data.get("predictions"):
            traffic_fig.add_trace(go.Scatter(
                x=traffic_data["prediction_timestamps"],
                y=traffic_data["predictions"],
                name="Predicted Traffic",
                mode="lines",
                line=dict(dash="dot", color="green")
            ))
            st.write(f"Prediction data points: {len(traffic_data['prediction_timestamps'])}")
        else:
            st.warning("No prediction data available")
        
        # Add anomalies
        if traffic_data.get("anomaly_timestamps") and traffic_data.get("anomaly_points"):
            traffic_fig.add_trace(go.Scatter(
                x=traffic_data["anomaly_timestamps"],
                y=traffic_data["anomaly_points"],
                name="Anomalies",
                mode="markers",
                marker=dict(size=12, color="red", symbol="x")
            ))
            st.write(f"Anomaly data points: {len(traffic_data['anomaly_timestamps'])}")
        
        traffic_fig.update_layout(
            title="Traffic Analysis with AI Predictions",
            xaxis_title="Time",
            yaxis_title="Request Count",
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(traffic_fig, use_container_width=True)
        
        # Predicted vs Actual Traffic
        st.write("### Predicted vs Actual Traffic")
        
        if traffic_data.get("predictions") and traffic_data.get("timestamps"):
            pred_vs_actual_fig = go.Figure()
            
            pred_vs_actual_fig.add_trace(go.Scatter(
                x=traffic_data["timestamps"],
                y=traffic_data["requests"],
                mode="lines+markers",
                name="Actual Traffic",
                line=dict(color="blue")
            ))
            
            pred_vs_actual_fig.add_trace(go.Scatter(
                x=traffic_data["prediction_timestamps"],
                y=traffic_data["predictions"],
                mode="lines",
                name="Predicted Traffic",
                line=dict(dash="dot", color="green")
            ))
            
            pred_vs_actual_fig.update_layout(
                title="Predicted vs Actual Traffic",
                xaxis_title="Time",
                yaxis_title="Request Count",
                height=400
            )
            
            st.plotly_chart(pred_vs_actual_fig, use_container_width=True)
        else:
            st.warning("Not enough data to display predicted vs actual traffic")
        
        # Create two columns for the next visualizations
        col1, col2 = st.columns(2)
        
        # Display anomaly history if available
        with col1:
            st.subheader("Anomaly Detection Results")
            
            if "anomaly_history" in dashboard_data["stats"] and dashboard_data["stats"]["anomaly_history"]:
                anomaly_history = dashboard_data["stats"]["anomaly_history"]
                anomaly_df = pd.DataFrame(anomaly_history)
                
                # Format timestamp
                anomaly_df["timestamp"] = pd.to_datetime(anomaly_df["timestamp"])
                anomaly_df["formatted_time"] = anomaly_df["timestamp"].dt.strftime("%H:%M:%S")
                
                # Create anomaly score chart
                anomaly_fig = go.Figure()
                
                anomaly_fig.add_trace(go.Scatter(
                    x=anomaly_df["formatted_time"],
                    y=anomaly_df["anomaly_score"],
                    mode="lines+markers",
                    name="Anomaly Score",
                    marker=dict(
                        size=8,
                        color=anomaly_df["anomaly_score"],
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Score")
                    )
                ))
                
                # Add threshold line
                anomaly_fig.add_hline(
                    y=0.5, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="Anomaly Threshold"
                )
                
                anomaly_fig.update_layout(
                    title="Anomaly Scores Over Time",
                    xaxis_title="Time",
                    yaxis_title="Anomaly Score",
                    height=400
                )
                
                st.plotly_chart(anomaly_fig, use_container_width=True)
                
                # Count of anomalies
                anomalies_count = sum(anomaly_df["is_anomalous"])
                st.metric("Total Anomalies Detected", anomalies_count)
            else:
                st.info("No anomaly detection data available yet")
        
        # Add Token Bucket Adjustments Over Time visualization
        with col2:
            st.subheader("Token Bucket Adjustments")
            
            # Check if token adjustment history is available
            if "token_adjustment_history" in dashboard_data and dashboard_data["token_adjustment_history"]:
                adj_history = dashboard_data["token_adjustment_history"]
                adj_df = pd.DataFrame(adj_history)
                
                # Format timestamp
                adj_df["timestamp"] = pd.to_datetime(adj_df["timestamp"])
                adj_df["formatted_time"] = adj_df["timestamp"].dt.strftime("%H:%M:%S")
                
                # Create token adjustment chart
                adj_fig = go.Figure()
                
                for tier in adj_df["tier"].unique():
                    tier_data = adj_df[adj_df["tier"] == tier]
                    
                    adj_fig.add_trace(go.Scatter(
                        x=tier_data["formatted_time"],
                        y=tier_data["capacity"],
                        mode="lines+markers",
                        name=f"{tier} Capacity"
                    ))
                
                adj_fig.update_layout(
                    title="Token Bucket Capacity Adjustments",
                    xaxis_title="Time",
                    yaxis_title="Capacity",
                    height=400
                )
                
                st.plotly_chart(adj_fig, use_container_width=True)
            else:
                # Create a static chart showing current token bucket parameters
                token_data = dashboard_data["token_data"]
                
                # Create a bar chart for token bucket adjustments
                adj_fig = go.Figure()
                
                adj_fig.add_trace(go.Bar(
                    x=token_data["tiers"],
                    y=token_data["capacity"],
                    name="Bucket Capacity",
                    marker_color="orange"
                ))
                
                # Calculate refill rate per minute for better visualization
                refill_per_min = []
                for i, tier in enumerate(token_data["tiers"]):
                    config = tier_stats.get(tier, {}).get("config", {})
                    if config:
                        rate = (config.get("refill_tokens", 0) * 60) / config.get("refill_duration", 60)
                        refill_per_min.append(rate)
                    else:
                        refill_per_min.append(0)
                
                adj_fig.add_trace(go.Bar(
                    x=token_data["tiers"],
                    y=refill_per_min,
                    name="Refill Rate (per min)",
                    marker_color="green"
                ))
                
                adj_fig.update_layout(
                    barmode="group",
                    title="Current Token Bucket Configuration",
                    xaxis_title="User Tier",
                    yaxis_title="Value",
                    height=400
                )
                
                st.plotly_chart(adj_fig, use_container_width=True)
                
                st.info("Token adjustment history is not yet available. The chart above shows the current configuration.")
        
        # AI Model Performance Metrics
        st.subheader("AI Model Performance")
        
        # Create metrics for AI model performance
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        
        with perf_col1:
            # Prediction accuracy (if available)
            if "prediction_accuracy" in dashboard_data["stats"]:
                accuracy = dashboard_data["stats"]["prediction_accuracy"]
                st.metric("Prediction Accuracy", f"{accuracy:.2f}%")
            else:
                # Calculate a simple prediction error metric if raw data is available
                if traffic_data.get("actual_vs_predicted", []):
                    avp = traffic_data["actual_vs_predicted"]
                    errors = [abs(item["actual"] - item["predicted"]) / max(1, item["actual"]) for item in avp]
                    avg_error = sum(errors) / len(errors) if errors else 0
                    st.metric("Avg. Prediction Error", f"{avg_error:.2f}%")
                else:
                    st.metric("Prediction Accuracy", "N/A")
        
        with perf_col2:
            # Anomaly detection rate
            if "anomaly_detection_rate" in dashboard_data["stats"]:
                anom_rate = dashboard_data["stats"]["anomaly_detection_rate"]
                st.metric("Anomaly Detection Rate", f"{anom_rate:.2f}%")
            else:
                if "anomaly_history" in dashboard_data["stats"] and dashboard_data["stats"]["anomaly_history"]:
                    anomaly_history = dashboard_data["stats"]["anomaly_history"]
                    anomaly_df = pd.DataFrame(anomaly_history)
                    anomaly_rate = (anomaly_df["is_anomalous"].sum() / len(anomaly_df)) * 100
                    st.metric("Anomaly Detection Rate", f"{anomaly_rate:.2f}%")
                else:
                    st.metric("Anomaly Detection Rate", "N/A")
        
        with perf_col3:
            # Rate limit effectiveness
            if "rate_limit_effectiveness" in dashboard_data["stats"]:
                effectiveness = dashboard_data["stats"]["rate_limit_effectiveness"]
                st.metric("Rate Limit Effectiveness", f"{effectiveness:.2f}%")
            else:
                st.metric("Rate Limit Effectiveness", "N/A")
    else:
        st.warning("Unable to fetch traffic analysis data. Make sure the API server is running.")

# Tab 3: Configuration
with tab3:
    if dashboard_data:
        st.subheader("Rate Limiter Configuration")
        
        config_data = dashboard_data["config_data"]
        
        # Create config figure
        config_fig = go.Figure()
        
        config_fig.add_trace(go.Bar(
            x=config_data["tiers"],
            y=config_data["capacity"],
            name="Bucket Capacity",
            marker_color="#2ca02c"
        ))
        
        config_fig.add_trace(go.Bar(
            x=config_data["tiers"],
            y=config_data["refill_rate"],
            name="Refill Rate (per min)",
            marker_color="#9467bd"
        ))
        
        config_fig.update_layout(
            barmode="group",
            title="Token Bucket Configuration by Tier",
            xaxis_title="User Tier",
            yaxis_title="Value",
            height=400
        )
        
        st.plotly_chart(config_fig, use_container_width=True)
        
        # Allow configuration updates
        st.subheader("Update Rate Limit Configuration")
        
        update_col1, update_col2 = st.columns(2)
        
        with update_col1:
            update_tier = st.selectbox("Select Tier", config_data["tiers"])
            update_capacity = st.number_input("New Capacity", min_value=1, value=100)
        
        with update_col2:
            update_refill_tokens = st.number_input("New Refill Tokens", min_value=1, value=10)
            update_refill_duration = st.number_input("New Refill Duration (seconds)", min_value=1, value=60)
        
        if st.button("Update Configuration"):
            try:
                response = requests.post(
                    f"{API_URL}/api/admin/update-tier",
                    params={
                        "tier": update_tier,
                        "capacity": update_capacity,
                        "refill_tokens": update_refill_tokens,
                        "refill_duration": update_refill_duration
                    }
                )
                
                if response.status_code == 200:
                    st.success(f"Updated configuration for {update_tier} tier!")
                else:
                    st.error(f"Failed to update configuration: {response.status_code}")
            except Exception as e:
                st.error(f"Error updating configuration: {str(e)}")
        
        # Display traffic threshold settings
        st.subheader("Traffic Threshold Settings")
        
        thresholds = settings.TRAFFIC_THRESHOLDS
        modifiers = settings.RATE_LIMIT_MODIFIERS
        
        threshold_df = pd.DataFrame({
            "Threshold": ["Low", "Medium", "High"],
            "Value": [thresholds["low"], 
                     f"{thresholds['low']} - {thresholds['high']}", 
                     thresholds["high"]]
        })
        
        st.table(threshold_df)
        
        # Display rate limit modifiers
        st.subheader("Rate Limit Modifiers by Traffic Level")
        
        modifiers_data = {
            "User Tier": ["Standard", "Premium"],
            "Low Traffic": [modifiers["low"]["STD"], modifiers["low"]["PRM"]],
            "Medium Traffic": [modifiers["medium"]["STD"], modifiers["medium"]["PRM"]],
            "High Traffic": [modifiers["high"]["STD"], modifiers["high"]["PRM"]]
        }
        
        modifiers_df = pd.DataFrame(modifiers_data)
        
        st.table(modifiers_df)
    else:
        st.warning("Unable to fetch configuration data. Make sure the API server is running.")

# Auto-refresh the dashboard
if st.sidebar.button("Refresh Now"):
    st.experimental_rerun()

# Add auto-refresh using JavaScript
st.markdown(
    f"""
    <script>
        var refreshRate = {refresh_rate * 1000};
        setInterval(function() {{
            window.location.reload();
        }}, refreshRate);
    </script>
    """,
    unsafe_allow_html=True
)

# Show last update time
st.sidebar.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# scripts/simulate_requests.py
import pandas as pd
import numpy as np
import requests
import time
import random
import argparse
import logging
import sys
import os
from datetime import datetime, timedelta
import threading
import signal
import json

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("request_simulator")

# Global variables
running = True
request_count = 0
success_count = 0
rate_limited_count = 0
error_count = 0

class RequestSimulator:
    """Simulates API requests based on synthetic data patterns."""
    
    def __init__(self, api_url, data_path, request_rate=10):
        """
        Initialize the request simulator.
        
        Args:
            api_url: Base URL of the API
            data_path: Path to the synthetic data file
            request_rate: Average requests per second
        """
        self.api_url = api_url
        self.data_path = data_path
        self.request_rate = request_rate
        self.data = None
        self.user_pools = {
            "STD": [],
            "PRM": []
        }
    
    def load_data(self):
        """Load and prepare synthetic data."""
        logger.info(f"Loading data from {self.data_path}")
        
        # Check file extension and load accordingly
        if self.data_path.endswith('.xlsx'):
            self.data = pd.read_excel(self.data_path)
        elif self.data_path.endswith('.csv'):
            self.data = pd.read_csv(self.data_path)
        else:
            raise ValueError("Unsupported file format. Use .xlsx or .csv")
        
        logger.info(f"Loaded {len(self.data)} records")
        
        # Extract unique users by tier
        users_by_tier = self.data.groupby('USER_TIER')['USER_ID'].unique()
        
        for tier in ['STD', 'PRM']:
            if tier in users_by_tier:
                self.user_pools[tier] = users_by_tier[tier].tolist()
                logger.info(f"Found {len(self.user_pools[tier])} {tier} users")
    
    def get_random_user(self, tier=None):
        """
        Get a random user, optionally filtered by tier.
        
        Args:
            tier: User tier (STD or PRM)
            
        Returns:
            Dictionary with user_id and user_tier
        """
        if tier is None:
            tier = random.choice(['STD', 'PRM'])
        
        if not self.user_pools[tier]:
            # Fallback to other tier if no users found
            other_tier = 'PRM' if tier == 'STD' else 'STD'
            if not self.user_pools[other_tier]:
                return {"user_id": f"user_{random.randint(1000, 9999)}", "user_tier": tier}
            tier = other_tier
        
        user_id = random.choice(self.user_pools[tier])
        return {"user_id": user_id, "user_tier": tier}
    
    def get_random_endpoint(self):
        """
        Get a random endpoint based on predefined probabilities.
        
        Returns:
            Tuple of (endpoint, http_method)
        """
        endpoint_choices = ['/api/payment-gateway', '/api/browse', '/api/test']
        endpoint_probs = [0.2, 0.7, 0.1]  # Same as in traffic.py
        
        endpoint = random.choices(endpoint_choices, weights=endpoint_probs, k=1)[0]
        
        # Determine HTTP method (POST for payment gateway, GET for others)
        if endpoint == '/api/payment-gateway':
            http_method = 'POST'
        else:
            http_method = 'GET'
            
        return endpoint, http_method
    
    def generate_request(self, user=None, endpoint=None, http_method=None):
        """
        Generate a request based on user and endpoint information.
        
        Args:
            user: User information (dict with user_id and user_tier)
            endpoint: API endpoint
            http_method: HTTP method
            
        Returns:
            Dict with request details
        """
        # Get random user if not provided
        if user is None:
            user = self.get_random_user()
        
        # Get random endpoint if not provided
        if endpoint is None or http_method is None:
            endpoint, http_method = self.get_random_endpoint()
        
        # Generate a realistic user agent string
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
            "PostmanRuntime/7.43.0"
        ]
        
        return {
            "user_id": user["user_id"],
            "user_tier": user["user_tier"],
            "endpoint": endpoint,
            "http_method": http_method,
            "user_agent": random.choice(user_agents),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        }
    
    def send_request(self, request_data):
        """
        Send a request to the API.
        
        Args:
            request_data: Dict with request details
            
        Returns:
            Response object
        """
        global request_count, success_count, rate_limited_count, error_count
        
        # Prepare request
        url = f"{self.api_url}{request_data['endpoint']}"
        headers = {
            "User-ID": str(request_data["user_id"]),
            "User-Tier": request_data["user_tier"],
            "User-Agent": request_data["user_agent"]
        }
        
        # Prepare payload for POST requests
        payload = None
        if request_data["http_method"] == "POST":
            payload = {
                "timestamp": request_data["timestamp"],
                "data": f"Sample payload from {request_data['user_id']}"
            }
        
        try:
            # Send request
            if request_data["http_method"] == "GET":
                response = requests.get(url, headers=headers, timeout=5)
            else:  # POST
                response = requests.post(url, headers=headers, json=payload, timeout=5)
            
            # Update counters
            request_count += 1
            if response.status_code == 200:
                success_count += 1
            elif response.status_code == 429:
                rate_limited_count += 1
            else:
                error_count += 1
            
            # Log the response
            log_level = logging.INFO if response.status_code == 200 else logging.WARNING
            logger.log(
                log_level,
                f"{request_data['http_method']} {request_data['endpoint']} - "
                f"User: {request_data['user_id']} ({request_data['user_tier']}) - "
                f"Status: {response.status_code}"
            )
            
            return response
        
        except Exception as e:
            error_count += 1
            logger.error(f"Request error: {str(e)}")
            return None
    
    def simulate_requests(self, duration=None, malicious_mode=False):
        """
        Simulate API requests for a specified duration.
        
        Args:
            duration: Duration in seconds (None for unlimited)
            malicious_mode: Whether to simulate malicious traffic
        """
        global running
        
        logger.info(f"Starting request simulation - Press Ctrl+C to stop")
        logger.info(f"Target request rate: {self.request_rate} requests/second")
        logger.info(f"Malicious mode: {'ON' if malicious_mode else 'OFF'}")
        
        start_time = time.time()
        last_report_time = start_time
        
        while running:
            # Check if duration has elapsed
            if duration is not None and time.time() - start_time >= duration:
                break
            
            # Generate a request
            if malicious_mode and random.random() < 0.1:  # 10% chance of malicious request
                # Malicious request - target payment gateway with a specific user
                user = self.get_random_user('STD')  # Usually standard users
                request_data = self.generate_request(
                    user=user,
                    endpoint='/api/payment-gateway',
                    http_method='POST'
                )
                
                # Send a burst of requests
                burst_count = random.randint(5, 15)
                logger.warning(f"Simulating malicious burst: {burst_count} requests")
                
                for _ in range(burst_count):
                    if not running:
                        break
                    self.send_request(request_data)
                    time.sleep(0.1)  # 100ms between requests in burst
            else:
                # Normal request
                request_data = self.generate_request()
                self.send_request(request_data)
            
            # Report stats every 5 seconds
            if time.time() - last_report_time >= 5:
                elapsed = time.time() - last_report_time
                rate = request_count / elapsed if elapsed > 0 else 0
                
                logger.info(
                    f"Stats: {request_count} requests, {success_count} success, "
                    f"{rate_limited_count} rate limited, {error_count} errors - "
                    f"Rate: {rate:.2f} req/sec"
                )
                
                # Reset counters
                last_report_time = time.time()
                
            # Sleep to maintain target request rate (with some randomness)
            sleep_time = random.uniform(0.5, 1.5) / self.request_rate
            time.sleep(sleep_time)
    
    def simulate_traffic_pattern(self, pattern='random'):
        """
        Simulate a specific traffic pattern.
        
        Args:
            pattern: Type of pattern to simulate ('random', 'spike', 'wave')
        """
        global running
        
        patterns = {
            'random': 'Random traffic pattern',
            'spike': 'Traffic spike pattern',
            'wave': 'Traffic wave pattern'
        }
        
        logger.info(f"Simulating {patterns.get(pattern, 'unknown')} - Press Ctrl+C to stop")
        
        base_rate = self.request_rate
        start_time = time.time()
        
        while running:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Adjust request rate based on pattern
            if pattern == 'spike':
                # Create periodic spikes
                if int(elapsed) % 60 < 10:  # Spike for 10 seconds every minute
                    self.request_rate = base_rate * 5  # 5x during spike
                else:
                    self.request_rate = base_rate
                    
            elif pattern == 'wave':
                # Sinusoidal wave pattern
                phase = (elapsed % 300) / 300  # 5-minute cycle
                self.request_rate = base_rate * (1 + 2 * np.sin(2 * np.pi * phase))
            
            # Generate and send a request
            request_data = self.generate_request()
            self.send_request(request_data)
            
            # Sleep to maintain current request rate
            sleep_time = random.uniform(0.8, 1.2) / self.request_rate
            time.sleep(sleep_time)

def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully stop simulation."""
    global running
    logger.info("Stopping simulation...")
    running = False

def main():
    """Main function to run the request simulator."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='API Request Simulator')
    parser.add_argument('--url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--data', default=settings.SYNTHETIC_DATA_PATH, help='Path to synthetic data file')
    parser.add_argument('--rate', type=int, default=10, help='Requests per second')
    parser.add_argument('--duration', type=int, default=None, help='Duration in seconds')
    parser.add_argument('--pattern', default='random', choices=['random', 'spike', 'wave'], 
                       help='Traffic pattern to simulate')
    parser.add_argument('--malicious', action='store_true', help='Simulate malicious traffic')
    
    args = parser.parse_args()
    
    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and run simulator
    simulator = RequestSimulator(
        api_url=args.url,
        data_path=args.data,
        request_rate=args.rate
    )
    
    # Load data
    simulator.load_data()
    
    # Choose simulation mode based on pattern
    if args.pattern == 'random':
        simulator.simulate_requests(duration=args.duration, malicious_mode=args.malicious)
    else:
        simulator.simulate_traffic_pattern(pattern=args.pattern)

if __name__ == "__main__":
    main()

# scripts/traffic_spike.py
import requests
import threading
import time

# Configuration
API_URL = "http://localhost:8000/api/test"
CONCURRENT_REQUESTS = 300
BURST_DURATION = 30  # seconds

# Function to send requests
def send_request(user_id, user_tier):
    headers = {
        "User-ID": f"test_user_{user_id}",
        "User-Tier": user_tier
    }
    try:
        response = requests.get(API_URL, headers=headers)
        print(f"User {user_id} ({user_tier}): {response.status_code}")
    except Exception as e:
        print(f"Error: {str(e)}")

# Create a mix of standard and premium users
def run_spike():
    threads = []
    for i in range(CONCURRENT_REQUESTS):
        # 70% standard, 30% premium users
        tier = "PRM" if i % 10 < 3 else "STD"
        thread = threading.Thread(target=send_request, args=(i, tier))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()

# Run traffic spike
print(f"Starting traffic spike with {CONCURRENT_REQUESTS} concurrent requests")
start_time = time.time()
while time.time() - start_time < BURST_DURATION:
    run_spike()
    time.sleep(0.5)  # Small delay between bursts

print("Traffic spike completed")

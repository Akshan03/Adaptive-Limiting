import pandas as pd
from faker import Faker
import numpy as np
from datetime import datetime, timedelta

# Initialize tools
fake = Faker()
np.random.seed(42)

# Configuration
NUM_RECORDS = 500_000  # Total records (~2 weeks of traffic)
USER_BASE = 4000       # Unique users
MALICIOUS_RATIO = 0.01  # 1% malicious traffic

# Define endpoints and probabilities
endpoints = {
    "high_priority": ["/payment-gateway"],
    "low_priority": ["/browse", "/test"]
}
endpoint_probs = [0.2, 0.7, 0.1]  # /payment, /browse, /test

# Generate user base
users = [{
    "user_id": str(1000 + i),
    "user_tier": np.random.choice(["STD", "PRM"], p=[0.7, 0.3]),
    "is_malicious": np.random.rand() < MALICIOUS_RATIO
} for i in range(USER_BASE)]

def generate_record():
    user = users[np.random.randint(0, USER_BASE)]
    is_malicious = user["is_malicious"] or (np.random.rand() < 0.001)
    
    # Generate endpoint and HTTP method
    endpoint = np.random.choice(
        ["/payment-gateway", "/browse", "/test"], 
        p=endpoint_probs
    )
    
    if is_malicious:
        # Enhanced malicious behavior patterns
        endpoint = "/payment-gateway"  # Force payment gateway target
        http_method = "POST"  # Only POST requests
        ip = fake.ipv4()
        error_code = np.random.choice([429, 500], p=[0.8, 0.2])  # 80% rate limit errors
        latency = np.random.randint(500, 2000)  # High latency
        
        # Generate 10x more requests per malicious call
        return [{
            "ID": fake.uuid4(),
            "TIMESTAMP": fake.date_time_between(
                start_date="-30d", 
                end_date="now"
            ).strftime("%Y-%m-%d %H:%M:%S.%f"),
            "USER_ID": user["user_id"],
            "ENDPOINT": endpoint,
            "HTTP_METHOD": http_method,
            "IP_ADDRESS": ip,
            "REQUEST_ID": fake.uuid4(),
            "USER_AGENT": fake.user_agent(),
            "USER_TIER": user["user_tier"],
            "is_malicious": True,
            "response_code": error_code,
            "latency": latency
        } for _ in range(10)]  # 10 requests per malicious event
        
    else:
        # Normal user behavior
        http_method = "POST" if (endpoint == "/payment-gateway" and user["user_tier"] == "PRM") else "GET"
        ip = "0:0:0:0:0:0:0:1" if np.random.rand() < 0.2 else fake.ipv6()
        
        return [{
            "ID": fake.uuid4(),
            "TIMESTAMP": fake.date_time_between(
                start_date="-30d", 
                end_date="now"
            ).strftime("%Y-%m-%d %H:%M:%S.%f"),
            "USER_ID": user["user_id"],
            "ENDPOINT": endpoint,
            "HTTP_METHOD": http_method,
            "IP_ADDRESS": ip,
            "REQUEST_ID": fake.uuid4(),
            "USER_AGENT": "PostmanRuntime/7.43.0" if np.random.rand() < 0.1 else fake.user_agent(),
            "USER_TIER": user["user_tier"],
            "is_malicious": False,
            "response_code": 200,
            "latency": np.random.randint(50, 300)  # Normal latency
        }]

# Generate DataFrame
records = []
for _ in range(NUM_RECORDS // 10):  # Account for 10x malicious requests
    records.extend(generate_record())
    
df = pd.DataFrame(records)

# Add temporal patterns (spikes every 7 days)
spike_days = pd.date_range(end=datetime.now(), periods=4, freq="7D")
for day in spike_days:
    spike_records = []
    for _ in range(200):  # 200 malicious users per spike
        spike_records.extend(generate_record())
    for rec in spike_records:
        rec["TIMESTAMP"] = day.strftime("%Y-%m-%d") + " " + fake.time()
    df = pd.concat([df, pd.DataFrame(spike_records)])

# Save to Excel
df.to_csv("synthetic_api_traffic.csv", index=False)
print(f"Generated {len(df)} records with:")
print(f"- {len(df[df['is_malicious']])} malicious users")
print(f"- {len(df[df['response_code'] >= 400])} error responses")
print(f"- Average malicious requests: {len(df[df['is_malicious']])/df['IP_ADDRESS'].nunique():.1f} per IP")
print(f"- Payment gateway attack ratio: {len(df[(df['ENDPOINT'] == '/payment-gateway') & df['is_malicious']])/len(df[df['is_malicious']]):.1%}")
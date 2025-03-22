# Adaptive AI-Based API Rate Limiter

## **Overview**
This project is an **Adaptive AI-Based API Rate Limiter** that dynamically adjusts rate limits for API requests based on traffic patterns and anomalies detected using machine learning models. It ensures fair usage for Standard (`STD`) and Premium (`PRM`) users while preventing abusive traffic from overwhelming the system.

The system uses:
- **LSTM (Long Short-Term Memory)** for traffic prediction.
- **Isolation Forest** for anomaly detection.
- **Token Bucket Algorithm** for rate limiting.

The project also includes a **real-time dashboard** built with Streamlit to visualize traffic metrics, token bucket statuses, and anomaly detection results.

---

## **Features**
1. **Dynamic Rate Limiting**:
   - Adjusts token bucket parameters (capacity and refill rate) based on predicted traffic levels.
   - Ensures Premium users maintain better access during high traffic.

2. **Anomaly Detection**:
   - Detects unusual traffic patterns using Isolation Forest.
   - Applies defensive rate limiting during anomalous traffic.

3. **Traffic Prediction**:
   - Predicts future traffic using LSTM based on recent request patterns.

4. **Real-Time Visualization**:
   - Displays token bucket statuses, traffic predictions, and anomaly scores on a dashboard.

5. **User Tier Differentiation**:
   - Standard (`STD`) users face stricter rate limits compared to Premium (`PRM`) users.

---

## **Technologies Used**

### Frameworks
- **FastAPI**: Backend framework for building APIs.
- **Streamlit**: Dashboard framework for real-time visualization.

### Libraries
- **TensorFlow**: For training and deploying the LSTM model.
- **Scikit-Learn**: For training and deploying the Isolation Forest model.
- **Plotly/Dash**: For creating interactive charts in the dashboard.
- **Pandas/Numpy**: For data manipulation and preprocessing.
- **Faker**: For generating synthetic data to simulate API requests.
- **Requests**: For simulating real-time API traffic.

---

## **Folder Structure**

```
adaptive-rate-limiter/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI main application
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_model.py        # LSTM prediction model
│   │   ├── isolation_forest.py  # Anomaly detection model
│   │   └── trained/             # Directory for saved models
│   │       ├── lstm_model.keras # Trained LSTM model
│   │       ├── lstm_scaler.pkl  # LSTM data scaler
│   │       └── isolation_forest.pkl  # Trained Isolation Forest model
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py            # Configuration settings
│   │   └── token_bucket.py      # Token bucket algorithm
│   ├── api/
│   │   ├── __init__.py
│   │   ├── endpoints.py         # API endpoints
│   │   └── rate_limiter.py      # Rate limiting middleware
│   └── utils/
│       ├── __init__.py
│       ├── traffic_monitor.py   # Traffic monitoring service
│       └── visualizer.py        # Visualization dashboard utilities
├── data/
│   ├── traffic.py               # Synthetic data generation script using Faker
│   └── synthetic_api_traffic.csv  # Generated synthetic data file (CSV)
├── scripts/
│   ├── train_models.py          # Script to train AI models
│   └── simulate_requests.py     # Script to simulate real-time API requests
├── dashboard/
│   ├── app.py                   # Streamlit dashboard application
├── tests/
│   ├── __init__.py
│   ├── test_rate_limiter.py     # Tests for rate limiter functionality
│   └── test_models.py           # Tests for AI models (LSTM & Isolation Forest)
├── logs/                        # Directory for application logs (if configured)
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation (this file)
```

---

## **How the Project Works**

### 1. Data Generation (`data/traffic.py`)
- Generates synthetic API request data using Faker.
- Simulates both normal and malicious traffic patterns.
- Saves the data to `data/synthetic_api_traffic.csv`.

### 2. Model Training (`scripts/train_models.py`)
- Trains the LSTM model on synthetic request counts for traffic prediction.
- Trains the Isolation Forest model on features like `request_count`, `avg_latency`, `error_rate`, etc., for anomaly detection.
- Saves trained models to `app/models/trained/`.

### 3. FastAPI Application (`app/main.py`)
- Loads trained models and initializes the token bucket algorithm.
- Starts the FastAPI server with endpoints for testing, monitoring, and configuration:
  - `/api/test`: Test endpoint subject to rate limiting.
  - `/api/browse`: Low-priority endpoint.
  - `/api/payment-gateway`: High-priority endpoint.
  - `/api/admin/update-tier`: Update tier configurations dynamically.

### 4. Traffic Monitoring (`app/utils/traffic_monitor.py`)
- Logs incoming requests to memory.
- Aggregates request counts into time bins (e.g., per minute).
- Uses LSTM predictions to adjust token bucket parameters dynamically.
- Detects anomalies using Isolation Forest and applies defensive rate limiting.

### 5. Real-Time Visualization (`dashboard/app.py`)
- Displays metrics like active users, requests, predicted traffic, and anomalies.
- Visualizes token bucket statuses by user tier (Standard vs Premium).
- Shows anomaly scores over time in a scatter plot.

---

## **Detailed Setup and Running Instructions**

### Step 1: Clone the Repository

```bash
git clone https://github.com/Akshan03/Adaptive-Limiting.git
cd Adaptive-Limiting
```

### Step 2: Set Up Virtual Environment (Recommended)

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

Install all required libraries using `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Step 4: Generate Synthetic Data

Run the `traffic.py` script to generate synthetic API request data:

```bash
python data/traffic.py
```

This will create a file at `data/synthetic_api_traffic.csv` with:
- Simulated API requests spread over various timestamps
- Different user tiers (STD/PRM)
- Response codes and latency values
- Simulated malicious traffic

### Step 5: Train Machine Learning Models

Train both the LSTM and Isolation Forest models using the synthetic data:

```bash
python scripts/train_models.py
```

This process:
- Trains an LSTM model to predict future traffic based on historical patterns
- Trains an Isolation Forest model to detect anomalous API usage
- Saves the trained models to `app/models/trained/`
- Generates visualizations of the training results in the `plots/` directory

Alternatively, if you want to train a model with smaller sequence length (for quicker predictions):

```bash
python scripts/retrain_lstm.py
```

### Step 6: Start the API Server

Start the FastAPI application:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API server will be accessible at `http://localhost:8000` with these endpoints:
- API Documentation: `http://localhost:8000/docs`
- `/api/test`: Basic test endpoint subject to rate limiting
- `/api/browse`: Endpoint for browsing resources
- `/api/payment-gateway`: Premium endpoint (higher priority)
- `/api/admin/metrics`: Get current traffic metrics
- `/api/admin/update-tier`: Update rate limit configurations dynamically

### Step 7: Launch the Dashboard (Optional)

In a separate terminal, start the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard will be accessible at `http://localhost:8501` and displays:
- Current active users and request counts
- Token bucket status for different user tiers
- Traffic predictions and anomaly detection results
- Visual indicators for traffic levels and system status

---

## **Testing the System**

### Basic Testing

You can test the API with simple curl commands:

```bash
# Make a test request (no rate limiting initially)
curl -X GET "http://localhost:8000/api/test?user_id=user123&user_tier=STD"

# Make multiple requests to trigger rate limiting
for i in {1..100}; do curl -X GET "http://localhost:8000/api/test?user_id=user123&user_tier=STD"; done
```

### Simulating Different Traffic Patterns

The system includes a request simulator to test various traffic patterns:

```bash
# Simulate normal traffic (10 requests per second for 60 seconds)
python scripts/simulate_requests.py --rate=10 --duration=60

# Simulate traffic spike (starts at 20 req/sec and spikes to 100 req/sec)
python scripts/simulate_requests.py --pattern=spike --rate=20 --duration=120

# Simulate wave pattern (traffic oscillates between low and high)
python scripts/simulate_requests.py --pattern=wave --rate=30 --duration=180

# Simulate malicious traffic (includes attack patterns)
python scripts/simulate_requests.py --pattern=random --malicious --duration=120
```

### Testing Anomaly Detection

To test the anomaly detection system:

1. Start with normal traffic:
```bash
python scripts/simulate_requests.py --rate=20 --duration=60
```

2. Then suddenly increase to a spike:
```bash
python scripts/simulate_requests.py --pattern=spike --rate=50 --malicious --duration=120
```

3. Watch the logs or dashboard to observe:
   - Detection of the traffic spike
   - Anomaly score calculation
   - Adaptive rate limit adjustments
   - Recovery after the spike ends

### Testing Dynamic Rate Adjustments

To test how the system adapts to different traffic levels:

1. Generate low traffic and observe the rate limits:
```bash
python scripts/simulate_requests.py --rate=5 --duration=120
```

2. Generate medium traffic:
```bash
python scripts/simulate_requests.py --rate=40 --duration=120
```

3. Generate high traffic:
```bash
python scripts/simulate_requests.py --rate=80 --duration=120
```

For each level, observe in the logs or dashboard:
- How token bucket capacities are adjusted
- Different treatment of STD vs PRM users
- LSTM predictions for future traffic

### Running Automated Tests

Execute the automated test suite to validate core functionality:

```bash
pytest tests/
```

These tests verify:
- Token bucket algorithm correctness
- Rate limiting middleware behavior
- Model prediction accuracy
- System response to anomalies

---

## **Monitoring and Troubleshooting**

### Logs

Application logs are stored in the `logs/` directory, providing detailed information about:
- Request processing
- Rate limit decisions
- Traffic predictions
- Anomaly detections
- System adaptations

View recent logs with:
```bash
tail -f logs/app.log
```

### Performance Metrics

Monitor system performance via the admin metrics endpoint:
```bash
curl -X GET "http://localhost:8000/api/admin/metrics"
```

The dashboard also provides visual indicators of system performance and rate limit effectiveness.

---

## **Configuration**

Key configuration settings are in `app/core/config.py`, including:

- Token bucket parameters for different user tiers
- Traffic thresholds for low/medium/high classification
- Rate limit adjustment factors
- Monitoring intervals
- Model paths

For quick adjustments without modifying code, use the admin endpoint:
```bash
curl -X POST "http://localhost:8000/api/admin/update-tier" \
  -H "Content-Type: application/json" \
  -d '{"tier": "STD", "capacity": 60, "refill_tokens": 12, "refill_duration": 60}'
```

---

## **Future Improvements**
1. Integrate more advanced anomaly detection techniques like DBSCAN or Autoencoders.
2. Add support for live retraining of models based on real-time feedback loops.
3. Improve visualization with interactive charts in Streamlit or Dash.
4. Implement distributed rate limiting using Redis for multi-server deployments.
5. Add user authentication and role-based access for admin controls.

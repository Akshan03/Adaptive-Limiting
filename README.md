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

## **How to Run the Project**

### Step 1: Install Dependencies

Install all required libraries using `requirements.txt`:

```bash
pip install -r requirements.txt
```

Ensure Python version is `3.11.9`.

---

### Step 2: Generate Synthetic Data

Run the `traffic.py` script to generate synthetic API request data:

```bash
python data/traffic.py
```

This creates a CSV file at `data/synthetic_api_traffic.csv`.

---

### Step 3: Train AI Models

Train the LSTM and Isolation Forest models using the synthetic data:

```bash
python scripts/train_models.py
```

This saves the trained models to `app/models/trained/`.

---

### Step 4: Start FastAPI Server

Start the FastAPI application:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The server will be accessible at `http://127.0.0.1:8000`.

---

### Step 5: Simulate Real-Time Traffic

Generate real-time API requests using `simulate_requests.py`:

```bash
python scripts/simulate_requests.py --rate 50 --pattern spike --duration 300
```

You can also use tools like Postman or Apache JMeter for load testing.

---

### Step 6: Access Dashboard

Open your browser and navigate to:

```
http://127.0.0.1:8000/dashboard
```

The dashboard will display:
1. Active users and request counts.
2. Token bucket statuses by user tier (STD vs PRM).
3. Predicted vs actual traffic trends.
4. Anomaly scores over time.

---

### Step 7: Test Endpoints

Use Swagger UI at `http://127.0.0.1:8000/docs` to test individual endpoints:
1. `/api/test`: Simulates a test request subject to rate limiting.
2. `/api/admin/update-tier`: Updates token bucket configurations dynamically.

---

## **Key Features of Dashboard**
1. Real-time visualization of token bucket statuses (available vs used tokens).
2. Graph of predicted vs actual traffic trends using LSTM model outputs.
3. Scatter plot of anomaly scores over time from Isolation Forest detections.

---

## **Testing**
Run unit tests for rate limiter functionality and AI models:

```bash
pytest tests/
```

This ensures that:
1. Token bucket algorithm behaves correctly under different scenarios.
2. LSTM predictions align with expected trends in synthetic data.
3. Isolation Forest detects anomalies accurately.

---

## **Future Improvements**
1. Integrate more advanced anomaly detection techniques like DBSCAN or Autoencoders.
2. Add support for live retraining of models based on real-time feedback loops.
3. Improve visualization with interactive charts in Streamlit or Dash. 
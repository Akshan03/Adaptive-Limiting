import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.core.config import settings
from app.api.endpoints import router as api_router, set_traffic_monitor
from app.api.rate_limiter import RateLimitMiddleware
from app.models.lstm_model import TrafficPredictor
from app.models.isolation_forest import AnomalyDetector
from app.utils.traffic_monitor import TrafficMonitor
from app.utils.visualizer import DashboardManager

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),  # Save logs to a file
        logging.StreamHandler()  # Also show logs in the terminal
    ]
)

logger = logging.getLogger(__name__)
logger.info("Starting Adaptive Rate Limiter API")

# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
app.add_middleware(RateLimitMiddleware)

# Load the trained models
lstm_model = TrafficPredictor(
    model_path=settings.LSTM_MODEL_PATH, 
    scaler_path=settings.LSTM_SCALER_PATH
)

anomaly_model = AnomalyDetector(
    model_path=settings.ISOLATION_FOREST_PATH
)

# Initialize traffic monitor with the models
traffic_monitor = TrafficMonitor(
    lstm_model=lstm_model,
    anomaly_model=anomaly_model,
    monitoring_interval=settings.MONITORING_INTERVAL,
    traffic_window=settings.TRAFFIC_WINDOW
)

# Create dashboard
dashboard = DashboardManager(app, traffic_monitor)

# Set traffic monitor in endpoints module
set_traffic_monitor(traffic_monitor)

# Start traffic monitoring on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Starting API rate limiter service")
    traffic_monitor.start()
    logger.info(f"Loaded models from {settings.MODEL_DIR}")
    logger.info(f"Rate limiter configured with buckets: {settings.TOKEN_BUCKET_CONFIGS}")

# Stop traffic monitoring on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down API rate limiter service")
    traffic_monitor.stop()

# Include API routes
app.include_router(api_router)

# Mount static files for dashboard
from fastapi.staticfiles import StaticFiles
os.makedirs("app/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app", 
        host=settings.HOST, 
        port=settings.PORT, 
        reload=settings.DEBUG
    )

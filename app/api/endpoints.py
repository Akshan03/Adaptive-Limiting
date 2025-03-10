# app/api/endpoints.py
from fastapi import APIRouter, Request, Depends, HTTPException, Header
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import time
from pydantic import BaseModel
from app.core.token_bucket import bucket_manager
from app.utils.traffic_monitor import TrafficMonitor
import logging

# Create API router
router = APIRouter()

# Access to the traffic monitor (will be set in main.py)
traffic_monitor = None

# Configure logger
logger = logging.getLogger(__name__)

def set_traffic_monitor(monitor: TrafficMonitor):
    global traffic_monitor
    traffic_monitor = monitor

# Models for API responses
class RateLimitStatus(BaseModel):
    user_id: str
    user_tier: str
    available_tokens: float
    capacity: int
    refill_rate: str
    
class APIResponse(BaseModel):
    message: str
    data: Optional[Dict[str, Any]] = None
    
class ErrorResponse(BaseModel):
    detail: str

# Test endpoints that will be rate-limited
@router.get("/api/test", response_model=APIResponse)
async def test_endpoint(
    request: Request,
    user_id: str = Header(default="anonymous", description="Unique user ID"),
    user_tier: str = Header(default="STD", description="User tier (STD or PRM)")
):
    """Simple test endpoint that's subject to rate limiting."""
    
    # Add request to traffic logs with consistent timestamp format
    if traffic_monitor:
        traffic_monitor.log_request({
            "timestamp": time.time(),  # Use Unix timestamp consistently
            "user_id": user_id,
            "user_tier": user_tier,
            "endpoint": "/api/test",
            "method": "GET",
            "status_code": 200,
            "ip_address": request.client.host,
            "user_agent": request.headers.get("User-Agent", "")
        })
        
        # Log for debugging
        logger.info(f"Request logged: User {user_id} ({user_tier}) - Traffic logs size: {len(traffic_monitor.traffic_logs)}")
    
    # Get bucket status for this user
    bucket = bucket_manager.get_bucket(user_id, user_tier)
    status = bucket.get_status()
    
    return {
        "message": "Test endpoint accessed successfully",
        "data": {
            "user_id": user_id,
            "user_tier": user_tier,
            "available_tokens": status["available_tokens"],
            "capacity": status["capacity"],
            "refill_rate": status["refill_rate"]
        }
    }

@router.get("/api/browse", response_model=APIResponse)
async def browse_endpoint(request: Request):
    """Browse endpoint that simulates a low-priority endpoint."""
    # Extract user information from headers
    user_id = request.headers.get("User-ID", "anonymous")
    user_tier = request.headers.get("User-Tier", "STD")
    
    # Add request to traffic logs
    if traffic_monitor:
        traffic_monitor.log_request({
            "timestamp": time.time(),
            "user_id": user_id,
            "user_tier": user_tier,
            "endpoint": "/api/browse",
            "method": "GET",
            "status_code": 200,
            "ip_address": request.client.host,
            "user_agent": request.headers.get("User-Agent", "")
        })
    
    # Simulate some processing time
    time.sleep(0.05)  # 50ms delay
    
    return {
        "message": "Browse endpoint accessed successfully",
        "data": {
            "items": [
                {"id": 1, "name": "Item 1"},
                {"id": 2, "name": "Item 2"},
                {"id": 3, "name": "Item 3"}
            ]
        }
    }

@router.post("/api/payment-gateway", response_model=APIResponse)
async def payment_gateway_endpoint(request: Request):
    """Payment gateway endpoint that simulates a high-priority endpoint."""
    # Extract user information from headers
    user_id = request.headers.get("User-ID", "anonymous")
    user_tier = request.headers.get("User-Tier", "STD")
    
    # Add request to traffic logs
    if traffic_monitor:
        traffic_monitor.log_request({
            "timestamp": time.time(),
            "user_id": user_id,
            "user_tier": user_tier,
            "endpoint": "/api/payment-gateway",
            "method": "POST",
            "status_code": 200,
            "ip_address": request.client.host,
            "user_agent": request.headers.get("User-Agent", "")
        })
    
    # Simulate some processing time
    time.sleep(0.2)  # 200ms delay for more complex operation
    
    return {
        "message": "Payment processed successfully",
        "data": {
            "transaction_id": "txn_12345",
            "status": "completed",
            "timestamp": time.time()
        }
    }

# Status and monitoring endpoints
@router.get("/health", response_model=APIResponse)
async def health_check():
    """Health check endpoint that's not subject to rate limiting."""
    return {
        "message": "Service is healthy",
        "data": {
            "timestamp": time.time(),
            "status": "UP"
        }
    }

@router.get("/api/rate-limit-status", response_model=Dict[str, Any])
async def rate_limit_status(request: Request):
    """Get the current rate limit status for the requesting user."""
    # Extract user information from headers
    user_id = request.headers.get("User-ID", "anonymous")
    user_tier = request.headers.get("User-Tier", "STD")
    
    # Get bucket for this user
    bucket = bucket_manager.get_bucket(user_id, user_tier)
    status = bucket.get_status()
    
    return {
        "user_id": user_id,
        "user_tier": user_tier,
        "status": status,
        "tier_config": bucket_manager.tier_configs.get(user_tier, {})
    }

# Admin endpoints for rate limit configuration
@router.get("/api/admin/tier-stats", response_model=Dict[str, Any])
async def tier_stats():
    """Get aggregated statistics for all user tiers."""
    return bucket_manager.get_tier_stats()

@router.post("/api/admin/update-tier")
async def update_tier(tier: str, capacity: Optional[int] = None, 
                     refill_tokens: Optional[float] = None, 
                     refill_duration: Optional[int] = None):
    """Update the configuration for a user tier."""
    bucket_manager.update_tier_config(
        tier=tier,
        capacity=capacity,
        refill_tokens=refill_tokens,
        refill_duration=refill_duration
    )
    
    return {
        "message": f"Updated configuration for tier {tier}",
        "config": bucket_manager.tier_configs.get(tier, {})
    }

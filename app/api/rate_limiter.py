# app/api/rate_limiter.py
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import time
from app.core.token_bucket import bucket_manager

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to apply rate limiting to incoming requests."""
    
    async def dispatch(self, request: Request, call_next):
        """
        Process a request through rate limiting middleware.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            The response from the next handler, or a rate limit exceeded response
        """
        start_time = time.time()
        
        # Skip rate limiting for specific paths (like health checks or dashboard)
        path = request.url.path
        if path == "/health" or path.startswith("/dashboard") or path.startswith("/api/admin"):
            return await call_next(request)
        
        # Extract user identification and tier from request headers
        user_id = request.headers.get("User-ID", "anonymous")
        user_tier = request.headers.get("User-Tier", "STD")
        
        # Get the user's token bucket
        bucket = bucket_manager.get_bucket(user_id, user_tier)
        
        # Try to consume a token
        if not bucket.consume():
            # If rate limit exceeded, return 429 response
            logger.warning(f"Rate limit exceeded for user {user_id} ({user_tier}) on {path}")
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please try again later."}
            )
        
        # Process the request if rate limit not exceeded
        response = await call_next(request)
        
        # Log request processing time
        process_time = time.time() - start_time
        logger.info(
            f"Request: {request.method} {path} - "
            f"User: {user_id} ({user_tier}) - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.4f}s"
        )
        
        return response

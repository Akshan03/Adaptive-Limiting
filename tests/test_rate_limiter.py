# tests/test_rate_limiter.py
import pytest
import time
import asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import app modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.token_bucket import TokenBucket, BucketManager
from app.api.rate_limiter import RateLimitMiddleware
from app.main import app as main_app

# Configure test client
client = TestClient(main_app)

class TestTokenBucket:
    """Test the TokenBucket implementation"""
    
    def test_init(self):
        """Test bucket initialization"""
        bucket = TokenBucket(capacity=10, refill_tokens=2, refill_duration=60)
        
        assert bucket.capacity == 10
        assert bucket.tokens == 10  # Start with full bucket
        assert bucket.refill_tokens == 2
        assert bucket.refill_duration == 60
    
    def test_consume(self):
        """Test token consumption"""
        bucket = TokenBucket(capacity=10, refill_tokens=2, refill_duration=60)
        
        # Should succeed - enough tokens
        assert bucket.consume(5) == True
        assert bucket.tokens == 5
        
        # Should succeed - exactly enough tokens
        assert bucket.consume(5) == True
        assert bucket.tokens == 0
        
        # Should fail - not enough tokens
        assert bucket.consume(1) == False
        assert bucket.tokens == 0
    
    def test_refill(self):
        """Test token refill over time"""
        bucket = TokenBucket(capacity=10, refill_tokens=5, refill_duration=1)
        
        # Consume some tokens
        assert bucket.consume(6) == True
        assert bucket.tokens == 4
        
        # Wait for refill
        time.sleep(1.2)  # Wait a bit more than 1 second
        
        # Refill should happen automatically on next consume
        assert bucket.consume(6) == True  # 4 + 5 = 9, then consume 6
        assert bucket.tokens == 3
    
    def test_partial_refill(self):
        """Test partial token refill"""
        bucket = TokenBucket(capacity=100, refill_tokens=10, refill_duration=1)
        
        # Consume all tokens
        assert bucket.consume(100) == True
        assert bucket.tokens == 0
        
        # Wait for half refill duration
        time.sleep(0.5)  # Half second should refill 5 tokens
        
        # Should have 5 tokens (half of refill_tokens)
        assert bucket.consume(3) == True
        assert round(bucket.tokens) == 2  # May be slightly off due to timing
        
        # Not enough for 3 more
        assert bucket.consume(3) == False
    
    def test_update_config(self):
        """Test dynamic configuration update"""
        bucket = TokenBucket(capacity=10, refill_tokens=2, refill_duration=60)
        
        # Update capacity
        bucket.update_config(capacity=20)
        assert bucket.capacity == 20
        assert bucket.tokens == 20  # Should refill to new capacity
        
        # Consume some tokens
        assert bucket.consume(15) == True
        assert bucket.tokens == 5
        
        # Update refill rate
        bucket.update_config(refill_tokens=10, refill_duration=30)
        assert bucket.refill_tokens == 10
        assert bucket.refill_duration == 30
        
        # Tokens should remain the same after config update
        assert bucket.tokens == 5

class TestBucketManager:
    """Test the BucketManager implementation"""
    
    def test_get_bucket(self):
        """Test getting buckets for different users"""
        manager = BucketManager()
        
        # Get buckets for two different users
        bucket1 = manager.get_bucket("user1", "STD")
        bucket2 = manager.get_bucket("user2", "STD")
        
        # Should be different bucket instances
        assert bucket1 is not bucket2
        
        # Get the same user's bucket again
        bucket1_again = manager.get_bucket("user1", "STD")
        
        # Should be the same instance
        assert bucket1 is bucket1_again
    
    def test_user_tiers(self):
        """Test different user tiers have different configurations"""
        manager = BucketManager()
        
        # Get buckets for different tiers
        std_bucket = manager.get_bucket("user1", "STD")
        prm_bucket = manager.get_bucket("user2", "PRM")
        
        # Premium should have higher capacity
        assert prm_bucket.capacity > std_bucket.capacity
        
        # Premium should have higher refill rate
        assert prm_bucket.refill_tokens / prm_bucket.refill_duration > std_bucket.refill_tokens / std_bucket.refill_duration
    
    def test_update_tier_config(self):
        """Test updating configuration for all users of a tier"""
        manager = BucketManager()
        
        # Create multiple buckets for the same tier
        bucket1 = manager.get_bucket("user1", "STD")
        bucket2 = manager.get_bucket("user2", "STD")
        
        # Initial capacities should be the same
        assert bucket1.capacity == bucket2.capacity
        
        # Update tier configuration
        manager.update_tier_config("STD", capacity=200)
        
        # Both buckets should be updated
        assert bucket1.capacity == 200
        assert bucket2.capacity == 200
        
        # Premium tier should remain unchanged
        prm_bucket = manager.get_bucket("user3", "PRM")
        assert prm_bucket.capacity != 200

@pytest.mark.asyncio
async def test_rate_limit_middleware():
    """Test the rate limiting middleware"""
    # Create a simple FastAPI app with rate limiting
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware)
    
    @app.get("/test")
    async def test_endpoint():
        return {"message": "success"}
    
    # Use TestClient to make requests
    client = TestClient(app)
    
    # Create a bucket with small capacity for testing
    with patch('app.api.rate_limiter.bucket_manager.get_bucket') as mock_get_bucket:
        # Mock bucket with capacity of 3
        mock_bucket = MagicMock()
        mock_bucket.consume.side_effect = [True, True, True, False, False]  # First 3 succeed, then fail
        mock_get_bucket.return_value = mock_bucket
        
        # Make requests
        for i in range(3):
            response = client.get("/test", headers={"User-ID": "test", "User-Tier": "STD"})
            assert response.status_code == 200
        
        # This should be rate limited
        response = client.get("/test", headers={"User-ID": "test", "User-Tier": "STD"})
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.text

def test_different_user_rates():
    """Test that different users have independent rate limits"""
    # Create a test client for the main app
    
    # Make multiple requests from different users
    responses_user1 = []
    responses_user2 = []
    
    # Make requests alternating between users
    for i in range(10):
        resp1 = client.get("/api/test", headers={"User-ID": "test_user1", "User-Tier": "STD"})
        resp2 = client.get("/api/test", headers={"User-ID": "test_user2", "User-Tier": "STD"})
        
        responses_user1.append(resp1.status_code)
        responses_user2.append(resp2.status_code)
    
    # Both users should have some successful requests
    assert 200 in responses_user1
    assert 200 in responses_user2
    
    # The rate limits should not affect each other (one might be rate limited before the other)
    assert responses_user1 != responses_user2

def test_premium_vs_standard():
    """Test premium users get higher rate limits than standard users"""
    # Make requests with premium and standard users
    responses_std = []
    responses_prm = []
    
    # Make a burst of requests for both user types
    for i in range(20):
        resp_std = client.get("/api/test", headers={"User-ID": "test_std", "User-Tier": "STD"})
        resp_prm = client.get("/api/test", headers={"User-ID": "test_prm", "User-Tier": "PRM"})
        
        responses_std.append(resp_std.status_code)
        responses_prm.append(resp_prm.status_code)
    
    # Count the number of successful requests
    std_success = responses_std.count(200)
    prm_success = responses_prm.count(200)
    
    # Premium users should have more successful requests
    assert prm_success > std_success

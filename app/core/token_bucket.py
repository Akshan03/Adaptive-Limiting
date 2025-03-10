# app/core/token_bucket.py
import time
import threading
import logging
from typing import Dict, Any, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

class TokenBucket:
    """
    Implements the token bucket algorithm for API rate limiting.
    
    The token bucket has a capacity and refills at a specific rate over time.
    When a request comes in, it consumes tokens from the bucket.
    If there are not enough tokens, the request is rejected.
    """
    
    def __init__(self, capacity: int, refill_tokens: float, refill_duration: int):
        """
        Initialize a token bucket.
        
        Args:
            capacity: Maximum number of tokens the bucket can hold
            refill_tokens: Number of tokens to add during each refill cycle
            refill_duration: Duration of the refill cycle in seconds
        """
        self.capacity = capacity
        self.tokens = capacity  # Start with a full bucket
        self.refill_tokens = refill_tokens
        self.refill_duration = refill_duration  # in seconds
        self.last_refill_timestamp = time.time()
        self.lock = threading.RLock()  # For thread safety
        self.total_consumed = 0  # Track total tokens consumed
        
    def refill(self) -> None:
        """Refill tokens based on elapsed time."""
        with self.lock:
            now = time.time()
            time_passed = now - self.last_refill_timestamp
            
            # Calculate how many tokens to add based on time passed
            refill_amount = (time_passed / self.refill_duration) * self.refill_tokens
            
            if refill_amount > 0:
                self.tokens = min(self.capacity, self.tokens + refill_amount)
                self.last_refill_timestamp = now
                logger.debug(f"Refilled {refill_amount:.2f} tokens. Current: {self.tokens:.2f}/{self.capacity}")
        
    def consume(self, tokens: int = 1) -> bool:
        """
        Attempt to consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            bool: True if tokens were consumed, False if not enough tokens
        """
        with self.lock:
            self.refill()  # Refill tokens based on elapsed time
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                self.total_consumed += tokens
                logger.debug(f"Consumed {tokens} tokens. Remaining: {self.tokens:.2f}/{self.capacity}")
                return True
            
            logger.debug(f"Rate limit exceeded. Required: {tokens}, Available: {self.tokens:.2f}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the token bucket."""
        with self.lock:
            self.refill()  # Make sure tokens are up to date
            return {
                "capacity": self.capacity,
                "available_tokens": self.tokens,
                "used_tokens": self.capacity - self.tokens,
                "refill_rate": f"{self.refill_tokens}/{self.refill_duration}s",
                "total_consumed": self.total_consumed
            }
    
    def update_config(self, capacity: Optional[int] = None, 
                    refill_tokens: Optional[float] = None, 
                    refill_duration: Optional[int] = None) -> None:
        """
        Update the configuration of the token bucket.
        
        Args:
            capacity: New maximum capacity
            refill_tokens: New refill tokens amount
            refill_duration: New refill duration in seconds
        """
        with self.lock:
            self.refill()  # Make sure tokens are up to date before changing config
            
            if capacity is not None:
                # Preserve the same fill percentage when changing capacity
                fill_percentage = self.tokens / self.capacity if self.capacity > 0 else 1.0
                self.capacity = capacity
                self.tokens = min(self.capacity, self.capacity * fill_percentage)
                
            if refill_tokens is not None:
                self.refill_tokens = refill_tokens
                
            if refill_duration is not None:
                self.refill_duration = refill_duration
            
            logger.info(f"Updated token bucket config: capacity={self.capacity}, "
                      f"refill_rate={self.refill_tokens}/{self.refill_duration}s")


class BucketManager:
    """Manages token buckets for different users and user tiers."""
    
    def __init__(self):
        """Initialize the bucket manager with default configurations."""
        self.buckets = {}  # Dict to store user buckets
        self.tier_configs = settings.TOKEN_BUCKET_CONFIGS.copy()
        self.lock = threading.RLock()
        
    def get_bucket(self, user_id: str, user_tier: str = "STD") -> TokenBucket:
        """
        Get or create a token bucket for a specific user.
        
        Args:
            user_id: Unique identifier for the user
            user_tier: User tier (STD or PRM)
            
        Returns:
            TokenBucket: The user's token bucket
        """
        with self.lock:
            bucket_key = f"{user_id}:{user_tier}"
            
            if bucket_key not in self.buckets:
                # Get config for this tier
                config = self.tier_configs.get(user_tier, self.tier_configs["STD"])
                
                # Create new bucket with this config
                self.buckets[bucket_key] = TokenBucket(
                    capacity=config["capacity"],
                    refill_tokens=config["refill_tokens"],
                    refill_duration=config["refill_duration"]
                )
                
            return self.buckets[bucket_key]
    
    def update_tier_config(self, tier: str, capacity: Optional[int] = None, 
                         refill_tokens: Optional[float] = None, 
                         refill_duration: Optional[int] = None) -> None:
        """
        Update the configuration for a user tier.
        
        Args:
            tier: User tier to update (STD or PRM)
            capacity: New capacity value
            refill_tokens: New refill tokens value
            refill_duration: New refill duration value
        """
        with self.lock:
            if tier not in self.tier_configs:
                self.tier_configs[tier] = self.tier_configs["STD"].copy()
                
            if capacity is not None:
                self.tier_configs[tier]["capacity"] = capacity
                
            if refill_tokens is not None:
                self.tier_configs[tier]["refill_tokens"] = refill_tokens
                
            if refill_duration is not None:
                self.tier_configs[tier]["refill_duration"] = refill_duration
            
            # Update all existing buckets for this tier
            for bucket_key, bucket in self.buckets.items():
                if bucket_key.endswith(f":{tier}"):
                    bucket.update_config(
                        capacity=self.tier_configs[tier]["capacity"],
                        refill_tokens=self.tier_configs[tier]["refill_tokens"],
                        refill_duration=self.tier_configs[tier]["refill_duration"]
                    )
            
            logger.info(f"Updated tier config for {tier}: {self.tier_configs[tier]}")
    
    def get_tier_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get aggregated statistics for each user tier.
        
        Returns:
            Dict mapping each tier to its aggregated stats
        """
        with self.lock:
            # Initialize stats for each tier
            tier_stats = {
                tier: {
                    "user_count": 0,
                    "total_capacity": 0,
                    "total_available": 0,
                    "total_used": 0,
                    "total_consumed": 0,
                    "config": config.copy()
                } 
                for tier, config in self.tier_configs.items()
            }
            
            # Aggregate stats from all buckets
            for bucket_key, bucket in self.buckets.items():
                tier = bucket_key.split(":")[-1]
                if tier in tier_stats:
                    status = bucket.get_status()
                    tier_stats[tier]["user_count"] += 1
                    tier_stats[tier]["total_capacity"] += status["capacity"]
                    tier_stats[tier]["total_available"] += status["available_tokens"]
                    tier_stats[tier]["total_used"] += status["used_tokens"]
                    tier_stats[tier]["total_consumed"] += status["total_consumed"]
            
            # Calculate averages for non-empty tiers
            for tier, stats in tier_stats.items():
                if stats["user_count"] > 0:
                    stats["avg_capacity"] = stats["total_capacity"] / stats["user_count"]
                    stats["avg_available"] = stats["total_available"] / stats["user_count"]
                    stats["avg_used"] = stats["total_used"] / stats["user_count"]
                    stats["avg_consumed"] = stats["total_consumed"] / stats["user_count"]
                else:
                    stats["avg_capacity"] = stats["config"]["capacity"]
                    stats["avg_available"] = stats["config"]["capacity"]
                    stats["avg_used"] = 0
                    stats["avg_consumed"] = 0
            
            return tier_stats


# Create a global instance of the bucket manager
bucket_manager = BucketManager()

"""Production-grade middleware for FastAPI claims processing API."""

import time
import uuid
import asyncio
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import gzip
import ipaddress
from dataclasses import dataclass, field

import structlog
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import redis.asyncio as redis

from src.core.config import settings
from src.monitoring.metrics.comprehensive_metrics import metrics_collector

logger = structlog.get_logger(__name__)


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration."""
    requests_per_minute: int
    requests_per_hour: int
    burst_allowance: int = 0
    window_size: int = 60  # seconds


@dataclass
class ClientStats:
    """Client request statistics."""
    requests: deque = field(default_factory=lambda: deque(maxlen=1000))
    last_request: Optional[datetime] = None
    total_requests: int = 0
    blocked_count: int = 0


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers for HIPAA compliance and general security."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self'; "
                "connect-src 'self'; "
                "frame-ancestors 'none';"
            ),
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": (
                "geolocation=(), microphone=(), camera=(), "
                "payment=(), usb=(), magnetometer=(), gyroscope=()"
            ),
            "Cache-Control": "no-store, no-cache, must-revalidate, private",
            "Pragma": "no-cache",
            "Expires": "0"
        }

    async def dispatch(self, request: Request, call_next):
        """Add security headers to all responses."""
        response = await call_next(request)
        
        # Add security headers
        for header_name, header_value in self.security_headers.items():
            response.headers[header_name] = header_value
        
        # Add request ID to response headers
        if hasattr(request.state, 'request_id'):
            response.headers["X-Request-ID"] = str(request.state.request_id)
        
        # Remove server identification
        response.headers.pop("server", None)
        
        return response


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Advanced rate limiting middleware with multiple algorithms."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.redis_client = None
        self.local_cache: Dict[str, ClientStats] = {}
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
        
        # Rate limit rules by endpoint pattern
        self.rate_limits = {
            "/auth/login": RateLimitRule(
                requests_per_minute=5,
                requests_per_hour=20,
                burst_allowance=2
            ),
            "/api/v1/claims": RateLimitRule(
                requests_per_minute=60,
                requests_per_hour=1000,
                burst_allowance=10
            ),
            "/api/v1/claims/batch": RateLimitRule(
                requests_per_minute=10,
                requests_per_hour=100,
                burst_allowance=2
            ),
            "default": RateLimitRule(
                requests_per_minute=30,
                requests_per_hour=500,
                burst_allowance=5
            )
        }
        
        # Initialize Redis connection
        asyncio.create_task(self._init_redis())

    async def _init_redis(self):
        """Initialize Redis connection for distributed rate limiting."""
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=getattr(settings, 'REDIS_PASSWORD', None),
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis connection established for rate limiting")
        except Exception as e:
            logger.warning("Redis unavailable, using local rate limiting", error=str(e))
            self.redis_client = None

    def _get_client_identifier(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Use X-Forwarded-For if behind proxy, otherwise use direct IP
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host
        
        # For authenticated requests, use user ID
        auth_header = request.headers.get("Authorization")
        if auth_header:
            # Would extract user ID from JWT token here
            # For now, use IP + auth indicator
            return f"auth:{client_ip}"
        
        return f"anon:{client_ip}"

    def _get_rate_limit_rule(self, request: Request) -> RateLimitRule:
        """Get rate limit rule for the request."""
        path = request.url.path
        
        # Find matching rule
        for pattern, rule in self.rate_limits.items():
            if pattern == "default":
                continue
            if path.startswith(pattern):
                return rule
        
        return self.rate_limits["default"]

    async def _check_rate_limit_redis(self, client_id: str, rule: RateLimitRule) -> bool:
        """Check rate limit using Redis sliding window."""
        if not self.redis_client:
            return await self._check_rate_limit_local(client_id, rule)
        
        try:
            current_time = time.time()
            minute_window = f"{client_id}:minute:{int(current_time // 60)}"
            hour_window = f"{client_id}:hour:{int(current_time // 3600)}"
            
            # Use pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            
            # Check and increment minute counter
            pipe.incr(minute_window)
            pipe.expire(minute_window, 120)  # 2 minute expiry
            
            # Check and increment hour counter
            pipe.incr(hour_window)
            pipe.expire(hour_window, 7200)  # 2 hour expiry
            
            results = await pipe.execute()
            
            minute_count = results[0]
            hour_count = results[2]
            
            # Check limits
            if minute_count > rule.requests_per_minute + rule.burst_allowance:
                return False
            if hour_count > rule.requests_per_hour:
                return False
            
            return True
            
        except Exception as e:
            logger.warning("Redis rate limit check failed, falling back to local", error=str(e))
            return await self._check_rate_limit_local(client_id, rule)

    async def _check_rate_limit_local(self, client_id: str, rule: RateLimitRule) -> bool:
        """Check rate limit using local memory."""
        current_time = datetime.utcnow()
        
        # Initialize client stats if not exists
        if client_id not in self.local_cache:
            self.local_cache[client_id] = ClientStats()
        
        client_stats = self.local_cache[client_id]
        
        # Clean old requests (sliding window)
        cutoff_time = current_time - timedelta(minutes=1)
        while client_stats.requests and client_stats.requests[0] < cutoff_time:
            client_stats.requests.popleft()
        
        # Check minute limit
        if len(client_stats.requests) >= rule.requests_per_minute + rule.burst_allowance:
            client_stats.blocked_count += 1
            return False
        
        # Add current request
        client_stats.requests.append(current_time)
        client_stats.last_request = current_time
        client_stats.total_requests += 1
        
        return True

    def _cleanup_local_cache(self):
        """Clean up old entries from local cache."""
        if time.time() - self.last_cleanup < self.cleanup_interval:
            return
        
        current_time = datetime.utcnow()
        cutoff_time = current_time - timedelta(hours=2)
        
        # Remove old client entries
        expired_clients = [
            client_id for client_id, stats in self.local_cache.items()
            if stats.last_request and stats.last_request < cutoff_time
        ]
        
        for client_id in expired_clients:
            del self.local_cache[client_id]
        
        self.last_cleanup = time.time()
        logger.debug("Rate limit cache cleanup completed", removed_clients=len(expired_clients))

    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting to requests."""
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/ready", "/live"]:
            return await call_next(request)
        
        client_id = self._get_client_identifier(request)
        rule = self._get_rate_limit_rule(request)
        
        # Check rate limit
        allowed = await self._check_rate_limit_redis(client_id, rule)
        
        if not allowed:
            # Record rate limit violation
            metrics_collector.http_requests_total.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code="429"
            ).inc()
            
            logger.warning("Rate limit exceeded",
                         client_id=client_id,
                         path=request.url.path,
                         method=request.method)
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": 60,
                    "message": "Too many requests. Please try again later."
                },
                headers={"Retry-After": "60"}
            )
        
        # Clean up local cache periodically
        self._cleanup_local_cache()
        
        return await call_next(request)


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """Track requests for monitoring and debugging."""
    
    async def dispatch(self, request: Request, call_next):
        """Track request lifecycle."""
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Record request start time
        start_time = time.perf_counter()
        
        # Log request start
        logger.info("Request started",
                   request_id=request_id,
                   method=request.method,
                   path=request.url.path,
                   client_ip=request.client.host,
                   user_agent=request.headers.get("User-Agent", ""))
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.perf_counter() - start_time
            
            # Record metrics
            metrics_collector.http_requests_total.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=str(response.status_code)
            ).inc()
            
            metrics_collector.http_request_duration.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(process_time)
            
            # Log request completion
            logger.info("Request completed",
                       request_id=request_id,
                       status_code=response.status_code,
                       process_time=f"{process_time:.3f}s")
            
            return response
            
        except Exception as e:
            # Calculate processing time for failed request
            process_time = time.perf_counter() - start_time
            
            # Record error metrics
            metrics_collector.http_requests_total.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code="500"
            ).inc()
            
            # Log request error
            logger.error("Request failed",
                        request_id=request_id,
                        error=str(e),
                        process_time=f"{process_time:.3f}s")
            
            raise


class HIPAAComplianceMiddleware(BaseHTTPMiddleware):
    """Ensure HIPAA compliance for all requests."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.phi_endpoints = {
            "/api/v1/claims/",
            "/api/v1/patients/",
            "/api/v1/failed-claims"
        }

    def _is_phi_endpoint(self, path: str) -> bool:
        """Check if endpoint handles PHI data."""
        return any(path.startswith(phi_path) for phi_path in self.phi_endpoints)

    async def dispatch(self, request: Request, call_next):
        """Apply HIPAA compliance checks."""
        # Check if this is a PHI endpoint
        if self._is_phi_endpoint(request.url.path):
            # Ensure HTTPS for PHI endpoints
            if request.url.scheme != "https" and not settings.DEBUG:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "error": "HTTPS required for PHI data access",
                        "message": "All PHI endpoints must use HTTPS encryption"
                    }
                )
            
            # Log PHI endpoint access
            logger.info("PHI endpoint accessed",
                       path=request.url.path,
                       method=request.method,
                       client_ip=request.client.host,
                       request_id=getattr(request.state, 'request_id', 'unknown'))
        
        return await call_next(request)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Centralized error handling with proper logging."""
    
    async def dispatch(self, request: Request, call_next):
        """Handle errors consistently."""
        try:
            return await call_next(request)
            
        except HTTPException as e:
            # Log HTTP exceptions
            logger.warning("HTTP exception",
                          status_code=e.status_code,
                          detail=e.detail,
                          path=request.url.path,
                          request_id=getattr(request.state, 'request_id', 'unknown'))
            
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": e.detail,
                    "status_code": e.status_code,
                    "request_id": getattr(request.state, 'request_id', 'unknown'),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except Exception as e:
            # Log unexpected errors
            logger.exception("Unhandled exception",
                           error=str(e),
                           path=request.url.path,
                           request_id=getattr(request.state, 'request_id', 'unknown'))
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Internal server error",
                    "message": "An unexpected error occurred",
                    "request_id": getattr(request.state, 'request_id', 'unknown'),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
"""
Logging middleware for comprehensive request/response logging.
"""

import time
import json
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from src.core.logging import get_logger, log_error

logger = get_logger(__name__, "api", structured=True)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive request/response logging."""
    
    def __init__(self, app):
        super().__init__(app)
        self.sensitive_headers = {
            'authorization', 'cookie', 'x-api-key', 'x-auth-token'
        }
        self.sensitive_params = {
            'password', 'token', 'secret', 'key', 'auth'
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response details."""
        start_time = time.time()
        request_id = self._generate_request_id()
        
        # Extract request details
        request_data = await self._extract_request_data(request, request_id)
        
        # Log incoming request
        logger.info(
            "Request started",
            request_id=request_id,
            **request_data
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Extract response details
            response_data = self._extract_response_data(response, process_time)
            
            # Log completed request
            logger.info(
                "Request completed",
                request_id=request_id,
                **response_data
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            
            # Log failed request
            logger.error(
                "Request failed",
                request_id=request_id,
                error=str(e),
                process_time=process_time,
                **request_data
            )
            
            # Log error with context
            log_error(__name__, e, {
                "request_id": request_id,
                "process_time": process_time,
                **request_data
            })
            
            raise
    
    async def _extract_request_data(self, request: Request, request_id: str) -> dict:
        """Extract relevant request data for logging."""
        # Basic request info
        data = {
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_host": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "content_type": request.headers.get("content-type"),
            "content_length": request.headers.get("content-length")
        }
        
        # Filter sensitive query parameters
        if data["query_params"]:
            data["query_params"] = self._filter_sensitive_data(data["query_params"])
        
        # Headers (filtered)
        headers = dict(request.headers)
        data["headers"] = self._filter_sensitive_headers(headers)
        
        # Body (for POST/PUT requests, if JSON and not too large)
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                content_type = request.headers.get("content-type", "")
                if "application/json" in content_type:
                    # Read body
                    body = await request.body()
                    if len(body) < 10000:  # Only log if less than 10KB
                        try:
                            json_body = json.loads(body)
                            data["body"] = self._filter_sensitive_data(json_body)
                        except json.JSONDecodeError:
                            data["body"] = "[Invalid JSON]"
                    else:
                        data["body"] = f"[Body too large: {len(body)} bytes]"
            except Exception:
                data["body"] = "[Could not read body]"
        
        return data
    
    def _extract_response_data(self, response: Response, process_time: float) -> dict:
        """Extract relevant response data for logging."""
        return {
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type"),
            "content_length": response.headers.get("content-length"),
            "process_time": process_time
        }
    
    def _filter_sensitive_headers(self, headers: dict) -> dict:
        """Filter out sensitive headers from logging."""
        filtered = {}
        for key, value in headers.items():
            if key.lower() in self.sensitive_headers:
                filtered[key] = "[REDACTED]"
            else:
                filtered[key] = value
        return filtered
    
    def _filter_sensitive_data(self, data: dict) -> dict:
        """Recursively filter sensitive data from dictionaries."""
        if not isinstance(data, dict):
            return data
            
        filtered = {}
        for key, value in data.items():
            key_lower = str(key).lower()
            
            # Check if key contains sensitive information
            is_sensitive = any(sensitive in key_lower for sensitive in self.sensitive_params)
            
            if is_sensitive:
                filtered[key] = "[REDACTED]"
            elif isinstance(value, dict):
                filtered[key] = self._filter_sensitive_data(value)
            elif isinstance(value, list):
                filtered[key] = [
                    self._filter_sensitive_data(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                filtered[key] = value
                
        return filtered
    
    def _generate_request_id(self) -> str:
        """Generate a unique request ID."""
        import uuid
        return str(uuid.uuid4())[:8]
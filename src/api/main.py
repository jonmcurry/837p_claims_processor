"""FastAPI application for claims processing system."""

import time
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from src.api.endpoints import auth, claims, analytics, health
from src.api.middleware.logging import LoggingMiddleware
from src.api.middleware.rate_limiting import RateLimitMiddleware
from src.api.middleware.security import SecurityMiddleware
from src.cache.redis_cache import cache_manager
from src.core.config import settings
from src.core.database.base import close_db, init_db
from src.monitoring.metrics.prometheus_metrics import metrics
from src.processing.ml_pipeline.predictor import claim_predictor

logger = structlog.get_logger(__name__)

# Application startup time for uptime metrics
app_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    # Startup
    logger.info("Starting Smart Claims Processor API", version="1.0.0")
    
    try:
        # Initialize database connections
        await init_db()
        logger.info("Database connections initialized")
        
        # Connect to Redis cache
        await cache_manager.connect()
        logger.info("Redis cache connected")
        
        # Warm up cache
        await cache_manager.warm_cache()
        logger.info("Cache warm-up completed")
        
        # Load ML models
        await claim_predictor.load_models()
        logger.info("ML models loaded")
        
        # Start metrics server
        if settings.enable_metrics:
            metrics.start_metrics_server()
            metrics.set_application_info(version="1.0.0", environment=settings.app_env)
            logger.info("Metrics server started", port=settings.prometheus_port)
        
        logger.info("Application startup completed successfully")
        
        yield
        
    except Exception as e:
        logger.exception("Application startup failed", error=str(e))
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Smart Claims Processor API")
        
        try:
            # Close database connections
            await close_db()
            logger.info("Database connections closed")
            
            # Disconnect from Redis
            await cache_manager.disconnect()
            logger.info("Redis cache disconnected")
            
        except Exception as e:
            logger.exception("Error during shutdown", error=str(e))
        
        logger.info("Application shutdown completed")


# Create FastAPI application
app = FastAPI(
    title="Smart Claims Processor API",
    description="High-performance HIPAA-compliant claims processing system",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if not settings.is_production else None,
    redoc_url="/redoc" if not settings.is_production else None,
    openapi_url="/openapi.json" if not settings.is_production else None,
)

# Security middleware (must be first)
app.add_middleware(SecurityMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Trusted host middleware for production
if settings.is_production:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure with actual trusted hosts
    )

# Rate limiting middleware
if settings.enable_rate_limiting:
    app.add_middleware(RateLimitMiddleware)

# Logging middleware (should be last)
app.add_middleware(LoggingMiddleware)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header and update metrics."""
    start_time = time.time()
    
    # Update uptime metric
    metrics.update_uptime(app_start_time)
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Record API metrics
    if settings.enable_metrics:
        metrics.observe_api_request_duration(
            method=request.method,
            endpoint=request.url.path,
            duration=process_time
        )
        
        metrics.increment_api_requests(
            method=request.method,
            endpoint=request.url.path,
            status_code=str(response.status_code)
        )
    
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured logging."""
    logger.warning(
        "HTTP exception occurred",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
        method=request.method,
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "timestamp": time.time(),
                "path": request.url.path,
            }
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with structured logging."""
    logger.exception(
        "Unhandled exception occurred",
        error=str(exc),
        path=request.url.path,
        method=request.method,
    )
    
    # Don't expose internal error details in production
    error_message = "Internal server error" if settings.is_production else str(exc)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": 500,
                "message": error_message,
                "timestamp": time.time(),
                "path": request.url.path,
            }
        },
    )


# Include API routers
app.include_router(
    health.router,
    prefix=settings.api_prefix,
    tags=["health"]
)

app.include_router(
    auth.router,
    prefix=settings.api_prefix,
    tags=["authentication"]
)

app.include_router(
    claims.router,
    prefix=settings.api_prefix,
    tags=["claims"]
)

app.include_router(
    analytics.router,
    prefix=settings.api_prefix,
    tags=["analytics"]
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Smart Claims Processor API",
        "version": "1.0.0",
        "status": "operational",
        "environment": settings.app_env,
        "docs_url": "/docs" if not settings.is_production else None,
    }


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    if not settings.enable_metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metrics are disabled"
        )
    
    return JSONResponse(
        content=metrics.get_metrics(),
        media_type="text/plain"
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True,
    )
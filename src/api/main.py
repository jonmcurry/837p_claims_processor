"""FastAPI application for claims processing system."""

import time
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

# Simplified imports - commenting out missing modules
# from src.api.endpoints import auth, claims, analytics, health
# from src.api.middleware.logging import LoggingMiddleware
# from src.api.middleware.rate_limiting import RateLimitMiddleware
# from src.api.middleware.security import SecurityMiddleware
# from src.cache.redis_cache import cache_manager
# from src.core.config import settings
# from src.core.database.base import close_db, init_db
# from src.monitoring.metrics.prometheus_metrics import metrics
# from src.processing.ml_pipeline.predictor import claim_predictor

# Import the processing pipeline directly
from src.processing.batch_processor.pipeline import processing_pipeline

logger = structlog.get_logger(__name__)

# Application startup time for uptime metrics
app_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Simplified application lifespan context manager."""
    # Startup
    logger.info("Starting Smart Claims Processor API", version="1.0.0")
    
    try:
        logger.info("Application startup completed successfully")
        yield
        
    except Exception as e:
        logger.exception("Application startup failed", error=str(e))
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Smart Claims Processor API")
        logger.info("Application shutdown completed")


# Create FastAPI application
app = FastAPI(
    title="Smart Claims Processor API",
    description="High-performance HIPAA-compliant claims processing system",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Basic CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Simplified for development
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
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
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": 500,
                "message": str(exc),
                "timestamp": time.time(),
                "path": request.url.path,
            }
        },
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Smart Claims Processor API",
        "version": "1.0.0",
        "status": "operational",
        "environment": "development",
        "docs_url": "/docs",
    }


@app.post("/api/v1/claims/process-batch")
async def process_batch(batch_id: str):
    """Process a batch of claims."""
    try:
        result = await processing_pipeline.process_batch(batch_id)
        return {
            "batch_id": result.batch_id,
            "total_claims": result.total_claims,
            "processed_claims": result.processed_claims,
            "failed_claims": result.failed_claims,
            "processing_time": result.processing_time,
            "throughput": result.throughput,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True,
    )
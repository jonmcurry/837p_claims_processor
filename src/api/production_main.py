"""Production-ready FastAPI application for claims processing system."""

import time
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import uuid

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.core.config import settings
from src.core.security.hipaa_security import security_manager
from src.core.security.access_control import rbac_system, Permission, require_permission, require_phi_access
from src.monitoring.metrics.comprehensive_metrics import metrics_collector
from src.api.middleware.production_middleware import (
    SecurityHeadersMiddleware,
    RateLimitingMiddleware, 
    RequestTrackingMiddleware,
    HIPAAComplianceMiddleware,
    ErrorHandlingMiddleware
)

logger = structlog.get_logger(__name__)

# Application startup time for uptime metrics
app_start_time = time.time()
security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Production application lifespan management."""
    # Startup
    logger.info("Starting Claims Processing API", version="1.0.0", environment="production")
    
    try:
        # Initialize database connections
        await init_database_connections()
        
        # Initialize caching
        await init_caching_layer()
        
        # Load ML models
        await init_ml_models()
        
        # Start metrics collection
        metrics_collector.start_metrics_server(port=8001)
        
        # Warm up critical caches
        await warm_application_caches()
        
        logger.info("Claims Processing API started successfully")
        
        yield
        
    except Exception as e:
        logger.exception("Failed to start application", error=str(e))
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Claims Processing API")
        await cleanup_resources()


async def init_database_connections():
    """Initialize database connection pools."""
    try:
        from src.core.database.base import init_db
        await init_db()
        logger.info("Database connections initialized")
    except Exception as e:
        logger.error("Failed to initialize database connections", error=str(e))
        raise


async def init_caching_layer():
    """Initialize Redis caching."""
    try:
        from src.cache.redis_cache import cache_manager
        await cache_manager.ping()
        logger.info("Cache layer initialized")
    except Exception as e:
        logger.error("Failed to initialize cache layer", error=str(e))
        raise


async def init_ml_models():
    """Initialize ML models."""
    try:
        from src.processing.ml_pipeline.advanced_predictor import advanced_predictor
        # Models will auto-load if available
        logger.info("ML models initialized")
    except Exception as e:
        logger.warning("ML models failed to initialize", error=str(e))
        # Don't fail startup for ML models


async def warm_application_caches():
    """Warm up critical application caches."""
    try:
        # Warm facility cache
        await rbac_system._load_lookup_data()
        logger.info("Application caches warmed up")
    except Exception as e:
        logger.warning("Cache warming failed", error=str(e))


async def cleanup_resources():
    """Clean up application resources."""
    try:
        from src.core.database.base import close_db
        await close_db()
        logger.info("Resources cleaned up")
    except Exception as e:
        logger.error("Error during cleanup", error=str(e))


# Create FastAPI application
app = FastAPI(
    title="Claims Processing API",
    description="High-performance HIPAA-compliant claims processing system",
    version="1.0.0",
    lifespan=lifespan,
    openapi_url="/api/openapi.json",
    docs_url=None,  # Disable default docs for security
    redoc_url=None,
    swagger_ui_oauth2_redirect_url=None
)

# Add production middleware stack
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(HIPAAComplianceMiddleware)
app.add_middleware(RequestTrackingMiddleware)
app.add_middleware(RateLimitingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

# CORS configuration for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://claims.company.com",
        "https://dashboard.company.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"]
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        "claims-api.company.com",
        "localhost",
        "127.0.0.1"
    ]
)


# Dependency for getting current user
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request: Request = None
) -> Dict[str, Any]:
    """Get current authenticated user."""
    try:
        # Get access context from security manager
        access_context = await security_manager.get_access_context(
            token=credentials.credentials,
            ip_address=request.client.host,
            request_id=str(request.state.request_id)
        )
        
        if not access_context:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return {
            "user_id": access_context.user_id,
            "user_role": access_context.user_role,
            "session_id": access_context.session_id,
            "access_context": access_context
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Authentication error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Health and monitoring endpoints
@app.get("/health", status_code=200, tags=["Health"])
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "uptime": time.time() - app_start_time,
        "version": "1.0.0"
    }


@app.get("/health/detailed", tags=["Health"])
@require_permission(Permission.SYSTEM_MONITOR)
async def detailed_health_check(current_user: dict = Depends(get_current_user)):
    """Detailed health check with system status."""
    try:
        # Check database connectivity
        from src.core.database.base import get_postgres_session
        db_healthy = True
        try:
            async with get_postgres_session() as session:
                await session.execute("SELECT 1")
        except Exception:
            db_healthy = False
        
        # Check cache connectivity
        cache_healthy = True
        try:
            from src.cache.redis_cache import cache_manager
            await cache_manager.ping()
        except Exception:
            cache_healthy = False
        
        health_status = {
            "status": "healthy" if db_healthy and cache_healthy else "degraded",
            "timestamp": time.time(),
            "uptime": time.time() - app_start_time,
            "version": "1.0.0",
            "components": {
                "database": "healthy" if db_healthy else "unhealthy",
                "cache": "healthy" if cache_healthy else "unhealthy",
                "api": "healthy"
            },
            "metrics": {
                "memory_usage_mb": __import__('psutil').virtual_memory().used / (1024 * 1024),
                "cpu_percent": __import__('psutil').cpu_percent(),
                "active_connections": len(getattr(app.state, 'active_connections', []))
            }
        }
        
        return health_status
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }


@app.get("/metrics", response_class=Response, tags=["Monitoring"])
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    metrics_data = metrics_collector.get_metrics()
    return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)


@app.get("/ready", status_code=200, tags=["Health"])
async def readiness_check():
    """Kubernetes readiness probe."""
    # Check if application is ready to serve traffic
    try:
        # Quick database check
        from src.core.database.base import get_postgres_session
        async with get_postgres_session() as session:
            await session.execute("SELECT 1")
        
        return {"status": "ready"}
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )


@app.get("/live", status_code=200, tags=["Health"])
async def liveness_check():
    """Kubernetes liveness probe."""
    return {"status": "alive", "timestamp": time.time()}


# Authentication endpoints
@app.post("/auth/login", tags=["Authentication"])
async def login(
    request: Request,
    credentials: Dict[str, str]
):
    """Authenticate user and return JWT tokens."""
    try:
        username = credentials.get("username")
        password = credentials.get("password")
        
        if not username or not password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username and password required"
            )
        
        # Authenticate user
        auth_result = await security_manager.authenticate_user(
            username=username,
            password=password,
            ip_address=request.client.host
        )
        
        if not auth_result:
            # Log failed attempt
            await security_manager.audit_logger.log_system_event(
                "login_failed",
                f"Failed login attempt for {username}",
                additional_data={"ip_address": request.client.host}
            )
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        # Record successful login metrics
        metrics_collector.login_attempts_total.labels(
            status="success",
            user_role=auth_result["user_data"]["role"],
            ip_subnet=request.client.host.split('.')[0:2]  # First two octets for privacy
        ).inc()
        
        return {
            "access_token": auth_result["access_token"],
            "refresh_token": auth_result["refresh_token"],
            "token_type": "bearer",
            "expires_in": 1800,  # 30 minutes
            "user_info": {
                "user_id": auth_result["user_data"]["user_id"],
                "username": auth_result["user_data"]["username"],
                "role": auth_result["user_data"]["role"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Login error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        )


@app.post("/auth/refresh", tags=["Authentication"])
async def refresh_token(refresh_token: str):
    """Refresh access token using refresh token."""
    try:
        payload = security_manager.jwt_manager.verify_token(refresh_token)
        
        if not payload or payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Create new access token
        new_access_token = security_manager.jwt_manager.create_access_token({
            "sub": payload["sub"],
            "session_id": payload["session_id"]
        })
        
        return {
            "access_token": new_access_token,
            "token_type": "bearer",
            "expires_in": 1800
        }
        
    except Exception as e:
        logger.error("Token refresh error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token refresh failed"
        )


@app.post("/auth/logout", tags=["Authentication"])
async def logout(current_user: dict = Depends(get_current_user)):
    """Logout user and invalidate session."""
    try:
        session_id = current_user["session_id"]
        
        # Invalidate session
        if session_id in security_manager.active_sessions:
            del security_manager.active_sessions[session_id]
        
        # Log logout
        await security_manager.audit_logger.log_system_event(
            "logout",
            f"User {current_user['user_id']} logged out",
            additional_data={"session_id": session_id}
        )
        
        return {"message": "Logged out successfully"}
        
    except Exception as e:
        logger.error("Logout error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


# Claims processing endpoints
@app.post("/api/v1/claims/batch", tags=["Claims Processing"])
@require_permission(Permission.SUBMIT_BATCH)
async def submit_claims_batch(
    batch_data: Dict[str, Any],
    current_user: dict = Depends(get_current_user),
    request: Request = None
):
    """Submit a batch of claims for processing."""
    try:
        from src.processing.batch_processor.pipeline import processing_pipeline
        
        batch_id = str(uuid.uuid4())
        
        # Record batch submission
        metrics_collector.record_batch_processing(
            facility_id=batch_data.get("facility_id", "unknown"),
            batch_size=len(batch_data.get("claims", [])),
            processing_time=0,  # Will be updated when processing completes
            throughput=0
        )
        
        # Process batch asynchronously
        asyncio.create_task(
            processing_pipeline.process_batch(batch_id)
        )
        
        return {
            "batch_id": batch_id,
            "status": "submitted",
            "claims_count": len(batch_data.get("claims", [])),
            "estimated_completion": "15 seconds"
        }
        
    except Exception as e:
        logger.error("Batch submission error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch submission failed"
        )


@app.get("/api/v1/claims/{claim_id}", tags=["Claims Processing"])
@require_permission(Permission.READ_CLAIMS)
async def get_claim(
    claim_id: str,
    current_user: dict = Depends(get_current_user),
    business_justification: Optional[str] = None
):
    """Get claim details by ID."""
    try:
        from src.core.database.base import get_postgres_session
        from src.core.database.models import Claim
        from sqlalchemy import select
        
        async with get_postgres_session() as session:
            query = select(Claim).where(Claim.claim_id == claim_id)
            result = await session.execute(query)
            claim = result.scalar_one_or_none()
            
            if not claim:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Claim not found"
                )
            
            # Check PHI access authorization
            access_context = current_user["access_context"]
            if business_justification:
                access_context.business_justification = business_justification
            
            # Convert claim to dict and handle PHI decryption if authorized
            claim_data = {
                "claim_id": claim.claim_id,
                "facility_id": claim.facility_id,
                "patient_account_number": claim.patient_account_number,
                "total_charges": float(claim.total_charges),
                "processing_status": claim.processing_status.value,
                "created_at": claim.created_at.isoformat()
            }
            
            # Add PHI data if user has access
            if rbac_system.can_access_phi(current_user["user_id"]):
                # Log PHI access
                await security_manager.audit_logger.log_phi_access(
                    access_context,
                    ["patient_name", "patient_dob"],
                    success=True
                )
                
                claim_data.update({
                    "patient_first_name": claim.patient_first_name,
                    "patient_last_name": claim.patient_last_name,
                    "patient_date_of_birth": claim.patient_date_of_birth.isoformat()
                })
            
            return claim_data
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get claim error", claim_id=claim_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve claim"
        )


# Failed claims management
@app.get("/api/v1/failed-claims", tags=["Failed Claims"])
@require_permission(Permission.VIEW_FAILED_CLAIMS)
async def get_failed_claims(
    facility_id: Optional[str] = None,
    failure_category: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    current_user: dict = Depends(get_current_user)
):
    """Get list of failed claims with filtering."""
    try:
        from src.core.database.base import get_postgres_session
        from src.core.database.models import FailedClaim
        from sqlalchemy import select
        
        async with get_postgres_session() as session:
            query = select(FailedClaim)
            
            if facility_id:
                query = query.where(FailedClaim.facility_id == facility_id)
            if failure_category:
                query = query.where(FailedClaim.failure_category == failure_category)
            
            query = query.limit(limit).offset(offset).order_by(FailedClaim.failed_at.desc())
            
            result = await session.execute(query)
            failed_claims = result.scalars().all()
            
            return {
                "failed_claims": [
                    {
                        "id": claim.id,
                        "claim_reference": claim.claim_reference,
                        "facility_id": claim.facility_id,
                        "failure_category": claim.failure_category.value,
                        "failure_reason": claim.failure_reason,
                        "charge_amount": float(claim.charge_amount),
                        "failed_at": claim.failed_at.isoformat(),
                        "resolution_status": claim.resolution_status
                    }
                    for claim in failed_claims
                ],
                "total_count": len(failed_claims),
                "limit": limit,
                "offset": offset
            }
            
    except Exception as e:
        logger.error("Get failed claims error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve failed claims"
        )


# Secured API documentation
@app.get("/docs", include_in_schema=False)
async def get_documentation(current_user: dict = Depends(get_current_user)):
    """Secured API documentation."""
    if not rbac_system.has_permission(current_user["user_id"], Permission.SYSTEM_MONITOR):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient privileges for API documentation"
        )
    
    return get_swagger_ui_html(
        openapi_url="/api/openapi.json",
        title="Claims Processing API - Documentation"
    )


@app.get("/api/openapi.json", include_in_schema=False)
async def get_openapi_schema(current_user: dict = Depends(get_current_user)):
    """Secured OpenAPI schema."""
    if not rbac_system.has_permission(current_user["user_id"], Permission.SYSTEM_MONITOR):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient privileges for API schema"
        )
    
    return get_openapi(
        title="Claims Processing API",
        version="1.0.0",
        description="High-performance HIPAA-compliant claims processing system",
        routes=app.routes,
    )


if __name__ == "__main__":
    uvicorn.run(
        "src.api.production_main:app",
        host="0.0.0.0",
        port=8080,
        workers=4,
        loop="uvloop",
        http="httptools",
        access_log=True,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )
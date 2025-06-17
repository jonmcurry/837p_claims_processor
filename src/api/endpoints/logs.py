"""
Log collection endpoint for frontend errors and events.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from src.core.logging import get_logger, log_error

router = APIRouter(prefix="/logs", tags=["logs"])
logger = get_logger(__name__, "api", structured=True)


class LogEntry(BaseModel):
    """Frontend log entry model."""
    timestamp: str
    level: str = Field(..., regex="^(debug|info|warn|error)$")
    message: str
    context: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, str]] = None
    userAgent: Optional[str] = None
    url: Optional[str] = None
    userId: Optional[str] = None
    sessionId: Optional[str] = None


class LogBatch(BaseModel):
    """Batch of log entries from frontend."""
    logs: List[LogEntry]


class FrontendLogHandler:
    """Handles frontend log processing and storage."""
    
    def __init__(self):
        self.frontend_logger = get_logger("frontend", "frontend")
        self.security_logger = get_logger("frontend.security", "security")
    
    def process_log_batch(self, log_batch: LogBatch) -> Dict[str, Any]:
        """Process a batch of frontend logs."""
        processed = 0
        errors = 0
        
        for log_entry in log_batch.logs:
            try:
                self._process_single_log(log_entry)
                processed += 1
            except Exception as e:
                errors += 1
                log_error(__name__, e, {"log_entry": log_entry.dict()})
        
        logger.info(
            "Frontend log batch processed",
            total_logs=len(log_batch.logs),
            processed=processed,
            errors=errors
        )
        
        return {
            "processed": processed,
            "errors": errors,
            "total": len(log_batch.logs)
        }
    
    def _process_single_log(self, log_entry: LogEntry):
        """Process a single frontend log entry."""
        log_data = {
            "timestamp": log_entry.timestamp,
            "message": log_entry.message,
            "level": log_entry.level,
            "user_id": log_entry.userId,
            "session_id": log_entry.sessionId,
            "url": log_entry.url,
            "user_agent": log_entry.userAgent,
            "context": log_entry.context or {},
            "error_details": log_entry.error
        }
        
        # Log based on level
        if log_entry.level == "error":
            self.frontend_logger.error(
                f"Frontend Error: {log_entry.message}",
                extra=log_data
            )
            
            # Also log security-related errors
            if self._is_security_related(log_entry):
                self.security_logger.warning(
                    f"Security-related frontend error: {log_entry.message}",
                    extra=log_data
                )
                
        elif log_entry.level == "warn":
            self.frontend_logger.warning(
                f"Frontend Warning: {log_entry.message}",
                extra=log_data
            )
            
        elif log_entry.level == "info":
            self.frontend_logger.info(
                f"Frontend Info: {log_entry.message}",
                extra=log_data
            )
            
        else:  # debug
            self.frontend_logger.debug(
                f"Frontend Debug: {log_entry.message}",
                extra=log_data
            )
    
    def _is_security_related(self, log_entry: LogEntry) -> bool:
        """Check if a log entry is security-related."""
        security_keywords = [
            "unauthorized", "forbidden", "authentication", "permission",
            "csrf", "xss", "injection", "malicious", "suspicious",
            "blocked", "rate limit", "abuse"
        ]
        
        message_lower = log_entry.message.lower()
        context_str = str(log_entry.context or {}).lower()
        
        return any(keyword in message_lower or keyword in context_str 
                  for keyword in security_keywords)


# Create handler instance
log_handler = FrontendLogHandler()


@router.post("/frontend")
async def receive_frontend_logs(log_batch: LogBatch):
    """Receive and process frontend logs."""
    try:
        result = log_handler.process_log_batch(log_batch)
        return {
            "success": True,
            "message": "Logs processed successfully",
            "result": result
        }
    except Exception as e:
        logger.error("Failed to process frontend logs", error=str(e))
        log_error(__name__, e, {"batch_size": len(log_batch.logs)})
        raise HTTPException(
            status_code=500,
            detail="Failed to process frontend logs"
        )


@router.get("/frontend/stats")
async def get_frontend_log_stats():
    """Get statistics about frontend logs."""
    try:
        # This would typically query a database or log files
        # For now, return placeholder stats
        return {
            "total_logs_today": 0,
            "error_count_today": 0,
            "warning_count_today": 0,
            "active_sessions": 0,
            "top_errors": [],
            "last_updated": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error("Failed to get frontend log stats", error=str(e))
        log_error(__name__, e, {"operation": "get_stats"})
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve log statistics"
        )
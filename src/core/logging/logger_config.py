"""
Centralized logging configuration for the 837P Claims Processor.

Provides file-based logging with rotation, structured logging, and
different log levels for various components of the system.
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import structlog
from structlog.stdlib import LoggerFactory
import json


class LoggerConfig:
    """Centralized logger configuration and management."""
    
    def __init__(self, base_log_dir: str = "logs"):
        self.base_log_dir = Path(base_log_dir)
        self.base_log_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.log_dirs = {
            "claims": self.base_log_dir / "claims",
            "api": self.base_log_dir / "api",
            "frontend": self.base_log_dir / "frontend",
            "system": self.base_log_dir / "system",
            "analytics": self.base_log_dir / "analytics",
            "security": self.base_log_dir / "security"
        }
        
        for dir_path in self.log_dirs.values():
            dir_path.mkdir(exist_ok=True)
            
        # Log file configurations
        self.log_configs = {
            "claims": {
                "filename": "claims_processing.log",
                "max_bytes": 100 * 1024 * 1024,  # 100MB
                "backup_count": 10,
                "level": logging.INFO
            },
            "api": {
                "filename": "api_requests.log",
                "max_bytes": 50 * 1024 * 1024,  # 50MB
                "backup_count": 10,
                "level": logging.INFO
            },
            "frontend": {
                "filename": "frontend_errors.log",
                "max_bytes": 50 * 1024 * 1024,  # 50MB
                "backup_count": 5,
                "level": logging.WARNING
            },
            "system": {
                "filename": "system.log",
                "max_bytes": 100 * 1024 * 1024,  # 100MB
                "backup_count": 10,
                "level": logging.INFO
            },
            "analytics": {
                "filename": "analytics.log",
                "max_bytes": 50 * 1024 * 1024,  # 50MB
                "backup_count": 5,
                "level": logging.INFO
            },
            "security": {
                "filename": "security_audit.log",
                "max_bytes": 100 * 1024 * 1024,  # 100MB
                "backup_count": 20,  # Keep more security logs
                "level": logging.INFO
            },
            "error": {
                "filename": "errors.log",
                "max_bytes": 100 * 1024 * 1024,  # 100MB
                "backup_count": 10,
                "level": logging.ERROR
            }
        }
        
        # Track created loggers
        self._loggers: Dict[str, logging.Logger] = {}
        
    def get_file_handler(self, log_type: str, config: Dict[str, Any]) -> logging.Handler:
        """Create a rotating file handler for the specified log type."""
        log_dir = self.log_dirs.get(log_type, self.base_log_dir)
        log_path = log_dir / config["filename"]
        
        handler = logging.handlers.RotatingFileHandler(
            filename=str(log_path),
            maxBytes=config["max_bytes"],
            backupCount=config["backup_count"],
            encoding="utf-8"
        )
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        handler.setLevel(config["level"])
        
        return handler
    
    def get_error_handler(self) -> logging.Handler:
        """Create a dedicated error handler that captures all errors."""
        error_config = self.log_configs["error"]
        error_path = self.base_log_dir / error_config["filename"]
        
        handler = logging.handlers.RotatingFileHandler(
            filename=str(error_path),
            maxBytes=error_config["max_bytes"],
            backupCount=error_config["backup_count"],
            encoding="utf-8"
        )
        
        # Detailed error formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(funcName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        handler.setLevel(logging.ERROR)
        
        return handler
    
    def setup_python_logger(self, name: str, log_type: str) -> logging.Logger:
        """Set up a standard Python logger with file handlers."""
        if name in self._loggers:
            return self._loggers[name]
            
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        logger.handlers = []
        
        # Add file handler for specific log type
        if log_type in self.log_configs:
            config = self.log_configs[log_type]
            handler = self.get_file_handler(log_type, config)
            logger.addHandler(handler)
        
        # Always add error handler
        error_handler = self.get_error_handler()
        logger.addHandler(error_handler)
        
        # Add console handler for warnings and above
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        self._loggers[name] = logger
        return logger
    
    def setup_structlog(self, log_type: str = "system") -> structlog.BoundLogger:
        """Set up structlog with file output for structured logging."""
        # Get log configuration
        config = self.log_configs.get(log_type, self.log_configs["system"])
        log_dir = self.log_dirs.get(log_type, self.base_log_dir)
        log_path = log_dir / config["filename"]
        
        # Create file handler
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_path),
            maxBytes=config["max_bytes"],
            backupCount=config["backup_count"],
            encoding="utf-8"
        )
        file_handler.setLevel(config["level"])
        
        # Create error handler
        error_handler = self.get_error_handler()
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.CallsiteParameterAdder(
                    parameters=[
                        structlog.processors.CallsiteParameter.FILENAME,
                        structlog.processors.CallsiteParameter.LINENO,
                        structlog.processors.CallsiteParameter.FUNC_NAME,
                    ]
                ),
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Set up stdlib logging to use our handlers
        logging.basicConfig(
            level=logging.DEBUG,
            handlers=[file_handler, error_handler]
        )
        
        return structlog.get_logger()
    
    def get_logger(self, name: str, log_type: str = "system", structured: bool = False) -> Any:
        """Get a logger instance with file output."""
        if structured:
            return self.setup_structlog(log_type)
        else:
            return self.setup_python_logger(name, log_type)
    
    def log_error(self, logger_name: str, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Log an error with full context to the error log."""
        error_logger = self.setup_python_logger(f"{logger_name}.error", "error")
        
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.utcnow().isoformat(),
            "logger": logger_name
        }
        
        if context:
            error_info["context"] = context
            
        error_logger.error(
            f"Error in {logger_name}: {type(error).__name__} - {str(error)}",
            exc_info=True,
            extra={"error_details": json.dumps(error_info)}
        )
    
    def create_daily_log(self, log_type: str, prefix: str = "") -> logging.Logger:
        """Create a logger that writes to daily rotating files."""
        date_str = datetime.now().strftime("%Y%m%d")
        log_name = f"{prefix}{date_str}" if prefix else date_str
        
        logger = logging.getLogger(f"{log_type}.{log_name}")
        logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        logger.handlers = []
        
        # Create daily file handler
        log_dir = self.log_dirs.get(log_type, self.base_log_dir)
        log_path = log_dir / f"{log_name}.log"
        
        handler = logging.handlers.TimedRotatingFileHandler(
            filename=str(log_path),
            when="midnight",
            interval=1,
            backupCount=30,  # Keep 30 days of logs
            encoding="utf-8"
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def get_audit_logger(self) -> logging.Logger:
        """Get a specialized audit logger for security events."""
        audit_logger = self.setup_python_logger("security.audit", "security")
        
        # Add special audit formatter
        for handler in audit_logger.handlers:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                audit_formatter = logging.Formatter(
                    '%(asctime)s - AUDIT - %(levelname)s - User: %(user)s - Action: %(action)s - '
                    'Resource: %(resource)s - Result: %(result)s - Details: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                handler.setFormatter(audit_formatter)
                
        return audit_logger
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up log files older than specified days."""
        import time
        
        current_time = time.time()
        cutoff_time = current_time - (days_to_keep * 24 * 60 * 60)
        
        cleaned = 0
        for log_dir in self.log_dirs.values():
            for log_file in log_dir.glob("*.log*"):
                if log_file.stat().st_mtime < cutoff_time:
                    try:
                        log_file.unlink()
                        cleaned += 1
                    except Exception as e:
                        print(f"Failed to delete {log_file}: {e}")
                        
        return cleaned


# Global logger configuration instance
logger_config = LoggerConfig()

# Convenience functions
def get_logger(name: str, log_type: str = "system", structured: bool = False) -> Any:
    """Get a logger instance."""
    return logger_config.get_logger(name, log_type, structured)

def log_error(logger_name: str, error: Exception, context: Optional[Dict[str, Any]] = None):
    """Log an error with context."""
    logger_config.log_error(logger_name, error, context)

def get_audit_logger() -> logging.Logger:
    """Get the audit logger."""
    return logger_config.get_audit_logger()
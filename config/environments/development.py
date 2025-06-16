"""Development environment configuration."""

import os
from pathlib import Path
from typing import Optional

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "development"
LOGS_DIR = BASE_DIR / "logs" / "development"
TEMP_DIR = BASE_DIR / "temp" / "development"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

class DevelopmentConfig:
    """Development environment configuration."""
    
    # Environment
    ENVIRONMENT = "development"
    DEBUG = True
    TESTING = False
    
    # Database Configuration
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB = os.getenv("POSTGRES_DB", "claims_processor_dev")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "claims_user")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "dev_password_123")
    
    # SQL Server Configuration (for testing dual database setup)
    SQLSERVER_HOST = os.getenv("SQLSERVER_HOST", "localhost")
    SQLSERVER_PORT = int(os.getenv("SQLSERVER_PORT", "1433"))
    SQLSERVER_DB = os.getenv("SQLSERVER_DB", "claims_analytics_dev")
    SQLSERVER_USER = os.getenv("SQLSERVER_USER", "sa")
    SQLSERVER_PASSWORD = os.getenv("SQLSERVER_PASSWORD", "DevPassword123!")
    
    # Connection URLs
    DATABASE_URL = f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    ANALYTICS_DATABASE_URL = f"mssql+pyodbc://{SQLSERVER_USER}:{SQLSERVER_PASSWORD}@{SQLSERVER_HOST}:{SQLSERVER_PORT}/{SQLSERVER_DB}?driver=ODBC+Driver+17+for+SQL+Server"
    
    # Redis Configuration
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
    REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    
    # Security Configuration
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-jwt-secret-key-change-in-production")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 60  # Longer for development
    JWT_REFRESH_TOKEN_EXPIRE_DAYS = 7
    ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", "dev-encryption-key-32-chars-long!")
    
    # CORS settings (permissive for development)
    CORS_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://0.0.0.0:3000"
    ]
    CORS_ALLOW_CREDENTIALS = True
    CORS_ALLOW_METHODS = ["*"]
    CORS_ALLOW_HEADERS = ["*"]
    
    # API Configuration
    API_V1_PREFIX = "/api/v1"
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    API_RELOAD = True  # Auto-reload for development
    API_WORKERS = 1  # Single worker for development
    
    # Batch Processing Configuration
    MAX_BATCH_SIZE = 1000  # Smaller batches for development
    MAX_CONCURRENT_BATCHES = 2
    BATCH_TIMEOUT_SECONDS = 300
    ENABLE_ASYNC_PROCESSING = True
    
    # ML Model Configuration
    ML_MODEL_PATH = str(BASE_DIR / "models" / "development")
    ML_BATCH_SIZE = 100  # Smaller for development
    ML_ENABLE_GPU = False  # Usually disabled in dev
    ML_MODEL_CACHE_TTL = 3600  # 1 hour
    
    # Validation Configuration
    VALIDATION_STRICT_MODE = False  # More lenient in development
    VALIDATION_CACHE_ENABLED = True
    VALIDATION_CACHE_TTL = 1800  # 30 minutes
    
    # Monitoring Configuration
    METRICS_ENABLED = True
    PROMETHEUS_PORT = 9090
    GRAFANA_PORT = 3000
    LOG_LEVEL = "DEBUG"
    LOG_FORMAT = "detailed"
    
    # File paths
    UPLOAD_PATH = str(DATA_DIR / "uploads")
    EXPORT_PATH = str(DATA_DIR / "exports")
    BACKUP_PATH = str(DATA_DIR / "backups")
    LOG_FILE_PATH = str(LOGS_DIR / "application.log")
    
    # Rate Limiting (more permissive for development)
    RATE_LIMIT_ENABLED = True
    RATE_LIMIT_REQUESTS_PER_MINUTE = 1000
    RATE_LIMIT_BURST = 100
    
    # WebSocket Configuration
    WEBSOCKET_ENABLED = True
    WEBSOCKET_PING_INTERVAL = 30
    WEBSOCKET_PING_TIMEOUT = 10
    
    # External Services (usually mocked in development)
    EXTERNAL_API_TIMEOUT = 30
    EXTERNAL_API_RETRIES = 3
    EXTERNAL_API_MOCK_MODE = True
    
    # Email Configuration (for notifications)
    SMTP_HOST = "localhost"
    SMTP_PORT = 1025  # MailHog for development
    SMTP_USERNAME = ""
    SMTP_PASSWORD = ""
    SMTP_USE_TLS = False
    EMAIL_FROM = "dev@claims-processor.local"
    
    # Development Tools
    ENABLE_PROFILING = True
    ENABLE_DEBUG_TOOLBAR = True
    ENABLE_SQL_ECHO = True  # Echo SQL queries
    ENABLE_MOCK_DATA = True
    
    # Feature Flags
    FEATURE_ML_PIPELINE = True
    FEATURE_REAL_TIME_UPDATES = True
    FEATURE_ADVANCED_ANALYTICS = True
    FEATURE_AUDIT_DASHBOARD = True
    
    # Performance Settings
    CONNECTION_POOL_SIZE = 5
    CONNECTION_POOL_MAX_OVERFLOW = 10
    CONNECTION_POOL_TIMEOUT = 30
    QUERY_TIMEOUT = 30
    
    # Caching
    CACHE_DEFAULT_TIMEOUT = 300  # 5 minutes
    CACHE_KEY_PREFIX = "claims_dev"
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate development configuration."""
        required_vars = [
            'DATABASE_URL', 'REDIS_URL', 'SECRET_KEY', 'JWT_SECRET_KEY', 'ENCRYPTION_KEY'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not getattr(cls, var, None):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required configuration variables: {missing_vars}")
        
        return True
    
    @classmethod
    def get_database_settings(cls) -> dict:
        """Get database connection settings."""
        return {
            "url": cls.DATABASE_URL,
            "echo": cls.ENABLE_SQL_ECHO,
            "pool_size": cls.CONNECTION_POOL_SIZE,
            "max_overflow": cls.CONNECTION_POOL_MAX_OVERFLOW,
            "pool_timeout": cls.CONNECTION_POOL_TIMEOUT
        }
    
    @classmethod
    def get_redis_settings(cls) -> dict:
        """Get Redis connection settings."""
        return {
            "url": cls.REDIS_URL,
            "decode_responses": True,
            "health_check_interval": 30
        }
    
    @classmethod
    def get_security_settings(cls) -> dict:
        """Get security settings."""
        return {
            "secret_key": cls.SECRET_KEY,
            "jwt_secret": cls.JWT_SECRET_KEY,
            "jwt_access_expire": cls.JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
            "jwt_refresh_expire": cls.JWT_REFRESH_TOKEN_EXPIRE_DAYS,
            "encryption_key": cls.ENCRYPTION_KEY
        }


# Export configuration instance
config = DevelopmentConfig()
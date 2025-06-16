"""Staging environment configuration."""

import os
from pathlib import Path
from typing import Optional

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = Path("/app/data/staging")
LOGS_DIR = Path("/app/logs/staging")
TEMP_DIR = Path("/app/temp/staging")

# Ensure directories exist in containerized environment
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

class StagingConfig:
    """Staging environment configuration - mirrors production with test data."""
    
    # Environment
    ENVIRONMENT = "staging"
    DEBUG = False
    TESTING = False
    
    # Database Configuration
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres-staging")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB = os.getenv("POSTGRES_DB", "claims_processor_staging")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "claims_user")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")  # Required from env
    
    # SQL Server Configuration
    SQLSERVER_HOST = os.getenv("SQLSERVER_HOST", "sqlserver-staging")
    SQLSERVER_PORT = int(os.getenv("SQLSERVER_PORT", "1433"))
    SQLSERVER_DB = os.getenv("SQLSERVER_DB", "claims_analytics_staging")
    SQLSERVER_USER = os.getenv("SQLSERVER_USER", "sa")
    SQLSERVER_PASSWORD = os.getenv("SQLSERVER_PASSWORD")  # Required from env
    
    # Connection URLs
    @property
    def DATABASE_URL(self) -> str:
        if not self.POSTGRES_PASSWORD:
            raise ValueError("POSTGRES_PASSWORD environment variable is required")
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    @property
    def ANALYTICS_DATABASE_URL(self) -> str:
        if not self.SQLSERVER_PASSWORD:
            raise ValueError("SQLSERVER_PASSWORD environment variable is required")
        return f"mssql+pyodbc://{self.SQLSERVER_USER}:{self.SQLSERVER_PASSWORD}@{self.SQLSERVER_HOST}:{self.SQLSERVER_PORT}/{self.SQLSERVER_DB}?driver=ODBC+Driver+17+for+SQL+Server"
    
    # Redis Configuration
    REDIS_HOST = os.getenv("REDIS_HOST", "redis-staging")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
    
    @property
    def REDIS_URL(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    # Security Configuration
    SECRET_KEY = os.getenv("SECRET_KEY")  # Required from env
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")  # Required from env
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS = 7
    ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")  # Required from env
    
    # CORS settings (restricted for staging)
    CORS_ORIGINS = [
        "https://staging.claims-processor.company.com",
        "https://staging-admin.claims-processor.company.com"
    ]
    CORS_ALLOW_CREDENTIALS = True
    CORS_ALLOW_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH"]
    CORS_ALLOW_HEADERS = ["*"]
    
    # API Configuration
    API_V1_PREFIX = "/api/v1"
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    API_RELOAD = False
    API_WORKERS = 4  # Multiple workers for staging
    
    # Batch Processing Configuration
    MAX_BATCH_SIZE = 50000  # Production-like but smaller
    MAX_CONCURRENT_BATCHES = 3
    BATCH_TIMEOUT_SECONDS = 600
    ENABLE_ASYNC_PROCESSING = True
    
    # ML Model Configuration
    ML_MODEL_PATH = "/app/models/staging"
    ML_BATCH_SIZE = 1000
    ML_ENABLE_GPU = os.getenv("ML_ENABLE_GPU", "false").lower() == "true"
    ML_MODEL_CACHE_TTL = 7200  # 2 hours
    
    # Validation Configuration
    VALIDATION_STRICT_MODE = True  # Production-like validation
    VALIDATION_CACHE_ENABLED = True
    VALIDATION_CACHE_TTL = 3600  # 1 hour
    
    # Monitoring Configuration
    METRICS_ENABLED = True
    PROMETHEUS_PORT = 9090
    GRAFANA_PORT = 3000
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "json"
    
    # File paths
    UPLOAD_PATH = str(DATA_DIR / "uploads")
    EXPORT_PATH = str(DATA_DIR / "exports")
    BACKUP_PATH = str(DATA_DIR / "backups")
    LOG_FILE_PATH = str(LOGS_DIR / "application.log")
    
    # Rate Limiting
    RATE_LIMIT_ENABLED = True
    RATE_LIMIT_REQUESTS_PER_MINUTE = 500
    RATE_LIMIT_BURST = 50
    
    # WebSocket Configuration
    WEBSOCKET_ENABLED = True
    WEBSOCKET_PING_INTERVAL = 30
    WEBSOCKET_PING_TIMEOUT = 10
    
    # External Services
    EXTERNAL_API_TIMEOUT = 30
    EXTERNAL_API_RETRIES = 3
    EXTERNAL_API_MOCK_MODE = False
    
    # Email Configuration
    SMTP_HOST = os.getenv("SMTP_HOST", "smtp.company.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME = os.getenv("SMTP_USERNAME")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
    SMTP_USE_TLS = True
    EMAIL_FROM = "staging@claims-processor.company.com"
    
    # Staging specific settings
    ENABLE_PROFILING = False
    ENABLE_DEBUG_TOOLBAR = False
    ENABLE_SQL_ECHO = False
    ENABLE_MOCK_DATA = False
    
    # Feature Flags (all enabled for staging testing)
    FEATURE_ML_PIPELINE = True
    FEATURE_REAL_TIME_UPDATES = True
    FEATURE_ADVANCED_ANALYTICS = True
    FEATURE_AUDIT_DASHBOARD = True
    
    # Performance Settings
    CONNECTION_POOL_SIZE = 20
    CONNECTION_POOL_MAX_OVERFLOW = 30
    CONNECTION_POOL_TIMEOUT = 30
    QUERY_TIMEOUT = 60
    
    # Caching
    CACHE_DEFAULT_TIMEOUT = 3600  # 1 hour
    CACHE_KEY_PREFIX = "claims_staging"
    
    # Backup Configuration
    BACKUP_ENABLED = True
    BACKUP_SCHEDULE = "0 2 * * *"  # Daily at 2 AM
    BACKUP_RETENTION_DAYS = 30
    
    # Alert Configuration
    ALERT_ENABLED = True
    ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL")
    ALERT_EMAIL_RECIPIENTS = os.getenv("ALERT_EMAIL_RECIPIENTS", "").split(",")
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate staging configuration."""
        required_env_vars = [
            'POSTGRES_PASSWORD', 'SQLSERVER_PASSWORD', 'SECRET_KEY', 
            'JWT_SECRET_KEY', 'ENCRYPTION_KEY'
        ]
        
        missing_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        # Validate security settings
        instance = cls()
        if len(instance.SECRET_KEY) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters")
        
        if len(instance.JWT_SECRET_KEY) < 32:
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters")
        
        if len(instance.ENCRYPTION_KEY) < 32:
            raise ValueError("ENCRYPTION_KEY must be at least 32 characters")
        
        return True
    
    @classmethod
    def get_database_settings(cls) -> dict:
        """Get database connection settings."""
        instance = cls()
        return {
            "url": instance.DATABASE_URL,
            "echo": instance.ENABLE_SQL_ECHO,
            "pool_size": instance.CONNECTION_POOL_SIZE,
            "max_overflow": instance.CONNECTION_POOL_MAX_OVERFLOW,
            "pool_timeout": instance.CONNECTION_POOL_TIMEOUT
        }
    
    @classmethod
    def get_redis_settings(cls) -> dict:
        """Get Redis connection settings."""
        instance = cls()
        return {
            "url": instance.REDIS_URL,
            "decode_responses": True,
            "health_check_interval": 30,
            "socket_keepalive": True,
            "socket_keepalive_options": {}
        }
    
    @classmethod
    def get_security_settings(cls) -> dict:
        """Get security settings."""
        instance = cls()
        return {
            "secret_key": instance.SECRET_KEY,
            "jwt_secret": instance.JWT_SECRET_KEY,
            "jwt_access_expire": instance.JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
            "jwt_refresh_expire": instance.JWT_REFRESH_TOKEN_EXPIRE_DAYS,
            "encryption_key": instance.ENCRYPTION_KEY
        }


# Export configuration instance
config = StagingConfig()
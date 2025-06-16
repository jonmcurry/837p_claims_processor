"""Production environment configuration."""

import os
from pathlib import Path
from typing import Optional, List

# Base paths for production
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = Path("/opt/claims-processor/data")
LOGS_DIR = Path("/opt/claims-processor/logs")
TEMP_DIR = Path("/opt/claims-processor/temp")

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

class ProductionConfig:
    """Production environment configuration - maximum security and performance."""
    
    # Environment
    ENVIRONMENT = "production"
    DEBUG = False
    TESTING = False
    
    # Database Configuration
    POSTGRES_HOST = os.getenv("POSTGRES_HOST")  # Required
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_DB = os.getenv("POSTGRES_DB", "claims_processor")
    POSTGRES_USER = os.getenv("POSTGRES_USER")  # Required
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")  # Required
    
    # SQL Server Configuration
    SQLSERVER_HOST = os.getenv("SQLSERVER_HOST")  # Required
    SQLSERVER_PORT = int(os.getenv("SQLSERVER_PORT", "1433"))
    SQLSERVER_DB = os.getenv("SQLSERVER_DB", "smart_pro_claims")
    SQLSERVER_USER = os.getenv("SQLSERVER_USER", "claims_analytics_user")  # Required
    SQLSERVER_PASSWORD = os.getenv("SQLSERVER_PASSWORD")  # Required
    
    # Connection URLs with SSL enforcement
    @property
    def DATABASE_URL(self) -> str:
        required_vars = [self.POSTGRES_HOST, self.POSTGRES_USER, self.POSTGRES_PASSWORD]
        if not all(required_vars):
            raise ValueError("POSTGRES_HOST, POSTGRES_USER, and POSTGRES_PASSWORD are required")
        return (f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@"
                f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}?"
                f"sslmode=require&sslcert=/opt/certs/client-cert.pem&"
                f"sslkey=/opt/certs/client-key.pem&sslrootcert=/opt/certs/ca-cert.pem")
    
    @property
    def ANALYTICS_DATABASE_URL(self) -> str:
        required_vars = [self.SQLSERVER_HOST, self.SQLSERVER_USER, self.SQLSERVER_PASSWORD]
        if not all(required_vars):
            raise ValueError("SQLSERVER_HOST, SQLSERVER_USER, and SQLSERVER_PASSWORD are required")
        return (f"mssql+pyodbc://{self.SQLSERVER_USER}:{self.SQLSERVER_PASSWORD}@"
                f"{self.SQLSERVER_HOST}:{self.SQLSERVER_PORT}/{self.SQLSERVER_DB}?"
                f"driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no")
    
    # Redis Configuration with cluster support
    REDIS_CLUSTER_NODES = os.getenv("REDIS_CLUSTER_NODES", "").split(",")
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")  # Required
    REDIS_SSL = os.getenv("REDIS_SSL", "true").lower() == "true"
    
    @property
    def REDIS_URL(self) -> str:
        if not self.REDIS_PASSWORD:
            raise ValueError("REDIS_PASSWORD is required in production")
        
        if self.REDIS_CLUSTER_NODES and self.REDIS_CLUSTER_NODES[0]:
            # Use first node for URL, cluster mode handled separately
            node = self.REDIS_CLUSTER_NODES[0]
            protocol = "rediss" if self.REDIS_SSL else "redis"
            return f"{protocol}://:{self.REDIS_PASSWORD}@{node}/0"
        
        # Fallback to single node
        host = os.getenv("REDIS_HOST", "redis-prod")
        port = os.getenv("REDIS_PORT", "6379")
        protocol = "rediss" if self.REDIS_SSL else "redis"
        return f"{protocol}://:{self.REDIS_PASSWORD}@{host}:{port}/0"
    
    # Security Configuration - All required from secure vault
    SECRET_KEY = os.getenv("SECRET_KEY")  # Required - 64+ chars
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")  # Required - 64+ chars
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 15  # Short expiry for production
    JWT_REFRESH_TOKEN_EXPIRE_DAYS = 1  # 24 hours max
    ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")  # Required - 32+ chars
    
    # Additional security keys
    API_KEY_ENCRYPTION_KEY = os.getenv("API_KEY_ENCRYPTION_KEY")  # Required
    AUDIT_LOG_SIGNING_KEY = os.getenv("AUDIT_LOG_SIGNING_KEY")  # Required
    
    # CORS settings (very restrictive)
    CORS_ORIGINS = [
        "https://claims.company.com",
        "https://admin.claims.company.com"
    ]
    CORS_ALLOW_CREDENTIALS = True
    CORS_ALLOW_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    CORS_ALLOW_HEADERS = ["Authorization", "Content-Type", "X-Request-ID"]
    
    # API Configuration
    API_V1_PREFIX = "/api/v1"
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    API_RELOAD = False
    API_WORKERS = int(os.getenv("API_WORKERS", "16"))  # Scale based on CPU
    
    # High-performance batch processing
    MAX_BATCH_SIZE = 100000  # Full production capacity
    MAX_CONCURRENT_BATCHES = 8
    BATCH_TIMEOUT_SECONDS = 900  # 15 minutes
    ENABLE_ASYNC_PROCESSING = True
    BATCH_PROCESSING_MEMORY_LIMIT_GB = 8
    
    # ML Model Configuration
    ML_MODEL_PATH = "/opt/claims-processor/models"
    ML_BATCH_SIZE = 5000  # Optimized for production
    ML_ENABLE_GPU = os.getenv("ML_ENABLE_GPU", "true").lower() == "true"
    ML_MODEL_CACHE_TTL = 86400  # 24 hours
    ML_MODEL_WARMUP_ENABLED = True
    
    # Validation Configuration
    VALIDATION_STRICT_MODE = True
    VALIDATION_CACHE_ENABLED = True
    VALIDATION_CACHE_TTL = 7200  # 2 hours
    VALIDATION_PARALLEL_WORKERS = 8
    
    # Monitoring Configuration
    METRICS_ENABLED = True
    PROMETHEUS_PORT = 9090
    GRAFANA_PORT = 3000
    LOG_LEVEL = "WARNING"  # Reduced logging for performance
    LOG_FORMAT = "json"
    
    # Advanced monitoring
    ENABLE_DISTRIBUTED_TRACING = True
    JAEGER_ENDPOINT = os.getenv("JAEGER_ENDPOINT")
    HEALTH_CHECK_INTERVAL = 30
    
    # File paths
    UPLOAD_PATH = str(DATA_DIR / "uploads")
    EXPORT_PATH = str(DATA_DIR / "exports")
    BACKUP_PATH = str(DATA_DIR / "backups")
    LOG_FILE_PATH = str(LOGS_DIR / "application.log")
    AUDIT_LOG_PATH = str(LOGS_DIR / "audit.log")
    
    # Rate Limiting (strict for production)
    RATE_LIMIT_ENABLED = True
    RATE_LIMIT_REQUESTS_PER_MINUTE = 100
    RATE_LIMIT_BURST = 20
    RATE_LIMIT_STORAGE = "redis"
    
    # WebSocket Configuration
    WEBSOCKET_ENABLED = True
    WEBSOCKET_PING_INTERVAL = 30
    WEBSOCKET_PING_TIMEOUT = 10
    WEBSOCKET_MAX_CONNECTIONS = 1000
    
    # External Services
    EXTERNAL_API_TIMEOUT = 15  # Shorter timeout for production
    EXTERNAL_API_RETRIES = 2
    EXTERNAL_API_MOCK_MODE = False
    
    # Email Configuration (production SMTP)
    SMTP_HOST = os.getenv("SMTP_HOST")  # Required
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME = os.getenv("SMTP_USERNAME")  # Required
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")  # Required
    SMTP_USE_TLS = True
    EMAIL_FROM = "noreply@claims-processor.company.com"
    
    # Production specific settings
    ENABLE_PROFILING = False
    ENABLE_DEBUG_TOOLBAR = False
    ENABLE_SQL_ECHO = False
    ENABLE_MOCK_DATA = False
    
    # Feature Flags
    FEATURE_ML_PIPELINE = True
    FEATURE_REAL_TIME_UPDATES = True
    FEATURE_ADVANCED_ANALYTICS = True
    FEATURE_AUDIT_DASHBOARD = True
    
    # High-performance settings
    CONNECTION_POOL_SIZE = 50
    CONNECTION_POOL_MAX_OVERFLOW = 100
    CONNECTION_POOL_TIMEOUT = 30
    CONNECTION_POOL_RECYCLE = 3600  # 1 hour
    QUERY_TIMEOUT = 120
    
    # Caching (aggressive for production)
    CACHE_DEFAULT_TIMEOUT = 7200  # 2 hours
    CACHE_KEY_PREFIX = "claims_prod"
    CACHE_COMPRESSION_ENABLED = True
    
    # Backup Configuration
    BACKUP_ENABLED = True
    BACKUP_SCHEDULE = "0 1 * * *"  # Daily at 1 AM
    BACKUP_RETENTION_DAYS = 90
    BACKUP_ENCRYPTION_ENABLED = True
    BACKUP_STORAGE_TYPE = "s3"  # or "azure", "gcp"
    BACKUP_STORAGE_BUCKET = os.getenv("BACKUP_STORAGE_BUCKET")
    
    # Disaster Recovery
    DR_ENABLED = True
    DR_REPLICA_HOSTS = os.getenv("DR_REPLICA_HOSTS", "").split(",")
    DR_SYNC_INTERVAL = 300  # 5 minutes
    
    # Alert Configuration
    ALERT_ENABLED = True
    ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL")
    ALERT_EMAIL_RECIPIENTS = os.getenv("ALERT_EMAIL_RECIPIENTS", "").split(",")
    ALERT_SMS_ENABLED = True
    ALERT_SMS_NUMBERS = os.getenv("ALERT_SMS_NUMBERS", "").split(",")
    
    # Compliance settings
    HIPAA_COMPLIANCE_MODE = True
    AUDIT_ALL_ACCESS = True
    DATA_RETENTION_DAYS = 2555  # 7 years
    AUTOMATIC_PHI_MASKING = True
    
    # Performance targets
    TARGET_THROUGHPUT_CLAIMS_PER_SECOND = 6667
    TARGET_RESPONSE_TIME_MS = 500
    TARGET_UPTIME_PERCENTAGE = 99.9
    
    # SSL/TLS Configuration
    SSL_CERT_PATH = "/opt/certs/server.crt"
    SSL_KEY_PATH = "/opt/certs/server.key"
    SSL_CA_PATH = "/opt/certs/ca.crt"
    SSL_VERIFY_MODE = "CERT_REQUIRED"
    
    @classmethod
    def validate_config(cls) -> bool:
        """Comprehensive production configuration validation."""
        
        # Required environment variables
        required_env_vars = [
            'POSTGRES_HOST', 'POSTGRES_USER', 'POSTGRES_PASSWORD',
            'SQLSERVER_HOST', 'SQLSERVER_USER', 'SQLSERVER_PASSWORD',
            'REDIS_PASSWORD', 'SECRET_KEY', 'JWT_SECRET_KEY', 'ENCRYPTION_KEY',
            'API_KEY_ENCRYPTION_KEY', 'AUDIT_LOG_SIGNING_KEY',
            'SMTP_HOST', 'SMTP_USERNAME', 'SMTP_PASSWORD'
        ]
        
        missing_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        # Security validation
        instance = cls()
        
        # Key length validation
        security_keys = {
            'SECRET_KEY': 64,
            'JWT_SECRET_KEY': 64,
            'ENCRYPTION_KEY': 32,
            'API_KEY_ENCRYPTION_KEY': 32,
            'AUDIT_LOG_SIGNING_KEY': 64
        }
        
        for key_name, min_length in security_keys.items():
            key_value = getattr(instance, key_name)
            if not key_value or len(key_value) < min_length:
                raise ValueError(f"{key_name} must be at least {min_length} characters")
        
        # SSL certificate validation
        ssl_files = [instance.SSL_CERT_PATH, instance.SSL_KEY_PATH, instance.SSL_CA_PATH]
        for ssl_file in ssl_files:
            if not Path(ssl_file).exists():
                raise ValueError(f"SSL file not found: {ssl_file}")
        
        # Performance validation
        if instance.API_WORKERS < 4:
            raise ValueError("API_WORKERS should be at least 4 for production")
        
        if instance.CONNECTION_POOL_SIZE < 20:
            raise ValueError("CONNECTION_POOL_SIZE should be at least 20 for production")
        
        return True
    
    @classmethod
    def get_database_settings(cls) -> dict:
        """Get production database connection settings."""
        instance = cls()
        return {
            "url": instance.DATABASE_URL,
            "echo": False,
            "pool_size": instance.CONNECTION_POOL_SIZE,
            "max_overflow": instance.CONNECTION_POOL_MAX_OVERFLOW,
            "pool_timeout": instance.CONNECTION_POOL_TIMEOUT,
            "pool_recycle": instance.CONNECTION_POOL_RECYCLE,
            "pool_pre_ping": True,
            "connect_args": {
                "command_timeout": instance.QUERY_TIMEOUT,
                "server_settings": {
                    "application_name": "claims_processor_prod"
                }
            }
        }
    
    @classmethod
    def get_redis_settings(cls) -> dict:
        """Get production Redis connection settings."""
        instance = cls()
        
        settings = {
            "url": instance.REDIS_URL,
            "decode_responses": True,
            "health_check_interval": 30,
            "socket_keepalive": True,
            "socket_keepalive_options": {
                "TCP_KEEPINTVL": 1,
                "TCP_KEEPCNT": 3,
                "TCP_KEEPIDLE": 1
            },
            "retry_on_timeout": True,
            "socket_connect_timeout": 5,
            "socket_timeout": 5
        }
        
        # Add cluster configuration if available
        if instance.REDIS_CLUSTER_NODES and instance.REDIS_CLUSTER_NODES[0]:
            settings["cluster_nodes"] = instance.REDIS_CLUSTER_NODES
            settings["skip_full_coverage_check"] = True
            settings["max_connections_per_node"] = 50
        
        return settings
    
    @classmethod
    def get_security_settings(cls) -> dict:
        """Get production security settings."""
        instance = cls()
        return {
            "secret_key": instance.SECRET_KEY,
            "jwt_secret": instance.JWT_SECRET_KEY,
            "jwt_access_expire": instance.JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
            "jwt_refresh_expire": instance.JWT_REFRESH_TOKEN_EXPIRE_DAYS,
            "encryption_key": instance.ENCRYPTION_KEY,
            "api_key_encryption_key": instance.API_KEY_ENCRYPTION_KEY,
            "audit_log_signing_key": instance.AUDIT_LOG_SIGNING_KEY,
            "ssl_cert_path": instance.SSL_CERT_PATH,
            "ssl_key_path": instance.SSL_KEY_PATH,
            "ssl_ca_path": instance.SSL_CA_PATH
        }


# Export configuration instance
config = ProductionConfig()
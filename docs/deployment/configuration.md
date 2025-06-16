# Configuration Management

This document describes the comprehensive configuration management system for the 837P Claims Processing System across different environments.

## Configuration Architecture

### Configuration Hierarchy

```mermaid
graph TB
    subgraph "Configuration Sources"
        ENV[Environment Variables]
        CONFIG[Config Files]
        SECRETS[Secrets Manager]
        DEFAULTS[Default Values]
    end
    
    subgraph "Configuration Loader"
        LOADER[Config Loader]
        VALIDATOR[Config Validator]
        MERGER[Config Merger]
    end
    
    subgraph "Application"
        APP[Application Code]
        COMPONENTS[Components]
    end
    
    ENV --> LOADER
    CONFIG --> LOADER
    SECRETS --> LOADER
    DEFAULTS --> LOADER
    
    LOADER --> VALIDATOR
    VALIDATOR --> MERGER
    MERGER --> APP
    APP --> COMPONENTS
```

### Configuration Precedence (Highest to Lowest)

1. **Environment Variables**: Runtime environment settings
2. **Secrets Manager**: Encrypted secrets and sensitive data
3. **Environment-Specific Files**: `config/.env.production`, `config/.env.staging`
4. **Base Configuration**: `config/.env.base`
5. **Default Values**: Hard-coded application defaults

## Environment-Specific Configuration

### Development Environment

```bash
# config/.env.development
# ===================================
# DEVELOPMENT ENVIRONMENT CONFIGURATION
# ===================================

# Environment
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Database Configuration
DATABASE_URL=postgresql://claims_user:dev_password@localhost:5432/claims_processor_dev
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20
DATABASE_ECHO=true  # SQL logging for development

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=
REDIS_MAX_CONNECTIONS=20

# API Configuration
API_HOST=127.0.0.1
API_PORT=8000
API_WORKERS=2
API_RELOAD=true  # Auto-reload on code changes
API_DEBUG=true

# Processing Configuration
BATCH_SIZE=100  # Smaller batches for development
MAX_WORKERS=2
ASYNC_WORKERS=2
ENABLE_ML_PIPELINE=false  # Disable ML for faster development

# Security Configuration (Development Only)
SECRET_KEY=development-secret-key-not-for-production
JWT_SECRET_KEY=development-jwt-secret
JWT_EXPIRATION_HOURS=24
ENCRYPTION_KEY=development-encryption-key-32-chars

# External Services (Mock/Test)
ENABLE_EXTERNAL_APIS=false
MOCK_EXTERNAL_SERVICES=true

# Monitoring
PROMETHEUS_ENABLED=false
GRAFANA_ENABLED=false
METRICS_COLLECTION_INTERVAL=60

# Caching
CACHE_TTL_SECONDS=300
ENABLE_CACHE_PRELOADING=false

# File Storage
UPLOAD_DIR=./uploads/dev/
MAX_FILE_SIZE_MB=10
```

### Staging Environment

```bash
# config/.env.staging
# ===================================
# STAGING ENVIRONMENT CONFIGURATION
# ===================================

# Environment
ENVIRONMENT=staging
DEBUG=false
LOG_LEVEL=INFO

# Database Configuration
DATABASE_URL=postgresql://claims_user:${DB_PASSWORD}@staging-db:5432/claims_processor_staging
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40
DATABASE_ECHO=false

# Redis Configuration
REDIS_URL=redis://:${REDIS_PASSWORD}@staging-redis:6379/0
REDIS_MAX_CONNECTIONS=50

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_RELOAD=false
API_DEBUG=false

# Processing Configuration
BATCH_SIZE=500
MAX_WORKERS=4
ASYNC_WORKERS=4
ENABLE_ML_PIPELINE=true

# Security Configuration
SECRET_KEY=${SECRET_KEY}
JWT_SECRET_KEY=${JWT_SECRET_KEY}
JWT_EXPIRATION_HOURS=8
ENCRYPTION_KEY=${ENCRYPTION_KEY}

# SSL/TLS Configuration
SSL_ENABLED=true
SSL_CERT_PATH=/etc/ssl/certs/staging.crt
SSL_KEY_PATH=/etc/ssl/private/staging.key

# External Services
ENABLE_EXTERNAL_APIS=true
MOCK_EXTERNAL_SERVICES=false
API_RATE_LIMIT_PER_MINUTE=1000

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
GRAFANA_ENABLED=true
GRAFANA_PORT=3000
METRICS_COLLECTION_INTERVAL=30

# Caching
CACHE_TTL_SECONDS=1800
ENABLE_CACHE_PRELOADING=true
CACHE_WARM_UP_ON_START=true

# File Storage
UPLOAD_DIR=/app/uploads/staging/
MAX_FILE_SIZE_MB=50

# Backup Configuration
ENABLE_AUTOMATED_BACKUPS=true
BACKUP_RETENTION_DAYS=7
BACKUP_SCHEDULE="0 2 * * *"  # Daily at 2 AM
```

### Production Environment

```bash
# config/.env.production
# ===================================
# PRODUCTION ENVIRONMENT CONFIGURATION
# ===================================

# Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING

# Database Configuration
DATABASE_URL=postgresql://claims_user:${DB_PASSWORD}@prod-db-cluster:5432/claims_processor
DATABASE_POOL_SIZE=50
DATABASE_MAX_OVERFLOW=100
DATABASE_ECHO=false
DATABASE_SSL_MODE=require

# Redis Configuration
REDIS_URL=redis://:${REDIS_PASSWORD}@prod-redis-cluster:6379/0
REDIS_MAX_CONNECTIONS=100
REDIS_SSL=true

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=16
API_RELOAD=false
API_DEBUG=false

# Processing Configuration
BATCH_SIZE=2000
MAX_WORKERS=20
ASYNC_WORKERS=8
ENABLE_ML_PIPELINE=true

# Security Configuration
SECRET_KEY=${SECRET_KEY}
JWT_SECRET_KEY=${JWT_SECRET_KEY}
JWT_EXPIRATION_HOURS=4
ENCRYPTION_KEY=${ENCRYPTION_KEY}

# SSL/TLS Configuration
SSL_ENABLED=true
SSL_CERT_PATH=/etc/ssl/certs/production.crt
SSL_KEY_PATH=/etc/ssl/private/production.key
TLS_VERSION=1.3

# External Services
ENABLE_EXTERNAL_APIS=true
MOCK_EXTERNAL_SERVICES=false
API_RATE_LIMIT_PER_MINUTE=10000

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
GRAFANA_ENABLED=true
GRAFANA_PORT=3000
METRICS_COLLECTION_INTERVAL=15
ENABLE_DISTRIBUTED_TRACING=true

# Caching
CACHE_TTL_SECONDS=3600
ENABLE_CACHE_PRELOADING=true
CACHE_WARM_UP_ON_START=true
CACHE_PRELOAD_MINUTES_AHEAD=30

# File Storage
UPLOAD_DIR=/mnt/claims_storage/uploads/
MAX_FILE_SIZE_MB=100
ENABLE_FILE_ENCRYPTION=true

# Backup Configuration
ENABLE_AUTOMATED_BACKUPS=true
BACKUP_RETENTION_DAYS=30
BACKUP_SCHEDULE="0 1,13 * * *"  # Twice daily
ENABLE_CROSS_REGION_BACKUP=true

# High Availability
ENABLE_CIRCUIT_BREAKER=true
CIRCUIT_BREAKER_FAILURE_THRESHOLD=10
CIRCUIT_BREAKER_RECOVERY_TIMEOUT=60

# Performance
ENABLE_REQUEST_COMPRESSION=true
ENABLE_RESPONSE_CACHING=true
CONNECTION_POOL_RECYCLE=3600
```

## Configuration Management System

### Configuration Loader

```python
# src/core/config/config_loader.py

import os
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"

@dataclass
class ConfigSection:
    """Represents a configuration section."""
    name: str
    required: bool = True
    sensitive: bool = False
    default: Optional[Any] = None
    validator: Optional[callable] = None

class ConfigurationManager:
    """Centralized configuration management."""
    
    def __init__(self, environment: Optional[str] = None):
        self.environment = Environment(environment or os.getenv('ENVIRONMENT', 'development'))
        self.config_dir = Path(__file__).parent.parent.parent.parent / 'config'
        self._config_cache = {}
        self._load_configuration()
    
    def _load_configuration(self):
        """Load configuration from multiple sources."""
        
        # Step 1: Load base configuration
        base_config = self._load_env_file(self.config_dir / '.env.base')
        
        # Step 2: Load environment-specific configuration
        env_config = self._load_env_file(
            self.config_dir / f'.env.{self.environment.value}'
        )
        
        # Step 3: Load secrets from environment or secrets manager
        secrets_config = self._load_secrets()
        
        # Step 4: Override with environment variables
        env_vars = self._load_environment_variables()
        
        # Step 5: Merge configurations (order matters)
        self._config_cache = {
            **base_config,
            **env_config,
            **secrets_config,
            **env_vars
        }
        
        # Step 6: Validate configuration
        self._validate_configuration()
    
    def _load_env_file(self, file_path: Path) -> Dict[str, str]:
        """Load configuration from .env file."""
        config = {}
        
        if not file_path.exists():
            return config
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip().strip('"\'')
        
        return config
    
    def _load_secrets(self) -> Dict[str, str]:
        """Load secrets from secrets manager or environment."""
        secrets = {}
        
        # Load from environment variables for secrets
        secret_keys = [
            'DB_PASSWORD',
            'REDIS_PASSWORD', 
            'SECRET_KEY',
            'JWT_SECRET_KEY',
            'ENCRYPTION_KEY',
            'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY'
        ]
        
        for key in secret_keys:
            value = os.getenv(key)
            if value:
                secrets[key] = value
        
        return secrets
    
    def _load_environment_variables(self) -> Dict[str, str]:
        """Load configuration from environment variables."""
        return dict(os.environ)
    
    def _validate_configuration(self):
        """Validate configuration values."""
        
        required_configs = [
            'DATABASE_URL',
            'REDIS_URL',
            'SECRET_KEY',
            'JWT_SECRET_KEY'
        ]
        
        missing_configs = []
        for config_key in required_configs:
            if not self.get(config_key):
                missing_configs.append(config_key)
        
        if missing_configs:
            raise ValueError(f"Missing required configuration: {missing_configs}")
        
        # Validate specific configurations
        self._validate_database_config()
        self._validate_security_config()
        self._validate_performance_config()
    
    def _validate_database_config(self):
        """Validate database configuration."""
        pool_size = self.get_int('DATABASE_POOL_SIZE', 20)
        max_overflow = self.get_int('DATABASE_MAX_OVERFLOW', 40)
        
        if pool_size <= 0:
            raise ValueError("DATABASE_POOL_SIZE must be positive")
        
        if max_overflow < pool_size:
            raise ValueError("DATABASE_MAX_OVERFLOW must be >= DATABASE_POOL_SIZE")
    
    def _validate_security_config(self):
        """Validate security configuration."""
        secret_key = self.get('SECRET_KEY')
        jwt_secret = self.get('JWT_SECRET_KEY')
        encryption_key = self.get('ENCRYPTION_KEY')
        
        if len(secret_key) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters")
        
        if len(jwt_secret) < 32:
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters")
        
        if encryption_key and len(encryption_key) != 32:
            raise ValueError("ENCRYPTION_KEY must be exactly 32 characters")
    
    def _validate_performance_config(self):
        """Validate performance configuration."""
        batch_size = self.get_int('BATCH_SIZE', 1000)
        max_workers = self.get_int('MAX_WORKERS', 8)
        
        if batch_size <= 0:
            raise ValueError("BATCH_SIZE must be positive")
        
        if max_workers <= 0:
            raise ValueError("MAX_WORKERS must be positive")
    
    def get(self, key: str, default: Any = None) -> str:
        """Get configuration value."""
        return self._config_cache.get(key, default)
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get configuration value as integer."""
        value = self.get(key, default)
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get configuration value as boolean."""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        return str(value).lower() in ('true', '1', 'yes', 'on')
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get configuration value as float."""
        value = self.get(key, default)
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def get_list(self, key: str, default: list = None, separator: str = ',') -> list:
        """Get configuration value as list."""
        value = self.get(key)
        if not value:
            return default or []
        return [item.strip() for item in value.split(separator)]
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return {
            'url': self.get('DATABASE_URL'),
            'pool_size': self.get_int('DATABASE_POOL_SIZE', 20),
            'max_overflow': self.get_int('DATABASE_MAX_OVERFLOW', 40),
            'pool_timeout': self.get_int('DATABASE_POOL_TIMEOUT', 30),
            'pool_recycle': self.get_int('DATABASE_POOL_RECYCLE', 3600),
            'echo': self.get_bool('DATABASE_ECHO', False),
            'ssl_mode': self.get('DATABASE_SSL_MODE', 'prefer')
        }
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration."""
        return {
            'url': self.get('REDIS_URL'),
            'password': self.get('REDIS_PASSWORD'),
            'max_connections': self.get_int('REDIS_MAX_CONNECTIONS', 50),
            'socket_timeout': self.get_int('REDIS_SOCKET_TIMEOUT', 30),
            'socket_connect_timeout': self.get_int('REDIS_SOCKET_CONNECT_TIMEOUT', 30),
            'retry_on_timeout': self.get_bool('REDIS_RETRY_ON_TIMEOUT', True),
            'ssl': self.get_bool('REDIS_SSL', False)
        }
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration."""
        return {
            'host': self.get('API_HOST', '0.0.0.0'),
            'port': self.get_int('API_PORT', 8000),
            'workers': self.get_int('API_WORKERS', 4),
            'reload': self.get_bool('API_RELOAD', False),
            'debug': self.get_bool('API_DEBUG', False),
            'ssl_enabled': self.get_bool('SSL_ENABLED', False),
            'ssl_cert_path': self.get('SSL_CERT_PATH'),
            'ssl_key_path': self.get('SSL_KEY_PATH')
        }
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration."""
        return {
            'secret_key': self.get('SECRET_KEY'),
            'jwt_secret_key': self.get('JWT_SECRET_KEY'),
            'jwt_expiration_hours': self.get_int('JWT_EXPIRATION_HOURS', 8),
            'encryption_key': self.get('ENCRYPTION_KEY'),
            'password_min_length': self.get_int('PASSWORD_MIN_LENGTH', 8),
            'session_timeout_minutes': self.get_int('SESSION_TIMEOUT_MINUTES', 480),
            'max_login_attempts': self.get_int('MAX_LOGIN_ATTEMPTS', 5),
            'lockout_duration_minutes': self.get_int('LOCKOUT_DURATION_MINUTES', 30)
        }
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        return {
            'prometheus_enabled': self.get_bool('PROMETHEUS_ENABLED', False),
            'prometheus_port': self.get_int('PROMETHEUS_PORT', 9090),
            'grafana_enabled': self.get_bool('GRAFANA_ENABLED', False),
            'grafana_port': self.get_int('GRAFANA_PORT', 3000),
            'metrics_collection_interval': self.get_int('METRICS_COLLECTION_INTERVAL', 30),
            'enable_distributed_tracing': self.get_bool('ENABLE_DISTRIBUTED_TRACING', False),
            'log_level': self.get('LOG_LEVEL', 'INFO'),
            'log_format': self.get('LOG_FORMAT', 'json')
        }


# Global configuration instance
config = ConfigurationManager()
```

## Secrets Management

### Development Environment
For development, secrets can be stored in environment files (not committed to git):

```bash
# config/.env.development.local (git-ignored)
DB_PASSWORD=dev_password_123
REDIS_PASSWORD=dev_redis_password
SECRET_KEY=development-secret-key-32-characters-long
JWT_SECRET_KEY=development-jwt-secret-key-32-chars
ENCRYPTION_KEY=development-encryption-key-32chars
```

### Production Environment
Use proper secrets management:

#### AWS Secrets Manager
```python
# Integration with AWS Secrets Manager
import boto3
from botocore.exceptions import ClientError

class AWSSecretsManager:
    def __init__(self, region_name='us-east-1'):
        self.client = boto3.client('secretsmanager', region_name=region_name)
    
    def get_secret(self, secret_name: str) -> Dict[str, Any]:
        """Retrieve secret from AWS Secrets Manager."""
        try:
            response = self.client.get_secret_value(SecretId=secret_name)
            return json.loads(response['SecretString'])
        except ClientError as e:
            logger.error(f"Failed to retrieve secret {secret_name}: {e}")
            raise
```

#### Kubernetes Secrets
```yaml
# k8s/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: claims-processor-secrets
type: Opaque
data:
  db-password: <base64-encoded-password>
  redis-password: <base64-encoded-password>
  secret-key: <base64-encoded-secret>
  jwt-secret-key: <base64-encoded-jwt-secret>
  encryption-key: <base64-encoded-encryption-key>
```

## Configuration Validation

### Schema Validation
```python
# config/schema.py
from marshmallow import Schema, fields, validate

class DatabaseConfigSchema(Schema):
    """Database configuration validation schema."""
    url = fields.Url(required=True)
    pool_size = fields.Integer(validate=validate.Range(min=1, max=100))
    max_overflow = fields.Integer(validate=validate.Range(min=0, max=200))
    pool_timeout = fields.Integer(validate=validate.Range(min=1, max=300))

class SecurityConfigSchema(Schema):
    """Security configuration validation schema."""
    secret_key = fields.String(required=True, validate=validate.Length(min=32))
    jwt_secret_key = fields.String(required=True, validate=validate.Length(min=32))
    encryption_key = fields.String(validate=validate.Length(equal=32))
    jwt_expiration_hours = fields.Integer(validate=validate.Range(min=1, max=72))

class APIConfigSchema(Schema):
    """API configuration validation schema."""
    host = fields.String(required=True)
    port = fields.Integer(validate=validate.Range(min=1024, max=65535))
    workers = fields.Integer(validate=validate.Range(min=1, max=32))
    debug = fields.Boolean()
```

## Configuration Templates

### Docker Compose Configuration
```yaml
# docker-compose.yml
version: '3.8'

services:
  claims_processor:
    build: .
    environment:
      - ENVIRONMENT=${ENVIRONMENT:-development}
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
      - SECRET_KEY=${SECRET_KEY}
    env_file:
      - config/.env.${ENVIRONMENT:-development}
    volumes:
      - ./logs:/app/logs
      - ./uploads:/app/uploads
    ports:
      - "${API_PORT:-8000}:8000"
    depends_on:
      - postgres
      - redis
```

### Kubernetes ConfigMap
```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: claims-processor-config
data:
  ENVIRONMENT: "production"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  API_WORKERS: "16"
  BATCH_SIZE: "2000"
  MAX_WORKERS: "20"
  PROMETHEUS_ENABLED: "true"
  GRAFANA_ENABLED: "true"
```

## Best Practices

### Configuration Security
1. **Never commit secrets** to version control
2. **Use environment-specific** configuration files
3. **Validate all configurations** at startup
4. **Use secrets managers** in production
5. **Rotate secrets regularly**

### Configuration Management
1. **Environment parity**: Keep environments similar
2. **Configuration as code**: Store configurations in version control
3. **Immutable deployments**: Don't modify running configurations
4. **Centralized management**: Use configuration management tools
5. **Documentation**: Document all configuration options

### Performance Considerations
1. **Cache configurations**: Avoid repeated file reads
2. **Lazy loading**: Load configurations when needed
3. **Validation caching**: Cache validation results
4. **Configuration hot-reloading**: For development environments

---

For related documentation, see:
- [Installation Guide](./installation.md)
- [Production Deployment](./production-deployment.md)
- [Security Configuration](../security/)
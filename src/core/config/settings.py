"""Application configuration and settings management."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, PostgresDsn, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation and type hints."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = Field(default="smart-claims-processor")
    app_env: str = Field(default="development")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_prefix: str = Field(default="/api/v1")
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    # PostgreSQL Configuration - Staging Database
    pg_host: str = Field(default="localhost")
    pg_port: int = Field(default=5432)
    pg_database: str = Field(default="claims_staging")
    pg_user: str = Field(default="claims_user")
    pg_password: SecretStr
    pg_pool_min: int = Field(default=10)
    pg_pool_max: int = Field(default=50)
    pg_pool_timeout: int = Field(default=30)
    pg_command_timeout: int = Field(default=60)

    # PostgreSQL Configuration - Production Database (formerly SQL Server)
    pg_prod_host: str = Field(default="localhost")
    pg_prod_port: int = Field(default=5432)
    pg_prod_database: str = Field(default="smart_pro_claims")
    pg_prod_user: str = Field(default="claims_user")
    pg_prod_password: SecretStr
    pg_prod_pool_min: int = Field(default=10)
    pg_prod_pool_max: int = Field(default=50)
    pg_prod_pool_timeout: int = Field(default=30)
    pg_prod_command_timeout: int = Field(default=120)

    # Memcached Configuration
    memcached_host: str = Field(default="localhost")
    memcached_port: int = Field(default=11211)
    memcached_pool_min: int = Field(default=10)
    memcached_pool_max: int = Field(default=50)
    memcached_ttl_seconds: int = Field(default=3600)

    # Security Configuration
    secret_key: SecretStr
    jwt_secret_key: SecretStr
    jwt_algorithm: str = Field(default="HS256")
    jwt_expiration_minutes: int = Field(default=30)
    encryption_key: SecretStr
    enable_mfa: bool = Field(default=True)

    # Processing Configuration
    batch_size: int = Field(default=500)
    worker_count: int = Field(default=8)
    max_retries: int = Field(default=5)
    retry_delay_seconds: int = Field(default=1)
    circuit_breaker_threshold: int = Field(default=5)
    circuit_breaker_timeout: int = Field(default=30)

    # ML Configuration
    ml_model_path: Path = Field(default=Path("/models/claims_filter_model.h5"))
    ml_prediction_threshold: float = Field(default=0.85)
    ml_batch_size: int = Field(default=100)

    # Performance Configuration
    enable_caching: bool = Field(default=True)
    cache_ttl_rvu: int = Field(default=3600)
    cache_ttl_facility: int = Field(default=7200)
    cache_ttl_rules: int = Field(default=14400)
    connection_pool_recycle: int = Field(default=3600)

    # Monitoring Configuration
    prometheus_port: int = Field(default=9090)
    enable_metrics: bool = Field(default=True)
    enable_tracing: bool = Field(default=True)
    jaeger_agent_host: str = Field(default="localhost")
    jaeger_agent_port: int = Field(default=6831)

    # Feature Flags
    enable_ml_predictions: bool = Field(default=True)
    enable_async_processing: bool = Field(default=True)
    enable_circuit_breaker: bool = Field(default=True)
    enable_rate_limiting: bool = Field(default=True)

    # Rate Limiting
    rate_limit_requests: int = Field(default=1000)
    rate_limit_period: int = Field(default=60)

    # File Storage
    failed_claims_path: Path = Field(default=Path("/data/failed_claims"))
    audit_logs_path: Path = Field(default=Path("/data/audit_logs"))
    export_path: Path = Field(default=Path("/data/exports"))

    # External Services
    cms_api_url: str = Field(default="https://api.cms.gov")
    cms_api_key: Optional[SecretStr] = Field(default=None)
    npi_registry_url: str = Field(default="https://npiregistry.cms.hhs.gov")

    # Performance Targets
    target_throughput: int = Field(default=6667)
    target_latency_p99: int = Field(default=100)
    sla_uptime: float = Field(default=99.9)

    # Notification Settings
    smtp_host: str = Field(default="localhost")
    smtp_port: int = Field(default=587)
    smtp_user: Optional[str] = Field(default=None)
    smtp_password: Optional[SecretStr] = Field(default=None)
    alert_email: str = Field(default="ops-team@company.com")

    @field_validator("app_env")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate application environment."""
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"app_env must be one of {allowed}")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper()
        if v not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v

    @property
    def postgres_url(self) -> str:
        """Construct PostgreSQL connection URL."""
        return (
            f"postgresql+asyncpg://{self.pg_user}:"
            f"{self.pg_password.get_secret_value()}@"
            f"{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )

    @property
    def postgres_prod_url(self) -> str:
        """Construct PostgreSQL production database connection URL."""
        return (
            f"postgresql+asyncpg://{self.pg_prod_user}:"
            f"{self.pg_prod_password.get_secret_value()}@"
            f"{self.pg_prod_host}:{self.pg_prod_port}/{self.pg_prod_database}"
        )

    @property
    def memcached_url(self) -> str:
        """Construct Memcached connection URL."""
        return f"{self.memcached_host}:{self.memcached_port}"

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.app_env == "development"

    def get_db_config(self, db_type: str) -> Dict[str, Any]:
        """Get database configuration by type."""
        if db_type == "postgresql" or db_type == "staging":
            return {
                "host": self.pg_host,
                "port": self.pg_port,
                "database": self.pg_database,
                "user": self.pg_user,
                "password": self.pg_password.get_secret_value(),
                "min_size": self.pg_pool_min,
                "max_size": self.pg_pool_max,
                "timeout": self.pg_pool_timeout,
                "command_timeout": self.pg_command_timeout,
            }
        elif db_type == "postgresql_prod" or db_type == "production":
            return {
                "host": self.pg_prod_host,
                "port": self.pg_prod_port,
                "database": self.pg_prod_database,
                "user": self.pg_prod_user,
                "password": self.pg_prod_password.get_secret_value(),
                "min_size": self.pg_prod_pool_min,
                "max_size": self.pg_prod_pool_max,
                "timeout": self.pg_prod_pool_timeout,
                "command_timeout": self.pg_prod_command_timeout,
            }
        else:
            raise ValueError(f"Unknown database type: {db_type}")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Export commonly used settings
settings = get_settings()
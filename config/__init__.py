"""Configuration management for claims processing system."""

import os
from typing import Union, Type

# Import all environment configurations
from .environments.development import DevelopmentConfig
from .environments.staging import StagingConfig
from .environments.production import ProductionConfig


class ConfigManager:
    """Central configuration manager for environment-specific settings."""
    
    _config_classes = {
        "development": DevelopmentConfig,
        "staging": StagingConfig,
        "production": ProductionConfig,
        "dev": DevelopmentConfig,  # Alias
        "prod": ProductionConfig,  # Alias
        "test": DevelopmentConfig,  # Use dev config for testing
    }
    
    def __init__(self):
        self._current_config = None
        self._environment = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration based on environment variable."""
        environment = os.getenv("ENVIRONMENT", "development").lower()
        self._environment = environment
        
        if environment not in self._config_classes:
            raise ValueError(
                f"Unknown environment: {environment}. "
                f"Valid options: {list(self._config_classes.keys())}"
            )
        
        config_class = self._config_classes[environment]
        self._current_config = config_class()
        
        # Validate configuration
        try:
            config_class.validate_config()
        except Exception as e:
            raise ValueError(f"Configuration validation failed for {environment}: {e}")
    
    @property
    def config(self) -> Union[DevelopmentConfig, StagingConfig, ProductionConfig]:
        """Get current configuration instance."""
        return self._current_config
    
    @property
    def environment(self) -> str:
        """Get current environment name."""
        return self._environment
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self._environment in ["development", "dev", "test"]
    
    def is_staging(self) -> bool:
        """Check if running in staging environment."""
        return self._environment == "staging"
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self._environment in ["production", "prod"]
    
    def get_database_url(self) -> str:
        """Get primary database URL."""
        return self._current_config.DATABASE_URL
    
    def get_analytics_database_url(self) -> str:
        """Get analytics database URL."""
        return self._current_config.ANALYTICS_DATABASE_URL
    
    def get_redis_url(self) -> str:
        """Get Redis URL."""
        return self._current_config.REDIS_URL
    
    def get_api_settings(self) -> dict:
        """Get API server settings."""
        return {
            "host": self._current_config.API_HOST,
            "port": self._current_config.API_PORT,
            "workers": getattr(self._current_config, "API_WORKERS", 1),
            "reload": getattr(self._current_config, "API_RELOAD", False),
            "debug": getattr(self._current_config, "DEBUG", False)
        }
    
    def get_security_settings(self) -> dict:
        """Get security settings."""
        return self._current_config.get_security_settings()
    
    def get_database_settings(self) -> dict:
        """Get database connection settings."""
        return self._current_config.get_database_settings()
    
    def get_redis_settings(self) -> dict:
        """Get Redis connection settings."""
        return self._current_config.get_redis_settings()
    
    def get_monitoring_settings(self) -> dict:
        """Get monitoring and metrics settings."""
        return {
            "metrics_enabled": getattr(self._current_config, "METRICS_ENABLED", True),
            "log_level": getattr(self._current_config, "LOG_LEVEL", "INFO"),
            "log_format": getattr(self._current_config, "LOG_FORMAT", "detailed"),
            "prometheus_port": getattr(self._current_config, "PROMETHEUS_PORT", 9090),
            "enable_tracing": getattr(self._current_config, "ENABLE_DISTRIBUTED_TRACING", False)
        }
    
    def get_processing_settings(self) -> dict:
        """Get batch processing settings."""
        return {
            "max_batch_size": getattr(self._current_config, "MAX_BATCH_SIZE", 10000),
            "max_concurrent_batches": getattr(self._current_config, "MAX_CONCURRENT_BATCHES", 2),
            "batch_timeout": getattr(self._current_config, "BATCH_TIMEOUT_SECONDS", 300),
            "enable_async": getattr(self._current_config, "ENABLE_ASYNC_PROCESSING", True),
            "memory_limit_gb": getattr(self._current_config, "BATCH_PROCESSING_MEMORY_LIMIT_GB", 4)
        }
    
    def get_ml_settings(self) -> dict:
        """Get ML pipeline settings."""
        return {
            "model_path": getattr(self._current_config, "ML_MODEL_PATH", "/tmp/models"),
            "batch_size": getattr(self._current_config, "ML_BATCH_SIZE", 100),
            "enable_gpu": getattr(self._current_config, "ML_ENABLE_GPU", False),
            "cache_ttl": getattr(self._current_config, "ML_MODEL_CACHE_TTL", 3600),
            "warmup_enabled": getattr(self._current_config, "ML_MODEL_WARMUP_ENABLED", False)
        }
    
    def get_feature_flags(self) -> dict:
        """Get feature flags."""
        return {
            "ml_pipeline": getattr(self._current_config, "FEATURE_ML_PIPELINE", True),
            "real_time_updates": getattr(self._current_config, "FEATURE_REAL_TIME_UPDATES", True),
            "advanced_analytics": getattr(self._current_config, "FEATURE_ADVANCED_ANALYTICS", True),
            "audit_dashboard": getattr(self._current_config, "FEATURE_AUDIT_DASHBOARD", True)
        }
    
    def reload_config(self):
        """Reload configuration (useful for config changes without restart)."""
        self._load_config()


# Global configuration manager instance
config_manager = ConfigManager()

# Convenience exports
config = config_manager.config
settings = config_manager.config  # Alias for backward compatibility

# Environment check functions
def is_development() -> bool:
    return config_manager.is_development()

def is_staging() -> bool:
    return config_manager.is_staging()

def is_production() -> bool:
    return config_manager.is_production()

def get_environment() -> str:
    return config_manager.environment


# Export configuration classes for direct access if needed
__all__ = [
    "ConfigManager",
    "config_manager",
    "config",
    "settings",
    "DevelopmentConfig",
    "StagingConfig", 
    "ProductionConfig",
    "is_development",
    "is_staging",
    "is_production",
    "get_environment"
]
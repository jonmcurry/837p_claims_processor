"""Redis-based caching system for high-performance data access."""

import json
import pickle
from typing import Any, Dict, List, Optional, Union

import redis.asyncio as redis
import structlog
from redis.asyncio import ConnectionPool

from src.core.config import settings

logger = structlog.get_logger(__name__)


class CacheManager:
    """High-performance Redis cache manager."""

    def __init__(self):
        """Initialize cache manager."""
        self.redis_client: Optional[redis.Redis] = None
        self.connection_pool: Optional[ConnectionPool] = None
        self.is_connected = False

    async def connect(self) -> None:
        """Establish Redis connection."""
        try:
            # Create connection pool
            self.connection_pool = ConnectionPool.from_url(
                settings.redis_url,
                max_connections=settings.redis_pool_max,
                retry_on_timeout=True,
                health_check_interval=30,
            )
            
            # Create Redis client
            self.redis_client = redis.Redis(
                connection_pool=self.connection_pool,
                decode_responses=False,  # We'll handle encoding ourselves for binary data
            )
            
            # Test connection
            await self.redis_client.ping()
            self.is_connected = True
            
            logger.info("Redis connection established", 
                       host=settings.redis_host, 
                       port=settings.redis_port)
            
        except Exception as e:
            logger.exception("Failed to connect to Redis", error=str(e))
            self.is_connected = False
            raise

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
        if self.connection_pool:
            await self.connection_pool.disconnect()
        self.is_connected = False
        logger.info("Redis connection closed")

    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        if not self.is_connected:
            return default
            
        try:
            value = await self.redis_client.get(key)
            if value is None:
                return default
                
            # Try to deserialize as JSON first, then pickle
            try:
                return json.loads(value.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return pickle.loads(value)
                
        except Exception as e:
            logger.warning("Cache get failed", key=key, error=str(e))
            return default

    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None, 
        serialize_json: bool = True
    ) -> bool:
        """Set value in cache."""
        if not self.is_connected:
            return False
            
        try:
            # Serialize value
            if serialize_json:
                try:
                    serialized_value = json.dumps(value).encode('utf-8')
                except (TypeError, ValueError):
                    # Fallback to pickle for complex objects
                    serialized_value = pickle.dumps(value)
                    serialize_json = False
            else:
                serialized_value = pickle.dumps(value)
            
            # Set with TTL
            ttl = ttl or settings.redis_ttl_seconds
            await self.redis_client.setex(key, ttl, serialized_value)
            
            logger.debug("Cache set successful", 
                        key=key, 
                        ttl=ttl, 
                        serialization="json" if serialize_json else "pickle")
            return True
            
        except Exception as e:
            logger.warning("Cache set failed", key=key, error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self.is_connected:
            return False
            
        try:
            result = await self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            logger.warning("Cache delete failed", key=key, error=str(e))
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.is_connected:
            return False
            
        try:
            return await self.redis_client.exists(key) > 0
        except Exception as e:
            logger.warning("Cache exists check failed", key=key, error=str(e))
            return False

    async def mget(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        if not self.is_connected or not keys:
            return {}
            
        try:
            values = await self.redis_client.mget(keys)
            result = {}
            
            for key, value in zip(keys, values):
                if value is not None:
                    try:
                        result[key] = json.loads(value.decode('utf-8'))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        result[key] = pickle.loads(value)
                        
            return result
            
        except Exception as e:
            logger.warning("Cache mget failed", keys=keys, error=str(e))
            return {}

    async def mset(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values in cache."""
        if not self.is_connected or not mapping:
            return False
            
        try:
            # Serialize all values
            serialized_mapping = {}
            for key, value in mapping.items():
                try:
                    serialized_mapping[key] = json.dumps(value).encode('utf-8')
                except (TypeError, ValueError):
                    serialized_mapping[key] = pickle.dumps(value)
            
            # Set all values
            await self.redis_client.mset(serialized_mapping)
            
            # Set TTL for each key if specified
            if ttl:
                ttl = ttl or settings.redis_ttl_seconds
                for key in mapping.keys():
                    await self.redis_client.expire(key, ttl)
            
            logger.debug("Cache mset successful", key_count=len(mapping), ttl=ttl)
            return True
            
        except Exception as e:
            logger.warning("Cache mset failed", error=str(e))
            return False

    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment counter in cache."""
        if not self.is_connected:
            return None
            
        try:
            return await self.redis_client.incrby(key, amount)
        except Exception as e:
            logger.warning("Cache increment failed", key=key, error=str(e))
            return None

    async def set_with_lock(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set value with distributed lock."""
        lock_key = f"lock:{key}"
        
        try:
            # Try to acquire lock
            lock_acquired = await self.redis_client.set(
                lock_key, 
                "locked", 
                nx=True, 
                ex=ttl
            )
            
            if not lock_acquired:
                return False
                
            # Set the actual value
            success = await self.set(key, value, ttl)
            
            # Release lock
            await self.redis_client.delete(lock_key)
            
            return success
            
        except Exception as e:
            logger.warning("Cache set with lock failed", key=key, error=str(e))
            return False

    # Domain-specific cache methods

    async def get_rvu_data(self, procedure_code: str) -> Optional[Dict[str, Any]]:
        """Get RVU data for a procedure code."""
        key = f"rvu:{procedure_code}"
        return await self.get(key)

    async def set_rvu_data(self, procedure_code: str, rvu_data: Dict[str, Any]) -> bool:
        """Cache RVU data for a procedure code."""
        key = f"rvu:{procedure_code}"
        return await self.set(key, rvu_data, ttl=settings.cache_ttl_rvu)

    async def get_facility_info(self, facility_id: str) -> Optional[Dict[str, Any]]:
        """Get facility information."""
        key = f"facility:{facility_id}"
        return await self.get(key)

    async def set_facility_info(self, facility_id: str, facility_data: Dict[str, Any]) -> bool:
        """Cache facility information."""
        key = f"facility:{facility_id}"
        return await self.set(key, facility_data, ttl=settings.cache_ttl_facility)

    async def get_npi_info(self, npi: str) -> Optional[Dict[str, Any]]:
        """Get NPI registry information."""
        key = f"npi:{npi}"
        return await self.get(key)

    async def set_npi_info(self, npi: str, npi_data: Dict[str, Any]) -> bool:
        """Cache NPI registry information."""
        key = f"npi:{npi}"
        return await self.set(key, npi_data, ttl=86400)  # 24 hours

    async def get_cpt_info(self, cpt_code: str) -> Optional[Dict[str, Any]]:
        """Get CPT code information."""
        key = f"cpt:{cpt_code}"
        return await self.get(key)

    async def set_cpt_info(self, cpt_code: str, cpt_data: Dict[str, Any]) -> bool:
        """Cache CPT code information."""
        key = f"cpt:{cpt_code}"
        return await self.set(key, cpt_data, ttl=settings.cache_ttl_rules)

    async def get_icd10_info(self, icd_code: str) -> Optional[Dict[str, Any]]:
        """Get ICD-10 code information."""
        key = f"icd10:{icd_code}"
        return await self.get(key)

    async def set_icd10_info(self, icd_code: str, icd_data: Dict[str, Any]) -> bool:
        """Cache ICD-10 code information."""
        key = f"icd10:{icd_code}"
        return await self.set(key, icd_data, ttl=settings.cache_ttl_rules)

    async def get_business_rules(self, facility_id: Optional[str] = None) -> Optional[List[Dict]]:
        """Get business rules for validation."""
        key = f"rules:{facility_id}" if facility_id else "rules:global"
        return await self.get(key)

    async def set_business_rules(self, rules: List[Dict], facility_id: Optional[str] = None) -> bool:
        """Cache business rules."""
        key = f"rules:{facility_id}" if facility_id else "rules:global"
        return await self.set(key, rules, ttl=settings.cache_ttl_rules)

    async def get_ml_model_result(self, claim_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached ML model prediction result."""
        key = f"ml_result:{claim_hash}"
        return await self.get(key)

    async def set_ml_model_result(self, claim_hash: str, result: Dict[str, Any]) -> bool:
        """Cache ML model prediction result."""
        key = f"ml_result:{claim_hash}"
        return await self.set(key, result, ttl=86400)  # 24 hours

    async def warm_cache(self) -> None:
        """Warm up cache with frequently accessed data."""
        logger.info("Starting cache warm-up process")
        
        try:
            # Warm up common RVU data
            await self._warm_rvu_cache()
            
            # Warm up facility data
            await self._warm_facility_cache()
            
            # Warm up business rules
            await self._warm_rules_cache()
            
            logger.info("Cache warm-up completed successfully")
            
        except Exception as e:
            logger.exception("Cache warm-up failed", error=str(e))

    async def _warm_rvu_cache(self) -> None:
        """Warm up RVU data cache."""
        # Common procedure codes that should be cached
        common_procedures = [
            "99213", "99214", "99215", "99223", "99232", "99233",
            "99283", "99284", "99285", "99291", "99292"
        ]
        
        # This would typically load from database
        # For now, we'll set placeholder data
        for proc_code in common_procedures:
            rvu_data = {
                "procedure_code": proc_code,
                "work_rvu": 1.0,
                "practice_expense_rvu": 1.0,
                "malpractice_rvu": 0.1,
                "status": "active"
            }
            await self.set_rvu_data(proc_code, rvu_data)

    async def _warm_facility_cache(self) -> None:
        """Warm up facility data cache."""
        # This would typically load active facilities from database
        pass

    async def _warm_rules_cache(self) -> None:
        """Warm up business rules cache."""
        # This would typically load active business rules from database
        pass

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        if not self.is_connected:
            return {"status": "disconnected"}
            
        try:
            info = await self.redis_client.info()
            
            return {
                "status": "connected",
                "redis_version": info.get("redis_version"),
                "connected_clients": info.get("connected_clients"),
                "used_memory_human": info.get("used_memory_human"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0),
                    info.get("keyspace_misses", 0)
                ),
                "total_commands_processed": info.get("total_commands_processed"),
            }
            
        except Exception as e:
            logger.warning("Failed to get cache stats", error=str(e))
            return {"status": "error", "error": str(e)}

    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate percentage."""
        total = hits + misses
        if total == 0:
            return 0.0
        return (hits / total) * 100


# Global cache manager instance
cache_manager = CacheManager()
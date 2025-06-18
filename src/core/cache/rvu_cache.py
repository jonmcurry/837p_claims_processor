"""High-performance RVU cache system with preloading and batch operations."""

import asyncio
import logging
import time
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple

import redis.asyncio as redis
from sqlalchemy import text

from src.core.config.settings import settings
from src.core.database.pool_manager import pool_manager

logger = logging.getLogger(__name__)


class RVUCacheData:
    """Optimized RVU data structure for caching."""
    
    def __init__(
        self,
        procedure_code: str,
        work_rvu: Decimal,
        practice_expense_rvu: Decimal,
        malpractice_rvu: Decimal,
        total_rvu: Optional[Decimal] = None,
        effective_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        self.procedure_code = procedure_code
        self.work_rvu = Decimal(str(work_rvu))
        self.practice_expense_rvu = Decimal(str(practice_expense_rvu))
        self.malpractice_rvu = Decimal(str(malpractice_rvu))
        self.total_rvu = total_rvu or (self.work_rvu + self.practice_expense_rvu + self.malpractice_rvu)
        self.effective_date = effective_date
        self.end_date = end_date
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for caching."""
        return {
            'procedure_code': self.procedure_code,
            'work_rvu': str(self.work_rvu),
            'practice_expense_rvu': str(self.practice_expense_rvu),
            'malpractice_rvu': str(self.malpractice_rvu),
            'total_rvu': str(self.total_rvu),
            'effective_date': self.effective_date,
            'end_date': self.end_date,
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'RVUCacheData':
        """Create from dictionary."""
        return cls(
            procedure_code=data['procedure_code'],
            work_rvu=Decimal(data['work_rvu']),
            practice_expense_rvu=Decimal(data['practice_expense_rvu']),
            malpractice_rvu=Decimal(data['malpractice_rvu']),
            total_rvu=Decimal(data['total_rvu']) if data.get('total_rvu') else None,
            effective_date=data.get('effective_date'),
            end_date=data.get('end_date'),
        )


class OptimizedRVUCache:
    """High-performance RVU cache with preloading and batch operations."""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.local_cache: Dict[str, RVUCacheData] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.batch_size = 1000
        self.preload_complete = False
        self._cache_lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize Redis connection and preload cache."""
        if self.redis_client:
            return
            
        logger.info("Initializing RVU cache system...")
        start_time = time.time()
        
        try:
            # Connect to Redis with shorter timeout for faster fallback
            self.redis_client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20,  # Reduced connections
                retry_on_timeout=False,  # Faster fallback
                socket_keepalive=False,  # Disable for faster startup
                socket_connect_timeout=1,  # 1 second timeout
            )
            
            # Test connection with timeout
            await asyncio.wait_for(self.redis_client.ping(), timeout=1.0)
            logger.info("Redis connection established")
            
            # Preload RVU data
            await self._preload_rvu_data()
            
            initialization_time = time.time() - start_time
            logger.info(f"RVU cache initialized in {initialization_time:.2f}s, "
                       f"loaded {len(self.local_cache)} procedure codes")
            
        except Exception as e:
            logger.error(f"Failed to initialize RVU cache: {e}")
            # Continue without Redis - use local cache only
            self.redis_client = None
            # Load defaults immediately instead of database query for faster startup
            self._load_default_rvus()
            self.preload_complete = True
            
    async def _preload_rvu_data(self):
        """Preload commonly used RVU data into cache."""
        logger.info("Preloading RVU data...")
        
        try:
            # Try loading from Redis first
            if self.redis_client:
                await self._preload_from_redis()
                
            # If Redis cache is empty, load from database in background
            if not self.local_cache:
                # Start with defaults for immediate availability
                self._load_default_rvus()
                # Then load additional data from database asynchronously
                asyncio.create_task(self._preload_from_database_background())
                
            self.preload_complete = True
            logger.info(f"RVU preload complete: {len(self.local_cache)} codes cached")
            
        except Exception as e:
            logger.error(f"RVU preload failed: {e}")
            # Load defaults as fallback
            self._load_default_rvus()
            
    async def _preload_from_redis(self):
        """Load RVU data from Redis cache."""
        if not self.redis_client:
            return
            
        try:
            # Get all RVU keys
            rvu_keys = await self.redis_client.keys("rvu:*")
            
            if rvu_keys:
                # Batch load from Redis
                pipeline = self.redis_client.pipeline()
                for key in rvu_keys:
                    pipeline.hgetall(key)
                    
                results = await pipeline.execute()
                
                for key, data in zip(rvu_keys, results):
                    if data:
                        procedure_code = key.replace("rvu:", "")
                        rvu_data = RVUCacheData.from_dict(data)
                        self.local_cache[procedure_code] = rvu_data
                        
                logger.info(f"Loaded {len(self.local_cache)} RVU codes from Redis")
                
        except Exception as e:
            logger.warning(f"Failed to preload from Redis: {e}")
            
    async def _preload_from_database(self):
        """Load RVU data from PostgreSQL database."""
        logger.info("Loading RVU data from database...")
        
        try:
            async with pool_manager.get_postgres_session() as session:
                # Check if rvu_data table exists first
                table_check = text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'rvu_data'
                    )
                """)
                
                table_exists = await session.execute(table_check)
                if not table_exists.scalar():
                    logger.warning("RVU table not found, using default values")
                    self._load_default_rvus()
                    return
                
                # Load only most common RVU data for faster startup
                query = text("""
                    SELECT 
                        procedure_code,
                        work_rvu,
                        practice_expense_rvu,
                        malpractice_rvu,
                        total_rvu,
                        effective_date,
                        end_date
                    FROM rvu_data
                    WHERE (end_date IS NULL OR end_date >= CURRENT_DATE)
                    AND procedure_code IN ('99213', '99214', '99215', '99223', '99233', 
                                          '99281', '99282', '99283', '99284', '99285',
                                          '10060', '36415', '71020', '85025', '80053')
                    ORDER BY procedure_code, effective_date DESC
                    LIMIT 50
                """)
                
                result = await session.execute(query)
                rows = result.fetchall()
                
                if rows:
                    # Process in batches to avoid memory issues
                    for i in range(0, len(rows), self.batch_size):
                        batch = rows[i:i + self.batch_size]
                        await self._process_rvu_batch(batch)
                else:
                    logger.info("No RVU data found in database, using defaults")
                    self._load_default_rvus()
                    
                logger.info(f"Loaded {len(self.local_cache)} RVU codes from database")
                
                # Cache to Redis for next time
                if self.redis_client:
                    await self._cache_batch_to_redis(list(self.local_cache.values()))
                    
        except Exception as e:
            logger.error(f"Failed to load RVU data from database: {e}")
            self._load_default_rvus()
            
    async def _preload_from_database_background(self):
        """Load RVU data from database in background without blocking startup."""
        try:
            logger.info("Loading additional RVU data from database in background...")
            await self._preload_from_database()
        except Exception as e:
            logger.warning(f"Background RVU loading failed: {e}")
            
    async def _process_rvu_batch(self, batch: List):
        """Process a batch of RVU data from database."""
        for row in batch:
            procedure_code = row[0]
            
            # Only store the most recent effective RVU data per code
            if procedure_code not in self.local_cache:
                rvu_data = RVUCacheData(
                    procedure_code=procedure_code,
                    work_rvu=row[1] or Decimal('0'),
                    practice_expense_rvu=row[2] or Decimal('0'),
                    malpractice_rvu=row[3] or Decimal('0'),
                    total_rvu=row[4],
                    effective_date=str(row[5]) if row[5] else None,
                    end_date=str(row[6]) if row[6] else None,
                )
                self.local_cache[procedure_code] = rvu_data
                
    async def _cache_batch_to_redis(self, rvu_data_list: List[RVUCacheData]):
        """Cache a batch of RVU data to Redis."""
        if not self.redis_client:
            return
            
        try:
            pipeline = self.redis_client.pipeline()
            
            for rvu_data in rvu_data_list:
                key = f"rvu:{rvu_data.procedure_code}"
                pipeline.hset(key, mapping=rvu_data.to_dict())
                pipeline.expire(key, settings.cache_ttl_rvu)
                
            await pipeline.execute()
            logger.debug(f"Cached {len(rvu_data_list)} RVU codes to Redis")
            
        except Exception as e:
            logger.warning(f"Failed to cache RVU data to Redis: {e}")
            
    def _load_default_rvus(self):
        """Load default RVU values for common procedure codes."""
        logger.info("Loading default RVU values")
        
        default_rvus = {
            # Most common E&M codes (optimized set for faster startup)
            "99213": RVUCacheData("99213", Decimal("0.97"), Decimal("0.85"), Decimal("0.04")),
            "99214": RVUCacheData("99214", Decimal("1.50"), Decimal("1.30"), Decimal("0.06")),
            "99223": RVUCacheData("99223", Decimal("3.05"), Decimal("1.20"), Decimal("0.12")),
            "99233": RVUCacheData("99233", Decimal("1.93"), Decimal("1.10"), Decimal("0.08")),
            
            # Most common ED codes
            "99283": RVUCacheData("99283", Decimal("1.42"), Decimal("1.95"), Decimal("0.07")),
            "99284": RVUCacheData("99284", Decimal("2.60"), Decimal("3.52"), Decimal("0.12")),
            
            # Common procedures
            "36415": RVUCacheData("36415", Decimal("0.17"), Decimal("0.13"), Decimal("0.01")),  # Venipuncture
            "71020": RVUCacheData("71020", Decimal("0.22"), Decimal("0.67"), Decimal("0.01")),  # Chest X-ray
            "85025": RVUCacheData("85025", Decimal("0.0"), Decimal("0.28"), Decimal("0.0")),    # CBC
            "80053": RVUCacheData("80053", Decimal("0.0"), Decimal("0.31"), Decimal("0.0")),    # Comprehensive metabolic
        }
        
        self.local_cache.update(default_rvus)
        logger.info(f"Loaded {len(default_rvus)} default RVU codes")
        
    async def get_rvu_data(self, procedure_code: str) -> Optional[RVUCacheData]:
        """Get RVU data for a single procedure code."""
        # Check local cache first
        if procedure_code in self.local_cache:
            self.cache_hits += 1
            return self.local_cache[procedure_code]
            
        self.cache_misses += 1
        
        # Try Redis if local cache misses
        if self.redis_client:
            try:
                data = await self.redis_client.hgetall(f"rvu:{procedure_code}")
                if data:
                    rvu_data = RVUCacheData.from_dict(data)
                    # Cache locally for future access
                    async with self._cache_lock:
                        self.local_cache[procedure_code] = rvu_data
                    return rvu_data
            except Exception as e:
                logger.warning(f"Redis lookup failed for {procedure_code}: {e}")
                
        # Fallback to database lookup
        return await self._lookup_from_database(procedure_code)
        
    async def get_batch_rvu_data(self, procedure_codes: List[str]) -> Dict[str, RVUCacheData]:
        """Get RVU data for multiple procedure codes efficiently."""
        results = {}
        missing_codes = []
        
        # Check local cache for all codes
        for code in procedure_codes:
            if code in self.local_cache:
                results[code] = self.local_cache[code]
                self.cache_hits += 1
            else:
                missing_codes.append(code)
                self.cache_misses += 1
                
        # Batch lookup missing codes
        if missing_codes:
            missing_results = await self._batch_lookup_missing_codes(missing_codes)
            results.update(missing_results)
            
        return results
        
    async def _batch_lookup_missing_codes(self, procedure_codes: List[str]) -> Dict[str, RVUCacheData]:
        """Batch lookup missing procedure codes from Redis and database."""
        results = {}
        
        # Try Redis first for all missing codes
        if self.redis_client and procedure_codes:
            try:
                pipeline = self.redis_client.pipeline()
                for code in procedure_codes:
                    pipeline.hgetall(f"rvu:{code}")
                    
                redis_results = await pipeline.execute()
                
                still_missing = []
                for code, data in zip(procedure_codes, redis_results):
                    if data:
                        rvu_data = RVUCacheData.from_dict(data)
                        results[code] = rvu_data
                        # Cache locally
                        async with self._cache_lock:
                            self.local_cache[code] = rvu_data
                    else:
                        still_missing.append(code)
                        
                procedure_codes = still_missing
                
            except Exception as e:
                logger.warning(f"Batch Redis lookup failed: {e}")
                
        # Fallback to database for remaining codes
        if procedure_codes:
            db_results = await self._batch_lookup_from_database(procedure_codes)
            results.update(db_results)
            
        return results
        
    async def _lookup_from_database(self, procedure_code: str) -> Optional[RVUCacheData]:
        """Lookup single RVU data from database."""
        try:
            async with pool_manager.get_postgres_session() as session:
                query = text("""
                    SELECT 
                        procedure_code,
                        work_rvu,
                        practice_expense_rvu,
                        malpractice_rvu,
                        total_rvu,
                        effective_date,
                        end_date
                    FROM rvu_data
                    WHERE procedure_code = :code
                    AND (end_date IS NULL OR end_date >= CURRENT_DATE)
                    ORDER BY effective_date DESC
                    LIMIT 1
                """)
                
                result = await session.execute(query, {"code": procedure_code})
                row = result.fetchone()
                
                if row:
                    rvu_data = RVUCacheData(
                        procedure_code=row[0],
                        work_rvu=row[1] or Decimal('0'),
                        practice_expense_rvu=row[2] or Decimal('0'),
                        malpractice_rvu=row[3] or Decimal('0'),
                        total_rvu=row[4],
                        effective_date=str(row[5]) if row[5] else None,
                        end_date=str(row[6]) if row[6] else None,
                    )
                    
                    # Cache the result
                    async with self._cache_lock:
                        self.local_cache[procedure_code] = rvu_data
                        
                    # Cache to Redis
                    if self.redis_client:
                        try:
                            key = f"rvu:{procedure_code}"
                            await self.redis_client.hset(key, mapping=rvu_data.to_dict())
                            await self.redis_client.expire(key, settings.cache_ttl_rvu)
                        except Exception as e:
                            logger.warning(f"Failed to cache to Redis: {e}")
                            
                    return rvu_data
                    
        except Exception as e:
            logger.error(f"Database lookup failed for {procedure_code}: {e}")
            
        return None
        
    async def _batch_lookup_from_database(self, procedure_codes: List[str]) -> Dict[str, RVUCacheData]:
        """Batch lookup RVU data from database."""
        results = {}
        
        try:
            async with pool_manager.get_postgres_session() as session:
                # Use parameterized query with IN clause
                placeholders = ','.join([f':code_{i}' for i in range(len(procedure_codes))])
                params = {f'code_{i}': code for i, code in enumerate(procedure_codes)}
                
                query = text(f"""
                    SELECT DISTINCT ON (procedure_code)
                        procedure_code,
                        work_rvu,
                        practice_expense_rvu,
                        malpractice_rvu,
                        total_rvu,
                        effective_date,
                        end_date
                    FROM rvu_data
                    WHERE procedure_code IN ({placeholders})
                    AND (end_date IS NULL OR end_date >= CURRENT_DATE)
                    ORDER BY procedure_code, effective_date DESC
                """)
                
                result = await session.execute(query, params)
                rows = result.fetchall()
                
                # Process results
                for row in rows:
                    procedure_code = row[0]
                    rvu_data = RVUCacheData(
                        procedure_code=procedure_code,
                        work_rvu=row[1] or Decimal('0'),
                        practice_expense_rvu=row[2] or Decimal('0'),
                        malpractice_rvu=row[3] or Decimal('0'),
                        total_rvu=row[4],
                        effective_date=str(row[5]) if row[5] else None,
                        end_date=str(row[6]) if row[6] else None,
                    )
                    results[procedure_code] = rvu_data
                    
                # Cache all results locally
                async with self._cache_lock:
                    self.local_cache.update(results)
                    
                # Cache to Redis
                if self.redis_client and results:
                    await self._cache_batch_to_redis(list(results.values()))
                    
        except Exception as e:
            logger.error(f"Batch database lookup failed: {e}")
            
        return results
        
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'local_cache_size': len(self.local_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate_percent': round(hit_rate, 2),
            'preload_complete': self.preload_complete,
        }
        
    async def refresh_cache(self, procedure_codes: Optional[List[str]] = None):
        """Refresh cache data from database."""
        if procedure_codes:
            # Refresh specific codes
            for code in procedure_codes:
                if code in self.local_cache:
                    del self.local_cache[code]
                    
            await self._batch_lookup_from_database(procedure_codes)
        else:
            # Full refresh
            self.local_cache.clear()
            await self._preload_from_database()
            
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None


# Global cache instance
rvu_cache = OptimizedRVUCache()
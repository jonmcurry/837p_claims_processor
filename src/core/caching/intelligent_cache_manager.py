"""Intelligent cache management with predictive preloading for peak processing periods."""

import asyncio
import json
import time
import hashlib
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
import pickle
import gzip

import redis.asyncio as redis
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.core.config import config
from src.monitoring.metrics.comprehensive_metrics import metrics_collector


logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategies for different data types."""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    TTL = "ttl"                    # Time To Live
    PREDICTIVE = "predictive"      # Predictive preloading
    PRIORITY = "priority"          # Priority-based


class CacheLevel(Enum):
    """Cache storage levels."""
    L1_MEMORY = "l1_memory"        # In-process memory
    L2_REDIS = "l2_redis"          # Redis cache
    L3_DISK = "l3_disk"            # Disk-based cache


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: Optional[int]
    size_bytes: int
    priority: int = 5  # 1-10, higher is more important
    cache_level: CacheLevel = CacheLevel.L1_MEMORY
    compression_enabled: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UsagePattern:
    """Usage pattern for predictive caching."""
    key_pattern: str
    hourly_access_counts: List[int]  # 24-hour pattern
    daily_access_counts: List[int]   # 7-day pattern
    peak_hours: List[int]
    peak_days: List[int]
    seasonal_multiplier: float = 1.0
    prediction_accuracy: float = 0.0


@dataclass
class PreloadJob:
    """Cache preload job configuration."""
    job_id: str
    key_patterns: List[str]
    target_time: datetime
    priority: int
    estimated_size_mb: float
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, completed, failed
    created_at: datetime = field(default_factory=datetime.now)


class IntelligentCacheManager:
    """Advanced cache manager with predictive preloading and multi-level storage."""
    
    def __init__(self, 
                 redis_client: Optional[redis.Redis] = None,
                 max_memory_cache_mb: float = 512,
                 enable_predictive_preloading: bool = True,
                 enable_compression: bool = True):
        
        self.redis_client = redis_client
        self.max_memory_cache_bytes = int(max_memory_cache_mb * 1024 * 1024)
        self.enable_predictive_preloading = enable_predictive_preloading
        self.enable_compression = enable_compression
        
        # Multi-level cache storage
        self._l1_cache: Dict[str, CacheEntry] = {}  # Memory cache
        self._l1_lock = threading.RLock()
        
        # Cache statistics
        self._stats = {
            'hits': defaultdict(int),
            'misses': defaultdict(int),
            'evictions': defaultdict(int),
            'preloads': defaultdict(int),
            'total_size_bytes': 0,
            'avg_access_time_ms': defaultdict(float)
        }
        
        # Usage pattern tracking for predictive caching
        self._usage_patterns: Dict[str, UsagePattern] = {}
        self._access_history: deque = deque(maxlen=10000)  # Recent access history
        
        # Preloading system
        self._preload_jobs: Dict[str, PreloadJob] = {}
        self._preload_queue = asyncio.Queue()
        self._preload_workers: List[asyncio.Task] = []
        
        # Background tasks
        self._pattern_analyzer_task = None
        self._preload_scheduler_task = None
        self._cache_optimizer_task = None
        self._start_background_tasks()
        
        # ML models for prediction
        self._access_predictor = None
        self._pattern_scaler = StandardScaler()
        self._prediction_history: deque = deque(maxlen=1000)
    
    async def get(self, 
                 key: str, 
                 default: Any = None,
                 cache_level: Optional[CacheLevel] = None) -> Any:
        """Get value from cache with intelligent level selection."""
        
        start_time = time.time()
        
        # Try L1 (memory) cache first
        with self._l1_lock:
            if key in self._l1_cache:
                entry = self._l1_cache[key]
                
                # Check TTL
                if self._is_expired(entry):
                    del self._l1_cache[key]
                    self._stats['evictions'][CacheLevel.L1_MEMORY] += 1
                else:
                    # Update access metadata
                    entry.last_accessed = datetime.now()
                    entry.access_count += 1
                    
                    self._record_access(key, CacheLevel.L1_MEMORY)
                    self._stats['hits'][CacheLevel.L1_MEMORY] += 1
                    
                    access_time = (time.time() - start_time) * 1000
                    self._update_access_time_stats(CacheLevel.L1_MEMORY, access_time)
                    
                    return entry.value
        
        # Try L2 (Redis) cache
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(f"cache:{key}")
                if cached_data:
                    # Deserialize and optionally decompress
                    entry_data = pickle.loads(cached_data)
                    
                    if entry_data.get('compressed', False):
                        value = pickle.loads(gzip.decompress(entry_data['value']))
                    else:
                        value = entry_data['value']
                    
                    # Promote to L1 cache if frequently accessed
                    await self._consider_l1_promotion(key, value, entry_data)
                    
                    self._record_access(key, CacheLevel.L2_REDIS)
                    self._stats['hits'][CacheLevel.L2_REDIS] += 1
                    
                    access_time = (time.time() - start_time) * 1000
                    self._update_access_time_stats(CacheLevel.L2_REDIS, access_time)
                    
                    return value
                    
            except Exception as e:
                logger.warning(f"Redis cache error for key {key}: {e}")
        
        # Cache miss
        self._stats['misses'][CacheLevel.L1_MEMORY] += 1
        self._record_access(key, None)  # Record miss
        
        return default
    
    async def set(self, 
                 key: str, 
                 value: Any,
                 ttl_seconds: Optional[int] = None,
                 priority: int = 5,
                 cache_strategy: CacheStrategy = CacheStrategy.LRU,
                 target_level: Optional[CacheLevel] = None) -> bool:
        """Set value in cache with intelligent level placement."""
        
        # Calculate value size
        value_size = self._calculate_size(value)
        
        # Determine optimal cache level
        if target_level is None:
            target_level = self._determine_cache_level(key, value_size, priority)
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            ttl_seconds=ttl_seconds,
            size_bytes=value_size,
            priority=priority,
            cache_level=target_level,
            compression_enabled=self.enable_compression and value_size > 1024
        )
        
        # Store in appropriate cache level
        if target_level == CacheLevel.L1_MEMORY:
            return await self._store_l1(entry)
        elif target_level == CacheLevel.L2_REDIS:
            return await self._store_l2(entry)
        else:
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from all cache levels."""
        deleted = False
        
        # Remove from L1
        with self._l1_lock:
            if key in self._l1_cache:
                del self._l1_cache[key]
                deleted = True
        
        # Remove from L2
        if self.redis_client:
            try:
                result = await self.redis_client.delete(f"cache:{key}")
                deleted = deleted or bool(result)
            except Exception as e:
                logger.warning(f"Redis delete error for key {key}: {e}")
        
        return deleted
    
    async def preload_for_peak_period(self, 
                                    peak_start: datetime,
                                    duration_minutes: int = 60,
                                    key_patterns: Optional[List[str]] = None) -> str:
        """Preload cache for an expected peak processing period."""
        
        job_id = f"preload_{int(peak_start.timestamp())}_{duration_minutes}"
        
        if key_patterns is None:
            # Use predictive patterns
            key_patterns = await self._predict_peak_patterns(peak_start, duration_minutes)
        
        # Create preload job
        preload_job = PreloadJob(
            job_id=job_id,
            key_patterns=key_patterns,
            target_time=peak_start,
            priority=8,  # High priority
            estimated_size_mb=self._estimate_preload_size(key_patterns)
        )
        
        self._preload_jobs[job_id] = preload_job
        
        # Schedule preload to start 15 minutes before peak
        preload_start = peak_start - timedelta(minutes=15)
        
        await self._schedule_preload_job(preload_job, preload_start)
        
        logger.info(f"Scheduled cache preload job {job_id} for {peak_start} "
                   f"({len(key_patterns)} patterns, ~{preload_job.estimated_size_mb:.1f}MB)")
        
        return job_id
    
    async def _predict_peak_patterns(self, 
                                   peak_start: datetime, 
                                   duration_minutes: int) -> List[str]:
        """Predict cache keys likely to be accessed during peak period."""
        
        peak_hour = peak_start.hour
        peak_day = peak_start.weekday()
        
        predicted_patterns = []
        
        # Analyze historical patterns
        for pattern_key, pattern in self._usage_patterns.items():
            # Check if this pattern is active during peak hours
            if peak_hour in pattern.peak_hours:
                confidence = pattern.prediction_accuracy
                
                # Adjust confidence based on day of week
                if peak_day in pattern.peak_days:
                    confidence *= 1.2
                
                # Include pattern if confidence is high enough
                if confidence > 0.6:
                    predicted_patterns.append(pattern_key)
        
        # Add common patterns based on time of day
        if 8 <= peak_hour <= 17:  # Business hours
            predicted_patterns.extend([
                "claims:facility:*",
                "validation:rules:*",
                "provider:performance:*",
                "ml:model:claims_classifier"
            ])
        
        if peak_hour in [9, 10, 14, 15]:  # Typical peak hours
            predicted_patterns.extend([
                "batch:metadata:*",
                "failed_claims:analysis:*",
                "real_time:metrics:*"
            ])
        
        return list(set(predicted_patterns))  # Remove duplicates
    
    async def _store_l1(self, entry: CacheEntry) -> bool:
        """Store entry in L1 (memory) cache."""
        
        with self._l1_lock:
            # Check if we need to evict entries
            if self._need_eviction_l1(entry.size_bytes):
                await self._evict_l1_entries(entry.size_bytes)
            
            # Store the entry
            self._l1_cache[entry.key] = entry
            self._stats['total_size_bytes'] += entry.size_bytes
            
            logger.debug(f"Stored {entry.key} in L1 cache ({entry.size_bytes} bytes)")
            return True
    
    async def _store_l2(self, entry: CacheEntry) -> bool:
        """Store entry in L2 (Redis) cache."""
        
        if not self.redis_client:
            return False
        
        try:
            # Prepare data for storage
            value_to_store = entry.value
            
            if entry.compression_enabled:
                compressed_value = gzip.compress(pickle.dumps(value_to_store))
                store_data = {
                    'value': compressed_value,
                    'compressed': True,
                    'original_size': entry.size_bytes,
                    'created_at': entry.created_at.isoformat(),
                    'priority': entry.priority
                }
            else:
                store_data = {
                    'value': value_to_store,
                    'compressed': False,
                    'created_at': entry.created_at.isoformat(),
                    'priority': entry.priority
                }
            
            # Store in Redis
            serialized_data = pickle.dumps(store_data)
            
            if entry.ttl_seconds:
                await self.redis_client.setex(
                    f"cache:{entry.key}", 
                    entry.ttl_seconds, 
                    serialized_data
                )
            else:
                await self.redis_client.set(f"cache:{entry.key}", serialized_data)
            
            logger.debug(f"Stored {entry.key} in L2 cache (Redis)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store {entry.key} in L2 cache: {e}")
            return False
    
    def _need_eviction_l1(self, new_entry_size: int) -> bool:
        """Check if L1 cache needs eviction for new entry."""
        current_size = sum(entry.size_bytes for entry in self._l1_cache.values())
        return current_size + new_entry_size > self.max_memory_cache_bytes
    
    async def _evict_l1_entries(self, space_needed: int):
        """Evict entries from L1 cache to make space."""
        
        with self._l1_lock:
            # Sort entries by eviction priority (LRU with priority consideration)
            sorted_entries = sorted(
                self._l1_cache.items(),
                key=lambda x: (
                    x[1].priority,  # Higher priority = less likely to evict
                    x[1].last_accessed.timestamp(),  # Older = more likely to evict
                    -x[1].access_count  # Less accessed = more likely to evict
                )
            )
            
            freed_space = 0
            entries_to_remove = []
            
            for key, entry in sorted_entries:
                entries_to_remove.append(key)
                freed_space += entry.size_bytes
                
                # Check if we've freed enough space
                if freed_space >= space_needed:
                    break
            
            # Remove selected entries
            for key in entries_to_remove:
                if key in self._l1_cache:
                    entry = self._l1_cache[key]
                    
                    # Optionally demote to L2 cache if it's high priority
                    if entry.priority >= 7 and self.redis_client:
                        await self._store_l2(entry)
                    
                    del self._l1_cache[key]
                    self._stats['evictions'][CacheLevel.L1_MEMORY] += 1
                    self._stats['total_size_bytes'] -= entry.size_bytes
            
            logger.debug(f"Evicted {len(entries_to_remove)} entries, freed {freed_space} bytes")
    
    async def _consider_l1_promotion(self, key: str, value: Any, entry_data: Dict):
        """Consider promoting frequently accessed L2 entry to L1."""
        
        # Check access frequency and priority
        priority = entry_data.get('priority', 5)
        
        # Simple promotion criteria
        if priority >= 7:  # High priority items
            value_size = self._calculate_size(value)
            
            # Only promote if it's not too large
            if value_size < (self.max_memory_cache_bytes * 0.1):  # Max 10% of cache
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    access_count=1,
                    ttl_seconds=None,
                    size_bytes=value_size,
                    priority=priority,
                    cache_level=CacheLevel.L1_MEMORY
                )
                
                await self._store_l1(entry)
                logger.debug(f"Promoted {key} from L2 to L1 cache")
    
    def _determine_cache_level(self, 
                             key: str, 
                             size_bytes: int, 
                             priority: int) -> CacheLevel:
        """Determine optimal cache level for new entry."""
        
        # High priority or small items go to L1
        if priority >= 8 or size_bytes < 1024:
            return CacheLevel.L1_MEMORY
        
        # Large items go to L2
        if size_bytes > (self.max_memory_cache_bytes * 0.1):
            return CacheLevel.L2_REDIS
        
        # Medium priority items go to L1 if there's space
        current_l1_size = sum(entry.size_bytes for entry in self._l1_cache.values())
        if current_l1_size + size_bytes <= self.max_memory_cache_bytes * 0.8:
            return CacheLevel.L1_MEMORY
        
        return CacheLevel.L2_REDIS
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            return len(pickle.dumps(value))
        except:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, dict):
                return sum(self._calculate_size(k) + self._calculate_size(v) 
                          for k, v in value.items())
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value)
            else:
                return 1024  # Default estimate
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry has expired."""
        if not entry.ttl_seconds:
            return False
        
        age_seconds = (datetime.now() - entry.created_at).total_seconds()
        return age_seconds > entry.ttl_seconds
    
    def _record_access(self, key: str, cache_level: Optional[CacheLevel]):
        """Record cache access for pattern analysis."""
        access_record = {
            'key': key,
            'timestamp': datetime.now(),
            'cache_level': cache_level.value if cache_level else 'miss',
            'hour': datetime.now().hour,
            'day_of_week': datetime.now().weekday()
        }
        
        self._access_history.append(access_record)
        
        # Update usage patterns
        self._update_usage_pattern(key, access_record)
    
    def _update_usage_pattern(self, key: str, access_record: Dict):
        """Update usage pattern for predictive caching."""
        
        # Extract pattern from key (e.g., "claims:facility:123" -> "claims:facility:*")
        pattern_key = self._extract_pattern(key)
        
        if pattern_key not in self._usage_patterns:
            self._usage_patterns[pattern_key] = UsagePattern(
                key_pattern=pattern_key,
                hourly_access_counts=[0] * 24,
                daily_access_counts=[0] * 7,
                peak_hours=[],
                peak_days=[]
            )
        
        pattern = self._usage_patterns[pattern_key]
        
        # Update hourly and daily access counts
        hour = access_record['hour']
        day = access_record['day_of_week']
        
        pattern.hourly_access_counts[hour] += 1
        pattern.daily_access_counts[day] += 1
        
        # Update peak hours and days
        self._update_peaks(pattern)
    
    def _extract_pattern(self, key: str) -> str:
        """Extract pattern from cache key."""
        parts = key.split(':')
        
        if len(parts) >= 2:
            # Replace specific IDs with wildcards
            pattern_parts = []
            for part in parts:
                if part.isdigit() or (part.startswith('CLM') and len(part) > 5):
                    pattern_parts.append('*')
                else:
                    pattern_parts.append(part)
            return ':'.join(pattern_parts)
        
        return key
    
    def _update_peaks(self, pattern: UsagePattern):
        """Update peak hours and days for pattern."""
        
        # Find peak hours (above average)
        avg_hourly = sum(pattern.hourly_access_counts) / 24
        pattern.peak_hours = [
            hour for hour, count in enumerate(pattern.hourly_access_counts)
            if count > avg_hourly * 1.5
        ]
        
        # Find peak days (above average)
        avg_daily = sum(pattern.daily_access_counts) / 7
        pattern.peak_days = [
            day for day, count in enumerate(pattern.daily_access_counts)
            if count > avg_daily * 1.5
        ]
    
    def _update_access_time_stats(self, cache_level: CacheLevel, access_time_ms: float):
        """Update access time statistics."""
        current_avg = self._stats['avg_access_time_ms'][cache_level]
        
        # Simple moving average
        alpha = 0.1
        self._stats['avg_access_time_ms'][cache_level] = (
            alpha * access_time_ms + (1 - alpha) * current_avg
        )
    
    def _estimate_preload_size(self, key_patterns: List[str]) -> float:
        """Estimate size of preload operation in MB."""
        
        # Rough estimation based on pattern types
        size_estimates = {
            'claims:*': 50,     # Claims data is typically large
            'ml:model:*': 100,  # ML models are very large
            'validation:*': 10, # Validation rules are smaller
            'provider:*': 20,   # Provider data medium size
            'facility:*': 15,   # Facility data medium size
            'batch:*': 30,      # Batch metadata medium-large
        }
        
        total_size_mb = 0
        for pattern in key_patterns:
            for size_pattern, size_mb in size_estimates.items():
                if any(part in pattern for part in size_pattern.split(':')):
                    total_size_mb += size_mb
                    break
            else:
                total_size_mb += 5  # Default estimate
        
        return total_size_mb
    
    async def _schedule_preload_job(self, job: PreloadJob, start_time: datetime):
        """Schedule preload job to run at specific time."""
        
        # Calculate delay
        delay_seconds = (start_time - datetime.now()).total_seconds()
        
        if delay_seconds > 0:
            # Schedule for future
            await asyncio.sleep(delay_seconds)
        
        # Execute preload
        job.status = "running"
        await self._execute_preload_job(job)
    
    async def _execute_preload_job(self, job: PreloadJob):
        """Execute cache preload job."""
        
        logger.info(f"Starting preload job {job.job_id} with {len(job.key_patterns)} patterns")
        
        try:
            preloaded_count = 0
            
            for pattern in job.key_patterns:
                # Simulate preloading based on pattern
                # In real implementation, this would query data sources
                keys_to_preload = await self._find_keys_matching_pattern(pattern)
                
                for key in keys_to_preload:
                    # Load data and cache it
                    data = await self._load_data_for_key(key)
                    if data is not None:
                        await self.set(key, data, ttl_seconds=3600, priority=8)
                        preloaded_count += 1
                
                # Throttle to avoid overwhelming the system
                await asyncio.sleep(0.1)
            
            job.status = "completed"
            self._stats['preloads']['successful'] += 1
            
            logger.info(f"Completed preload job {job.job_id}: {preloaded_count} items cached")
            
        except Exception as e:
            job.status = "failed"
            self._stats['preloads']['failed'] += 1
            logger.error(f"Preload job {job.job_id} failed: {e}")
    
    async def _find_keys_matching_pattern(self, pattern: str) -> List[str]:
        """Find existing keys that match the given pattern."""
        
        # This is a simplified implementation
        # In practice, this would query Redis or application data sources
        
        matching_keys = []
        
        # Check L1 cache
        with self._l1_lock:
            for key in self._l1_cache.keys():
                if self._key_matches_pattern(key, pattern):
                    matching_keys.append(key)
        
        # Check L2 cache (Redis)
        if self.redis_client:
            try:
                redis_pattern = pattern.replace('*', '*').replace(':', ':')
                async for key in self.redis_client.scan_iter(match=f"cache:{redis_pattern}"):
                    cache_key = key.decode('utf-8').replace('cache:', '')
                    matching_keys.append(cache_key)
            except Exception as e:
                logger.warning(f"Redis pattern scan error: {e}")
        
        return matching_keys
    
    def _key_matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches pattern."""
        key_parts = key.split(':')
        pattern_parts = pattern.split(':')
        
        if len(key_parts) != len(pattern_parts):
            return False
        
        for key_part, pattern_part in zip(key_parts, pattern_parts):
            if pattern_part != '*' and pattern_part != key_part:
                return False
        
        return True
    
    async def _load_data_for_key(self, key: str) -> Optional[Any]:
        """Load data for cache key from data source."""
        
        # This is a placeholder - in practice, this would:
        # 1. Determine data source based on key pattern
        # 2. Query appropriate database/service
        # 3. Return the data
        
        # For now, return None to indicate no data found
        return None
    
    def _start_background_tasks(self):
        """Start background cache optimization tasks."""
        
        async def pattern_analyzer():
            """Analyze access patterns and update predictions."""
            while True:
                try:
                    await asyncio.sleep(300)  # Analyze every 5 minutes
                    
                    # Update pattern analysis
                    if len(self._access_history) > 100:
                        await self._analyze_access_patterns()
                    
                    # Train prediction model
                    if len(self._prediction_history) > 50:
                        await self._train_access_predictor()
                
                except Exception as e:
                    logger.error(f"Pattern analyzer error: {e}")
        
        async def cache_optimizer():
            """Optimize cache configuration based on usage."""
            while True:
                try:
                    await asyncio.sleep(600)  # Optimize every 10 minutes
                    
                    # Analyze cache performance
                    performance_stats = await self._analyze_cache_performance()
                    
                    # Adjust cache parameters if needed
                    await self._optimize_cache_parameters(performance_stats)
                
                except Exception as e:
                    logger.error(f"Cache optimizer error: {e}")
        
        if self.enable_predictive_preloading:
            self._pattern_analyzer_task = asyncio.create_task(pattern_analyzer())
        
        self._cache_optimizer_task = asyncio.create_task(cache_optimizer())
        
        # Start preload workers
        for i in range(2):
            worker = asyncio.create_task(self._preload_worker())
            self._preload_workers.append(worker)
    
    async def _preload_worker(self):
        """Background worker for processing preload jobs."""
        while True:
            try:
                job = await self._preload_queue.get()
                await self._execute_preload_job(job)
                self._preload_queue.task_done()
            except Exception as e:
                logger.error(f"Preload worker error: {e}")
    
    async def _analyze_access_patterns(self):
        """Analyze recent access patterns for optimization."""
        
        # Group access history by pattern
        pattern_accesses = defaultdict(list)
        
        for access in list(self._access_history)[-1000:]:  # Last 1000 accesses
            pattern = self._extract_pattern(access['key'])
            pattern_accesses[pattern].append(access)
        
        # Update pattern predictions
        for pattern, accesses in pattern_accesses.items():
            if pattern in self._usage_patterns:
                usage_pattern = self._usage_patterns[pattern]
                
                # Calculate prediction accuracy
                # (This is simplified - real implementation would be more sophisticated)
                recent_hours = [access['hour'] for access in accesses[-10:]]
                predicted_hours = usage_pattern.peak_hours
                
                if predicted_hours:
                    accuracy = len(set(recent_hours) & set(predicted_hours)) / len(set(recent_hours) | set(predicted_hours))
                    usage_pattern.prediction_accuracy = accuracy
    
    async def _train_access_predictor(self):
        """Train ML model for access prediction."""
        
        try:
            # Prepare training data
            features = []
            targets = []
            
            for record in list(self._prediction_history)[-500:]:
                # Features: hour, day_of_week, cache_level, pattern_type
                feature_vector = [
                    record['hour'],
                    record['day_of_week'],
                    1 if record['cache_level'] != 'miss' else 0,
                    len(record['key'].split(':'))  # Pattern complexity
                ]
                features.append(feature_vector)
                targets.append(1 if record['cache_level'] != 'miss' else 0)
            
            if len(features) > 20:
                # Train simple linear regression model
                X = np.array(features)
                y = np.array(targets)
                
                X_scaled = self._pattern_scaler.fit_transform(X)
                
                if self._access_predictor is None:
                    self._access_predictor = LinearRegression()
                
                self._access_predictor.fit(X_scaled, y)
                
                logger.debug("Updated access prediction model")
        
        except Exception as e:
            logger.warning(f"Failed to train access predictor: {e}")
    
    async def _analyze_cache_performance(self) -> Dict[str, Any]:
        """Analyze cache performance metrics."""
        
        total_hits = sum(self._stats['hits'].values())
        total_misses = sum(self._stats['misses'].values())
        total_requests = total_hits + total_misses
        
        hit_rate = total_hits / total_requests if total_requests > 0 else 0
        
        l1_hits = self._stats['hits'][CacheLevel.L1_MEMORY]
        l2_hits = self._stats['hits'][CacheLevel.L2_REDIS]
        
        return {
            'overall_hit_rate': hit_rate,
            'l1_hit_rate': l1_hits / total_requests if total_requests > 0 else 0,
            'l2_hit_rate': l2_hits / total_requests if total_requests > 0 else 0,
            'avg_l1_access_time_ms': self._stats['avg_access_time_ms'][CacheLevel.L1_MEMORY],
            'avg_l2_access_time_ms': self._stats['avg_access_time_ms'][CacheLevel.L2_REDIS],
            'total_size_mb': self._stats['total_size_bytes'] / (1024 * 1024),
            'eviction_rate': sum(self._stats['evictions'].values()) / total_requests if total_requests > 0 else 0
        }
    
    async def _optimize_cache_parameters(self, performance_stats: Dict[str, Any]):
        """Optimize cache parameters based on performance analysis."""
        
        # Adjust cache size if hit rate is low and eviction rate is high
        if (performance_stats['overall_hit_rate'] < 0.8 and 
            performance_stats['eviction_rate'] > 0.1):
            
            # Consider increasing cache size (if system resources allow)
            system_memory_usage = psutil.virtual_memory().percent
            if system_memory_usage < 70:  # System not under memory pressure
                # Increase cache size by 10%
                new_size = int(self.max_memory_cache_bytes * 1.1)
                max_allowed = int(psutil.virtual_memory().total * 0.1)  # Max 10% of system memory
                
                if new_size <= max_allowed:
                    self.max_memory_cache_bytes = new_size
                    logger.info(f"Increased L1 cache size to {new_size / (1024*1024):.1f}MB")
    
    async def get_cache_status(self) -> Dict[str, Any]:
        """Get comprehensive cache status and statistics."""
        
        performance_stats = await self._analyze_cache_performance()
        
        # Usage patterns summary
        pattern_summary = {}
        for pattern_key, pattern in self._usage_patterns.items():
            pattern_summary[pattern_key] = {
                'peak_hours': pattern.peak_hours,
                'peak_days': pattern.peak_days,
                'prediction_accuracy': pattern.prediction_accuracy,
                'total_accesses': sum(pattern.hourly_access_counts)
            }
        
        # Preload jobs status
        preload_status = {}
        for job_id, job in self._preload_jobs.items():
            preload_status[job_id] = {
                'status': job.status,
                'target_time': job.target_time.isoformat(),
                'estimated_size_mb': job.estimated_size_mb,
                'patterns_count': len(job.key_patterns)
            }
        
        return {
            'performance': performance_stats,
            'cache_levels': {
                'l1_memory': {
                    'current_entries': len(self._l1_cache),
                    'max_size_mb': self.max_memory_cache_bytes / (1024 * 1024),
                    'current_size_mb': sum(e.size_bytes for e in self._l1_cache.values()) / (1024 * 1024)
                },
                'l2_redis': {
                    'available': self.redis_client is not None
                }
            },
            'usage_patterns': pattern_summary,
            'preload_jobs': preload_status,
            'predictive_preloading_enabled': self.enable_predictive_preloading,
            'compression_enabled': self.enable_compression
        }
    
    async def shutdown(self):
        """Gracefully shutdown cache manager."""
        
        # Cancel background tasks
        if self._pattern_analyzer_task:
            self._pattern_analyzer_task.cancel()
        
        if self._cache_optimizer_task:
            self._cache_optimizer_task.cancel()
        
        for worker in self._preload_workers:
            worker.cancel()
        
        # Close Redis connection if we own it
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Intelligent cache manager shut down")


# Global cache manager instance
intelligent_cache_manager: Optional[IntelligentCacheManager] = None

def initialize_cache_manager(redis_client: Optional[redis.Redis] = None) -> IntelligentCacheManager:
    """Initialize the global cache manager instance."""
    global intelligent_cache_manager
    intelligent_cache_manager = IntelligentCacheManager(
        redis_client=redis_client,
        max_memory_cache_mb=config.get_processing_settings().get('cache_size_mb', 512),
        enable_predictive_preloading=True,
        enable_compression=True
    )
    return intelligent_cache_manager
"""Advanced ML model caching and optimization system."""

import asyncio
import hashlib
import pickle
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging

import numpy as np
import torch
import tensorflow as tf
from sklearn.base import BaseEstimator
import redis.asyncio as redis
import psutil

from src.core.config import config
from src.monitoring.metrics.comprehensive_metrics import metrics_collector


logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for cached models."""
    model_name: str
    model_version: str
    model_type: str  # 'pytorch', 'tensorflow', 'sklearn'
    model_size_mb: float
    cache_timestamp: datetime
    last_accessed: datetime
    access_count: int
    avg_inference_time_ms: float
    memory_usage_mb: float
    gpu_memory_mb: Optional[float] = None
    optimization_level: str = "standard"  # 'standard', 'optimized', 'quantized'


@dataclass
class BatchInferenceJob:
    """Batch inference job configuration."""
    job_id: str
    model_name: str
    input_data: np.ndarray
    batch_size: int
    priority: str  # 'low', 'medium', 'high', 'critical'
    created_at: datetime
    timeout_seconds: int = 300
    callback: Optional[callable] = None


class ModelCacheManager:
    """Advanced model caching with intelligent memory management."""
    
    def __init__(self, 
                 cache_size_gb: float = 4.0,
                 redis_client: Optional[redis.Redis] = None,
                 enable_gpu_cache: bool = True):
        self.cache_size_bytes = int(cache_size_gb * 1024 * 1024 * 1024)
        self.redis_client = redis_client
        self.enable_gpu_cache = enable_gpu_cache and torch.cuda.is_available()
        
        # In-memory cache
        self._memory_cache: Dict[str, Any] = {}
        self._model_metadata: Dict[str, ModelMetadata] = {}
        self._cache_lock = threading.RLock()
        
        # GPU cache management
        if self.enable_gpu_cache:
            self._gpu_cache: Dict[str, torch.nn.Module] = {}
            self._gpu_memory_tracker = {}
        
        # Cache statistics
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_requests = 0
        
        # Background cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
        
        # Model optimization queue
        self._optimization_queue = asyncio.Queue()
        self._optimization_workers = []
        self._start_optimization_workers()
    
    async def get_model(self, 
                       model_name: str, 
                       model_version: str = "latest",
                       optimization_level: str = "standard") -> Tuple[Any, ModelMetadata]:
        """Get model from cache with intelligent loading."""
        cache_key = f"{model_name}:{model_version}:{optimization_level}"
        
        self._total_requests += 1
        
        # Check memory cache first
        with self._cache_lock:
            if cache_key in self._memory_cache:
                model = self._memory_cache[cache_key]
                metadata = self._model_metadata[cache_key]
                metadata.last_accessed = datetime.now()
                metadata.access_count += 1
                self._cache_hits += 1
                
                logger.debug(f"Cache hit for model: {cache_key}")
                return model, metadata
        
        # Check Redis cache
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(f"model_cache:{cache_key}")
                if cached_data:
                    model_data = pickle.loads(cached_data)
                    model = model_data['model']
                    metadata = model_data['metadata']
                    
                    # Load into memory cache
                    await self._store_in_memory_cache(cache_key, model, metadata)
                    self._cache_hits += 1
                    
                    logger.debug(f"Redis cache hit for model: {cache_key}")
                    return model, metadata
            except Exception as e:
                logger.warning(f"Redis cache error for {cache_key}: {e}")
        
        # Load from disk and optimize
        self._cache_misses += 1
        model, metadata = await self._load_and_optimize_model(
            model_name, model_version, optimization_level
        )
        
        # Store in caches
        await self._store_in_memory_cache(cache_key, model, metadata)
        
        if self.redis_client:
            try:
                model_data = {'model': model, 'metadata': metadata}
                await self.redis_client.setex(
                    f"model_cache:{cache_key}",
                    86400,  # 24 hours
                    pickle.dumps(model_data)
                )
            except Exception as e:
                logger.warning(f"Failed to store in Redis cache: {e}")
        
        logger.info(f"Loaded and cached model: {cache_key}")
        return model, metadata
    
    async def _load_and_optimize_model(self, 
                                     model_name: str, 
                                     model_version: str,
                                     optimization_level: str) -> Tuple[Any, ModelMetadata]:
        """Load model from disk and apply optimizations."""
        model_path = Path(config.ML_MODEL_PATH) / model_name / model_version
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        start_time = time.time()
        
        # Determine model type and load
        if (model_path / "model.pt").exists() or (model_path / "model.pth").exists():
            model = await self._load_pytorch_model(model_path, optimization_level)
            model_type = "pytorch"
        elif (model_path / "saved_model.pb").exists():
            model = await self._load_tensorflow_model(model_path, optimization_level)
            model_type = "tensorflow"
        elif (model_path / "model.pkl").exists():
            model = await self._load_sklearn_model(model_path, optimization_level)
            model_type = "sklearn"
        else:
            raise ValueError(f"Unknown model format in {model_path}")
        
        load_time = time.time() - start_time
        
        # Calculate model size
        model_size_mb = self._calculate_model_size(model, model_type)
        
        # Create metadata
        metadata = ModelMetadata(
            model_name=model_name,
            model_version=model_version,
            model_type=model_type,
            model_size_mb=model_size_mb,
            cache_timestamp=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            avg_inference_time_ms=0.0,
            memory_usage_mb=model_size_mb,
            optimization_level=optimization_level
        )
        
        # GPU optimization if available
        if self.enable_gpu_cache and model_type == "pytorch":
            model = await self._optimize_for_gpu(model, metadata)
        
        logger.info(f"Loaded {model_type} model {model_name}:{model_version} "
                   f"in {load_time:.2f}s ({model_size_mb:.1f}MB)")
        
        return model, metadata
    
    async def _load_pytorch_model(self, model_path: Path, optimization_level: str) -> torch.nn.Module:
        """Load and optimize PyTorch model."""
        model_file = model_path / "model.pt" if (model_path / "model.pt").exists() else model_path / "model.pth"
        
        model = torch.load(model_file, map_location='cpu')
        model.eval()
        
        if optimization_level == "optimized":
            # Apply TorchScript optimization
            try:
                model = torch.jit.script(model)
                logger.debug("Applied TorchScript optimization")
            except Exception as e:
                logger.warning(f"TorchScript optimization failed: {e}")
        
        elif optimization_level == "quantized":
            # Apply quantization
            try:
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.debug("Applied dynamic quantization")
            except Exception as e:
                logger.warning(f"Quantization failed: {e}")
        
        return model
    
    async def _load_tensorflow_model(self, model_path: Path, optimization_level: str) -> Any:
        """Load and optimize TensorFlow model."""
        model = tf.saved_model.load(str(model_path))
        
        if optimization_level == "optimized":
            # Apply TensorFlow Lite optimization
            try:
                converter = tf.lite.TFLiteConverter.from_saved_model(str(model_path))
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                tflite_model = converter.convert()
                
                # Create interpreter
                interpreter = tf.lite.Interpreter(model_content=tflite_model)
                interpreter.allocate_tensors()
                model = interpreter
                logger.debug("Applied TensorFlow Lite optimization")
            except Exception as e:
                logger.warning(f"TensorFlow Lite optimization failed: {e}")
        
        return model
    
    async def _load_sklearn_model(self, model_path: Path, optimization_level: str) -> BaseEstimator:
        """Load and optimize scikit-learn model."""
        with open(model_path / "model.pkl", 'rb') as f:
            model = pickle.load(f)
        
        # Scikit-learn models don't have significant optimization options
        # But we can ensure they're using optimized BLAS libraries
        
        return model
    
    async def _optimize_for_gpu(self, model: torch.nn.Module, metadata: ModelMetadata) -> torch.nn.Module:
        """Optimize model for GPU usage."""
        if not torch.cuda.is_available():
            return model
        
        try:
            device = torch.device('cuda')
            model = model.to(device)
            
            # Measure GPU memory usage
            torch.cuda.empty_cache()
            gpu_memory_before = torch.cuda.memory_allocated()
            
            # Warm up the model
            dummy_input = torch.randn(1, 100).to(device)  # Adjust based on your model
            with torch.no_grad():
                _ = model(dummy_input)
            
            gpu_memory_after = torch.cuda.memory_allocated()
            metadata.gpu_memory_mb = (gpu_memory_after - gpu_memory_before) / (1024 * 1024)
            
            logger.debug(f"Model loaded to GPU, using {metadata.gpu_memory_mb:.1f}MB GPU memory")
            
        except Exception as e:
            logger.warning(f"Failed to optimize for GPU: {e}")
            model = model.cpu()
        
        return model
    
    def _calculate_model_size(self, model: Any, model_type: str) -> float:
        """Calculate model size in MB."""
        if model_type == "pytorch":
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            return (param_size + buffer_size) / (1024 * 1024)
        
        elif model_type == "tensorflow":
            # Approximate size calculation for TensorFlow models
            try:
                if hasattr(model, 'get_concrete_function'):
                    # SavedModel
                    return 50.0  # Default estimate
                else:
                    # TFLite model
                    return len(model._get_buffer()) / (1024 * 1024)
            except:
                return 50.0  # Default estimate
        
        elif model_type == "sklearn":
            # Use pickle size as approximation
            return len(pickle.dumps(model)) / (1024 * 1024)
        
        return 0.0
    
    async def _store_in_memory_cache(self, cache_key: str, model: Any, metadata: ModelMetadata):
        """Store model in memory cache with size management."""
        with self._cache_lock:
            # Check if we need to free up space
            current_size = sum(meta.memory_usage_mb for meta in self._model_metadata.values())
            target_size = current_size + metadata.memory_usage_mb
            
            if target_size * 1024 * 1024 > self.cache_size_bytes:
                await self._evict_least_used_models(metadata.memory_usage_mb)
            
            # Store the model
            self._memory_cache[cache_key] = model
            self._model_metadata[cache_key] = metadata
            
            logger.debug(f"Stored model in memory cache: {cache_key} ({metadata.memory_usage_mb:.1f}MB)")
    
    async def _evict_least_used_models(self, required_space_mb: float):
        """Evict least recently used models to free up space."""
        with self._cache_lock:
            # Sort by last accessed time and access count
            sorted_models = sorted(
                self._model_metadata.items(),
                key=lambda x: (x[1].last_accessed, x[1].access_count)
            )
            
            freed_space = 0.0
            for cache_key, metadata in sorted_models:
                if freed_space >= required_space_mb:
                    break
                
                # Remove from memory cache
                if cache_key in self._memory_cache:
                    del self._memory_cache[cache_key]
                    freed_space += metadata.memory_usage_mb
                    logger.debug(f"Evicted model from cache: {cache_key}")
                
                # Remove from GPU cache if present
                if self.enable_gpu_cache and cache_key in self._gpu_cache:
                    del self._gpu_cache[cache_key]
                    torch.cuda.empty_cache()
            
            # Update metadata
            for cache_key, _ in sorted_models[:int(freed_space / required_space_mb) + 1]:
                if cache_key in self._model_metadata:
                    del self._model_metadata[cache_key]
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        async def cleanup_expired_models():
            while True:
                try:
                    await asyncio.sleep(3600)  # Run every hour
                    await self._cleanup_expired_models()
                except Exception as e:
                    logger.error(f"Cleanup task error: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_expired_models())
    
    async def _cleanup_expired_models(self):
        """Clean up expired models from cache."""
        cutoff_time = datetime.now() - timedelta(hours=6)  # 6 hours
        
        with self._cache_lock:
            expired_keys = [
                key for key, metadata in self._model_metadata.items()
                if metadata.last_accessed < cutoff_time and metadata.access_count < 10
            ]
            
            for key in expired_keys:
                if key in self._memory_cache:
                    del self._memory_cache[key]
                if key in self._model_metadata:
                    del self._model_metadata[key]
                if self.enable_gpu_cache and key in self._gpu_cache:
                    del self._gpu_cache[key]
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired models from cache")
                if self.enable_gpu_cache:
                    torch.cuda.empty_cache()
    
    def _start_optimization_workers(self):
        """Start background model optimization workers."""
        async def optimization_worker():
            while True:
                try:
                    job = await self._optimization_queue.get()
                    await self._process_optimization_job(job)
                    self._optimization_queue.task_done()
                except Exception as e:
                    logger.error(f"Optimization worker error: {e}")
        
        # Start 2 optimization workers
        for _ in range(2):
            worker = asyncio.create_task(optimization_worker())
            self._optimization_workers.append(worker)
    
    async def _process_optimization_job(self, job: Dict[str, Any]):
        """Process model optimization job."""
        model_name = job['model_name']
        optimization_type = job['optimization_type']
        
        logger.info(f"Processing optimization job: {model_name} -> {optimization_type}")
        
        try:
            # Load base model
            model, metadata = await self.get_model(model_name, "latest", "standard")
            
            # Apply optimization
            if optimization_type == "quantization" and metadata.model_type == "pytorch":
                optimized_model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                
                # Cache optimized version
                cache_key = f"{model_name}:latest:quantized"
                optimized_metadata = ModelMetadata(
                    model_name=model_name,
                    model_version="latest",
                    model_type=metadata.model_type,
                    model_size_mb=self._calculate_model_size(optimized_model, metadata.model_type),
                    cache_timestamp=datetime.now(),
                    last_accessed=datetime.now(),
                    access_count=0,
                    avg_inference_time_ms=0.0,
                    memory_usage_mb=metadata.memory_usage_mb * 0.7,  # Approx reduction
                    optimization_level="quantized"
                )
                
                await self._store_in_memory_cache(cache_key, optimized_model, optimized_metadata)
                logger.info(f"Created quantized version of {model_name}")
        
        except Exception as e:
            logger.error(f"Optimization job failed for {model_name}: {e}")
    
    async def schedule_optimization(self, model_name: str, optimization_type: str):
        """Schedule background model optimization."""
        job = {
            'model_name': model_name,
            'optimization_type': optimization_type,
            'scheduled_at': datetime.now()
        }
        
        await self._optimization_queue.put(job)
        logger.debug(f"Scheduled optimization: {model_name} -> {optimization_type}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        hit_rate = self._cache_hits / self._total_requests if self._total_requests > 0 else 0
        
        with self._cache_lock:
            total_memory_mb = sum(meta.memory_usage_mb for meta in self._model_metadata.values())
            model_count = len(self._model_metadata)
        
        gpu_memory_mb = 0
        if self.enable_gpu_cache and torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        
        return {
            'cache_hit_rate': hit_rate,
            'total_requests': self._total_requests,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cached_models': model_count,
            'total_memory_mb': total_memory_mb,
            'gpu_memory_mb': gpu_memory_mb,
            'cache_size_limit_mb': self.cache_size_bytes / (1024 * 1024)
        }
    
    async def preload_models(self, model_specs: List[Dict[str, str]]):
        """Preload models into cache for better performance."""
        logger.info(f"Preloading {len(model_specs)} models into cache")
        
        preload_tasks = []
        for spec in model_specs:
            task = self.get_model(
                spec['name'], 
                spec.get('version', 'latest'),
                spec.get('optimization_level', 'standard')
            )
            preload_tasks.append(task)
        
        try:
            await asyncio.gather(*preload_tasks)
            logger.info("Model preloading completed successfully")
        except Exception as e:
            logger.error(f"Model preloading failed: {e}")
    
    async def shutdown(self):
        """Cleanup resources on shutdown."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        for worker in self._optimization_workers:
            worker.cancel()
        
        if self.enable_gpu_cache:
            torch.cuda.empty_cache()
        
        logger.info("Model cache manager shut down")


# Global model cache manager instance
model_cache_manager = ModelCacheManager(
    cache_size_gb=config.get_ml_settings().get('cache_size_gb', 4.0),
    enable_gpu_cache=config.get_ml_settings().get('enable_gpu', False)
)
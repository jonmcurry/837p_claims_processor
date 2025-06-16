"""Advanced batch inference optimization for ML models."""

import asyncio
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from concurrent.futures import ThreadPoolExecutor
import logging
import threading

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
from sklearn.base import BaseEstimator

from .model_cache_manager import ModelCacheManager, BatchInferenceJob, model_cache_manager
from src.monitoring.metrics.comprehensive_metrics import metrics_collector


logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Individual inference request."""
    request_id: str
    model_name: str
    input_data: np.ndarray
    priority: str = "medium"
    created_at: datetime = field(default_factory=datetime.now)
    timeout_seconds: int = 30
    callback: Optional[Callable] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchInferenceResult:
    """Result of batch inference operation."""
    batch_id: str
    model_name: str
    predictions: np.ndarray
    request_ids: List[str]
    inference_time_ms: float
    batch_size: int
    throughput_per_second: float
    model_load_time_ms: float = 0.0
    preprocessing_time_ms: float = 0.0
    postprocessing_time_ms: float = 0.0


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_batch_size: int = 1000
    max_wait_time_ms: int = 100
    min_batch_size: int = 1
    priority_boost_factor: float = 2.0
    auto_scaling_enabled: bool = True
    gpu_memory_threshold: float = 0.8


class BatchInferenceOptimizer:
    """Advanced batch inference optimizer with intelligent batching and scheduling."""
    
    def __init__(self, 
                 model_cache: ModelCacheManager,
                 max_concurrent_batches: int = 4,
                 enable_gpu_optimization: bool = True):
        self.model_cache = model_cache
        self.max_concurrent_batches = max_concurrent_batches
        self.enable_gpu_optimization = enable_gpu_optimization and torch.cuda.is_available()
        
        # Request queues by priority
        self._request_queues = {
            'critical': deque(),
            'high': deque(),
            'medium': deque(),
            'low': deque()
        }
        self._queue_lock = threading.RLock()
        
        # Batching configuration per model
        self._batch_configs: Dict[str, BatchConfig] = {}
        self._default_batch_config = BatchConfig()
        
        # Active batches and scheduling
        self._active_batches: Dict[str, BatchInferenceJob] = {}
        self._batch_semaphore = asyncio.Semaphore(max_concurrent_batches)
        
        # Performance tracking
        self._model_performance: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._inference_history: deque = deque(maxlen=1000)
        
        # Background tasks
        self._batch_scheduler_task = None
        self._performance_monitor_task = None
        self._start_background_tasks()
        
        # Thread pool for CPU inference
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
    
    async def submit_inference_request(self, request: InferenceRequest) -> str:
        """Submit inference request and return request ID."""
        with self._queue_lock:
            self._request_queues[request.priority].append(request)
        
        logger.debug(f"Submitted inference request {request.request_id} for model {request.model_name}")
        
        # Record metrics
        metrics_collector.record_ml_request_submitted(
            model_name=request.model_name,
            priority=request.priority,
            queue_size=len(self._request_queues[request.priority])
        )
        
        return request.request_id
    
    async def get_inference_result(self, request_id: str, timeout: float = 30.0) -> Optional[np.ndarray]:
        """Get inference result for a specific request."""
        # This would typically use a result store (Redis, etc.)
        # For now, we'll use a simple in-memory approach
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check if result is available
            # In a real implementation, this would check a result store
            await asyncio.sleep(0.1)
        
        return None  # Timeout or not found
    
    async def process_batch_inference(self, model_name: str) -> Optional[BatchInferenceResult]:
        """Process a batch of inference requests for a specific model."""
        batch_config = self._get_batch_config(model_name)
        
        # Collect requests for batching
        requests = await self._collect_requests_for_batch(model_name, batch_config)
        
        if not requests:
            return None
        
        batch_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            async with self._batch_semaphore:
                # Load model from cache
                model_load_start = time.time()
                model, model_metadata = await self.model_cache.get_model(model_name)
                model_load_time = (time.time() - model_load_start) * 1000
                
                # Prepare batch input
                prep_start = time.time()
                batch_input = self._prepare_batch_input(requests, model_metadata.model_type)
                prep_time = (time.time() - prep_start) * 1000
                
                # Run inference
                inference_start = time.time()
                predictions = await self._run_batch_inference(
                    model, batch_input, model_metadata, batch_config
                )
                inference_time = (time.time() - inference_start) * 1000
                
                # Post-process results
                postproc_start = time.time()
                individual_results = self._split_batch_results(predictions, requests)
                postproc_time = (time.time() - postproc_start) * 1000
                
                total_time = (time.time() - start_time) * 1000
                throughput = len(requests) / (total_time / 1000)
                
                # Store individual results
                await self._store_individual_results(requests, individual_results)
                
                # Create batch result
                result = BatchInferenceResult(
                    batch_id=batch_id,
                    model_name=model_name,
                    predictions=predictions,
                    request_ids=[req.request_id for req in requests],
                    inference_time_ms=inference_time,
                    batch_size=len(requests),
                    throughput_per_second=throughput,
                    model_load_time_ms=model_load_time,
                    preprocessing_time_ms=prep_time,
                    postprocessing_time_ms=postproc_time
                )
                
                # Update performance tracking
                self._update_performance_metrics(model_name, result)
                
                # Record metrics
                metrics_collector.record_ml_batch_inference(
                    model_name=model_name,
                    batch_size=len(requests),
                    inference_time_ms=inference_time,
                    throughput=throughput
                )
                
                logger.info(f"Completed batch inference {batch_id} for {model_name}: "
                           f"{len(requests)} requests in {total_time:.1f}ms "
                           f"({throughput:.1f} req/s)")
                
                return result
                
        except Exception as e:
            logger.error(f"Batch inference failed for {model_name}: {e}")
            # Mark requests as failed
            for request in requests:
                if request.callback:
                    try:
                        await request.callback(request.request_id, None, str(e))
                    except Exception as callback_error:
                        logger.error(f"Callback error: {callback_error}")
            
            metrics_collector.record_ml_batch_error(model_name=model_name, error=str(e))
            return None
    
    async def _collect_requests_for_batch(self, 
                                        model_name: str, 
                                        batch_config: BatchConfig) -> List[InferenceRequest]:
        """Collect requests for batching with intelligent selection."""
        requests = []
        max_wait_start = time.time()
        
        # Priority order for request collection
        priority_order = ['critical', 'high', 'medium', 'low']
        
        while len(requests) < batch_config.max_batch_size:
            found_request = False
            
            with self._queue_lock:
                for priority in priority_order:
                    queue = self._request_queues[priority]
                    
                    # Look for requests for this model
                    for i, request in enumerate(queue):
                        if request.model_name == model_name:
                            requests.append(queue.popleft() if i == 0 else queue.pop(i))
                            found_request = True
                            break
                    
                    if found_request:
                        break
            
            if not found_request:
                # No more requests for this model
                break
            
            # Check if we should wait for more requests
            current_wait = (time.time() - max_wait_start) * 1000
            if (len(requests) >= batch_config.min_batch_size and 
                current_wait >= batch_config.max_wait_time_ms):
                break
        
        return requests
    
    def _prepare_batch_input(self, 
                           requests: List[InferenceRequest], 
                           model_type: str) -> Union[np.ndarray, torch.Tensor, Any]:
        """Prepare batch input from individual requests."""
        # Stack all input arrays
        input_arrays = [req.input_data for req in requests]
        
        if model_type == "pytorch":
            # Convert to PyTorch tensor
            batch_array = np.stack(input_arrays, axis=0)
            return torch.from_numpy(batch_array).float()
        
        elif model_type == "tensorflow":
            # Keep as numpy array for TensorFlow
            return np.stack(input_arrays, axis=0)
        
        elif model_type == "sklearn":
            # Stack for sklearn
            return np.vstack(input_arrays)
        
        else:
            return np.stack(input_arrays, axis=0)
    
    async def _run_batch_inference(self, 
                                 model: Any, 
                                 batch_input: Union[np.ndarray, torch.Tensor], 
                                 model_metadata: Any,
                                 batch_config: BatchConfig) -> np.ndarray:
        """Run inference on a batch of inputs."""
        
        if model_metadata.model_type == "pytorch":
            return await self._run_pytorch_inference(model, batch_input, batch_config)
        
        elif model_metadata.model_type == "tensorflow":
            return await self._run_tensorflow_inference(model, batch_input, batch_config)
        
        elif model_metadata.model_type == "sklearn":
            return await self._run_sklearn_inference(model, batch_input, batch_config)
        
        else:
            raise ValueError(f"Unsupported model type: {model_metadata.model_type}")
    
    async def _run_pytorch_inference(self, 
                                   model: torch.nn.Module, 
                                   batch_input: torch.Tensor,
                                   batch_config: BatchConfig) -> np.ndarray:
        """Run PyTorch model inference with optimizations."""
        
        def _inference():
            model.eval()
            with torch.no_grad():
                if self.enable_gpu_optimization and torch.cuda.is_available():
                    # GPU inference
                    device = torch.device('cuda')
                    batch_input_gpu = batch_input.to(device)
                    
                    # Check GPU memory usage
                    if torch.cuda.memory_fraction() > batch_config.gpu_memory_threshold:
                        # Use smaller sub-batches
                        sub_batch_size = batch_config.max_batch_size // 2
                        results = []
                        
                        for i in range(0, len(batch_input_gpu), sub_batch_size):
                            sub_batch = batch_input_gpu[i:i + sub_batch_size]
                            sub_result = model(sub_batch)
                            results.append(sub_result.cpu())
                        
                        return torch.cat(results, dim=0).numpy()
                    else:
                        # Full batch GPU inference
                        output = model(batch_input_gpu)
                        return output.cpu().numpy()
                else:
                    # CPU inference
                    output = model(batch_input)
                    return output.numpy()
        
        # Run inference in thread pool for better async handling
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self._thread_pool, _inference)
        return result
    
    async def _run_tensorflow_inference(self, 
                                      model: Any, 
                                      batch_input: np.ndarray,
                                      batch_config: BatchConfig) -> np.ndarray:
        """Run TensorFlow model inference with optimizations."""
        
        def _inference():
            if hasattr(model, 'signatures'):
                # SavedModel inference
                infer = model.signatures['serving_default']
                input_tensor = tf.constant(batch_input)
                output = infer(input_tensor)
                
                # Extract output tensor (assuming single output)
                output_key = list(output.keys())[0]
                return output[output_key].numpy()
            
            elif hasattr(model, 'allocate_tensors'):
                # TensorFlow Lite inference
                input_details = model.get_input_details()
                output_details = model.get_output_details()
                
                # Process in sub-batches for TFLite
                results = []
                for i in range(len(batch_input)):
                    model.set_tensor(input_details[0]['index'], batch_input[i:i+1])
                    model.invoke()
                    output_data = model.get_tensor(output_details[0]['index'])
                    results.append(output_data)
                
                return np.concatenate(results, axis=0)
            
            else:
                # Direct model call
                return model(batch_input).numpy()
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self._thread_pool, _inference)
        return result
    
    async def _run_sklearn_inference(self, 
                                   model: BaseEstimator, 
                                   batch_input: np.ndarray,
                                   batch_config: BatchConfig) -> np.ndarray:
        """Run scikit-learn model inference."""
        
        def _inference():
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(batch_input)
            else:
                return model.predict(batch_input)
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(self._thread_pool, _inference)
        
        # Ensure 2D output
        if result.ndim == 1:
            result = result.reshape(-1, 1)
        
        return result
    
    def _split_batch_results(self, 
                           predictions: np.ndarray, 
                           requests: List[InferenceRequest]) -> List[np.ndarray]:
        """Split batch predictions into individual results."""
        individual_results = []
        
        for i, request in enumerate(requests):
            individual_result = predictions[i]
            individual_results.append(individual_result)
        
        return individual_results
    
    async def _store_individual_results(self, 
                                      requests: List[InferenceRequest], 
                                      results: List[np.ndarray]):
        """Store individual results and trigger callbacks."""
        for request, result in zip(requests, results):
            # In a real implementation, store in Redis or similar
            # For now, just trigger callbacks
            if request.callback:
                try:
                    await request.callback(request.request_id, result, None)
                except Exception as e:
                    logger.error(f"Callback error for request {request.request_id}: {e}")
    
    def _get_batch_config(self, model_name: str) -> BatchConfig:
        """Get batching configuration for a specific model."""
        return self._batch_configs.get(model_name, self._default_batch_config)
    
    def set_batch_config(self, model_name: str, config: BatchConfig):
        """Set batching configuration for a specific model."""
        self._batch_configs[model_name] = config
        logger.info(f"Updated batch config for {model_name}: "
                   f"max_batch_size={config.max_batch_size}, "
                   f"max_wait_time={config.max_wait_time_ms}ms")
    
    def _update_performance_metrics(self, model_name: str, result: BatchInferenceResult):
        """Update performance metrics for adaptive optimization."""
        perf = self._model_performance[model_name]
        
        # Update moving averages
        alpha = 0.1  # Smoothing factor
        
        if 'avg_inference_time' in perf:
            perf['avg_inference_time'] = (
                alpha * result.inference_time_ms + 
                (1 - alpha) * perf['avg_inference_time']
            )
        else:
            perf['avg_inference_time'] = result.inference_time_ms
        
        if 'avg_throughput' in perf:
            perf['avg_throughput'] = (
                alpha * result.throughput_per_second + 
                (1 - alpha) * perf['avg_throughput']
            )
        else:
            perf['avg_throughput'] = result.throughput_per_second
        
        perf['last_batch_size'] = result.batch_size
        perf['total_inferences'] = perf.get('total_inferences', 0) + result.batch_size
        
        # Store in history
        self._inference_history.append({
            'model_name': model_name,
            'timestamp': datetime.now(),
            'batch_size': result.batch_size,
            'inference_time_ms': result.inference_time_ms,
            'throughput': result.throughput_per_second
        })
        
        # Adaptive batch size optimization
        self._optimize_batch_config(model_name, result)
    
    def _optimize_batch_config(self, model_name: str, result: BatchInferenceResult):
        """Optimize batch configuration based on performance metrics."""
        config = self._get_batch_config(model_name)
        perf = self._model_performance[model_name]
        
        # Simple adaptive algorithm
        if result.throughput_per_second > perf.get('best_throughput', 0):
            perf['best_throughput'] = result.throughput_per_second
            perf['best_batch_size'] = result.batch_size
        
        # Adjust batch size based on throughput trends
        recent_history = [
            h for h in self._inference_history 
            if h['model_name'] == model_name and 
               h['timestamp'] > datetime.now() - timedelta(minutes=5)
        ]
        
        if len(recent_history) >= 3:
            throughputs = [h['throughput'] for h in recent_history[-3:]]
            if all(t < throughputs[0] * 0.9 for t in throughputs[1:]):
                # Decreasing throughput, reduce batch size
                new_batch_size = max(config.min_batch_size, int(config.max_batch_size * 0.8))
                config.max_batch_size = new_batch_size
                logger.info(f"Reduced batch size for {model_name} to {new_batch_size}")
    
    def _start_background_tasks(self):
        """Start background processing tasks."""
        
        async def batch_scheduler():
            """Main batch scheduling loop."""
            while True:
                try:
                    # Check each model for pending requests
                    model_requests = defaultdict(int)
                    
                    with self._queue_lock:
                        for priority_queue in self._request_queues.values():
                            for request in priority_queue:
                                model_requests[request.model_name] += 1
                    
                    # Process models with pending requests
                    tasks = []
                    for model_name, request_count in model_requests.items():
                        if request_count > 0:
                            task = self.process_batch_inference(model_name)
                            tasks.append(task)
                    
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                    
                    await asyncio.sleep(0.05)  # 50ms scheduling interval
                    
                except Exception as e:
                    logger.error(f"Batch scheduler error: {e}")
                    await asyncio.sleep(1)
        
        async def performance_monitor():
            """Monitor and log performance metrics."""
            while True:
                try:
                    await asyncio.sleep(60)  # Monitor every minute
                    
                    # Log performance summary
                    total_requests = sum(len(q) for q in self._request_queues.values())
                    
                    if total_requests > 0 or self._model_performance:
                        logger.info(f"Batch inference status: "
                                   f"{total_requests} pending requests, "
                                   f"{len(self._model_performance)} active models")
                        
                        for model_name, perf in self._model_performance.items():
                            if 'avg_throughput' in perf:
                                logger.info(f"  {model_name}: "
                                           f"{perf['avg_throughput']:.1f} req/s avg, "
                                           f"{perf.get('total_inferences', 0)} total")
                
                except Exception as e:
                    logger.error(f"Performance monitor error: {e}")
        
        self._batch_scheduler_task = asyncio.create_task(batch_scheduler())
        self._performance_monitor_task = asyncio.create_task(performance_monitor())
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status and performance metrics."""
        with self._queue_lock:
            queue_sizes = {
                priority: len(queue) 
                for priority, queue in self._request_queues.items()
            }
        
        total_pending = sum(queue_sizes.values())
        
        # Performance summary
        performance_summary = {}
        for model_name, perf in self._model_performance.items():
            performance_summary[model_name] = {
                'avg_throughput': perf.get('avg_throughput', 0),
                'avg_inference_time_ms': perf.get('avg_inference_time', 0),
                'total_inferences': perf.get('total_inferences', 0),
                'best_throughput': perf.get('best_throughput', 0)
            }
        
        return {
            'queue_sizes': queue_sizes,
            'total_pending_requests': total_pending,
            'active_batches': len(self._active_batches),
            'model_performance': performance_summary,
            'recent_throughput_history': list(self._inference_history)[-10:]  # Last 10 entries
        }
    
    async def shutdown(self):
        """Gracefully shutdown the batch processor."""
        if self._batch_scheduler_task:
            self._batch_scheduler_task.cancel()
        
        if self._performance_monitor_task:
            self._performance_monitor_task.cancel()
        
        self._thread_pool.shutdown(wait=True)
        
        logger.info("Batch inference optimizer shut down")


# Global batch inference optimizer
batch_inference_optimizer = BatchInferenceOptimizer(model_cache_manager)
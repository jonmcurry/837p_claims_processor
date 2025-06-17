"""Ultra high-performance ML predictor with model optimization, async processing, and caching."""

import asyncio
import hashlib
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import structlog
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from src.core.config.settings import settings
from src.core.cache.rvu_cache import rvu_cache
from src.core.database.models import Claim, ClaimLineItem
from src.processing.ml_pipeline.predictor import FeatureExtractor

logger = structlog.get_logger(__name__)


@dataclass
class MLPredictionResult:
    """ML prediction result with metadata."""
    
    should_process: bool
    confidence: float
    reason: str
    model_used: str
    prediction_time_ms: float
    cache_hit: bool = False
    feature_hash: Optional[str] = None


@dataclass 
class DynamicBatchConfig:
    """Dynamic batch configuration based on system resources."""
    
    base_batch_size: int = 500
    max_batch_size: int = 5000
    min_batch_size: int = 50
    cpu_threshold: float = 80.0  # CPU usage threshold
    memory_threshold: float = 75.0  # Memory usage threshold
    current_batch_size: int = field(default_factory=lambda: 500)
    
    def adjust_batch_size(self, cpu_usage: float, memory_usage: float, processing_time: float):
        """Dynamically adjust batch size based on system resources."""
        # Increase batch size if resources are available and processing is fast
        if cpu_usage < 60 and memory_usage < 60 and processing_time < 1.0:
            self.current_batch_size = min(
                self.current_batch_size + 100,
                self.max_batch_size
            )
        # Decrease batch size if system is stressed or processing is slow
        elif cpu_usage > self.cpu_threshold or memory_usage > self.memory_threshold or processing_time > 3.0:
            self.current_batch_size = max(
                self.current_batch_size - 100,
                self.min_batch_size
            )
            
        logger.debug("Adjusted ML batch size",
                    current_size=self.current_batch_size,
                    cpu_usage=cpu_usage,
                    memory_usage=memory_usage,
                    processing_time=processing_time)


class ModelOptimizer:
    """Model optimization for faster inference."""
    
    def __init__(self):
        self.optimized_models = {}
        
    def optimize_tensorflow_model(self, model: tf.keras.Model) -> tf.keras.Model:
        """Optimize TensorFlow model with quantization and pruning."""
        try:
            logger.info("Optimizing TensorFlow model...")
            
            # Apply post-training quantization
            optimized_model = self._quantize_model(model)
            
            # Additional optimizations
            optimized_model = self._optimize_inference(optimized_model)
            
            logger.info("TensorFlow model optimization completed")
            return optimized_model
            
        except Exception as e:
            logger.warning(f"Model optimization failed: {e}, using original model")
            return model
            
    def _quantize_model(self, model: tf.keras.Model) -> tf.keras.Model:
        """Apply post-training quantization to reduce model size and increase speed."""
        try:
            # Create a representative dataset for quantization
            # (In production, this would be real data)
            def representative_data_gen():
                for _ in range(100):
                    # Generate sample data matching model input shape
                    sample = np.random.random((1, model.input_shape[1])).astype(np.float32)
                    yield [sample]
            
            # Convert to TensorFlow Lite with quantization
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            
            tflite_model = converter.convert()
            
            # Create TFLite interpreter
            interpreter = tf.lite.Interpreter(model_content=tflite_model)
            interpreter.allocate_tensors()
            
            # Wrap interpreter in a Keras-like interface
            return self._create_tflite_wrapper(interpreter)
            
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            return model
            
    def _create_tflite_wrapper(self, interpreter):
        """Create a wrapper around TFLite interpreter to mimic Keras model interface."""
        class TFLiteWrapper:
            def __init__(self, interpreter):
                self.interpreter = interpreter
                self.input_details = interpreter.get_input_details()
                self.output_details = interpreter.get_output_details()
                
            def predict(self, x, verbose=0):
                """Predict using TFLite interpreter."""
                # Convert input to int8 if needed
                input_data = x.astype(self.input_details[0]['dtype'])
                
                # Set input tensor
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                
                # Run inference
                self.interpreter.invoke()
                
                # Get output
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
                
                # Convert back to float32 for compatibility
                return output_data.astype(np.float32)
                
        return TFLiteWrapper(interpreter)
        
    def _optimize_inference(self, model) -> tf.keras.Model:
        """Additional inference optimizations."""
        try:
            # For TensorFlow models, we can enable XLA compilation
            if hasattr(model, 'compile'):
                model.compile(
                    optimizer=model.optimizer,
                    loss=model.loss,
                    metrics=model.metrics,
                    jit_compile=True  # Enable XLA compilation
                )
                
            return model
            
        except Exception as e:
            logger.warning(f"Inference optimization failed: {e}")
            return model


class MLPredictionCache:
    """High-performance caching for ML predictions."""
    
    def __init__(self, max_size: int = 50000, ttl_seconds: int = 3600):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
        
    def _generate_feature_hash(self, features: Dict[str, Any]) -> str:
        """Generate a hash for feature dictionary."""
        # Create a stable hash of the features
        feature_str = str(sorted(features.items()))
        return hashlib.md5(feature_str.encode()).hexdigest()
        
    def get(self, features: Dict[str, Any]) -> Optional[MLPredictionResult]:
        """Get cached prediction for features."""
        feature_hash = self._generate_feature_hash(features)
        current_time = time.time()
        
        if feature_hash in self.cache:
            result, timestamp = self.cache[feature_hash]
            
            # Check TTL
            if current_time - timestamp < self.ttl_seconds:
                self.access_times[feature_hash] = current_time
                self.hits += 1
                
                # Mark as cache hit
                result.cache_hit = True
                result.feature_hash = feature_hash
                
                return result
            else:
                # Expired entry
                del self.cache[feature_hash]
                del self.access_times[feature_hash]
                
        self.misses += 1
        return None
        
    def put(self, features: Dict[str, Any], result: MLPredictionResult):
        """Cache a prediction result."""
        feature_hash = self._generate_feature_hash(features)
        current_time = time.time()
        
        # Evict old entries if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_lru()
            
        result.feature_hash = feature_hash
        self.cache[feature_hash] = (result, current_time)
        self.access_times[feature_hash] = current_time
        
    def _evict_lru(self):
        """Evict least recently used entries."""
        # Remove oldest 10% of entries
        num_to_remove = max(1, len(self.cache) // 10)
        
        # Sort by access time and remove oldest
        sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
        
        for feature_hash, _ in sorted_items[:num_to_remove]:
            if feature_hash in self.cache:
                del self.cache[feature_hash]
            if feature_hash in self.access_times:
                del self.access_times[feature_hash]
                
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate_percent': round(hit_rate, 2),
            'utilization_percent': round(len(self.cache) / self.max_size * 100, 2)
        }
        
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_times.clear()
        self.hits = 0
        self.misses = 0


class OptimizedClaimPredictor:
    """Ultra high-performance ML predictor with optimizations."""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.model_optimizer = ModelOptimizer()
        self.prediction_cache = MLPredictionCache()
        self.batch_config = DynamicBatchConfig()
        
        self.models = {}
        self.optimized_models = {}
        self.is_loaded = False
        
        # Async processing
        self.executor = ThreadPoolExecutor(max_workers=min(8, psutil.cpu_count()))
        self.prediction_semaphore = asyncio.Semaphore(32)  # Limit concurrent predictions
        
        # Performance tracking
        self.performance_stats = {
            'total_predictions': 0,
            'total_prediction_time': 0.0,
            'cache_enabled_predictions': 0,
            'avg_prediction_time_ms': 0.0
        }
        
    async def initialize(self):
        """Initialize the optimized predictor."""
        logger.info("Initializing ultra high-performance ML predictor...")
        start_time = time.time()
        
        try:
            # Load and optimize models
            await self._load_and_optimize_models()
            
            # Pre-warm the cache with common patterns
            await self._prewarm_cache()
            
            self.is_loaded = True
            
            init_time = time.time() - start_time
            logger.info(f"ML predictor initialized in {init_time:.2f}s")
            
        except Exception as e:
            logger.exception("ML predictor initialization failed", error=str(e))
            
    async def _load_and_optimize_models(self):
        """Load and optimize ML models."""
        try:
            model_path = settings.ml_model_path
            
            if model_path.exists():
                # Load TensorFlow model
                tf_model_path = model_path.parent / "tensorflow_model.h5"
                if tf_model_path.exists():
                    original_model = tf.keras.models.load_model(str(tf_model_path))
                    self.models["tensorflow"] = original_model
                    
                    # Optimize the model
                    optimized_model = self.model_optimizer.optimize_tensorflow_model(original_model)
                    self.optimized_models["tensorflow"] = optimized_model
                    
                    logger.info("TensorFlow model loaded and optimized")
                    
            # Initialize fallback model if no models available
            if not self.models and not self.optimized_models:
                await self._initialize_fast_fallback()
                
        except Exception as e:
            logger.exception("Model loading failed", error=str(e))
            await self._initialize_fast_fallback()
            
    async def _initialize_fast_fallback(self):
        """Initialize a fast rule-based fallback."""
        self.models["fast_fallback"] = FastRuleBasedPredictor()
        logger.info("Fast rule-based fallback initialized")
        
    async def _prewarm_cache(self):
        """Pre-warm the cache with common feature patterns."""
        try:
            # Generate common feature patterns for caching
            common_patterns = self._generate_common_patterns()
            
            logger.info(f"Pre-warming cache with {len(common_patterns)} common patterns")
            
            for features in common_patterns:
                # Generate a prediction and cache it
                result = await self._predict_features_internal(features)
                self.prediction_cache.put(features, result)
                
            logger.info("Cache pre-warming completed")
            
        except Exception as e:
            logger.warning(f"Cache pre-warming failed: {e}")
            
    def _generate_common_patterns(self) -> List[Dict[str, Any]]:
        """Generate common feature patterns for cache warming."""
        patterns = []
        
        # Common insurance types and financial classes
        insurance_types = ["Medicare", "Medicaid", "Commercial", "Self Pay"]
        financial_classes = ["A", "B", "MA", "HM", "BC", "SP"]
        
        # Generate combinations
        for insurance_type in insurance_types:
            for financial_class in financial_classes:
                pattern = {
                    "insurance_type_encoded": hash(insurance_type) % 1000,
                    "financial_class_encoded": hash(financial_class) % 1000,
                    "patient_age": 45,  # Common age
                    "total_charges": 1500.0,  # Common charge amount
                    "line_item_count": 3,  # Common line item count
                    "service_duration_days": 1,
                    "unique_procedures": 2,
                    "has_surgery_codes": 0,
                }
                patterns.append(pattern)
                
        return patterns[:50]  # Limit to 50 patterns
        
    async def predict_batch_optimized(self, claims: List[Claim], line_items_map: Dict[int, List[ClaimLineItem]] = None) -> List[MLPredictionResult]:
        """Ultra high-performance batch prediction with all optimizations."""
        start_time = time.time()
        
        if not self.is_loaded:
            await self.initialize()
            
        line_items_map = line_items_map or {}
        
        # Adjust batch size dynamically
        self._adjust_batch_size()
        
        # Process in optimized batches
        results = []
        batch_size = self.batch_config.current_batch_size
        
        for i in range(0, len(claims), batch_size):
            batch_claims = claims[i:i + batch_size]
            batch_start = time.time()
            
            # Process batch with async optimization
            batch_results = await self._process_batch_async(batch_claims, line_items_map)
            results.extend(batch_results)
            
            batch_time = time.time() - batch_start
            
            # Update batch configuration based on performance
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_usage = psutil.virtual_memory().percent
            self.batch_config.adjust_batch_size(cpu_usage, memory_usage, batch_time)
            
            logger.debug(f"Processed batch of {len(batch_claims)} claims in {batch_time:.2f}s")
            
        # Update performance stats
        total_time = time.time() - start_time
        self._update_performance_stats(len(claims), total_time)
        
        logger.info(f"ML batch prediction completed: {len(claims)} claims in {total_time:.2f}s "
                   f"({len(claims)/total_time:.0f} claims/sec)")
        
        return results
        
    async def _process_batch_async(self, claims: List[Claim], line_items_map: Dict[int, List[ClaimLineItem]]) -> List[MLPredictionResult]:
        """Process a batch of claims with async optimization."""
        # Extract features for all claims in parallel
        feature_tasks = []
        
        for claim in claims:
            task = asyncio.create_task(
                self._extract_features_async(claim, line_items_map.get(claim.id, []))
            )
            feature_tasks.append(task)
            
        batch_features = await asyncio.gather(*feature_tasks)
        
        # Process predictions with caching and async execution
        prediction_tasks = []
        
        for features in batch_features:
            task = asyncio.create_task(
                self._predict_with_cache_async(features)
            )
            prediction_tasks.append(task)
            
        results = await asyncio.gather(*prediction_tasks)
        return results
        
    async def _extract_features_async(self, claim: Claim, line_items: List[ClaimLineItem]) -> Dict[str, Any]:
        """Extract features asynchronously."""
        async with self.prediction_semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                self.feature_extractor.extract_claim_features,
                claim,
                line_items
            )
            
    async def _predict_with_cache_async(self, features: Dict[str, Any]) -> MLPredictionResult:
        """Predict with caching optimization."""
        # Check cache first
        cached_result = self.prediction_cache.get(features)
        if cached_result:
            return cached_result
            
        # Make prediction
        result = await self._predict_features_internal(features)
        
        # Cache the result
        self.prediction_cache.put(features, result)
        
        return result
        
    async def _predict_features_internal(self, features: Dict[str, Any]) -> MLPredictionResult:
        """Internal prediction method."""
        start_time = time.time()
        
        try:
            # Use optimized model if available
            if "tensorflow" in self.optimized_models:
                prediction = await self._predict_tensorflow_optimized(features)
            elif "tensorflow" in self.models:
                prediction = await self._predict_tensorflow_standard(features)
            elif "fast_fallback" in self.models:
                prediction = await self._predict_fast_fallback(features)
            else:
                # Default safe prediction
                prediction = {
                    "should_process": True,
                    "confidence": 0.5,
                    "reason": "no_model_available",
                    "model_used": "default"
                }
                
            prediction_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return MLPredictionResult(
                should_process=prediction["should_process"],
                confidence=prediction["confidence"],
                reason=prediction["reason"],
                model_used=prediction["model_used"],
                prediction_time_ms=prediction_time,
                cache_hit=False
            )
            
        except Exception as e:
            logger.exception("Prediction failed", error=str(e))
            
            return MLPredictionResult(
                should_process=True,
                confidence=0.5,
                reason="prediction_error",
                model_used="error_fallback",
                prediction_time_ms=(time.time() - start_time) * 1000,
                cache_hit=False
            )
            
    async def _predict_tensorflow_optimized(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict using optimized TensorFlow model."""
        async with self.prediction_semaphore:
            loop = asyncio.get_event_loop()
            
            def predict_sync():
                model = self.optimized_models["tensorflow"]
                feature_vector = self._features_to_vector(features)
                feature_array = np.array([feature_vector])
                
                prediction = model.predict(feature_array, verbose=0)[0]
                
                return {
                    "should_process": prediction[0] > settings.ml_prediction_threshold,
                    "confidence": float(prediction[0]),
                    "reason": "ml_prediction",
                    "model_used": "tensorflow_optimized"
                }
                
            return await loop.run_in_executor(self.executor, predict_sync)
            
    async def _predict_tensorflow_standard(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict using standard TensorFlow model."""
        async with self.prediction_semaphore:
            loop = asyncio.get_event_loop()
            
            def predict_sync():
                model = self.models["tensorflow"]
                feature_vector = self._features_to_vector(features)
                feature_array = np.array([feature_vector])
                
                prediction = model.predict(feature_array, verbose=0)[0]
                
                return {
                    "should_process": prediction[0] > settings.ml_prediction_threshold,
                    "confidence": float(prediction[0]),
                    "reason": "ml_prediction",
                    "model_used": "tensorflow_standard"
                }
                
            return await loop.run_in_executor(self.executor, predict_sync)
            
    async def _predict_fast_fallback(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict using fast rule-based fallback."""
        predictor = self.models["fast_fallback"]
        return await predictor.predict_features(features)
        
    def _features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert feature dictionary to numpy vector (optimized)."""
        # Use a pre-defined feature order for faster conversion
        expected_features = [
            "total_charges", "line_item_count", "patient_age", "service_duration_days",
            "unique_procedures", "has_surgery_codes", "insurance_type_encoded",
            "financial_class_encoded", "avg_line_item_charge", "max_line_item_charge"
        ]
        
        # Fast vector creation
        vector = np.zeros(len(expected_features), dtype=np.float32)
        
        for i, feature_name in enumerate(expected_features):
            value = features.get(feature_name, 0)
            if isinstance(value, str):
                value = hash(value) % 1000
            elif value is None:
                value = 0
            vector[i] = float(value)
            
        return vector
        
    def _adjust_batch_size(self):
        """Adjust batch size based on current system resources."""
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_usage = psutil.virtual_memory().percent
        
        # Simple adjustment based on current load
        if cpu_usage > 80 or memory_usage > 80:
            self.batch_config.current_batch_size = max(
                self.batch_config.min_batch_size,
                self.batch_config.current_batch_size - 50
            )
        elif cpu_usage < 50 and memory_usage < 50:
            self.batch_config.current_batch_size = min(
                self.batch_config.max_batch_size,
                self.batch_config.current_batch_size + 50
            )
            
    def _update_performance_stats(self, claim_count: int, total_time: float):
        """Update performance statistics."""
        self.performance_stats['total_predictions'] += claim_count
        self.performance_stats['total_prediction_time'] += total_time
        
        if self.performance_stats['total_predictions'] > 0:
            self.performance_stats['avg_prediction_time_ms'] = (
                self.performance_stats['total_prediction_time'] / 
                self.performance_stats['total_predictions'] * 1000
            )
            
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        cache_stats = self.prediction_cache.get_stats()
        
        return {
            'prediction_stats': self.performance_stats,
            'cache_stats': cache_stats,
            'batch_config': {
                'current_batch_size': self.batch_config.current_batch_size,
                'min_batch_size': self.batch_config.min_batch_size,
                'max_batch_size': self.batch_config.max_batch_size,
            },
            'models_loaded': list(self.models.keys()),
            'optimized_models': list(self.optimized_models.keys()),
        }
        
    async def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=False)
        self.prediction_cache.clear()


class FastRuleBasedPredictor:
    """Ultra-fast rule-based predictor for fallback."""
    
    async def predict_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Fast rule-based prediction using features."""
        total_charges = features.get("total_charges", 0)
        line_item_count = features.get("line_item_count", 0)
        patient_age = features.get("patient_age", 0)
        
        # Fast rules
        if total_charges > 50000:
            return {
                "should_process": False,
                "confidence": 0.9,
                "reason": "high_charges",
                "model_used": "fast_rules"
            }
            
        if line_item_count == 0:
            return {
                "should_process": False,
                "confidence": 0.95,
                "reason": "no_line_items",
                "model_used": "fast_rules"
            }
            
        if patient_age > 120 or patient_age < 0:
            return {
                "should_process": False,
                "confidence": 0.8,
                "reason": "invalid_age",
                "model_used": "fast_rules"
            }
            
        # Accept most other claims
        return {
            "should_process": True,
            "confidence": 0.75,
            "reason": "fast_rules_approval",
            "model_used": "fast_rules"
        }


# Global optimized predictor instance
optimized_predictor = OptimizedClaimPredictor()
"""Async ML processing manager for seamless pipeline integration."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import psutil
import structlog

from src.core.config.settings import settings
from src.core.database.models import Claim, ClaimLineItem
from src.processing.ml_pipeline.optimized_predictor import optimized_predictor, MLPredictionResult

logger = structlog.get_logger(__name__)


@dataclass
class MLProcessingMetrics:
    """Metrics for ML processing performance."""
    
    total_claims_processed: int = 0
    total_processing_time: float = 0.0
    cache_hit_rate: float = 0.0
    avg_prediction_time_ms: float = 0.0
    throughput_claims_per_sec: float = 0.0
    concurrent_predictions: int = 0
    peak_concurrent_predictions: int = 0


class AsyncMLManager:
    """High-performance async ML processing manager."""
    
    def __init__(self):
        self.is_initialized = False
        self.processing_queue = asyncio.Queue(maxsize=10000)
        self.result_cache = {}
        self.processing_semaphore = asyncio.Semaphore(50)  # Control concurrency
        
        # Performance tracking
        self.metrics = MLProcessingMetrics()
        self.active_predictions = 0
        
        # Background processing
        self.background_tasks = []
        self.processing_enabled = True
        
    async def initialize(self):
        """Initialize the async ML manager."""
        if self.is_initialized:
            return
            
        logger.info("Initializing async ML processing manager...")
        start_time = time.time()
        
        try:
            # Initialize the optimized predictor
            await optimized_predictor.initialize()
            
            # Start background processing workers
            await self._start_background_workers()
            
            self.is_initialized = True
            
            init_time = time.time() - start_time
            logger.info(f"Async ML manager initialized in {init_time:.2f}s")
            
        except Exception as e:
            logger.exception("Async ML manager initialization failed", error=str(e))
            
    async def _start_background_workers(self):
        """Start background processing workers."""
        # Start multiple worker tasks for concurrent processing
        num_workers = min(8, psutil.cpu_count())
        
        for i in range(num_workers):
            task = asyncio.create_task(self._background_worker(f"worker-{i}"))
            self.background_tasks.append(task)
            
        logger.info(f"Started {num_workers} background ML workers")
        
    async def _background_worker(self, worker_id: str):
        """Background worker for processing ML prediction requests."""
        logger.debug(f"Background ML worker {worker_id} started")
        
        try:
            while self.processing_enabled:
                try:
                    # Get next batch from queue with timeout
                    batch_request = await asyncio.wait_for(
                        self.processing_queue.get(),
                        timeout=1.0
                    )
                    
                    if batch_request is None:  # Shutdown signal
                        break
                        
                    # Process the batch
                    await self._process_batch_request(batch_request, worker_id)
                    
                    # Mark task as done
                    self.processing_queue.task_done()
                    
                except asyncio.TimeoutError:
                    # No work available, continue
                    continue
                except Exception as e:
                    logger.exception(f"Worker {worker_id} error", error=str(e))
                    
        except Exception as e:
            logger.exception(f"Background worker {worker_id} failed", error=str(e))
            
        logger.debug(f"Background ML worker {worker_id} stopped")
        
    async def _process_batch_request(self, batch_request: Dict, worker_id: str):
        """Process a batch prediction request."""
        claims = batch_request['claims']
        line_items_map = batch_request['line_items_map']
        result_future = batch_request['result_future']
        
        try:
            start_time = time.time()
            
            # Track concurrent predictions
            self.active_predictions += len(claims)
            self.metrics.peak_concurrent_predictions = max(
                self.metrics.peak_concurrent_predictions,
                self.active_predictions
            )
            
            # Process predictions with optimized predictor
            predictions = await optimized_predictor.predict_batch_optimized(
                claims, line_items_map
            )
            
            processing_time = time.time() - start_time
            
            # Update metrics
            self._update_metrics(len(claims), processing_time, predictions)
            
            # Set result
            result_future.set_result(predictions)
            
            logger.debug(f"Worker {worker_id} processed {len(claims)} claims in {processing_time:.2f}s")
            
        except Exception as e:
            logger.exception(f"Batch processing failed in worker {worker_id}", error=str(e))
            result_future.set_exception(e)
        finally:
            self.active_predictions -= len(claims)
            
    def _update_metrics(self, claim_count: int, processing_time: float, predictions: List[MLPredictionResult]):
        """Update processing metrics."""
        self.metrics.total_claims_processed += claim_count
        self.metrics.total_processing_time += processing_time
        
        # Calculate throughput
        if processing_time > 0:
            self.metrics.throughput_claims_per_sec = claim_count / processing_time
            
        # Calculate average prediction time
        if predictions:
            total_prediction_time = sum(p.prediction_time_ms for p in predictions)
            self.metrics.avg_prediction_time_ms = total_prediction_time / len(predictions)
            
        # Calculate cache hit rate
        if predictions:
            cache_hits = sum(1 for p in predictions if p.cache_hit)
            self.metrics.cache_hit_rate = (cache_hits / len(predictions)) * 100
            
    async def predict_claims_async(self, claims: List[Claim], line_items_map: Dict[int, List[ClaimLineItem]] = None) -> List[MLPredictionResult]:
        """Async prediction method for pipeline integration."""
        if not self.is_initialized:
            await self.initialize()
            
        if not claims:
            return []
            
        line_items_map = line_items_map or {}
        
        # For small batches, process directly
        if len(claims) <= 100:
            return await optimized_predictor.predict_batch_optimized(claims, line_items_map)
            
        # For larger batches, use background processing
        return await self._process_large_batch_async(claims, line_items_map)
        
    async def _process_large_batch_async(self, claims: List[Claim], line_items_map: Dict[int, List[ClaimLineItem]]) -> List[MLPredictionResult]:
        """Process large batches using background workers."""
        # Split into smaller chunks for parallel processing
        chunk_size = 500  # Process in chunks of 500
        chunks = [claims[i:i + chunk_size] for i in range(0, len(claims), chunk_size)]
        
        # Create futures for all chunks
        chunk_futures = []
        
        for chunk in chunks:
            # Create chunk-specific line items map
            chunk_line_items_map = {}
            for claim in chunk:
                if claim.id in line_items_map:
                    chunk_line_items_map[claim.id] = line_items_map[claim.id]
                    
            # Create future for result
            result_future = asyncio.Future()
            
            # Submit to processing queue
            batch_request = {
                'claims': chunk,
                'line_items_map': chunk_line_items_map,
                'result_future': result_future
            }
            
            await self.processing_queue.put(batch_request)
            chunk_futures.append(result_future)
            
        # Wait for all chunks to complete
        chunk_results = await asyncio.gather(*chunk_futures)
        
        # Combine results
        all_predictions = []
        for chunk_result in chunk_results:
            all_predictions.extend(chunk_result)
            
        return all_predictions
        
    async def predict_claims_pipeline_optimized(self, claims_data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Optimized prediction for parallel pipeline integration."""
        if not claims_data:
            return [], []
            
        start_time = time.time()
        
        # Convert dict data to prediction format
        predictions = await self._predict_claims_dict_format(claims_data)
        
        # Separate approved and rejected claims
        approved_claims = []
        rejected_claims = []
        
        for i, prediction in enumerate(predictions):
            claim_data = claims_data[i]
            
            if prediction.should_process:
                # Add ML metadata to claim
                claim_data['ml_prediction_score'] = prediction.confidence
                claim_data['ml_prediction_result'] = 'approved'
                claim_data['ml_model_used'] = prediction.model_used
                claim_data['ml_prediction_time_ms'] = prediction.prediction_time_ms
                claim_data['ml_cache_hit'] = prediction.cache_hit
                
                approved_claims.append(claim_data)
            else:
                # Create failed claim record
                failed_claim = {
                    'claim_id': claim_data.get('claim_id'),
                    'facility_id': claim_data.get('facility_id'),
                    'failure_category': 'ml_rejection',
                    'failure_reason': f"ML rejection: {prediction.reason} (confidence: {prediction.confidence:.2f})",
                    'claim_data': claim_data,
                    'ml_metadata': {
                        'model_used': prediction.model_used,
                        'confidence': prediction.confidence,
                        'reason': prediction.reason,
                        'prediction_time_ms': prediction.prediction_time_ms,
                        'cache_hit': prediction.cache_hit
                    }
                }
                rejected_claims.append(failed_claim)
                
        processing_time = time.time() - start_time
        
        logger.info(f"ML pipeline processing: {len(claims_data)} claims in {processing_time:.2f}s, "
                   f"{len(approved_claims)} approved, {len(rejected_claims)} rejected")
        
        return approved_claims, rejected_claims
        
    async def _predict_claims_dict_format(self, claims_data: List[Dict]) -> List[MLPredictionResult]:
        """Predict on claims in dictionary format."""
        # Extract features directly from dictionary data
        features_list = []
        
        for claim_data in claims_data:
            features = self._extract_features_from_dict(claim_data)
            features_list.append(features)
            
        # Batch prediction on features
        predictions = []
        
        # Process in batches for efficiency
        batch_size = 1000
        for i in range(0, len(features_list), batch_size):
            batch_features = features_list[i:i + batch_size]
            
            # Process batch with caching
            batch_predictions = await self._predict_features_batch(batch_features)
            predictions.extend(batch_predictions)
            
        return predictions
        
    def _extract_features_from_dict(self, claim_data: Dict) -> Dict[str, Any]:
        """Extract ML features from claim dictionary data."""
        # Quick feature extraction for performance
        line_items = claim_data.get('line_items', [])
        
        features = {
            'total_charges': float(claim_data.get('total_charges', 0)),
            'line_item_count': len(line_items),
            'patient_age': self._calculate_age(claim_data.get('patient_date_of_birth')),
            'service_duration_days': self._calculate_service_duration(claim_data),
            'insurance_type_encoded': hash(claim_data.get('insurance_type', '')) % 1000,
            'financial_class_encoded': hash(claim_data.get('financial_class', '')) % 1000,
            'unique_procedures': len(set(item.get('procedure_code', '') for item in line_items)),
            'has_surgery_codes': 1 if any(
                item.get('procedure_code', '').isdigit() and 10000 <= int(item.get('procedure_code', '0')) <= 69999
                for item in line_items
            ) else 0,
        }
        
        if line_items:
            charges = [float(item.get('charge_amount', 0)) for item in line_items]
            features.update({
                'avg_line_item_charge': sum(charges) / len(charges) if charges else 0,
                'max_line_item_charge': max(charges) if charges else 0,
            })
        else:
            features.update({
                'avg_line_item_charge': 0,
                'max_line_item_charge': 0,
            })
            
        return features
        
    def _calculate_age(self, birth_date) -> int:
        """Calculate age from birth date."""
        if not birth_date:
            return 45  # Default age
            
        try:
            from datetime import datetime
            if isinstance(birth_date, str):
                birth_date = datetime.fromisoformat(birth_date.replace('Z', '+00:00'))
            elif hasattr(birth_date, 'year'):
                pass  # Already a date object
            else:
                return 45
                
            return (datetime.now() - birth_date).days // 365
        except:
            return 45
            
    def _calculate_service_duration(self, claim_data: Dict) -> int:
        """Calculate service duration in days."""
        try:
            from datetime import datetime
            
            from_date = claim_data.get('service_from_date')
            to_date = claim_data.get('service_to_date')
            
            if not from_date or not to_date:
                return 1
                
            if isinstance(from_date, str):
                from_date = datetime.fromisoformat(from_date.replace('Z', '+00:00'))
            if isinstance(to_date, str):
                to_date = datetime.fromisoformat(to_date.replace('Z', '+00:00'))
                
            return (to_date - from_date).days + 1
        except:
            return 1
            
    async def _predict_features_batch(self, features_list: List[Dict]) -> List[MLPredictionResult]:
        """Predict on a batch of feature dictionaries."""
        predictions = []
        
        for features in features_list:
            # Check cache first
            cached_result = optimized_predictor.prediction_cache.get(features)
            if cached_result:
                predictions.append(cached_result)
            else:
                # Make prediction
                result = await optimized_predictor._predict_features_internal(features)
                optimized_predictor.prediction_cache.put(features, result)
                predictions.append(result)
                
        return predictions
        
    def get_processing_metrics(self) -> Dict:
        """Get comprehensive ML processing metrics."""
        # Get predictor stats
        predictor_stats = optimized_predictor.get_performance_stats()
        
        return {
            'ml_manager_metrics': {
                'total_claims_processed': self.metrics.total_claims_processed,
                'total_processing_time': self.metrics.total_processing_time,
                'avg_throughput_claims_per_sec': (
                    self.metrics.total_claims_processed / self.metrics.total_processing_time
                    if self.metrics.total_processing_time > 0 else 0
                ),
                'cache_hit_rate_percent': self.metrics.cache_hit_rate,
                'avg_prediction_time_ms': self.metrics.avg_prediction_time_ms,
                'active_predictions': self.active_predictions,
                'peak_concurrent_predictions': self.metrics.peak_concurrent_predictions,
                'queue_size': self.processing_queue.qsize(),
                'background_workers': len(self.background_tasks),
            },
            'predictor_stats': predictor_stats,
            'system_resources': {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'cpu_count': psutil.cpu_count(),
            }
        }
        
    async def shutdown(self):
        """Shutdown the async ML manager."""
        logger.info("Shutting down async ML manager...")
        
        self.processing_enabled = False
        
        # Signal workers to stop
        for _ in self.background_tasks:
            await self.processing_queue.put(None)
            
        # Wait for workers to finish
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Cleanup predictor
        await optimized_predictor.cleanup()
        
        logger.info("Async ML manager shutdown completed")


# Global async ML manager instance
async_ml_manager = AsyncMLManager()
"""Ultra high-performance async processing pipeline for 100k+ claims/15s."""

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor

import structlog
from sqlalchemy import select, update, text
from sqlalchemy.ext.asyncio import AsyncSession
from aiocache import cached, Cache
from aiocache.serializers import PickleSerializer

from src.core.config import settings
from src.core.database.base import get_postgres_session, get_sqlserver_session
from src.core.database.models import (
    BatchMetadata,
    Claim,
    ClaimLineItem,
    FailedClaim,
    PerformanceMetrics,
    ProcessingStatus,
)
from src.processing.calculations.rvu_calculator import RVUCalculator
from src.processing.ml_pipeline.predictor import ClaimPredictor
from src.processing.validation.rule_engine import ClaimValidator
from src.monitoring.metrics.prometheus_metrics import (
    claims_processed_total,
    claims_failed_total,
    processing_latency,
    throughput_gauge,
    batch_processing_time
)

logger = structlog.get_logger(__name__)


@dataclass
class ProcessingResult:
    """Results from processing a batch of claims."""

    batch_id: str
    total_claims: int
    processed_claims: int = 0
    failed_claims: int = 0
    processing_time: float = 0.0
    throughput: float = 0.0
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineStage:
    """Configuration for a pipeline stage."""

    name: str
    workers: int
    batch_size: int
    timeout: int = 30


class ClaimProcessingPipeline:
    """High-performance async claims processing pipeline."""

    def __init__(self):
        """Initialize the processing pipeline."""
        self.validator = ClaimValidator()
        self.predictor = ClaimPredictor()
        self.calculator = RVUCalculator()
        
        # Optimized pipeline stage configurations for 100k+ claims/15s
        # Target: 6,667 claims/second requires maximum parallelization
        cpu_count = psutil.cpu_count(logical=True)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Dynamic worker allocation based on system resources
        max_workers = min(cpu_count * 2, 32)  # Cap at 32 to prevent resource exhaustion
        
        self.stages = {
            "data_fetch": PipelineStage("Data Fetch", max_workers // 4, 2000, 45),
            "validation": PipelineStage("Validation", max_workers, 1000, 20),
            "ml_prediction": PipelineStage("ML Prediction", max_workers // 2, 500, 30),
            "calculation": PipelineStage("RVU Calculation", max_workers, 1000, 15),
            "data_transfer": PipelineStage("Data Transfer", max_workers // 2, 500, 25),
        }
        
        # Initialize thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Memory optimization settings
        self.memory_threshold = memory_gb * 0.8  # Use 80% of available memory
        self.batch_memory_limit = min(10000, int(memory_gb * 1000))  # Claims per batch
        
        # Performance tracking
        self.performance_stats = {
            "total_processed": 0,
            "total_failed": 0,
            "average_throughput": 0.0,
            "peak_throughput": 0.0
        }

    async def process_batch(self, batch_id: str) -> ProcessingResult:
        """Process a complete batch of claims through ultra high-performance pipeline."""
        start_time = time.perf_counter()
        
        # Start Prometheus metrics tracking
        with batch_processing_time.time():
            logger.info("Starting ultra high-performance batch processing", 
                       batch_id=batch_id,
                       target_throughput="6667 claims/sec")
            
            try:
                # Get batch metadata with connection pooling optimization
                batch = await self._get_batch_metadata_optimized(batch_id)
                if not batch:
                    raise ValueError(f"Batch {batch_id} not found")

                # Initialize result tracking
                result = ProcessingResult(
                    batch_id=batch_id,
                    total_claims=batch.total_claims,
                )

                # Check memory requirements and adjust batch size if needed
                if batch.total_claims > self.batch_memory_limit:
                    logger.warning("Large batch detected, will process in chunks",
                                 batch_id=batch_id,
                                 total_claims=batch.total_claims,
                                 chunk_size=self.batch_memory_limit)
                    return await self._process_large_batch_chunked(batch_id, batch)

                # Update batch status to processing
                await self._update_batch_status(batch_id, ProcessingStatus.PROCESSING)

                # Execute ultra high-performance pipeline with concurrent stages
                claims = await self._fetch_claims_optimized(batch_id)
                logger.info("Fetched claims for ultra high-speed processing", 
                           batch_id=batch_id, 
                           claim_count=len(claims))

                # Concurrent pipeline execution for maximum throughput
                stage_tasks = []
                
                # Stage 1: Validation (ultra parallel processing)
                validation_task = asyncio.create_task(
                    self._validate_claims_ultra_parallel(claims)
                )
                stage_tasks.append(("validation", validation_task))
                
                validated_claims, validation_failures = await validation_task
                result.failed_claims += len(validation_failures)
                claims_failed_total.inc(len(validation_failures))
                
                if not validated_claims:
                    logger.warning("No claims passed validation", batch_id=batch_id)
                    await self._finalize_batch(batch_id, result)
                    return result
                
                # Stage 2: ML Prediction (optimized batch processing)
                prediction_task = asyncio.create_task(
                    self._predict_claims_ultra_parallel(validated_claims)
                )
                stage_tasks.append(("ml_prediction", prediction_task))
                
                predicted_claims, prediction_failures = await prediction_task
                result.failed_claims += len(prediction_failures)
                claims_failed_total.inc(len(prediction_failures))
                
                # Stage 3: RVU Calculation (vectorized processing)
                calculation_task = asyncio.create_task(
                    self._calculate_claims_ultra_parallel(predicted_claims)
                )
                stage_tasks.append(("calculation", calculation_task))
                
                calculated_claims, calculation_failures = await calculation_task
                result.failed_claims += len(calculation_failures)
                claims_failed_total.inc(len(calculation_failures))
                
                # Stage 4: Data Transfer (bulk insert optimization)
                transfer_task = asyncio.create_task(
                    self._transfer_claims_ultra_parallel(calculated_claims)
                )
                stage_tasks.append(("data_transfer", transfer_task))
                
                transfer_success, transfer_failures = await transfer_task
                result.failed_claims += len(transfer_failures)
                result.processed_claims = len(transfer_success)
                
                # Update Prometheus metrics
                claims_processed_total.inc(result.processed_claims)
                claims_failed_total.inc(len(transfer_failures))

                # Calculate performance metrics
                result.processing_time = time.perf_counter() - start_time
                result.throughput = result.total_claims / result.processing_time if result.processing_time > 0 else 0
                
                # Update performance stats
                self.performance_stats["total_processed"] += result.processed_claims
                self.performance_stats["total_failed"] += result.failed_claims
                self.performance_stats["peak_throughput"] = max(
                    self.performance_stats["peak_throughput"], 
                    result.throughput
                )
                
                # Update Prometheus gauges
                throughput_gauge.set(result.throughput)
                processing_latency.observe(result.processing_time)

                # Update batch completion status
                await self._finalize_batch(batch_id, result)
                
                # Record performance metrics
                await self._record_performance_metrics(result)
                
                # Trigger garbage collection for memory optimization
                if result.total_claims > 10000:
                    gc.collect()

                logger.info("Ultra high-performance batch processing completed",
                           batch_id=batch_id,
                           processed=result.processed_claims,
                           failed=result.failed_claims,
                           throughput=f"{result.throughput:.2f} claims/sec",
                           duration=f"{result.processing_time:.3f}s",
                           target_met=result.throughput >= 6667)

                return result

            except Exception as e:
                logger.exception("Ultra high-performance batch processing failed", 
                               batch_id=batch_id, error=str(e))
                await self._update_batch_status(batch_id, ProcessingStatus.FAILED)
                raise

    async def _fetch_claims(self, batch_id: str) -> List[Claim]:
        """Fetch claims for processing from staging database."""
        async with get_postgres_session() as session:
            query = (
                select(Claim, ClaimLineItem)
                .join(ClaimLineItem, Claim.id == ClaimLineItem.claim_id)
                .where(Claim.batch_id == batch_id)
                .where(Claim.processing_status == ProcessingStatus.PENDING)
            )
            
            result = await session.execute(query)
            return result.scalars().all()

    async def _validate_claims_parallel(self, claims: List[Claim]) -> Tuple[List[Claim], List[Dict]]:
        """Validate claims in parallel using multiple workers."""
        stage = self.stages["validation"]
        
        async def validate_batch(claim_batch: List[Claim]) -> Tuple[List[Claim], List[Dict]]:
            valid_claims = []
            failures = []
            
            for claim in claim_batch:
                try:
                    is_valid, errors = await self.validator.validate_claim(claim)
                    if is_valid:
                        valid_claims.append(claim)
                    else:
                        failures.append({
                            "claim_id": claim.claim_id,
                            "errors": errors,
                            "stage": "validation"
                        })
                        await self._store_failed_claim(claim, "validation_error", errors)
                except Exception as e:
                    logger.exception("Validation error", claim_id=claim.claim_id)
                    failures.append({
                        "claim_id": claim.claim_id,
                        "errors": [str(e)],
                        "stage": "validation"
                    })
            
            return valid_claims, failures

        # Split claims into batches for parallel processing
        claim_batches = [
            claims[i:i + stage.batch_size]
            for i in range(0, len(claims), stage.batch_size)
        ]

        # Process batches in parallel with limited concurrency
        semaphore = asyncio.Semaphore(stage.workers)
        
        async def process_with_semaphore(batch):
            async with semaphore:
                return await validate_batch(batch)

        tasks = [process_with_semaphore(batch) for batch in claim_batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        all_valid_claims = []
        all_failures = []
        
        for result in results:
            if isinstance(result, Exception):
                logger.exception("Batch validation failed", error=result)
                continue
                
            valid_claims, failures = result
            all_valid_claims.extend(valid_claims)
            all_failures.extend(failures)

        return all_valid_claims, all_failures

    async def _predict_claims_parallel(self, claims: List[Claim]) -> Tuple[List[Claim], List[Dict]]:
        """Run ML predictions on claims in parallel."""
        stage = self.stages["ml_prediction"]
        
        if not settings.enable_ml_predictions:
            return claims, []

        async def predict_batch(claim_batch: List[Claim]) -> Tuple[List[Claim], List[Dict]]:
            predicted_claims = []
            failures = []
            
            try:
                # Batch prediction for efficiency
                predictions = await self.predictor.predict_batch(claim_batch)
                
                for claim, prediction in zip(claim_batch, predictions):
                    if prediction["should_process"]:
                        claim.ml_prediction_score = prediction["confidence"]
                        claim.ml_prediction_result = "approved"
                        predicted_claims.append(claim)
                    else:
                        failures.append({
                            "claim_id": claim.claim_id,
                            "errors": [f"ML rejection: {prediction['reason']}"],
                            "stage": "ml_prediction"
                        })
                        await self._store_failed_claim(claim, "ml_rejection", prediction)
                        
            except Exception as e:
                logger.exception("ML prediction batch failed")
                for claim in claim_batch:
                    failures.append({
                        "claim_id": claim.claim_id,
                        "errors": [f"ML prediction error: {str(e)}"],
                        "stage": "ml_prediction"
                    })
            
            return predicted_claims, failures

        # Process in smaller batches for ML efficiency
        claim_batches = [
            claims[i:i + stage.batch_size]
            for i in range(0, len(claims), stage.batch_size)
        ]

        semaphore = asyncio.Semaphore(stage.workers)
        
        async def process_with_semaphore(batch):
            async with semaphore:
                return await predict_batch(batch)

        tasks = [process_with_semaphore(batch) for batch in claim_batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        all_predicted_claims = []
        all_failures = []
        
        for result in results:
            if isinstance(result, Exception):
                logger.exception("Batch prediction failed", error=result)
                continue
                
            predicted_claims, failures = result
            all_predicted_claims.extend(predicted_claims)
            all_failures.extend(failures)

        return all_predicted_claims, all_failures

    async def _calculate_claims_parallel(self, claims: List[Claim]) -> Tuple[List[Claim], List[Dict]]:
        """Calculate RVU and reimbursement amounts in parallel."""
        stage = self.stages["calculation"]
        
        async def calculate_batch(claim_batch: List[Claim]) -> Tuple[List[Claim], List[Dict]]:
            calculated_claims = []
            failures = []
            
            for claim in claim_batch:
                try:
                    # Calculate RVU and expected reimbursement
                    await self.calculator.calculate_claim_rvus(claim)
                    calculated_claims.append(claim)
                    
                except Exception as e:
                    logger.exception("RVU calculation failed", claim_id=claim.claim_id)
                    failures.append({
                        "claim_id": claim.claim_id,
                        "errors": [f"Calculation error: {str(e)}"],
                        "stage": "calculation"
                    })
                    await self._store_failed_claim(claim, "financial_error", str(e))
            
            return calculated_claims, failures

        # Split into batches
        claim_batches = [
            claims[i:i + stage.batch_size]
            for i in range(0, len(claims), stage.batch_size)
        ]

        semaphore = asyncio.Semaphore(stage.workers)
        
        async def process_with_semaphore(batch):
            async with semaphore:
                return await calculate_batch(batch)

        tasks = [process_with_semaphore(batch) for batch in claim_batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        all_calculated_claims = []
        all_failures = []
        
        for result in results:
            if isinstance(result, Exception):
                logger.exception("Batch calculation failed", error=result)
                continue
                
            calculated_claims, failures = result
            all_calculated_claims.extend(calculated_claims)
            all_failures.extend(failures)

        return all_calculated_claims, all_failures

    async def _transfer_claims_parallel(self, claims: List[Claim]) -> Tuple[List[str], List[Dict]]:
        """Transfer processed claims to production database in parallel."""
        stage = self.stages["data_transfer"]
        
        async def transfer_batch(claim_batch: List[Claim]) -> Tuple[List[str], List[Dict]]:
            transferred = []
            failures = []
            
            try:
                async with get_sqlserver_session() as session:
                    for claim in claim_batch:
                        try:
                            # Transform and insert into production schema
                            await self._insert_claim_to_production(session, claim)
                            
                            # Update staging record status
                            async with get_postgres_session() as staging_session:
                                await staging_session.execute(
                                    update(Claim)
                                    .where(Claim.id == claim.id)
                                    .values(
                                        processing_status=ProcessingStatus.COMPLETED,
                                        processed_at=datetime.utcnow()
                                    )
                                )
                                await staging_session.commit()
                            
                            transferred.append(claim.claim_id)
                            
                        except Exception as e:
                            logger.exception("Individual claim transfer failed", 
                                           claim_id=claim.claim_id)
                            failures.append({
                                "claim_id": claim.claim_id,
                                "errors": [f"Transfer error: {str(e)}"],
                                "stage": "data_transfer"
                            })
                    
                    await session.commit()
                    
            except Exception as e:
                logger.exception("Batch transfer failed")
                for claim in claim_batch:
                    failures.append({
                        "claim_id": claim.claim_id,
                        "errors": [f"Batch transfer error: {str(e)}"],
                        "stage": "data_transfer"
                    })
            
            return transferred, failures

        # Split into batches
        claim_batches = [
            claims[i:i + stage.batch_size]
            for i in range(0, len(claims), stage.batch_size)
        ]

        semaphore = asyncio.Semaphore(stage.workers)
        
        async def process_with_semaphore(batch):
            async with semaphore:
                return await transfer_batch(batch)

        tasks = [process_with_semaphore(batch) for batch in claim_batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        all_transferred = []
        all_failures = []
        
        for result in results:
            if isinstance(result, Exception):
                logger.exception("Batch transfer failed", error=result)
                continue
                
            transferred, failures = result
            all_transferred.extend(transferred)
            all_failures.extend(failures)

        return all_transferred, all_failures

    async def _get_batch_metadata(self, session: AsyncSession, batch_id: str) -> Optional[BatchMetadata]:
        """Get batch metadata from database."""
        query = select(BatchMetadata).where(BatchMetadata.batch_id == batch_id)
        result = await session.execute(query)
        return result.scalar_one_or_none()

    async def _update_batch_status(self, batch_id: str, status: ProcessingStatus) -> None:
        """Update batch processing status."""
        async with get_postgres_session() as session:
            await session.execute(
                update(BatchMetadata)
                .where(BatchMetadata.batch_id == batch_id)
                .values(
                    status=status,
                    started_at=datetime.utcnow() if status == ProcessingStatus.PROCESSING else None,
                    completed_at=datetime.utcnow() if status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED] else None
                )
            )
            await session.commit()

    async def _finalize_batch(self, batch_id: str, result: ProcessingResult) -> None:
        """Finalize batch processing with results."""
        async with get_postgres_session() as session:
            await session.execute(
                update(BatchMetadata)
                .where(BatchMetadata.batch_id == batch_id)
                .values(
                    status=ProcessingStatus.COMPLETED,
                    completed_at=datetime.utcnow(),
                    processed_claims=result.processed_claims,
                    failed_claims=result.failed_claims,
                    processing_time_seconds=result.processing_time,
                    throughput_per_second=result.throughput,
                )
            )
            await session.commit()

    async def _store_failed_claim(self, claim: Claim, failure_category: str, error_details: Any) -> None:
        """Store failed claim for investigation."""
        async with get_postgres_session() as session:
            failed_claim = FailedClaim(
                original_claim_id=claim.id,
                claim_reference=claim.claim_id,
                facility_id=claim.facility_id,
                failure_category=failure_category,
                failure_reason=str(error_details),
                failure_details={"errors": error_details},
                claim_data={"claim_id": claim.claim_id, "facility_id": claim.facility_id},
                charge_amount=claim.total_charges,
                expected_reimbursement=claim.expected_reimbursement,
            )
            session.add(failed_claim)
            await session.commit()

    async def _insert_claim_to_production(self, session: AsyncSession, claim: Claim) -> None:
        """Insert claim into production SQL Server database."""
        # This would contain the actual SQL Server insert logic
        # For now, we'll simulate the operation
        pass

    async def _record_performance_metrics(self, result: ProcessingResult) -> None:
        """Record performance metrics for monitoring."""
        async with get_postgres_session() as session:
            metrics = [
                PerformanceMetrics(
                    metric_type="throughput",
                    metric_name="claims_per_second",
                    metric_value=result.throughput,
                    unit="claims/sec",
                    batch_id=result.batch_id,
                    service_name="claims_processor",
                ),
                PerformanceMetrics(
                    metric_type="latency",
                    metric_name="processing_time",
                    metric_value=result.processing_time,
                    unit="seconds",
                    batch_id=result.batch_id,
                    service_name="claims_processor",
                ),
                PerformanceMetrics(
                    metric_type="quality",
                    metric_name="success_rate",
                    metric_value=(result.processed_claims / result.total_claims * 100) if result.total_claims > 0 else 0,
                    unit="percent",
                    batch_id=result.batch_id,
                    service_name="claims_processor",
                ),
            ]
            
            session.add_all(metrics)
            await session.commit()


# Global pipeline instance
processing_pipeline = ClaimProcessingPipeline()
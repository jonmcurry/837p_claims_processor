"""High-performance parallel processing pipeline for 6,667+ claims/second throughput."""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import aiofiles
import structlog
from aiocache import cached, Cache
from aiocache.serializers import PickleSerializer

from src.core.config.settings import settings
from src.core.database.batch_operations import batch_ops
from src.core.database.pool_manager import pool_manager
from src.core.cache.rvu_cache import rvu_cache
from src.processing.ml_pipeline.async_ml_manager import async_ml_manager

logger = structlog.get_logger(__name__)


@dataclass
class ParallelProcessingResult:
    """Result from parallel processing pipeline."""
    
    total_claims: int = 0
    processed_claims: int = 0
    failed_claims: int = 0
    processing_time: float = 0.0
    throughput: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)
    errors: List[Dict] = field(default_factory=list)


class WorkerPool:
    """Optimized worker pool for CPU and I/O intensive tasks."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, (asyncio.cpu_count() or 4) * 4)
        self.io_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.cpu_executor = ProcessPoolExecutor(max_workers=min(16, asyncio.cpu_count() or 4))
        self.semaphore = asyncio.Semaphore(self.max_workers)
        
    async def submit_io_task(self, func, *args, **kwargs):
        """Submit I/O intensive task to thread pool."""
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.io_executor, func, *args, **kwargs)
            
    async def submit_cpu_task(self, func, *args, **kwargs):
        """Submit CPU intensive task to process pool."""
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.cpu_executor, func, *args, **kwargs)
            
    def shutdown(self):
        """Shutdown worker pools."""
        self.io_executor.shutdown(wait=False)
        self.cpu_executor.shutdown(wait=False)


class ParallelClaimsProcessor:
    """Ultra high-performance parallel claims processor targeting 6,667+ claims/second."""
    
    def __init__(self):
        self.worker_pool = WorkerPool()
        self.conversion_factor = Decimal("38.87")  # Default Medicare conversion factor
        
        # Performance tuning parameters
        self.batch_sizes = {
            'fetch': 5000,          # Large fetch batches
            'validation': 1000,     # Validation batch size
            'rvu_calculation': 2000,  # RVU calculation batch size
            'transfer': 1000,       # Transfer batch size
        }
        
        # Concurrency limits for optimal throughput
        self.concurrency_limits = {
            'validation': 50,       # High concurrency for validation
            'rvu_calculation': 40,  # High concurrency for calculations
            'transfer': 30,         # Moderate concurrency for database writes
        }
        
    async def process_claims_parallel(self, batch_id: str = None, limit: int = None) -> ParallelProcessingResult:
        """Process claims with maximum parallelization for 6,667+ claims/second."""
        start_time = time.time()
        result = ParallelProcessingResult()
        
        try:
            logger.info("Starting ultra high-performance parallel processing", 
                       target_throughput="6,667+ claims/sec")
            
            # Initialize systems
            await self._initialize_systems()
            
            # Stage 1: Parallel data fetching
            stage_start = time.time()
            claims_data = await self._fetch_claims_parallel(batch_id, limit)
            result.stage_times['fetch'] = time.time() - stage_start
            result.total_claims = len(claims_data)
            
            if not claims_data:
                logger.warning("No claims to process")
                return result
                
            logger.info(f"Fetched {len(claims_data)} claims for parallel processing")
            
            # Stage 2: Parallel validation + ML processing (combined for efficiency)
            stage_start = time.time()
            validated_claims, validation_failures = await self._validate_and_ml_process_parallel(claims_data)
            result.stage_times['validation_ml'] = time.time() - stage_start
            result.failed_claims += len(validation_failures)
            
            if validation_failures:
                await self._store_failed_claims_batch(validation_failures)
                
            logger.info(f"Validated and ML processed {len(validated_claims)} claims, {len(validation_failures)} failed")
            
            # Stage 3: Parallel RVU calculation
            stage_start = time.time()
            calculated_claims, calculation_failures = await self._calculate_claims_parallel(validated_claims)
            result.stage_times['calculation'] = time.time() - stage_start
            result.failed_claims += len(calculation_failures)
            
            if calculation_failures:
                await self._store_failed_claims_batch(calculation_failures)
                
            logger.info(f"Calculated {len(calculated_claims)} claims, {len(calculation_failures)} failed")
            
            # Stage 4: Parallel data transfer
            stage_start = time.time()
            successful_transfers, transfer_failures = await self._transfer_claims_parallel(calculated_claims)
            result.stage_times['transfer'] = time.time() - stage_start
            result.processed_claims = successful_transfers
            result.failed_claims += len(transfer_failures)
            
            if transfer_failures:
                await self._store_failed_claims_batch(transfer_failures)
                
            logger.info(f"Transferred {successful_transfers} claims, {len(transfer_failures)} failed")
            
            # Stage 5: Parallel status updates
            stage_start = time.time()
            await self._update_claim_statuses_parallel(calculated_claims)
            result.stage_times['status_update'] = time.time() - stage_start
            
            # Calculate final metrics
            result.processing_time = time.time() - start_time
            result.throughput = result.total_claims / result.processing_time if result.processing_time > 0 else 0
            
            # Log performance results
            self._log_performance_results(result)
            
            return result
            
        except Exception as e:
            logger.exception("Parallel processing failed", error=str(e))
            result.processing_time = time.time() - start_time
            result.errors.append({"stage": "pipeline", "error": str(e)})
            return result
            
    async def _initialize_systems(self):
        """Initialize all required systems for optimal performance."""
        initialization_tasks = [
            pool_manager.initialize(),
            rvu_cache.initialize(),
            async_ml_manager.initialize(),
        ]
        await asyncio.gather(*initialization_tasks, return_exceptions=True)
        
    async def _fetch_claims_parallel(self, batch_id: str = None, limit: int = None) -> List[Dict]:
        """Fetch claims data using optimized parallel queries."""
        return await batch_ops.fetch_claims_batch(batch_id, limit)
        
    async def _validate_and_ml_process_parallel(self, claims_data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Combined validation and ML processing for maximum efficiency."""
        logger.info(f"Starting combined validation and ML processing for {len(claims_data)} claims")
        
        # Step 1: Fast validation first to filter out obviously invalid claims
        validated_claims, validation_failures = await self._fast_validate_claims_parallel(claims_data)
        
        if not validated_claims:
            return [], validation_failures
            
        # Step 2: ML processing on validated claims using optimized ML manager
        ml_approved_claims, ml_rejected_claims = await async_ml_manager.predict_claims_pipeline_optimized(validated_claims)
        
        # Combine all failures
        all_failures = validation_failures + ml_rejected_claims
        
        logger.info(f"Combined validation+ML: {len(claims_data)} → {len(ml_approved_claims)} approved, {len(all_failures)} rejected")
        
        return ml_approved_claims, all_failures
        
    async def _fast_validate_claims_parallel(self, claims_data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Validate claims in parallel with high concurrency."""
        validated_claims = []
        failed_claims = []
        
        async def validate_claim_batch(claim_batch: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
            """Validate a batch of claims."""
            batch_validated = []
            batch_failed = []
            
            for claim in claim_batch:
                try:
                    # Fast validation checks
                    validation_result = await self._fast_validate_claim(claim)
                    if validation_result['is_valid']:
                        batch_validated.append(claim)
                    else:
                        batch_failed.append({
                            'claim_id': claim['claim_id'],
                            'facility_id': claim['facility_id'],
                            'failure_category': 'validation_error',
                            'failure_reason': '; '.join(validation_result['errors']),
                            'claim_data': claim,
                        })
                except Exception as e:
                    logger.exception(f"Validation failed for claim {claim.get('claim_id')}")
                    batch_failed.append({
                        'claim_id': claim.get('claim_id'),
                        'facility_id': claim.get('facility_id'),
                        'failure_category': 'system_error',
                        'failure_reason': str(e),
                        'claim_data': claim,
                    })
                    
            return batch_validated, batch_failed
            
        # Split into batches and process in parallel
        batch_size = self.batch_sizes['validation']
        claim_batches = [claims_data[i:i + batch_size] for i in range(0, len(claims_data), batch_size)]
        
        # Process batches with controlled concurrency
        semaphore = asyncio.Semaphore(self.concurrency_limits['validation'])
        
        async def process_with_semaphore(batch):
            async with semaphore:
                return await validate_claim_batch(batch)
                
        tasks = [process_with_semaphore(batch) for batch in claim_batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch validation failed: {result}")
                continue
                
            batch_validated, batch_failed = result
            validated_claims.extend(batch_validated)
            failed_claims.extend(batch_failed)
            
        return validated_claims, failed_claims
        
    async def _fast_validate_claim(self, claim: Dict) -> Dict:
        """Fast validation checks optimized for high throughput."""
        errors = []
        
        try:
            # Critical validation checks only
            if not claim.get('claim_id'):
                errors.append("Missing claim ID")
                
            if not claim.get('facility_id'):
                errors.append("Missing facility ID")
                
            if not claim.get('patient_account_number'):
                errors.append("Missing patient account number")
                
            # Date validations
            service_from = claim.get('service_from_date')
            service_to = claim.get('service_to_date')
            
            if service_from and service_to and service_from > service_to:
                errors.append("Service from date after service to date")
                
            # Financial validations
            total_charges = claim.get('total_charges', 0)
            if total_charges <= 0:
                errors.append("Total charges must be greater than zero")
                
            # Line item validations
            line_items = claim.get('line_items', [])
            if not line_items:
                errors.append("No line items found")
            else:
                for line_item in line_items:
                    if not line_item.get('procedure_code'):
                        errors.append(f"Missing procedure code in line {line_item.get('line_number')}")
                        
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            
        return {'is_valid': len(errors) == 0, 'errors': errors}
        
    async def _calculate_claims_parallel(self, claims_data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Calculate RVU values for claims in parallel with vectorized operations."""
        calculated_claims = []
        failed_claims = []
        
        # Extract all unique procedure codes for batch RVU lookup
        all_procedure_codes = set()
        for claim in claims_data:
            for line_item in claim.get('line_items', []):
                code = line_item.get('procedure_code')
                if code:
                    all_procedure_codes.add(code)
                    
        # Batch lookup RVU data
        logger.info(f"Batch lookup for {len(all_procedure_codes)} unique procedure codes")
        rvu_data = await batch_ops.batch_rvu_lookup(list(all_procedure_codes))
        
        async def calculate_claim_batch(claim_batch: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
            """Calculate RVU for a batch of claims."""
            batch_calculated = []
            batch_failed = []
            
            for claim in claim_batch:
                try:
                    # Calculate RVU values using cached data
                    total_expected_reimbursement = Decimal('0')
                    
                    for line_item in claim.get('line_items', []):
                        procedure_code = line_item.get('procedure_code')
                        units = Decimal(str(line_item.get('units', 1)))
                        
                        if procedure_code in rvu_data:
                            rvu_info = rvu_data[procedure_code]
                            total_rvu = Decimal(str(rvu_info.get('total_rvu', 0)))
                            line_reimbursement = total_rvu * units * self.conversion_factor
                            total_expected_reimbursement += line_reimbursement
                            
                            # Store calculated values in line item
                            line_item['rvu_total'] = float(total_rvu)
                            line_item['expected_reimbursement'] = float(line_reimbursement)
                        else:
                            # Use default minimal RVU for unknown codes
                            default_rvu = Decimal('0.1')
                            line_reimbursement = default_rvu * units * self.conversion_factor
                            total_expected_reimbursement += line_reimbursement
                            
                            line_item['rvu_total'] = float(default_rvu)
                            line_item['expected_reimbursement'] = float(line_reimbursement)
                            
                    # Update claim with total expected reimbursement
                    claim['expected_reimbursement'] = float(total_expected_reimbursement)
                    batch_calculated.append(claim)
                    
                except Exception as e:
                    logger.exception(f"RVU calculation failed for claim {claim.get('claim_id')}")
                    batch_failed.append({
                        'claim_id': claim.get('claim_id'),
                        'facility_id': claim.get('facility_id'),
                        'failure_category': 'financial_error',
                        'failure_reason': f"RVU calculation error: {str(e)}",
                        'claim_data': claim,
                    })
                    
            return batch_calculated, batch_failed
            
        # Process in parallel batches
        batch_size = self.batch_sizes['rvu_calculation']
        claim_batches = [claims_data[i:i + batch_size] for i in range(0, len(claims_data), batch_size)]
        
        semaphore = asyncio.Semaphore(self.concurrency_limits['rvu_calculation'])
        
        async def process_with_semaphore(batch):
            async with semaphore:
                return await calculate_claim_batch(batch)
                
        tasks = [process_with_semaphore(batch) for batch in claim_batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch calculation failed: {result}")
                continue
                
            batch_calculated, batch_failed = result
            calculated_claims.extend(batch_calculated)
            failed_claims.extend(batch_failed)
            
        return calculated_claims, failed_claims
        
    async def _transfer_claims_parallel(self, claims_data: List[Dict]) -> Tuple[int, List[Dict]]:
        """Transfer claims to SQL Server in parallel with bulk operations."""
        successful_transfers = 0
        failed_claims = []
        
        # Process in batches with parallel execution
        batch_size = self.batch_sizes['transfer']
        claim_batches = [claims_data[i:i + batch_size] for i in range(0, len(claims_data), batch_size)]
        
        async def transfer_batch(claim_batch: List[Dict]) -> Tuple[int, List[Dict]]:
            """Transfer a batch of claims to SQL Server."""
            try:
                success_count, fail_count = await batch_ops.bulk_insert_claims_sqlserver(claim_batch)
                return success_count, []
            except Exception as e:
                logger.exception("Batch transfer failed")
                failed_batch = []
                for claim in claim_batch:
                    failed_batch.append({
                        'claim_id': claim.get('claim_id'),
                        'facility_id': claim.get('facility_id'),
                        'failure_category': 'transfer_error',
                        'failure_reason': f"Transfer error: {str(e)}",
                        'claim_data': claim,
                    })
                return 0, failed_batch
                
        # Execute transfers with controlled concurrency
        semaphore = asyncio.Semaphore(self.concurrency_limits['transfer'])
        
        async def process_with_semaphore(batch):
            async with semaphore:
                return await transfer_batch(batch)
                
        tasks = [process_with_semaphore(batch) for batch in claim_batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch transfer failed: {result}")
                continue
                
            success_count, batch_failed = result
            successful_transfers += success_count
            failed_claims.extend(batch_failed)
            
        return successful_transfers, failed_claims
        
    async def _update_claim_statuses_parallel(self, claims_data: List[Dict]):
        """Update claim processing statuses in parallel."""
        if not claims_data:
            return
            
        # Prepare status updates
        status_updates = []
        for claim in claims_data:
            status_updates.append({
                'claim_id': claim['id'],
                'status': 'completed',
                'processed_at': 'NOW()',
                'expected_reimbursement': claim.get('expected_reimbursement', 0),
            })
            
        # Execute bulk status update
        await batch_ops.bulk_update_claim_status(status_updates)
        
    async def _store_failed_claims_batch(self, failed_claims: List[Dict]):
        """Store failed claims in batch for investigation."""
        if failed_claims:
            await batch_ops.bulk_insert_failed_claims(failed_claims)
            
    def _log_performance_results(self, result: ParallelProcessingResult):
        """Log detailed performance results."""
        logger.info("Parallel processing completed",
                   total_claims=result.total_claims,
                   processed_claims=result.processed_claims,
                   failed_claims=result.failed_claims,
                   throughput=f"{result.throughput:.2f} claims/sec",
                   processing_time=f"{result.processing_time:.3f}s",
                   target_met=result.throughput >= 6667)
                   
        # Log stage timings
        for stage, timing in result.stage_times.items():
            logger.info(f"Stage timing: {stage}", duration=f"{timing:.3f}s")
            
        # Performance assessment
        if result.throughput >= 6667:
            logger.info("🎯 TARGET ACHIEVED: Processing rate exceeds 6,667 claims/second!")
        else:
            improvement_needed = 6667 - result.throughput
            logger.warning(f"Target not met. Need {improvement_needed:.0f} more claims/sec")
            
    async def shutdown(self):
        """Shutdown parallel processor."""
        self.worker_pool.shutdown()
        await async_ml_manager.shutdown()


# Global parallel processor instance
parallel_processor = ParallelClaimsProcessor()
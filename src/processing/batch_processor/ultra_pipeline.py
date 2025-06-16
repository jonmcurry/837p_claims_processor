"""Ultra high-performance pipeline methods for 100k+ claims/15s processing."""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import structlog
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession
from aiocache import cached, Cache
from aiocache.serializers import PickleSerializer

from src.core.database.base import get_postgres_session, get_sqlserver_session
from src.core.database.models import Claim, BatchMetadata, ProcessingStatus

logger = structlog.get_logger(__name__)


class UltraHighPerformancePipeline:
    """Ultra optimized pipeline methods for maximum throughput."""

    def __init__(self, parent_pipeline):
        self.parent = parent_pipeline
        self.cache = Cache(Cache.REDIS, endpoint="redis://localhost", port=6379, 
                          serializer=PickleSerializer())

    async def _get_batch_metadata_optimized(self, batch_id: str) -> Optional[BatchMetadata]:
        """Get batch metadata with connection pooling and caching optimization."""
        cache_key = f"batch_metadata:{batch_id}"
        
        # Try cache first
        cached_batch = await self.cache.get(cache_key)
        if cached_batch:
            return cached_batch
            
        async with get_postgres_session() as session:
            # Use prepared statement for faster execution
            query = text("""
                SELECT id, batch_id, total_claims, status, facility_id, 
                       submitted_by, submitted_at, priority
                FROM batch_metadata 
                WHERE batch_id = :batch_id
            """)
            result = await session.execute(query, {"batch_id": batch_id})
            row = result.fetchone()
            
            if row:
                batch = BatchMetadata(
                    id=row.id,
                    batch_id=row.batch_id,
                    total_claims=row.total_claims,
                    status=row.status,
                    facility_id=row.facility_id,
                    submitted_by=row.submitted_by,
                    submitted_at=row.submitted_at,
                    priority=row.priority
                )
                # Cache for 5 minutes
                await self.cache.set(cache_key, batch, ttl=300)
                return batch
                
        return None

    async def _fetch_claims_optimized(self, batch_id: str) -> List[Claim]:
        """Fetch claims with optimized SQL and minimal object creation."""
        async with get_postgres_session() as session:
            # Use raw SQL for maximum performance
            query = text("""
                SELECT c.id, c.claim_id, c.facility_id, c.patient_account_number,
                       c.patient_first_name, c.patient_last_name, c.patient_date_of_birth,
                       c.total_charges, c.financial_class, c.insurance_type,
                       c.billing_provider_npi, c.primary_diagnosis_code,
                       c.service_from_date, c.service_to_date,
                       COALESCE(
                           json_agg(
                               json_build_object(
                                   'line_number', cli.line_number,
                                   'procedure_code', cli.procedure_code,
                                   'units', cli.units,
                                   'charge_amount', cli.charge_amount,
                                   'service_date', cli.service_date
                               ) ORDER BY cli.line_number
                           ) FILTER (WHERE cli.id IS NOT NULL), 
                           '[]'::json
                       ) as line_items
                FROM claims c
                LEFT JOIN claim_line_items cli ON c.id = cli.claim_id
                WHERE c.batch_id = (SELECT id FROM batch_metadata WHERE batch_id = :batch_id)
                  AND c.processing_status = 'pending'
                GROUP BY c.id, c.claim_id, c.facility_id, c.patient_account_number,
                         c.patient_first_name, c.patient_last_name, c.patient_date_of_birth,
                         c.total_charges, c.financial_class, c.insurance_type,
                         c.billing_provider_npi, c.primary_diagnosis_code,
                         c.service_from_date, c.service_to_date
                ORDER BY c.id
            """)
            
            result = await session.execute(query, {"batch_id": batch_id})
            rows = result.fetchall()
            
            # Convert to lightweight claim objects
            claims = []
            for row in rows:
                claim = Claim(
                    id=row.id,
                    claim_id=row.claim_id,
                    facility_id=row.facility_id,
                    patient_account_number=row.patient_account_number,
                    patient_first_name=row.patient_first_name,
                    patient_last_name=row.patient_last_name,
                    patient_date_of_birth=row.patient_date_of_birth,
                    total_charges=row.total_charges,
                    financial_class=row.financial_class,
                    insurance_type=row.insurance_type,
                    billing_provider_npi=row.billing_provider_npi,
                    primary_diagnosis_code=row.primary_diagnosis_code,
                    service_from_date=row.service_from_date,
                    service_to_date=row.service_to_date
                )
                # Attach line items as attribute for processing
                claim._line_items_data = row.line_items
                claims.append(claim)
                
            return claims

    async def _process_large_batch_chunked(self, batch_id: str, batch: BatchMetadata) -> 'ProcessingResult':
        """Process large batches in optimized chunks to prevent memory exhaustion."""
        logger.info("Processing large batch in chunks", 
                   batch_id=batch_id, 
                   total_claims=batch.total_claims,
                   chunk_size=self.parent.batch_memory_limit)
        
        chunk_size = self.parent.batch_memory_limit
        total_chunks = (batch.total_claims + chunk_size - 1) // chunk_size
        
        # Initialize aggregated results
        total_result = ProcessingResult(
            batch_id=batch_id,
            total_claims=batch.total_claims
        )
        
        start_time = time.perf_counter()
        
        for chunk_idx in range(total_chunks):
            offset = chunk_idx * chunk_size
            logger.info("Processing chunk", 
                       batch_id=batch_id,
                       chunk=f"{chunk_idx + 1}/{total_chunks}",
                       offset=offset)
            
            # Fetch chunk with limit and offset
            chunk_claims = await self._fetch_claims_chunk(batch_id, offset, chunk_size)
            
            if not chunk_claims:
                continue
            
            # Process chunk through pipeline
            chunk_result = await self._process_claims_chunk(chunk_claims, batch_id, chunk_idx)
            
            # Aggregate results
            total_result.processed_claims += chunk_result.processed_claims
            total_result.failed_claims += chunk_result.failed_claims
            total_result.errors.extend(chunk_result.errors)
            
            # Force garbage collection between chunks
            gc.collect()
        
        total_result.processing_time = time.perf_counter() - start_time
        total_result.throughput = total_result.total_claims / total_result.processing_time
        
        await self.parent._finalize_batch(batch_id, total_result)
        return total_result

    async def _fetch_claims_chunk(self, batch_id: str, offset: int, limit: int) -> List[Claim]:
        """Fetch a specific chunk of claims with pagination."""
        async with get_postgres_session() as session:
            query = text("""
                SELECT c.id, c.claim_id, c.facility_id, c.patient_account_number,
                       c.patient_first_name, c.patient_last_name, c.patient_date_of_birth,
                       c.total_charges, c.financial_class, c.insurance_type,
                       c.billing_provider_npi, c.primary_diagnosis_code,
                       c.service_from_date, c.service_to_date
                FROM claims c
                WHERE c.batch_id = (SELECT id FROM batch_metadata WHERE batch_id = :batch_id)
                  AND c.processing_status = 'pending'
                ORDER BY c.id
                LIMIT :limit OFFSET :offset
            """)
            
            result = await session.execute(query, {
                "batch_id": batch_id,
                "limit": limit,
                "offset": offset
            })
            
            return [self._row_to_claim(row) for row in result.fetchall()]

    async def _validate_claims_ultra_parallel(self, claims: List[Claim]) -> Tuple[List[Claim], List[Dict]]:
        """Ultra parallel validation using numpy vectorization and thread pools."""
        stage = self.parent.stages["validation"]
        
        # Convert claims to numpy arrays for vectorized operations
        claim_data = self._claims_to_numpy_arrays(claims)
        
        # Vectorized validation using numpy operations
        validation_results = await self._vectorized_validation(claim_data)
        
        # Parallel processing of complex validations
        complex_validation_tasks = []
        semaphore = asyncio.Semaphore(stage.workers)
        
        async def validate_complex_batch(claim_batch: List[Claim]) -> Tuple[List[Claim], List[Dict]]:
            async with semaphore:
                return await self._validate_complex_rules(claim_batch)
        
        # Split into optimal batch sizes
        batch_size = max(1, len(claims) // stage.workers)
        claim_batches = [claims[i:i + batch_size] for i in range(0, len(claims), batch_size)]
        
        # Execute validations in parallel
        tasks = [validate_complex_batch(batch) for batch in claim_batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        valid_claims = []
        failures = []
        
        for result in batch_results:
            if isinstance(result, Exception):
                logger.exception("Batch validation failed", error=result)
                continue
            
            batch_valid, batch_failures = result
            valid_claims.extend(batch_valid)
            failures.extend(batch_failures)
        
        return valid_claims, failures

    async def _predict_claims_ultra_parallel(self, claims: List[Claim]) -> Tuple[List[Claim], List[Dict]]:
        """Ultra parallel ML prediction with batch optimization."""
        stage = self.parent.stages["ml_prediction"]
        
        if not hasattr(self.parent, 'predictor') or not self.parent.predictor:
            return claims, []
        
        # Batch predictions for maximum GPU utilization
        optimal_batch_size = 100  # Optimized for ML model
        
        predicted_claims = []
        failures = []
        
        # Process in optimal batches for ML inference
        for i in range(0, len(claims), optimal_batch_size):
            batch = claims[i:i + optimal_batch_size]
            
            try:
                # Batch ML prediction
                predictions = await self.parent.predictor.predict_batch_optimized(batch)
                
                for claim, prediction in zip(batch, predictions):
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
                        
            except Exception as e:
                logger.exception("ML prediction batch failed", batch_size=len(batch))
                for claim in batch:
                    failures.append({
                        "claim_id": claim.claim_id,
                        "errors": [f"ML prediction error: {str(e)}"],
                        "stage": "ml_prediction"
                    })
        
        return predicted_claims, failures

    async def _calculate_claims_ultra_parallel(self, claims: List[Claim]) -> Tuple[List[Claim], List[Dict]]:
        """Ultra parallel RVU calculation using vectorized operations."""
        stage = self.parent.stages["calculation"]
        
        # Use pandas for vectorized RVU calculations
        claims_df = await self._claims_to_dataframe(claims)
        
        # Vectorized RVU lookup and calculation
        rvu_data = await self._get_rvu_lookup_table()
        
        # Merge and calculate in one vectorized operation
        calculated_df = await self._vectorized_rvu_calculation(claims_df, rvu_data)
        
        # Update claims with calculated values
        calculated_claims = []
        failures = []
        
        for idx, row in calculated_df.iterrows():
            try:
                claim = claims[idx]
                claim.expected_reimbursement = row['expected_reimbursement']
                calculated_claims.append(claim)
            except Exception as e:
                failures.append({
                    "claim_id": claims[idx].claim_id if idx < len(claims) else "unknown",
                    "errors": [f"Calculation error: {str(e)}"],
                    "stage": "calculation"
                })
        
        return calculated_claims, failures

    async def _transfer_claims_ultra_parallel(self, claims: List[Claim]) -> Tuple[List[str], List[Dict]]:
        """Ultra parallel data transfer with bulk insert optimization."""
        stage = self.parent.stages["data_transfer"]
        
        # Prepare bulk insert data
        insert_data = await self._prepare_bulk_insert_data(claims)
        
        # Execute bulk inserts in parallel
        batch_size = 1000  # Optimal for SQL Server bulk insert
        batches = [insert_data[i:i + batch_size] for i in range(0, len(insert_data), batch_size)]
        
        transferred = []
        failures = []
        
        # Parallel bulk inserts
        semaphore = asyncio.Semaphore(stage.workers)
        
        async def bulk_insert_batch(batch_data: List[Dict]) -> Tuple[List[str], List[Dict]]:
            async with semaphore:
                return await self._execute_bulk_insert(batch_data)
        
        tasks = [bulk_insert_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        for result in results:
            if isinstance(result, Exception):
                logger.exception("Bulk insert failed", error=result)
                continue
            
            batch_transferred, batch_failures = result
            transferred.extend(batch_transferred)
            failures.extend(batch_failures)
        
        return transferred, failures

    def _claims_to_numpy_arrays(self, claims: List[Claim]) -> Dict[str, np.ndarray]:
        """Convert claims to numpy arrays for vectorized operations."""
        return {
            'total_charges': np.array([float(c.total_charges) for c in claims]),
            'facility_ids': np.array([c.facility_id for c in claims]),
            'financial_classes': np.array([c.financial_class for c in claims]),
            'service_dates': np.array([c.service_from_date for c in claims])
        }

    async def _vectorized_validation(self, claim_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Perform vectorized validation checks using numpy."""
        # Example vectorized validations
        valid_charges = claim_data['total_charges'] > 0
        valid_facilities = np.isin(claim_data['facility_ids'], await self._get_valid_facility_ids())
        
        # Combine validation results
        return valid_charges & valid_facilities

    @cached(ttl=3600)  # Cache for 1 hour
    async def _get_valid_facility_ids(self) -> List[str]:
        """Get valid facility IDs with caching."""
        async with get_postgres_session() as session:
            result = await session.execute(text("SELECT facility_id FROM facilities WHERE is_active = true"))
            return [row[0] for row in result.fetchall()]

    async def _claims_to_dataframe(self, claims: List[Claim]) -> pd.DataFrame:
        """Convert claims to pandas DataFrame for vectorized operations."""
        data = []
        for claim in claims:
            if hasattr(claim, '_line_items_data'):
                for line_item in claim._line_items_data:
                    data.append({
                        'claim_id': claim.claim_id,
                        'procedure_code': line_item['procedure_code'],
                        'units': line_item['units'],
                        'charge_amount': line_item['charge_amount']
                    })
        
        return pd.DataFrame(data)

    @cached(ttl=1800)  # Cache for 30 minutes
    async def _get_rvu_lookup_table(self) -> pd.DataFrame:
        """Get RVU lookup table as DataFrame with caching."""
        async with get_postgres_session() as session:
            query = text("""
                SELECT procedure_code, work_rvu, practice_expense_rvu, 
                       malpractice_rvu, total_rvu, conversion_factor
                FROM rvu_data 
                WHERE is_active = true AND year = EXTRACT(YEAR FROM CURRENT_DATE)
            """)
            result = await session.execute(query)
            
            return pd.DataFrame(result.fetchall(), columns=[
                'procedure_code', 'work_rvu', 'practice_expense_rvu',
                'malpractice_rvu', 'total_rvu', 'conversion_factor'
            ])

    async def _vectorized_rvu_calculation(self, claims_df: pd.DataFrame, rvu_df: pd.DataFrame) -> pd.DataFrame:
        """Perform vectorized RVU calculations using pandas."""
        # Merge claims with RVU data
        merged_df = claims_df.merge(rvu_df, on='procedure_code', how='left')
        
        # Vectorized calculation
        merged_df['total_rvu_amount'] = merged_df['total_rvu'] * merged_df['units']
        merged_df['expected_reimbursement'] = (
            merged_df['total_rvu_amount'] * merged_df['conversion_factor']
        )
        
        # Group by claim_id and sum
        result_df = merged_df.groupby('claim_id').agg({
            'expected_reimbursement': 'sum'
        }).reset_index()
        
        return result_df

    def _row_to_claim(self, row) -> Claim:
        """Convert database row to Claim object."""
        return Claim(
            id=row.id,
            claim_id=row.claim_id,
            facility_id=row.facility_id,
            patient_account_number=row.patient_account_number,
            patient_first_name=row.patient_first_name,
            patient_last_name=row.patient_last_name,
            patient_date_of_birth=row.patient_date_of_birth,
            total_charges=row.total_charges,
            financial_class=row.financial_class,
            insurance_type=row.insurance_type,
            billing_provider_npi=row.billing_provider_npi,
            primary_diagnosis_code=row.primary_diagnosis_code,
            service_from_date=row.service_from_date,
            service_to_date=row.service_to_date
        )
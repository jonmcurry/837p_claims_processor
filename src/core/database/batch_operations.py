"""Optimized batch database operations for high-throughput claims processing."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from sqlalchemy import text, bindparam
from sqlalchemy.dialects.postgresql import insert as pg_insert

# Handle SQL Server dialect import gracefully
try:
    from sqlalchemy.dialects.sqlserver import insert as ss_insert
except ImportError:
    # SQL Server dialect not available, use generic insert
    from sqlalchemy import insert as ss_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config.settings import settings
from src.core.database.pool_manager import pool_manager
from src.core.cache.rvu_cache import rvu_cache

logger = logging.getLogger(__name__)


class BatchDatabaseOperations:
    """High-performance batch database operations for claims processing."""
    
    def __init__(self):
        self.batch_sizes = {
            'claim_fetch': 5000,  # Larger fetch batches
            'claim_insert': 1000,  # Optimized insert batches
            'line_item_insert': 2000,  # Line items can be larger
            'status_update': 2000,  # Bulk status updates
            'rvu_lookup': 1000,  # RVU batch lookups
        }
        
    async def fetch_claims_batch(self, batch_id: str = None, limit: int = None) -> List[Dict]:
        """Fetch claims in optimized batches using single query with joins."""
        start_time = time.time()
        
        try:
            async with pool_manager.get_postgres_session() as session:
                # Optimized query that fetches claims and line items in one go
                # Build query conditionally to avoid parameter type issues
                if batch_id is None:
                    query = text("""
                        SELECT 
                            c.id as claim_id,
                            c.claim_id as claim_reference,
                            c.facility_id,
                            c.patient_account_number,
                            c.medical_record_number,
                            c.patient_first_name,
                            c.patient_last_name,
                            c.patient_middle_name,
                            c.patient_date_of_birth,
                            c.admission_date,
                            c.discharge_date,
                            c.service_from_date,
                            c.service_to_date,
                            c.financial_class,
                            c.total_charges,
                            c.expected_reimbursement,
                            c.insurance_type,
                            c.insurance_plan_id,
                            c.subscriber_id,
                            c.billing_provider_npi,
                            c.billing_provider_name,
                            c.attending_provider_npi,
                            c.attending_provider_name,
                            c.primary_diagnosis_code,
                            c.diagnosis_codes,
                            c.batch_id,
                            c.processing_status,
                            c.priority,
                            cli.line_number,
                            cli.service_date,
                            cli.procedure_code,
                            cli.procedure_description,
                            cli.units,
                            cli.charge_amount,
                            cli.rendering_provider_npi,
                            cli.rendering_provider_name,
                            cli.diagnosis_pointers,
                            cli.modifier_codes
                        FROM claims c
                        LEFT JOIN claim_line_items cli ON c.id = cli.claim_id
                        WHERE c.processing_status = 'pending'
                        ORDER BY c.priority DESC, c.created_at ASC
                        LIMIT :limit_val
                    """)
                    params = {
                        'limit_val': limit or self.batch_sizes['claim_fetch']
                    }
                else:
                    query = text("""
                        SELECT 
                            c.id as claim_id,
                            c.claim_id as claim_reference,
                            c.facility_id,
                            c.patient_account_number,
                            c.medical_record_number,
                            c.patient_first_name,
                            c.patient_last_name,
                            c.patient_middle_name,
                            c.patient_date_of_birth,
                            c.admission_date,
                            c.discharge_date,
                            c.service_from_date,
                            c.service_to_date,
                            c.financial_class,
                            c.total_charges,
                            c.expected_reimbursement,
                            c.insurance_type,
                            c.insurance_plan_id,
                            c.subscriber_id,
                            c.billing_provider_npi,
                            c.billing_provider_name,
                            c.attending_provider_npi,
                            c.attending_provider_name,
                            c.primary_diagnosis_code,
                            c.diagnosis_codes,
                            c.batch_id,
                            c.processing_status,
                            c.priority,
                            cli.line_number,
                            cli.service_date,
                            cli.procedure_code,
                            cli.procedure_description,
                            cli.units,
                            cli.charge_amount,
                            cli.rendering_provider_npi,
                            cli.rendering_provider_name,
                            cli.diagnosis_pointers,
                            cli.modifier_codes
                        FROM claims c
                        LEFT JOIN claim_line_items cli ON c.id = cli.claim_id
                        WHERE c.processing_status = 'pending'
                        AND c.batch_id = :batch_id
                        ORDER BY c.priority DESC, c.created_at ASC
                        LIMIT :limit_val
                    """)
                    params = {
                        'batch_id': batch_id,
                        'limit_val': limit or self.batch_sizes['claim_fetch']
                    }
                
                result = await session.execute(query, params)
                rows = result.fetchall()
                
                # Group by claim to reconstruct claim objects with line items
                claims_dict = {}
                for row in rows:
                    claim_id = row.claim_id
                    
                    if claim_id not in claims_dict:
                        claims_dict[claim_id] = {
                            'id': row.claim_id,
                            'claim_id': row.claim_reference,
                            'facility_id': row.facility_id,
                            'patient_account_number': row.patient_account_number,
                            'medical_record_number': row.medical_record_number,
                            'patient_first_name': row.patient_first_name,
                            'patient_last_name': row.patient_last_name,
                            'patient_middle_name': row.patient_middle_name,
                            'patient_date_of_birth': row.patient_date_of_birth,
                            'admission_date': row.admission_date,
                            'discharge_date': row.discharge_date,
                            'service_from_date': row.service_from_date,
                            'service_to_date': row.service_to_date,
                            'financial_class': row.financial_class,
                            'total_charges': float(row.total_charges) if row.total_charges else 0,
                            'expected_reimbursement': float(row.expected_reimbursement) if row.expected_reimbursement else 0,
                            'insurance_type': row.insurance_type,
                            'insurance_plan_id': row.insurance_plan_id,
                            'subscriber_id': row.subscriber_id,
                            'billing_provider_npi': row.billing_provider_npi,
                            'billing_provider_name': row.billing_provider_name,
                            'attending_provider_npi': row.attending_provider_npi,
                            'attending_provider_name': row.attending_provider_name,
                            'primary_diagnosis_code': row.primary_diagnosis_code,
                            'diagnosis_codes': row.diagnosis_codes,
                            'batch_id': row.batch_id,
                            'processing_status': row.processing_status,
                            'priority': row.priority,
                            'line_items': []
                        }
                    
                    # Add line item if present
                    if row.line_number is not None:
                        line_item = {
                            'line_number': row.line_number,
                            'service_date': row.service_date,
                            'procedure_code': row.procedure_code,
                            'procedure_description': row.procedure_description,
                            'units': row.units or 1,
                            'charge_amount': float(row.charge_amount) if row.charge_amount else 0,
                            'rendering_provider_npi': row.rendering_provider_npi,
                            'rendering_provider_name': row.rendering_provider_name,
                            'diagnosis_pointers': row.diagnosis_pointers,
                            'modifier_codes': row.modifier_codes
                        }
                        claims_dict[claim_id]['line_items'].append(line_item)
                
                claims_list = list(claims_dict.values())
                
                fetch_time = time.time() - start_time
                logger.info(f"Fetched {len(claims_list)} claims with line items in {fetch_time:.2f}s")
                
                return claims_list
                
        except Exception as e:
            logger.error(f"Batch fetch failed: {e}")
            raise
            
    async def batch_rvu_lookup(self, procedure_codes: List[str]) -> Dict[str, Dict]:
        """Batch lookup RVU data for multiple procedure codes."""
        start_time = time.time()
        
        try:
            # Use optimized RVU cache for batch lookup
            rvu_results = await rvu_cache.get_batch_rvu_data(procedure_codes)
            
            # Convert to dictionary format expected by calculations
            rvu_dict = {}
            for code, rvu_data in rvu_results.items():
                if rvu_data:
                    rvu_dict[code] = {
                        'work_rvu': float(rvu_data.work_rvu),
                        'practice_expense_rvu': float(rvu_data.practice_expense_rvu),
                        'malpractice_rvu': float(rvu_data.malpractice_rvu),
                        'total_rvu': float(rvu_data.total_rvu),
                    }
                    
            lookup_time = time.time() - start_time
            logger.debug(f"Batch RVU lookup for {len(procedure_codes)} codes in {lookup_time:.2f}s")
            
            return rvu_dict
            
        except Exception as e:
            logger.error(f"Batch RVU lookup failed: {e}")
            return {}
            
    async def bulk_insert_claims_sqlserver(self, claims_data: List[Dict]) -> Tuple[int, int]:
        """Bulk insert claims into SQL Server with optimized batching."""
        start_time = time.time()
        successful_inserts = 0
        failed_inserts = 0
        
        try:
            # Check if SQL Server session maker is available
            if not pool_manager.sqlserver_session_maker:
                logger.warning("SQL Server not available, skipping bulk insert")
                return 0, len(claims_data)
                
            async with pool_manager.get_sqlserver_session() as session:
                # Prepare bulk insert for claims
                claims_to_insert = []
                line_items_to_insert = []
                
                for claim in claims_data:
                    # Prepare claim record
                    full_patient_name = f"{claim.get('patient_first_name', '')} {claim.get('patient_middle_name', '')} {claim.get('patient_last_name', '')}".strip()
                    
                    claim_record = {
                        'facility_id': claim['facility_id'],
                        'patient_account_number': claim['patient_account_number'],
                        'medical_record_number': claim.get('medical_record_number'),
                        'patient_name': full_patient_name,
                        'first_name': claim.get('patient_first_name'),
                        'last_name': claim.get('patient_last_name'),
                        'date_of_birth': claim.get('patient_date_of_birth'),
                        'gender': 'U',  # Unknown default
                        'financial_class_id': self._map_financial_class(claim.get('financial_class')),
                        'secondary_insurance': None,
                    }
                    claims_to_insert.append(claim_record)
                    
                    # Prepare line items
                    for line_item in claim.get('line_items', []):
                        line_record = {
                            'facility_id': claim['facility_id'],
                            'patient_account_number': claim['patient_account_number'],
                            'line_number': line_item['line_number'],
                            'procedure_code': line_item['procedure_code'],
                            'units': line_item.get('units', 1),
                            'charge_amount': line_item.get('charge_amount', 0),
                            'service_from_date': line_item.get('service_date'),
                            'service_to_date': line_item.get('service_date'),
                            'diagnosis_pointer': line_item.get('diagnosis_pointers'),
                            'rendering_provider_id': None,  # Skip provider lookup for performance
                        }
                        line_items_to_insert.append(line_record)
                
                # Execute bulk inserts
                if claims_to_insert:
                    await self._execute_bulk_insert(session, 'dbo.claims', claims_to_insert)
                    successful_inserts += len(claims_to_insert)
                    
                if line_items_to_insert:
                    await self._execute_bulk_insert(session, 'dbo.claims_line_items', line_items_to_insert)
                    
                await session.commit()
                
                insert_time = time.time() - start_time
                logger.info(f"Bulk inserted {successful_inserts} claims with {len(line_items_to_insert)} line items in {insert_time:.2f}s")
                
        except Exception as e:
            logger.error(f"Bulk insert failed: {e}")
            failed_inserts = len(claims_data)
            
        return successful_inserts, failed_inserts
        
    async def _execute_bulk_insert(self, session: AsyncSession, table_name: str, records: List[Dict]):
        """Execute bulk insert using SQL Server efficient methods."""
        if not records:
            return
            
        # For SQL Server, use bulk insert with VALUES clause
        try:
            if table_name == 'dbo.claims':
                # Bulk insert claims
                values_clause = []
                params = {}
                
                for i, record in enumerate(records):
                    value_params = []
                    for key, value in record.items():
                        param_name = f"{key}_{i}"
                        params[param_name] = value
                        value_params.append(f":{param_name}")
                    values_clause.append(f"({', '.join(value_params)})")
                
                query = text(f"""
                    INSERT INTO {table_name} (
                        facility_id, patient_account_number, medical_record_number,
                        patient_name, first_name, last_name, date_of_birth,
                        gender, financial_class_id, secondary_insurance
                    ) VALUES {', '.join(values_clause)}
                """)
                
                await session.execute(query, params)
                
            elif table_name == 'dbo.claims_line_items':
                # Bulk insert line items
                values_clause = []
                params = {}
                
                for i, record in enumerate(records):
                    value_params = []
                    for key, value in record.items():
                        param_name = f"{key}_{i}"
                        params[param_name] = value
                        value_params.append(f":{param_name}")
                    values_clause.append(f"({', '.join(value_params)})")
                
                query = text(f"""
                    INSERT INTO {table_name} (
                        facility_id, patient_account_number, line_number,
                        procedure_code, units, charge_amount,
                        service_from_date, service_to_date,
                        diagnosis_pointer, rendering_provider_id
                    ) VALUES {', '.join(values_clause)}
                """)
                
                await session.execute(query, params)
                
        except Exception as e:
            logger.error(f"Bulk insert to {table_name} failed: {e}")
            raise
            
    async def bulk_update_claim_status(self, claim_updates: List[Dict]) -> int:
        """Bulk update claim processing status in PostgreSQL."""
        start_time = time.time()
        
        try:
            async with pool_manager.get_postgres_session() as session:
                # Use efficient bulk update with CASE statements
                if not claim_updates:
                    return 0
                    
                # Prepare bulk update query
                claim_ids = [update['claim_id'] for update in claim_updates]
                
                # Build CASE statements for different fields
                status_cases = []
                processed_at_cases = []
                reimbursement_cases = []
                
                params = {'claim_ids': claim_ids}
                
                for i, update in enumerate(claim_updates):
                    claim_id_param = f"claim_id_{i}"
                    status_param = f"status_{i}"
                    processed_at_param = f"processed_at_{i}"
                    reimbursement_param = f"reimbursement_{i}"
                    
                    params[claim_id_param] = update['claim_id']
                    params[status_param] = update.get('status', 'completed')
                    params[processed_at_param] = update.get('processed_at')
                    params[reimbursement_param] = update.get('expected_reimbursement', 0)
                    
                    status_cases.append(f"WHEN id = :{claim_id_param} THEN CAST(:{status_param} AS processing_status)")
                    if update.get('processed_at') is None:
                        processed_at_cases.append(f"WHEN id = :{claim_id_param} THEN NOW()")
                    else:
                        processed_at_cases.append(f"WHEN id = :{claim_id_param} THEN CAST(:{processed_at_param} AS timestamp)")
                    reimbursement_cases.append(f"WHEN id = :{claim_id_param} THEN CAST(:{reimbursement_param} AS numeric)")
                
                query = text(f"""
                    UPDATE claims 
                    SET 
                        processing_status = CASE {' '.join(status_cases)} END,
                        processed_at = CASE {' '.join(processed_at_cases)} END,
                        expected_reimbursement = CASE {' '.join(reimbursement_cases)} END,
                        updated_at = NOW()
                    WHERE id = ANY(:claim_ids)
                """)
                
                result = await session.execute(query, params)
                await session.commit()
                
                updated_count = result.rowcount
                update_time = time.time() - start_time
                
                logger.info(f"Bulk updated {updated_count} claim statuses in {update_time:.2f}s")
                return updated_count
                
        except Exception as e:
            logger.error(f"Bulk status update failed: {e}")
            return 0
            
    async def bulk_insert_failed_claims(self, failed_claims: List[Dict]) -> int:
        """Bulk insert failed claims for investigation."""
        start_time = time.time()
        
        try:
            async with pool_manager.get_postgres_session() as session:
                if not failed_claims:
                    return 0
                    
                # Prepare bulk insert for failed claims
                insert_query = text("""
                    INSERT INTO failed_claims (
                        original_claim_id, claim_reference, facility_id,
                        failure_category, failure_reason, failure_details,
                        claim_data, charge_amount, expected_reimbursement,
                        created_at
                    ) VALUES 
                """ + ", ".join([f"""(
                    :original_claim_id_{i}, :claim_reference_{i}, :facility_id_{i},
                    :failure_category_{i}::failure_category, :failure_reason_{i}, :failure_details_{i}::jsonb,
                    :claim_data_{i}::jsonb, :charge_amount_{i}, :expected_reimbursement_{i},
                    NOW()
                )""" for i in range(len(failed_claims))]))
                
                # Prepare parameters
                params = {}
                for i, failed_claim in enumerate(failed_claims):
                    params.update({
                        f'original_claim_id_{i}': failed_claim.get('original_claim_id'),
                        f'claim_reference_{i}': failed_claim.get('claim_reference'),
                        f'facility_id_{i}': failed_claim.get('facility_id'),
                        f'failure_category_{i}': failed_claim.get('failure_category', 'system_error'),
                        f'failure_reason_{i}': failed_claim.get('failure_reason', ''),
                        f'failure_details_{i}': failed_claim.get('failure_details', {}),
                        f'claim_data_{i}': failed_claim.get('claim_data', {}),
                        f'charge_amount_{i}': failed_claim.get('charge_amount', 0),
                        f'expected_reimbursement_{i}': failed_claim.get('expected_reimbursement', 0),
                    })
                
                result = await session.execute(insert_query, params)
                await session.commit()
                
                insert_count = result.rowcount
                insert_time = time.time() - start_time
                
                logger.info(f"Bulk inserted {insert_count} failed claims in {insert_time:.2f}s")
                return insert_count
                
        except Exception as e:
            logger.error(f"Bulk failed claims insert failed: {e}")
            return 0
            
    async def update_batch_metadata(self, batch_id: str, metrics: Dict) -> bool:
        """Update batch metadata with processing results."""
        try:
            async with pool_manager.get_postgres_session() as session:
                query = text("""
                    UPDATE batch_metadata
                    SET 
                        processed_claims = :processed_claims,
                        failed_claims = :failed_claims,
                        status = CAST(:status AS processing_status),
                        completed_at = :completed_at,
                        processing_time_seconds = :processing_time,
                        throughput_per_second = :throughput,
                        updated_at = NOW()
                    WHERE id = :batch_id
                """)
                
                params = {
                    'batch_id': batch_id,
                    'processed_claims': metrics.get('processed_claims', 0),
                    'failed_claims': metrics.get('failed_claims', 0),
                    'status': metrics.get('status', 'completed'),
                    'completed_at': metrics.get('completed_at', 'NOW()'),
                    'processing_time': metrics.get('processing_time', 0),
                    'throughput': metrics.get('throughput', 0),
                }
                
                await session.execute(query, params)
                await session.commit()
                return True
                
        except Exception as e:
            logger.error(f"Batch metadata update failed: {e}")
            return False
            
    def _map_financial_class(self, financial_class_name: str) -> Optional[str]:
        """Map PostgreSQL financial class names to SQL Server IDs."""
        if not financial_class_name:
            return None
            
        mapping = {
            'Medicare Part A': 'A',
            'Medicare Part B': 'B',
            'Medicaid': 'MA',
            'Commercial HMO': 'HM',
            'Commercial PPO': 'BC',
            'Self Pay': 'SP',
            'Workers Comp': 'WC',
            # Direct mappings
            'A': 'A', 'BC': 'BC', 'MA': 'MA', 'SP': 'SP', 
            'WC': 'WC', 'CO': 'CO', 'HM': 'HM'
        }
        
        return mapping.get(financial_class_name)
        
    async def get_processing_statistics(self) -> Dict:
        """Get current processing statistics from both databases."""
        stats = {'postgres': {}, 'sqlserver': {}}
        
        try:
            # PostgreSQL stats
            async with pool_manager.get_postgres_session() as session:
                pg_query = text("""
                    SELECT 
                        processing_status,
                        COUNT(*) as count
                    FROM claims
                    GROUP BY processing_status
                """)
                result = await session.execute(pg_query)
                stats['postgres'] = {row.processing_status: row.count for row in result.fetchall()}
                
            # SQL Server stats (if available)
            if pool_manager.sqlserver_session_maker:
                async with pool_manager.get_sqlserver_session() as session:
                    ss_query = text("""
                        SELECT 
                            COUNT(*) as total_claims,
                            COUNT(DISTINCT facility_id) as facilities
                        FROM dbo.claims
                    """)
                    result = await session.execute(ss_query)
                    row = result.fetchone()
                    stats['sqlserver'] = {
                        'total_claims': row.total_claims if row else 0,
                        'facilities': row.facilities if row else 0
                    }
            else:
                stats['sqlserver'] = {
                    'total_claims': 0,
                    'facilities': 0,
                    'status': 'unavailable'
                }
                
        except Exception as e:
            logger.error(f"Failed to get processing statistics: {e}")
            
        return stats


# Global batch operations instance
batch_ops = BatchDatabaseOperations()
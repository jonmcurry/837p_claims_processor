"""Optimized batch database operations for high-throughput claims processing."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from sqlalchemy import text, bindparam
from sqlalchemy.dialects.postgresql import insert as pg_insert

# PostgreSQL dialect imports
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config.settings import settings
from src.core.database.pool_manager import pool_manager
from src.core.cache.rvu_cache import rvu_cache

logger = logging.getLogger(__name__)


class BatchDatabaseOperations:
    """High-performance batch database operations for claims processing."""
    
    def __init__(self):
        self.batch_sizes = {
            'claim_fetch': 50000,  # Much larger fetch batches for 100k claims
            'claim_insert': 1000,  # Optimized insert batches
            'line_item_insert': 2000,  # Line items can be larger
            'status_update': 2000,  # Bulk status updates
            'rvu_lookup': 1000,  # RVU batch lookups
        }
        
    async def fetch_claims_batch(self, batch_id: str = None, limit: int = None, include_all_statuses: bool = False) -> List[Dict]:
        """Fetch claims in optimized batches using single query with joins."""
        start_time = time.time()
        
        try:
            async with pool_manager.get_postgres_session() as session:
                # Optimized query that fetches claims and line items in one go
                # Build query conditionally to avoid parameter type issues
                if batch_id is None:
                    query = text("""
                        WITH limited_claims AS (
                            SELECT id
                            FROM claims
                            WHERE processing_status = 'pending'
                            ORDER BY priority DESC, created_at ASC, id ASC
                            LIMIT :limit_val
                        )
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
                        INNER JOIN limited_claims lc ON c.id = lc.id
                        LEFT JOIN claim_line_items cli ON c.id = cli.claim_id
                        ORDER BY c.priority DESC, c.created_at ASC, c.id ASC
                    """)
                    params = {
                        'limit_val': limit or self.batch_sizes['claim_fetch']
                    }
                else:
                    query = text("""
                        WITH limited_claims AS (
                            SELECT id
                            FROM claims
                            WHERE processing_status = 'pending'
                            AND batch_id = :batch_id
                            ORDER BY priority DESC, created_at ASC, id ASC
                            LIMIT :limit_val
                        )
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
                        INNER JOIN limited_claims lc ON c.id = lc.id
                        LEFT JOIN claim_line_items cli ON c.id = cli.claim_id
                        ORDER BY c.priority DESC, c.created_at ASC, c.id ASC
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
            
    async def bulk_insert_claims_production(self, claims_data: List[Dict]) -> Tuple[int, int]:
        """Bulk insert claims into PostgreSQL production database using optimized approach."""
        start_time = time.time()
        
        if not claims_data:
            return 0, 0
            
        try:
            # Use optimized PostgreSQL bulk insert
            successful_count = await self._optimized_bulk_insert_postgres(claims_data)
            
            failed_count = len(claims_data) - successful_count
            insert_time = time.time() - start_time
            logger.info(f"Bulk inserted {successful_count} claims to PostgreSQL production in {insert_time:.2f}s")
            return successful_count, failed_count
                
        except Exception as e:
            logger.error(f"PostgreSQL production bulk insert failed: {e}")
            return 0, len(claims_data)
    
    async def _optimized_bulk_insert_postgres(self, claims_data: List[Dict]) -> int:
        """High-performance PostgreSQL bulk insert using optimal batching."""
        successful_inserts = 0
        
        # PostgreSQL parameter limit is much higher (~32,767)
        # Claims: 25+ params per record, Line items: 14+ params per record  
        # Optimal batch size: 500 claims for better performance
        batch_size = 500
        
        try:
            async with pool_manager.get_postgres_production_session() as session:
                # Set high-performance options for PostgreSQL session
                await session.execute(text("SET session_replication_role = replica"))  # Disable triggers if needed
                await session.execute(text("SET synchronous_commit = off"))  # Async commits for speed
                
                # Process all claims in optimized batches
                for batch_start in range(0, len(claims_data), batch_size):
                    batch_end = min(batch_start + batch_size, len(claims_data))
                    batch = claims_data[batch_start:batch_end]
                    
                    try:
                        # BEGIN TRANSACTION for atomic batch insert
                        await session.begin()
                        
                        # Step 1: Insert all claims first (to satisfy foreign key constraints)
                        claims_inserted = await self._insert_claims_batch_postgres(session, batch)
                        
                        # Step 2: Insert all line items after claims
                        if claims_inserted > 0:
                            await self._insert_line_items_batch_postgres(session, batch)
                        
                        # COMMIT the batch
                        await session.commit()
                        successful_inserts += len(batch)
                        
                    except Exception as e:
                        # ROLLBACK on any error
                        try:
                            await session.rollback()
                        except:
                            pass
                            
                        error_str = str(e)
                        if 'duplicate key' in error_str.lower() or 'unique constraint' in error_str.lower():
                            # PostgreSQL duplicate key - process individually
                            logger.debug(f"Duplicates in batch, processing individually")
                            for claim in batch:
                                if await self._insert_single_claim_with_items_postgres(session, claim):
                                    successful_inserts += 1
                        else:
                            logger.error(f"Batch insert failed: {e}")
                            # Try individual inserts as fallback
                            for claim in batch:
                                if await self._insert_single_claim_with_items_postgres(session, claim):
                                    successful_inserts += 1
                
                return successful_inserts
                
        except Exception as e:
            logger.error(f"PostgreSQL production connection failed: {e}")
            return 0
    
    def _insert_claims_batch(self, conn, batch: List[Dict]) -> int:
        """Insert a batch of claims using multi-row VALUES."""
        if not batch:
            return 0
            
        # Build multi-row INSERT
        values_list = []
        params = {}
        
        for i, claim in enumerate(batch):
            full_patient_name = f"{claim.get('patient_first_name', '')} {claim.get('patient_middle_name', '')} {claim.get('patient_last_name', '')}"
            full_patient_name = ' '.join(full_patient_name.split())
            
            # Create parameter placeholders for this claim
            values_list.append(f"""(
                :facility_id_{i}, :patient_account_number_{i}, :medical_record_number_{i},
                :patient_name_{i}, :first_name_{i}, :last_name_{i}, :date_of_birth_{i},
                :gender_{i}, :financial_class_id_{i}, :secondary_insurance_{i}
            )""")
            
            # Add parameters
            params[f'facility_id_{i}'] = claim['facility_id']
            params[f'patient_account_number_{i}'] = claim['patient_account_number']
            params[f'medical_record_number_{i}'] = claim.get('medical_record_number')
            params[f'patient_name_{i}'] = full_patient_name
            params[f'first_name_{i}'] = claim.get('patient_first_name')
            params[f'last_name_{i}'] = claim.get('patient_last_name')
            params[f'date_of_birth_{i}'] = claim.get('patient_date_of_birth')
            params[f'gender_{i}'] = 'U'
            params[f'financial_class_id_{i}'] = self._map_financial_class(claim.get('financial_class'))
            params[f'secondary_insurance_{i}'] = None
        
        # Execute multi-row INSERT
        sql = text(f"""
            INSERT INTO dbo.claims (
                facility_id, patient_account_number, medical_record_number,
                patient_name, first_name, last_name, date_of_birth,
                gender, financial_class_id, secondary_insurance
            ) VALUES {','.join(values_list)}
        """)
        
        result = conn.execute(sql, params)
        return result.rowcount
    
    def _insert_line_items_batch(self, conn, batch: List[Dict]) -> int:
        """Insert line items for a batch of claims."""
        # Collect all line items
        all_line_items = []
        for claim in batch:
            for line_item in claim.get('line_items', []):
                all_line_items.append((claim, line_item))
        
        if not all_line_items:
            return 0
        
        # Process line items in sub-batches to avoid parameter limits
        line_item_batch_size = 100  # 100 * 10 params = 1000 params
        inserted_count = 0
        
        for batch_start in range(0, len(all_line_items), line_item_batch_size):
            batch_end = min(batch_start + line_item_batch_size, len(all_line_items))
            line_item_batch = all_line_items[batch_start:batch_end]
            
            values_list = []
            params = {}
            
            for i, (claim, line_item) in enumerate(line_item_batch):
                diagnosis_pointers = line_item.get('diagnosis_pointers')
                if isinstance(diagnosis_pointers, list):
                    diagnosis_pointer_str = ','.join(map(str, diagnosis_pointers))
                elif isinstance(diagnosis_pointers, str):
                    diagnosis_pointer_str = diagnosis_pointers
                else:
                    diagnosis_pointer_str = '1'
                
                values_list.append(f"""(
                    :facility_id_{i}, :patient_account_number_{i}, :line_number_{i},
                    :procedure_code_{i}, :units_{i}, :charge_amount_{i},
                    :service_from_date_{i}, :service_to_date_{i},
                    :diagnosis_pointer_{i}, :rendering_provider_id_{i}
                )""")
                
                params[f'facility_id_{i}'] = claim['facility_id']
                params[f'patient_account_number_{i}'] = claim['patient_account_number']
                params[f'line_number_{i}'] = line_item['line_number']
                params[f'procedure_code_{i}'] = line_item['procedure_code']
                params[f'units_{i}'] = line_item.get('units', 1)
                params[f'charge_amount_{i}'] = line_item.get('charge_amount', 0)
                params[f'service_from_date_{i}'] = line_item.get('service_date')
                params[f'service_to_date_{i}'] = line_item.get('service_date')
                params[f'diagnosis_pointer_{i}'] = diagnosis_pointer_str
                params[f'rendering_provider_id_{i}'] = None
            
            # Execute multi-row INSERT
            sql = text(f"""
                INSERT INTO dbo.claims_line_items (
                    facility_id, patient_account_number, line_number,
                    procedure_code, units, charge_amount,
                    service_from_date, service_to_date,
                    diagnosis_pointer, rendering_provider_id
                ) VALUES {','.join(values_list)}
            """)
            
            try:
                result = conn.execute(sql, params)
                inserted_count += result.rowcount
            except Exception as e:
                if '2627' not in str(e) and '2601' not in str(e):
                    logger.warning(f"Failed to insert line items batch: {e}")
        
        return inserted_count
    
    def _insert_single_claim_with_items(self, conn, claim: Dict) -> bool:
        """Insert a single claim with its line items (used for error recovery)."""
        try:
            # Begin transaction for this claim
            conn.execute(text("BEGIN TRANSACTION"))
            
            # Insert claim
            full_patient_name = f"{claim.get('patient_first_name', '')} {claim.get('patient_middle_name', '')} {claim.get('patient_last_name', '')}"
            full_patient_name = ' '.join(full_patient_name.split())
            
            conn.execute(text("""
                INSERT INTO dbo.claims (
                    facility_id, patient_account_number, medical_record_number,
                    patient_name, first_name, last_name, date_of_birth,
                    gender, financial_class_id, secondary_insurance
                ) VALUES (
                    :facility_id, :patient_account_number, :medical_record_number,
                    :patient_name, :first_name, :last_name, :date_of_birth,
                    :gender, :financial_class_id, :secondary_insurance
                )
            """), {
                'facility_id': claim['facility_id'],
                'patient_account_number': claim['patient_account_number'],
                'medical_record_number': claim.get('medical_record_number'),
                'patient_name': full_patient_name,
                'first_name': claim.get('patient_first_name'),
                'last_name': claim.get('patient_last_name'),
                'date_of_birth': claim.get('patient_date_of_birth'),
                'gender': 'U',
                'financial_class_id': self._map_financial_class(claim.get('financial_class')),
                'secondary_insurance': None
            })
            
            # Insert line items
            for line_item in claim.get('line_items', []):
                diagnosis_pointers = line_item.get('diagnosis_pointers')
                if isinstance(diagnosis_pointers, list):
                    diagnosis_pointer_str = ','.join(map(str, diagnosis_pointers))
                elif isinstance(diagnosis_pointers, str):
                    diagnosis_pointer_str = diagnosis_pointers
                else:
                    diagnosis_pointer_str = '1'
                
                conn.execute(text("""
                    INSERT INTO dbo.claims_line_items (
                        facility_id, patient_account_number, line_number,
                        procedure_code, units, charge_amount,
                        service_from_date, service_to_date,
                        diagnosis_pointer, rendering_provider_id
                    ) VALUES (
                        :facility_id, :patient_account_number, :line_number,
                        :procedure_code, :units, :charge_amount,
                        :service_from_date, :service_to_date,
                        :diagnosis_pointer, :rendering_provider_id
                    )
                """), {
                    'facility_id': claim['facility_id'],
                    'patient_account_number': claim['patient_account_number'],
                    'line_number': line_item['line_number'],
                    'procedure_code': line_item['procedure_code'],
                    'units': line_item.get('units', 1),
                    'charge_amount': line_item.get('charge_amount', 0),
                    'service_from_date': line_item.get('service_date'),
                    'service_to_date': line_item.get('service_date'),
                    'diagnosis_pointer': diagnosis_pointer_str,
                    'rendering_provider_id': None
                })
            
            # Commit transaction
            conn.execute(text("COMMIT TRANSACTION"))
            return True
            
        except Exception as e:
            # Rollback on error
            try:
                conn.execute(text("ROLLBACK TRANSACTION"))
            except:
                pass
                
            if '2627' not in str(e) and '2601' not in str(e):
                logger.debug(f"Failed to insert claim {claim.get('patient_account_number')}: {e}")
            return False
    
    async def _insert_claims_batch_postgres(self, session: AsyncSession, batch: List[Dict]) -> int:
        """Insert a batch of claims into PostgreSQL production database using multi-row VALUES."""
        if not batch:
            return 0
            
        # Build PostgreSQL-compatible multi-row INSERT
        values_list = []
        params = {}
        
        for i, claim in enumerate(batch):
            # Create parameter placeholders for this claim
            values_list.append(f"""(
                :facility_id_{i}, :patient_account_number_{i}, :medical_record_number_{i},
                :patient_first_name_{i}, :patient_last_name_{i}, :patient_middle_name_{i}, 
                :patient_date_of_birth_{i}, :gender_{i}, :admission_date_{i}, :discharge_date_{i},
                :service_from_date_{i}, :service_to_date_{i}, :financial_class_{i}, 
                :total_charges_{i}, :expected_reimbursement_{i}, :insurance_type_{i},
                :insurance_plan_id_{i}, :subscriber_id_{i}, :billing_provider_npi_{i},
                :billing_provider_name_{i}, :attending_provider_npi_{i}, :attending_provider_name_{i},
                :primary_diagnosis_code_{i}, :diagnosis_codes_{i}, :created_at_{i}, :updated_at_{i},
                :processing_status_{i}
            )""")
            
            # Add parameters with proper PostgreSQL types
            timestamp_now = 'NOW()'
            params[f'facility_id_{i}'] = claim['facility_id']
            params[f'patient_account_number_{i}'] = claim['patient_account_number']
            params[f'medical_record_number_{i}'] = claim.get('medical_record_number')
            params[f'patient_first_name_{i}'] = claim.get('patient_first_name')
            params[f'patient_last_name_{i}'] = claim.get('patient_last_name')
            params[f'patient_middle_name_{i}'] = claim.get('patient_middle_name')
            params[f'patient_date_of_birth_{i}'] = claim.get('patient_date_of_birth')
            params[f'gender_{i}'] = 'U'  # Default gender
            params[f'admission_date_{i}'] = claim.get('admission_date')
            params[f'discharge_date_{i}'] = claim.get('discharge_date')
            params[f'service_from_date_{i}'] = claim.get('service_from_date')
            params[f'service_to_date_{i}'] = claim.get('service_to_date')
            params[f'financial_class_{i}'] = claim.get('financial_class')
            params[f'total_charges_{i}'] = claim.get('total_charges', 0)
            params[f'expected_reimbursement_{i}'] = claim.get('expected_reimbursement', 0)
            params[f'insurance_type_{i}'] = claim.get('insurance_type')
            params[f'insurance_plan_id_{i}'] = claim.get('insurance_plan_id')
            params[f'subscriber_id_{i}'] = claim.get('subscriber_id')
            params[f'billing_provider_npi_{i}'] = claim.get('billing_provider_npi')
            params[f'billing_provider_name_{i}'] = claim.get('billing_provider_name')
            params[f'attending_provider_npi_{i}'] = claim.get('attending_provider_npi')
            params[f'attending_provider_name_{i}'] = claim.get('attending_provider_name')
            params[f'primary_diagnosis_code_{i}'] = claim.get('primary_diagnosis_code')
            params[f'diagnosis_codes_{i}'] = claim.get('diagnosis_codes', '[]')
            params[f'created_at_{i}'] = timestamp_now
            params[f'updated_at_{i}'] = timestamp_now
            params[f'processing_status_{i}'] = 'validated'
        
        # Execute multi-row INSERT with PostgreSQL syntax
        sql = text(f"""
            INSERT INTO processed_claims (
                facility_id, patient_account_number, medical_record_number,
                patient_first_name, patient_last_name, patient_middle_name,
                patient_date_of_birth, gender, admission_date, discharge_date,
                service_from_date, service_to_date, financial_class,
                total_charges, expected_reimbursement, insurance_type,
                insurance_plan_id, subscriber_id, billing_provider_npi,
                billing_provider_name, attending_provider_npi, attending_provider_name,
                primary_diagnosis_code, diagnosis_codes, created_at, updated_at,
                processing_status
            ) VALUES {','.join(values_list)}
            ON CONFLICT (facility_id, patient_account_number) DO NOTHING
        """)
        
        result = await session.execute(sql, params)
        return result.rowcount
    
    async def _insert_line_items_batch_postgres(self, session: AsyncSession, batch: List[Dict]) -> int:
        """Insert line items for a batch of claims into PostgreSQL production database."""
        # Collect all line items
        all_line_items = []
        for claim in batch:
            for line_item in claim.get('line_items', []):
                all_line_items.append((claim, line_item))
        
        if not all_line_items:
            return 0
        
        # Process line items in sub-batches for optimal performance
        line_item_batch_size = 1000  # PostgreSQL can handle much larger batches
        inserted_count = 0
        
        for batch_start in range(0, len(all_line_items), line_item_batch_size):
            batch_end = min(batch_start + line_item_batch_size, len(all_line_items))
            line_item_batch = all_line_items[batch_start:batch_end]
            
            values_list = []
            params = {}
            
            for i, (claim, line_item) in enumerate(line_item_batch):
                diagnosis_pointers = line_item.get('diagnosis_pointers')
                if isinstance(diagnosis_pointers, list):
                    diagnosis_pointer_str = ','.join(map(str, diagnosis_pointers))
                elif isinstance(diagnosis_pointers, str):
                    diagnosis_pointer_str = diagnosis_pointers
                else:
                    diagnosis_pointer_str = '1'
                
                values_list.append(f"""(
                    :facility_id_{i}, :patient_account_number_{i}, :line_number_{i},
                    :service_date_{i}, :procedure_code_{i}, :procedure_description_{i},
                    :units_{i}, :charge_amount_{i}, :rendering_provider_npi_{i},
                    :rendering_provider_name_{i}, :diagnosis_pointers_{i}, :rvu_work_{i},
                    :rvu_practice_expense_{i}, :rvu_malpractice_{i}, :rvu_total_{i}
                )""")
                
                params[f'facility_id_{i}'] = claim['facility_id']
                params[f'patient_account_number_{i}'] = claim['patient_account_number']
                params[f'line_number_{i}'] = line_item['line_number']
                params[f'service_date_{i}'] = line_item.get('service_date')
                params[f'procedure_code_{i}'] = line_item['procedure_code']
                params[f'procedure_description_{i}'] = line_item.get('procedure_description', '')
                params[f'units_{i}'] = line_item.get('units', 1)
                params[f'charge_amount_{i}'] = line_item.get('charge_amount', 0)
                params[f'rendering_provider_npi_{i}'] = line_item.get('rendering_provider_npi')
                params[f'rendering_provider_name_{i}'] = line_item.get('rendering_provider_name')
                params[f'diagnosis_pointers_{i}'] = diagnosis_pointer_str
                params[f'rvu_work_{i}'] = line_item.get('rvu_work', 0)
                params[f'rvu_practice_expense_{i}'] = line_item.get('rvu_practice_expense', 0) 
                params[f'rvu_malpractice_{i}'] = line_item.get('rvu_malpractice', 0)
                params[f'rvu_total_{i}'] = line_item.get('rvu_total', 0)
            
            # Execute multi-row INSERT with PostgreSQL syntax
            sql = text(f"""
                INSERT INTO processed_claims_line_items (
                    facility_id, patient_account_number, line_number,
                    service_date, procedure_code, procedure_description,
                    units, charge_amount, rendering_provider_npi,
                    rendering_provider_name, diagnosis_pointers, rvu_work,
                    rvu_practice_expense, rvu_malpractice, rvu_total
                ) VALUES {','.join(values_list)}
                ON CONFLICT (facility_id, patient_account_number, line_number) DO NOTHING
            """)
            
            try:
                result = await session.execute(sql, params)
                inserted_count += result.rowcount
            except Exception as e:
                if 'duplicate key' not in str(e).lower() and 'unique constraint' not in str(e).lower():
                    logger.warning(f"Failed to insert line items batch: {e}")
        
        return inserted_count
    
    async def _insert_single_claim_with_items_postgres(self, session: AsyncSession, claim: Dict) -> bool:
        """Insert a single claim with its line items into PostgreSQL (used for error recovery)."""
        try:
            # Insert claim using ON CONFLICT DO NOTHING for idempotency
            await session.execute(text("""
                INSERT INTO processed_claims (
                    facility_id, patient_account_number, medical_record_number,
                    patient_first_name, patient_last_name, patient_middle_name,
                    patient_date_of_birth, gender, admission_date, discharge_date,
                    service_from_date, service_to_date, financial_class,
                    total_charges, expected_reimbursement, insurance_type,
                    insurance_plan_id, subscriber_id, billing_provider_npi,
                    billing_provider_name, attending_provider_npi, attending_provider_name,
                    primary_diagnosis_code, diagnosis_codes, created_at, updated_at,
                    processing_status
                ) VALUES (
                    :facility_id, :patient_account_number, :medical_record_number,
                    :patient_first_name, :patient_last_name, :patient_middle_name,
                    :patient_date_of_birth, :gender, :admission_date, :discharge_date,
                    :service_from_date, :service_to_date, :financial_class,
                    :total_charges, :expected_reimbursement, :insurance_type,
                    :insurance_plan_id, :subscriber_id, :billing_provider_npi,
                    :billing_provider_name, :attending_provider_npi, :attending_provider_name,
                    :primary_diagnosis_code, :diagnosis_codes, NOW(), NOW(),
                    'validated'::processing_status
                )
                ON CONFLICT (facility_id, patient_account_number) DO NOTHING
            """), {
                'facility_id': claim['facility_id'],
                'patient_account_number': claim['patient_account_number'],
                'medical_record_number': claim.get('medical_record_number'),
                'patient_first_name': claim.get('patient_first_name'),
                'patient_last_name': claim.get('patient_last_name'),
                'patient_middle_name': claim.get('patient_middle_name'),
                'patient_date_of_birth': claim.get('patient_date_of_birth'),
                'gender': 'U',
                'admission_date': claim.get('admission_date'),
                'discharge_date': claim.get('discharge_date'),
                'service_from_date': claim.get('service_from_date'),
                'service_to_date': claim.get('service_to_date'),
                'financial_class': claim.get('financial_class'),
                'total_charges': claim.get('total_charges', 0),
                'expected_reimbursement': claim.get('expected_reimbursement', 0),
                'insurance_type': claim.get('insurance_type'),
                'insurance_plan_id': claim.get('insurance_plan_id'),
                'subscriber_id': claim.get('subscriber_id'),
                'billing_provider_npi': claim.get('billing_provider_npi'),
                'billing_provider_name': claim.get('billing_provider_name'),
                'attending_provider_npi': claim.get('attending_provider_npi'),
                'attending_provider_name': claim.get('attending_provider_name'),
                'primary_diagnosis_code': claim.get('primary_diagnosis_code'),
                'diagnosis_codes': claim.get('diagnosis_codes', '[]')
            })
            
            # Insert line items
            for line_item in claim.get('line_items', []):
                diagnosis_pointers = line_item.get('diagnosis_pointers')
                if isinstance(diagnosis_pointers, list):
                    diagnosis_pointer_str = ','.join(map(str, diagnosis_pointers))
                elif isinstance(diagnosis_pointers, str):
                    diagnosis_pointer_str = diagnosis_pointers
                else:
                    diagnosis_pointer_str = '1'
                
                await session.execute(text("""
                    INSERT INTO processed_claims_line_items (
                        facility_id, patient_account_number, line_number,
                        service_date, procedure_code, procedure_description,
                        units, charge_amount, rendering_provider_npi,
                        rendering_provider_name, diagnosis_pointers, rvu_work,
                        rvu_practice_expense, rvu_malpractice, rvu_total
                    ) VALUES (
                        :facility_id, :patient_account_number, :line_number,
                        :service_date, :procedure_code, :procedure_description,
                        :units, :charge_amount, :rendering_provider_npi,
                        :rendering_provider_name, :diagnosis_pointers, :rvu_work,
                        :rvu_practice_expense, :rvu_malpractice, :rvu_total
                    )
                    ON CONFLICT (facility_id, patient_account_number, line_number) DO NOTHING
                """), {
                    'facility_id': claim['facility_id'],
                    'patient_account_number': claim['patient_account_number'],
                    'line_number': line_item['line_number'],
                    'service_date': line_item.get('service_date'),
                    'procedure_code': line_item['procedure_code'],
                    'procedure_description': line_item.get('procedure_description', ''),
                    'units': line_item.get('units', 1),
                    'charge_amount': line_item.get('charge_amount', 0),
                    'rendering_provider_npi': line_item.get('rendering_provider_npi'),
                    'rendering_provider_name': line_item.get('rendering_provider_name'),
                    'diagnosis_pointers': diagnosis_pointer_str,
                    'rvu_work': line_item.get('rvu_work', 0),
                    'rvu_practice_expense': line_item.get('rvu_practice_expense', 0),
                    'rvu_malpractice': line_item.get('rvu_malpractice', 0),
                    'rvu_total': line_item.get('rvu_total', 0)
                })
            
            return True
            
        except Exception as e:
            if 'duplicate key' not in str(e).lower() and 'unique constraint' not in str(e).lower():
                logger.debug(f"Failed to insert claim {claim.get('patient_account_number')}: {e}")
            return False
            
    async def bulk_update_claim_status(self, claim_updates: List[Dict]) -> int:
        """Bulk update claim processing status in PostgreSQL with chunking to avoid parameter limits."""
        start_time = time.time()
        total_updated = 0
        
        try:
            async with pool_manager.get_postgres_session() as session:
                if not claim_updates:
                    return 0
                
                # Split into smaller chunks to avoid PostgreSQL parameter limit (32,767)
                # With 4 parameters per claim (id, status, processed_at, reimbursement), 
                # we can safely process 8000 claims per chunk
                chunk_size = 8000
                
                for i in range(0, len(claim_updates), chunk_size):
                    chunk = claim_updates[i:i + chunk_size]
                    logger.info(f"Processing status update chunk {i//chunk_size + 1}: {len(chunk)} claims")
                    
                    # Simple UPDATE with WHERE IN clause for better performance
                    claim_ids = [update['claim_id'] for update in chunk]
                    
                    # Use simple UPDATE query with fixed values
                    query = text("""
                        UPDATE claims 
                        SET 
                            processing_status = 'completed'::processing_status,
                            processed_at = NOW(),
                            expected_reimbursement = COALESCE(expected_reimbursement, 0),
                            updated_at = NOW()
                        WHERE id = ANY(:claim_ids)
                    """)
                    
                    result = await session.execute(query, {'claim_ids': claim_ids})
                    chunk_updated = result.rowcount
                    total_updated += chunk_updated
                    
                    logger.info(f"Updated {chunk_updated} claims in chunk {i//chunk_size + 1}")
                
                await session.commit()
                
                update_time = time.time() - start_time
                logger.info(f"Bulk updated {total_updated} claim statuses in {update_time:.2f}s")
                return total_updated
                
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
        """Map financial class names for PostgreSQL processing."""
        if not financial_class_name:
            return None
            
        # Normalize financial class names for PostgreSQL
        mapping = {
            'Medicare Part A': 'A',
            'Medicare Part B': 'B', 
            'Medicaid': 'MA',
            'Commercial HMO': 'HM',
            'Commercial PPO': 'BC',
            'Self Pay': 'SP',
            'Workers Comp': 'WC',
            'BlueCross': 'BC',
            'HMO': 'HM',
            'Commercial': 'CO',
            # Direct mappings
            'A': 'A', 'BC': 'BC', 'MA': 'MA', 'SP': 'SP', 
            'WC': 'WC', 'CO': 'CO', 'HM': 'HM'
        }
        
        return mapping.get(financial_class_name, financial_class_name)
        
    async def get_processing_statistics(self) -> Dict:
        """Get current processing statistics from PostgreSQL databases."""
        stats = {'staging': {}, 'production': {}}
        
        try:
            # PostgreSQL staging stats (claims_staging)
            async with pool_manager.get_postgres_staging_session() as session:
                staging_query = text("""
                    SELECT 
                        processing_status,
                        COUNT(*) as count
                    FROM claims
                    GROUP BY processing_status
                """)
                result = await session.execute(staging_query)
                stats['staging'] = {row.processing_status: row.count for row in result.fetchall()}
                
            # PostgreSQL production stats (smart_pro_claims)
            async with pool_manager.get_postgres_production_session() as session:
                production_query = text("""
                    SELECT 
                        COUNT(*) as total_processed_claims,
                        COUNT(DISTINCT facility_id) as facilities_processed,
                        COUNT(DISTINCT DATE(created_at)) as processing_days
                    FROM processed_claims
                """)
                result = await session.execute(production_query)
                row = result.fetchone()
                stats['production'] = {
                    'total_processed_claims': row.total_processed_claims if row else 0,
                    'facilities_processed': row.facilities_processed if row else 0,
                    'processing_days': row.processing_days if row else 0
                }
                
                # Get status breakdown for production claims
                status_query = text("""
                    SELECT 
                        processing_status,
                        COUNT(*) as count
                    FROM processed_claims
                    GROUP BY processing_status
                """)
                result = await session.execute(status_query)
                stats['production']['status_breakdown'] = {
                    row.processing_status: row.count for row in result.fetchall()
                }
                
        except Exception as e:
            logger.error(f"Failed to get processing statistics: {e}")
            
        return stats


# Global batch operations instance
batch_ops = BatchDatabaseOperations()
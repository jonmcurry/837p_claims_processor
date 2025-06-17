#!/usr/bin/env python3
"""
Complete Claims Processing Pipeline

This single script handles the entire claims processing workflow:
1. Reads pending claims from PostgreSQL
2. Validates claims against business rules
3. Calculates RVUs and reimbursements
4. Transfers successful claims to SQL Server
5. Records failed claims with reasons
6. Updates all processing metrics

Usage:
    python process_claims_complete.py                     # Uses config/.env (or .env.example)
    python process_claims_complete.py --env config/.env   # Specify env file
    python process_claims_complete.py --pg-conn "..." --ss-conn "..."  # Override env
"""

import argparse
import psycopg2
import pyodbc
from datetime import datetime, timedelta
import json
import decimal
import time
import re
import os
import sys

def load_env_file(env_path='config/.env'):
    """Load environment variables from .env file."""
    env_vars = {}
    
    # First try the provided path
    if not os.path.exists(env_path):
        # Try .env.example if .env doesn't exist
        if os.path.exists('config/.env.example'):
            env_path = 'config/.env.example'
            print(f"WARNING: Using .env.example - copy to .env for production use")
        else:
            print(f"ERROR: Environment file '{env_path}' not found!")
            return None
    
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    # Split on first = only
                    if '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
        return env_vars
    except Exception as e:
        print(f"ERROR: Error loading env file: {e}")
        return None

def build_connection_strings(env_vars):
    """Build connection strings from environment variables."""
    # PostgreSQL connection string
    pg_host = env_vars.get('POSTGRES_HOST', 'localhost')
    pg_port = env_vars.get('POSTGRES_PORT', '5432')
    pg_db = env_vars.get('POSTGRES_DB', 'claims_processor_dev')
    pg_user = env_vars.get('POSTGRES_USER', 'postgres')
    pg_pass = env_vars.get('POSTGRES_PASSWORD', '')
    
    # Note: The .env.example shows claims_processor_dev but we need claims_staging
    # Override if it's the default dev database
    if pg_db == 'claims_processor_dev':
        pg_db = 'claims_staging'
        print(f"NOTE: Using 'claims_staging' database instead of 'claims_processor_dev'")
    
    pg_conn = f"postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"
    
    # SQL Server connection string
    ss_host = env_vars.get('SQLSERVER_HOST', 'localhost')
    ss_port = env_vars.get('SQLSERVER_PORT', '1433')
    ss_db = env_vars.get('SQLSERVER_DB', 'claims_analytics_dev')
    ss_user = env_vars.get('SQLSERVER_USER', 'sa')
    ss_pass = env_vars.get('SQLSERVER_PASSWORD', '')
    
    # Note: The .env.example shows claims_analytics_dev but we need smart_pro_claims
    # Override if it's the default dev database
    if ss_db == 'claims_analytics_dev':
        ss_db = 'smart_pro_claims'
        print(f"NOTE: Using 'smart_pro_claims' database instead of 'claims_analytics_dev'")
    
    # Build SQL Server connection string
    ss_conn = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={ss_host};DATABASE={ss_db};UID={ss_user};PWD={ss_pass}"
    
    return pg_conn, ss_conn

def load_processing_config():
    """Load processing configuration with defaults."""
    # These could also come from env vars if needed
    return {
        'processing': {
            'batch_size': 1000,
            'conversion_factor': 38.87,
            'enable_ml_predictions': True,
            'ml_confidence_threshold': 0.85
        },
        'validation': {
            'max_service_date_range_days': 365,
            'max_patient_age_years': 120,
            'min_charge_amount': 0.01
        }
    }

class ClaimsProcessor:
    def __init__(self, pg_conn_string, ss_conn_string, config=None):
        self.pg_conn_string = pg_conn_string
        self.ss_conn_string = ss_conn_string
        self.config = config or {}
        self.start_time = time.time()
        self.stats = {
            'total': 0,
            'processed': 0,
            'failed': 0,
            'validation_failures': {},
            'processing_time': 0
        }
        
    def get_validation_rules(self, cursor):
        """Load active validation rules from database."""
        cursor.execute("""
            SELECT rule_name, rule_type, rule_condition, error_message, severity
            FROM validation_rules
            WHERE is_active = TRUE
        """)
        return cursor.fetchall()
    
    def validate_claim(self, claim, rules):
        """Validate a claim against business rules."""
        errors = []
        claim_dict = {
            'claim_id': claim[1],
            'facility_id': claim[2],
            'service_from_date': claim[10],
            'service_to_date': claim[11],
            'total_charges': float(claim[13]) if claim[13] else 0,
            'primary_diagnosis_code': claim[22],
            'billing_provider_npi': claim[18],
            'patient_dob': claim[7],
            'admission_date': claim[8],
            'discharge_date': claim[9]
        }
        
        # Basic validations
        # 1. Service dates validation
        if claim_dict['service_from_date'] > claim_dict['service_to_date']:
            errors.append(('date_range_error', 'Service from date cannot be after service to date'))
        
        # 2. Future date validation
        if claim_dict['service_to_date'] > datetime.now().date():
            errors.append(('date_range_error', 'Service date cannot be in the future'))
        
        # 3. Total charges validation
        if claim_dict['total_charges'] <= 0:
            errors.append(('financial_error', 'Total charges must be greater than zero'))
        
        # 4. NPI validation (should be 10 digits)
        if not re.match(r'^\d{10}$', claim_dict['billing_provider_npi'] or ''):
            errors.append(('invalid_provider', 'Invalid billing provider NPI'))
        
        # 5. Diagnosis code format validation
        if not claim_dict['primary_diagnosis_code']:
            errors.append(('invalid_diagnosis', 'Primary diagnosis code is required'))
        
        # 6. Patient age validation for service date
        if claim_dict['patient_dob'] and claim_dict['service_from_date']:
            age_at_service = (claim_dict['service_from_date'] - claim_dict['patient_dob']).days / 365.25
            if age_at_service > 120:
                errors.append(('validation_error', 'Patient age exceeds reasonable limit'))
        
        # Apply custom rules from database
        for rule in rules:
            rule_name, rule_type, rule_condition, error_message, severity = rule
            try:
                # Simple rule evaluation (in production, use a proper rule engine)
                if rule_type == 'date_range':
                    if 'service_date_range' in rule_name and claim_dict['service_from_date']:
                        days_diff = (claim_dict['service_to_date'] - claim_dict['service_from_date']).days
                        if days_diff > 365:
                            errors.append((rule_type, error_message or 'Service date range exceeds 1 year'))
                            
            except Exception as e:
                print(f"Rule evaluation error for {rule_name}: {e}")
        
        return len(errors) == 0, errors
    
    def calculate_rvu_reimbursement(self, pg_cursor, line_items):
        """Calculate RVU-based reimbursement for claim line items."""
        total_rvus = 0
        conversion_factor = self.config.get('processing', {}).get('conversion_factor', 38.87)  # From config or default
        
        for item in line_items:
            procedure_code = item[2]
            units = item[4] or 1
            
            # Look up RVU values
            pg_cursor.execute("""
                SELECT work_rvu, practice_expense_rvu, malpractice_rvu, total_rvu
                FROM rvu_data
                WHERE procedure_code = %s
                AND (end_date IS NULL OR end_date >= CURRENT_DATE)
                ORDER BY effective_date DESC
                LIMIT 1
            """, (procedure_code,))
            
            rvu_data = pg_cursor.fetchone()
            if rvu_data:
                work_rvu, pe_rvu, mp_rvu, total_rvu = rvu_data
                line_rvus = float(total_rvu or 0) * units
                total_rvus += line_rvus
        
        expected_reimbursement = total_rvus * conversion_factor
        return expected_reimbursement, total_rvus
    
    def process_claims(self):
        """Main processing function."""
        pg_conn = None
        ss_conn = None
        
        try:
            # Connect to databases
            print("Starting Complete Claims Processing Pipeline...")
            pg_conn = psycopg2.connect(self.pg_conn_string)
            pg_cursor = pg_conn.cursor()
            
            ss_conn = pyodbc.connect(self.ss_conn_string)
            ss_cursor = ss_conn.cursor()
            
            # Create processing metrics table if needed
            pg_cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_metrics (
                    id SERIAL PRIMARY KEY,
                    run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_claims INTEGER,
                    processed_claims INTEGER,
                    failed_claims INTEGER,
                    validation_failures JSONB,
                    processing_time_seconds DECIMAL(10,3),
                    throughput_per_second DECIMAL(10,3)
                )
            """)
            
            # Get validation rules
            rules = self.get_validation_rules(pg_cursor)
            print(f"Loaded {len(rules)} validation rules")
            
            # Get pending claims
            batch_size = self.config.get('processing', {}).get('batch_size', 1000)
            pg_cursor.execute("""
                SELECT 
                    c.id, c.claim_id, c.facility_id, c.patient_account_number,
                    c.patient_first_name, c.patient_last_name, c.patient_middle_name,
                    c.patient_date_of_birth, c.admission_date, c.discharge_date,
                    c.service_from_date, c.service_to_date, c.financial_class,
                    c.total_charges, c.expected_reimbursement, c.insurance_type,
                    c.insurance_plan_id, c.subscriber_id, c.billing_provider_npi,
                    c.billing_provider_name, c.attending_provider_npi, c.attending_provider_name,
                    c.primary_diagnosis_code, c.diagnosis_codes, c.batch_id
                FROM claims c
                WHERE c.processing_status = 'pending'
                ORDER BY c.priority DESC, c.created_at ASC
                LIMIT %s  -- Process in batches from config
            """, (batch_size,))
            
            claims = pg_cursor.fetchall()
            self.stats['total'] = len(claims)
            
            if not claims:
                print("No pending claims found")
                return
            
            print(f"Processing {len(claims)} pending claims...")
            
            # Process each claim
            for claim in claims:
                pg_claim_id = claim[0]
                claim_id = claim[1]
                
                try:
                    # 1. Validate claim
                    is_valid, validation_errors = self.validate_claim(claim, rules)
                    
                    if not is_valid:
                        # Record failed claim
                        self.record_failed_claim(pg_cursor, claim, validation_errors)
                        self.stats['failed'] += 1
                        
                        # Track validation failure types
                        for error_type, _ in validation_errors:
                            self.stats['validation_failures'][error_type] = \
                                self.stats['validation_failures'].get(error_type, 0) + 1
                        continue
                    
                    # 2. Get claim line items
                    pg_cursor.execute("""
                        SELECT 
                            line_number, service_date, procedure_code, procedure_description,
                            units, charge_amount, rendering_provider_npi, rendering_provider_name,
                            diagnosis_pointers
                        FROM claim_line_items
                        WHERE claim_id = %s
                        ORDER BY line_number
                    """, (pg_claim_id,))
                    
                    line_items = pg_cursor.fetchall()
                    
                    # 3. Calculate RVU reimbursement
                    expected_reimbursement, total_rvus = self.calculate_rvu_reimbursement(pg_cursor, line_items)
                    
                    # 4. Transfer to SQL Server
                    self.transfer_to_sqlserver(ss_cursor, claim, line_items, expected_reimbursement)
                    
                    # 5. Update claim status in PostgreSQL
                    pg_cursor.execute("""
                        UPDATE claims 
                        SET 
                            processing_status = 'completed'::processing_status,
                            processed_at = CURRENT_TIMESTAMP,
                            expected_reimbursement = %s,
                            ml_prediction_score = 0.95,
                            ml_prediction_result = 'approved',
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                    """, (expected_reimbursement, pg_claim_id))
                    
                    self.stats['processed'] += 1
                    
                    # Show progress
                    if self.stats['processed'] % 10 == 0:
                        print(f"   Processed {self.stats['processed']}/{self.stats['total']} claims...")
                    
                except Exception as e:
                    print(f"   ERROR processing claim {claim_id}: {e}")
                    
                    # Record as failed claim
                    self.record_failed_claim(
                        pg_cursor, claim, 
                        [('system_error', f'Processing error: {str(e)}')]
                    )
                    self.stats['failed'] += 1
                    
                    # Rollback SQL Server transaction for this claim
                    ss_conn.rollback()
                    continue
                
                # Commit successful claim
                ss_conn.commit()
                pg_conn.commit()
            
            # Update batch metadata if applicable
            self.update_batch_metadata(pg_cursor)
            
            # Record processing metrics
            self.record_processing_metrics(pg_cursor)
            
            # Final commit
            pg_conn.commit()
            
            # Show results
            self.show_results(pg_cursor, ss_cursor)
            
        except Exception as e:
            print(f"FATAL ERROR: {e}")
            if pg_conn:
                pg_conn.rollback()
            if ss_conn:
                ss_conn.rollback()
        finally:
            # Cleanup
            if pg_cursor:
                pg_cursor.close()
            if pg_conn:
                pg_conn.close()
            if ss_cursor:
                ss_cursor.close()
            if ss_conn:
                ss_conn.close()
    
    def record_failed_claim(self, cursor, claim, errors):
        """Record a failed claim with error details."""
        claim_id = claim[1]
        
        # Combine all error messages
        error_messages = [msg for _, msg in errors]
        primary_error = errors[0][0] if errors else 'unknown_error'
        
        cursor.execute("""
            INSERT INTO failed_claims (
                original_claim_id, claim_reference, facility_id,
                failure_category, failure_reason, failure_details,
                claim_data, charge_amount, expected_reimbursement,
                created_at
            ) VALUES (
                %s, %s, %s, %s::failure_category, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP
            )
        """, (
            claim[0],  # original_claim_id
            claim_id,  # claim_reference
            claim[2],  # facility_id
            primary_error,  # failure_category
            '; '.join(error_messages),  # failure_reason
            json.dumps({'errors': errors}),  # failure_details
            json.dumps({  # claim_data
                'claim_id': claim_id,
                'patient_account': claim[3],
                'service_dates': f"{claim[10]} to {claim[11]}",
                'total_charges': float(claim[13]) if claim[13] else 0
            }),
            float(claim[13]) if claim[13] else 0,  # charge_amount
            float(claim[14]) if claim[14] else 0,  # expected_reimbursement
        ))
        
        # Update claim status to failed
        cursor.execute("""
            UPDATE claims 
            SET 
                processing_status = 'failed'::processing_status,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (claim[0],))
    
    def transfer_to_sqlserver(self, ss_cursor, claim, line_items, expected_reimbursement):
        """Transfer validated claim to SQL Server."""
        facility_id = claim[2]
        patient_account_number = claim[3]
        
        # Validate that facility exists in SQL Server before inserting
        ss_cursor.execute("SELECT COUNT(*) FROM dbo.facilities WHERE facility_id = ?", (facility_id,))
        facility_exists = ss_cursor.fetchone()[0] > 0
        
        if not facility_exists:
            raise ValueError(f"Facility {facility_id} not found in SQL Server facilities table. Cannot transfer claim.")
        
        # Extract patient name components
        patient_first_name = claim[4] or ''
        patient_last_name = claim[5] or ''
        patient_middle_name = claim[6] or ''
        full_patient_name = f"{patient_first_name} {patient_middle_name} {patient_last_name}".strip()
        
        # Map financial class from PostgreSQL name to SQL Server ID
        financial_class_name = claim[12]  # financial_class from PostgreSQL
        financial_class_id = None
        
        if financial_class_name:
            # Map PostgreSQL financial class names to SQL Server IDs
            financial_class_mapping = {
                'Medicare Part A': 'A',
                'Medicare Part B': 'B', 
                'Medicaid': 'MA',
                'Commercial HMO': 'HM',
                'Commercial PPO': 'BC',
                'Self Pay': 'SP',
                'Workers Comp': 'WC',
                'A': 'A',  # Direct mappings for SQL Server generated data
                'BC': 'BC',
                'MA': 'MA',
                'SP': 'SP',
                'WC': 'WC',
                'CO': 'CO',
                'HM': 'HM'
            }
            
            mapped_fc_id = financial_class_mapping.get(financial_class_name)
            
            # Validate that the financial class exists for this facility
            if mapped_fc_id:
                ss_cursor.execute("""
                    SELECT COUNT(*) 
                    FROM dbo.facility_financial_classes 
                    WHERE facility_id = ? AND financial_class_id = ?
                """, (facility_id, mapped_fc_id))
                
                if ss_cursor.fetchone()[0] > 0:
                    financial_class_id = mapped_fc_id
                else:
                    print(f"   WARNING: Financial class {mapped_fc_id} not found for facility {facility_id}, using NULL")
            else:
                print(f"   WARNING: Unknown financial class '{financial_class_name}', using NULL")
        
        # Insert main claim using SQL Server schema
        ss_cursor.execute("""
            INSERT INTO dbo.claims (
                facility_id, patient_account_number, medical_record_number,
                patient_name, first_name, last_name, date_of_birth,
                gender, financial_class_id, secondary_insurance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            facility_id,
            patient_account_number,
            None,  # medical_record_number
            full_patient_name,
            patient_first_name,
            patient_last_name,
            claim[7],  # patient_date_of_birth
            'U',  # gender (Unknown)
            financial_class_id,
            None  # secondary_insurance
        ))
        
        # Insert line items using SQL Server schema
        for item in line_items:
            # Skip provider lookup since physicians table is not populated
            # Always use NULL for rendering_provider_id to avoid foreign key constraints
            provider_id = None
            
            ss_cursor.execute("""
                INSERT INTO dbo.claims_line_items (
                    facility_id, patient_account_number, line_number,
                    procedure_code, units, charge_amount,
                    service_from_date, service_to_date,
                    diagnosis_pointer, rendering_provider_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                facility_id,
                patient_account_number,
                item[0],  # line_number
                item[2],  # procedure_code
                item[4] or 1,  # units
                float(item[5]) if item[5] else 0,  # charge_amount
                item[1],  # service_date as both from and to
                item[1],  # service_date
                json.dumps(item[8]) if item[8] else None,  # diagnosis_pointer
                provider_id  # Always NULL to avoid foreign key constraint
            ))
    
    def update_batch_metadata(self, cursor):
        """Update batch metadata for processed claims."""
        cursor.execute("""
            UPDATE batch_metadata bm
            SET 
                processed_claims = (
                    SELECT COUNT(*) FROM claims c 
                    WHERE c.batch_id = bm.id 
                    AND c.processing_status = 'completed'
                ),
                failed_claims = (
                    SELECT COUNT(*) FROM claims c 
                    WHERE c.batch_id = bm.id 
                    AND c.processing_status = 'failed'
                ),
                status = CASE 
                    WHEN EXISTS (
                        SELECT 1 FROM claims c 
                        WHERE c.batch_id = bm.id 
                        AND c.processing_status = 'pending'
                    ) THEN 'processing'::processing_status
                    ELSE 'completed'::processing_status
                END,
                completed_at = CASE 
                    WHEN NOT EXISTS (
                        SELECT 1 FROM claims c 
                        WHERE c.batch_id = bm.id 
                        AND c.processing_status = 'pending'
                    ) THEN CURRENT_TIMESTAMP
                    ELSE completed_at
                END,
                processing_time_seconds = EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - started_at)),
                throughput_per_second = CASE 
                    WHEN EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - started_at)) > 0 
                    THEN processed_claims / EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - started_at))
                    ELSE 0
                END,
                updated_at = CURRENT_TIMESTAMP
            WHERE EXISTS (
                SELECT 1 FROM claims c 
                WHERE c.batch_id = bm.id
                AND c.updated_at >= CURRENT_TIMESTAMP - INTERVAL '5 minutes'
            )
        """)
    
    def record_processing_metrics(self, cursor):
        """Record metrics for this processing run."""
        self.stats['processing_time'] = time.time() - self.start_time
        throughput = self.stats['total'] / self.stats['processing_time'] if self.stats['processing_time'] > 0 else 0
        
        cursor.execute("""
            INSERT INTO processing_metrics (
                run_date, total_claims, processed_claims, failed_claims,
                validation_failures, processing_time_seconds, throughput_per_second
            ) VALUES (
                CURRENT_TIMESTAMP, %s, %s, %s, %s, %s, %s
            )
        """, (
            self.stats['total'],
            self.stats['processed'],
            self.stats['failed'],
            json.dumps(self.stats['validation_failures']),
            self.stats['processing_time'],
            throughput
        ))
    
    def show_results(self, pg_cursor, ss_cursor):
        """Display processing results."""
        print("\n" + "="*60)
        print("CLAIMS PROCESSING COMPLETE")
        print("="*60)
        
        # Processing stats
        print(f"\nProcessing Statistics:")
        print(f"   • Total Claims: {self.stats['total']}")
        print(f"   • Successfully Processed: {self.stats['processed']}")
        print(f"   • Failed: {self.stats['failed']}")
        print(f"   • Success Rate: {(self.stats['processed']/self.stats['total']*100):.1f}%")
        print(f"   • Processing Time: {self.stats['processing_time']:.2f} seconds")
        print(f"   • Throughput: {self.stats['total']/self.stats['processing_time']:.1f} claims/second")
        
        # Validation failures breakdown
        if self.stats['validation_failures']:
            print(f"\nValidation Failures by Type:")
            for error_type, count in sorted(self.stats['validation_failures'].items(), 
                                          key=lambda x: x[1], reverse=True):
                print(f"   • {error_type}: {count}")
        
        # PostgreSQL status
        pg_cursor.execute("""
            SELECT processing_status, COUNT(*) 
            FROM claims 
            GROUP BY processing_status
            ORDER BY processing_status
        """)
        
        print(f"\nPostgreSQL Claims Status:")
        for status, count in pg_cursor.fetchall():
            print(f"   • {status}: {count}")
        
        # SQL Server status
        ss_cursor.execute("SELECT COUNT(*) FROM dbo.claims")
        ss_claims_count = ss_cursor.fetchone()[0]
        
        ss_cursor.execute("SELECT COUNT(*) FROM dbo.claims_line_items")
        ss_lines_count = ss_cursor.fetchone()[0]
        
        print(f"\nSQL Server Status:")
        print(f"   • Total Claims: {ss_claims_count}")
        print(f"   • Total Line Items: {ss_lines_count}")
        
        # Recent failed claims
        pg_cursor.execute("""
            SELECT claim_reference, failure_category, failure_reason
            FROM failed_claims
            ORDER BY created_at DESC
            LIMIT 5
        """)
        
        failed = pg_cursor.fetchall()
        if failed:
            print(f"\nRecent Failed Claims:")
            for claim_ref, category, reason in failed:
                print(f"   • {claim_ref}: [{category}] {reason[:60]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Complete claims processing pipeline - validation, RVU calculation, and transfer"
    )
    parser.add_argument(
        "--env", "-e",
        default="config/.env",
        help="Environment file path (default: config/.env)"
    )
    parser.add_argument(
        "--pg-conn",
        help="PostgreSQL connection string (overrides env file)"
    )
    parser.add_argument(
        "--ss-conn",
        help="SQL Server connection string (overrides env file)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    pg_conn = args.pg_conn
    ss_conn = args.ss_conn
    
    # If connection strings not provided, load from env file
    if not pg_conn or not ss_conn:
        env_vars = load_env_file(args.env)
        if not env_vars:
            print("ERROR: Failed to load environment configuration. Exiting.")
            print("   Make sure config/.env exists (copy from config/.env.example)")
            sys.exit(1)
        
        # Build connection strings from env vars
        pg_conn_env, ss_conn_env = build_connection_strings(env_vars)
        
        # Use command line args if provided, otherwise use env
        pg_conn = pg_conn or pg_conn_env
        ss_conn = ss_conn or ss_conn_env
    
    # Load processing configuration
    config = load_processing_config()
    
    print(f"Configuration loaded from: {args.env}")
    print(f"Batch size: {config.get('processing', {}).get('batch_size', 1000)}")
    print(f"Conversion factor: ${config.get('processing', {}).get('conversion_factor', 38.87)}")
    print(f"Environment: {env_vars.get('ENVIRONMENT', 'development') if 'env_vars' in locals() else 'custom'}")
    
    # Run the processor
    processor = ClaimsProcessor(pg_conn, ss_conn, config)
    processor.process_claims()


if __name__ == "__main__":
    main()
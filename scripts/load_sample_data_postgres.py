#!/usr/bin/env python3
"""
PostgreSQL-Only Sample Data Loader for Smart Pro Claims

This script loads comprehensive sample data into both PostgreSQL databases:
- smart_claims_staging: Claims processing workflow
- smart_pro_claims: Production data (migrated from SQL Server schema)

Includes:
- Facilities and organizational data
- Providers and RVU data
- 100,000 sample claims with realistic healthcare data
- Configuration and validation rules

Usage:
    python scripts/load_sample_data_postgres.py
    python scripts/load_sample_data_postgres.py --skip-claims
    python scripts/load_sample_data_postgres.py --staging-only
    python scripts/load_sample_data_postgres.py --production-only
"""

import argparse
import json
import random
import sys
import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any
from pathlib import Path

from faker import Faker
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import psycopg2

# Initialize Faker for generating realistic data
fake = Faker()
fake.seed_instance(42)  # For reproducible data
random.seed(42)

# Healthcare-specific data sets
DIAGNOSIS_CODES = [
    ('Z00.00', 'Encounter for general adult medical examination without abnormal findings'),
    ('I10', 'Essential hypertension'),
    ('E11.9', 'Type 2 diabetes mellitus without complications'),
    ('Z12.31', 'Encounter for screening mammography for malignant neoplasm of breast'),
    ('M79.1', 'Myalgia'),
    ('R06.02', 'Shortness of breath'),
    ('K21.9', 'Gastro-esophageal reflux disease without esophagitis'),
    ('F32.9', 'Major depressive disorder, single episode, unspecified'),
    ('M25.511', 'Pain in right shoulder'),
    ('R51', 'Headache'),
    ('J06.9', 'Acute upper respiratory infection, unspecified'),
    ('N39.0', 'Urinary tract infection, site not specified'),
    ('M54.5', 'Low back pain'),
    ('K59.00', 'Constipation, unspecified'),
    ('R50.9', 'Fever, unspecified'),
    ('J44.1', 'Chronic obstructive pulmonary disease with acute exacerbation'),
    ('I25.10', 'Atherosclerotic heart disease of native coronary artery without angina pectoris'),
    ('E78.5', 'Hyperlipidemia, unspecified'),
    ('F41.9', 'Anxiety disorder, unspecified'),
    ('M79.3', 'Panniculitis, unspecified'),
    ('K30', 'Functional dyspepsia'),
    ('R10.9', 'Unspecified abdominal pain'),
    ('J02.9', 'Acute pharyngitis, unspecified'),
    ('L70.9', 'Acne, unspecified'),
    ('H52.4', 'Presbyopia'),
    ('M99.23', 'Subluxation stenosis of neural canal of lumbar region'),
    ('R11.10', 'Vomiting, unspecified'),
    ('K92.2', 'Gastrointestinal bleeding, unspecified'),
    ('R42', 'Dizziness and giddiness'),
    ('I63.9', 'Cerebral infarction, unspecified')
]

CPT_CODES = [
    ('99213', 'Office visit established patient level 3', 1.3, 1.05, 0.07, 2.42),
    ('99214', 'Office visit established patient level 4', 2.0, 1.48, 0.10, 3.58),
    ('99215', 'Office visit established patient level 5', 2.8, 1.98, 0.14, 4.92),
    ('99203', 'Office visit new patient level 3', 1.6, 1.30, 0.08, 2.98),
    ('99204', 'Office visit new patient level 4', 2.6, 1.84, 0.12, 4.56),
    ('99205', 'Office visit new patient level 5', 3.5, 2.43, 0.17, 6.10),
    ('99212', 'Office visit established patient level 2', 0.7, 0.72, 0.04, 1.46),
    ('99211', 'Office visit established patient level 1', 0.18, 0.48, 0.01, 0.67),
    ('99202', 'Office visit new patient level 2', 0.93, 0.94, 0.05, 1.92),
    ('99201', 'Office visit new patient level 1', 0.48, 0.67, 0.02, 1.17),
    ('93000', 'Electrocardiogram routine ECG with interpretation', 0.17, 0.15, 0.01, 0.33),
    ('36415', 'Collection of venous blood by venipuncture', 0.16, 0.05, 0.00, 0.21),
    ('85025', 'Blood count; complete (CBC), automated', 0.0, 0.12, 0.00, 0.12),
    ('80053', 'Comprehensive metabolic panel', 0.0, 0.25, 0.00, 0.25),
    ('85027', 'Blood count; complete (CBC), automated with differential WBC', 0.0, 0.14, 0.00, 0.14),
    ('81001', 'Urinalysis; automated, with microscopy', 0.0, 0.09, 0.00, 0.09),
    ('71020', 'Radiologic examination, chest, 2 views', 0.22, 0.32, 0.02, 0.56),
    ('73060', 'Radiologic examination; knee, 1 or 2 views', 0.22, 0.26, 0.02, 0.50),
    ('73030', 'Radiologic examination, shoulder; complete', 0.26, 0.32, 0.02, 0.60),
    ('74177', 'CT abdomen and pelvis with contrast', 1.42, 4.73, 0.31, 6.46),
    ('72148', 'MRI lumbar spine without contrast', 1.0, 7.58, 0.50, 9.08),
    ('76700', 'Abdominal ultrasound complete', 0.75, 1.63, 0.11, 2.49),
    ('45378', 'Colonoscopy flexible; diagnostic', 4.43, 3.32, 0.22, 7.97)
]

FINANCIAL_CLASSES = [
    ('A', 'Medicare Part A', '1'),
    ('B', 'Medicare Part B', '1'),
    ('MA', 'Medicaid', '2'),
    ('BC', 'BlueCross', '3'),
    ('HM', 'HMO', '6'),
    ('SP', 'Self Pay', '5'),
    ('WC', 'Workers Comp', '9'),
    ('CO', 'Commercial', '8')
]

FACILITIES = [
    ('FAC001', 'General Hospital Medical Center', 350, 'New York', 'NY'),
    ('FAC002', 'Regional Medical Center', 275, 'Los Angeles', 'CA'),
    ('FAC003', 'Community Health Hospital', 150, 'Chicago', 'IL'),
    ('FAC004', 'University Medical Center', 400, 'Houston', 'TX'),
    ('FAC005', 'Metropolitan Hospital', 200, 'Philadelphia', 'PA')
]

def load_env_file():
    """Load environment variables from .env file."""
    env_vars = {}
    env_path = 'config/.env'
    
    # First try .env, then .env.example
    if not os.path.exists(env_path):
        env_path = 'config/.env.example'
    
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    
    return env_vars

def get_database_connections():
    """Get database connections from environment variables."""
    env_vars = load_env_file()
    
    postgres_host = env_vars.get('POSTGRES_HOST', 'localhost')
    postgres_port = env_vars.get('POSTGRES_PORT', '5432')
    postgres_user = env_vars.get('POSTGRES_USER', 'postgres')
    postgres_password = env_vars.get('POSTGRES_PASSWORD', '')
    
    staging_url = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/smart_claims_staging"
    production_url = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/smart_pro_claims"
    
    return staging_url, production_url

class PostgreSQLDataLoader:
    def __init__(self, staging_url: str, production_url: str, load_staging=True, load_production=True):
        self.staging_url = staging_url
        self.production_url = production_url
        self.load_staging = load_staging
        self.load_production = load_production
        
        # Create engines
        if self.load_staging:
            self.staging_engine = create_engine(staging_url)
            print(f"Connected to staging database: smart_claims_staging")
        
        if self.load_production:
            self.production_engine = create_engine(production_url)
            print(f"Connected to production database: smart_pro_claims")

    def load_core_reference_data(self):
        """Load core reference data into production database."""
        if not self.load_production:
            return
            
        print("Loading core reference data into smart_pro_claims...")
        
        with self.production_engine.connect() as conn:
            # Load organizational hierarchy
            print("  - Loading facility organization...")
            conn.execute(text("""
                INSERT INTO facility_organization (org_name, active) VALUES 
                ('Smart Pro Healthcare System', TRUE),
                ('Regional Health Network', TRUE),
                ('Community Care Alliance', TRUE)
                ON CONFLICT DO NOTHING
            """))
            
            # Load facility regions
            print("  - Loading facility regions...")
            conn.execute(text("""
                INSERT INTO facility_region (region_name, org_id, active) VALUES 
                ('Northeast Region', 1, TRUE),
                ('West Coast Region', 1, TRUE),
                ('Midwest Region', 1, TRUE),
                ('South Region', 1, TRUE),
                ('Central Region', 2, TRUE)
                ON CONFLICT DO NOTHING
            """))
            
            # Load facilities
            print("  - Loading facilities...")
            for i, (facility_id, name, beds, city, state) in enumerate(FACILITIES):
                org_id = (i % 2) + 1  # Distribute between orgs
                region_id = (i % 5) + 1  # Distribute among regions
                
                conn.execute(text("""
                    INSERT INTO facilities (facility_id, facility_name, beds, city, state, org_id, region_id, active) VALUES 
                    (:facility_id, :name, :beds, :city, :state, :org_id, :region_id, TRUE)
                    ON CONFLICT (facility_id) DO NOTHING
                """), {
                    'facility_id': facility_id,
                    'name': name,
                    'beds': beds,
                    'city': city,
                    'state': state,
                    'org_id': org_id,
                    'region_id': region_id
                })
            
            # Load financial classes for each facility
            print("  - Loading facility financial classes...")
            for facility_id, _, _, _, _ in FACILITIES:
                for financial_class_id, financial_class_name, payer_id in FINANCIAL_CLASSES:
                    reimbursement_rate = random.uniform(0.75, 0.95)  # 75-95% reimbursement
                    
                    conn.execute(text("""
                        INSERT INTO facility_financial_classes 
                        (facility_id, financial_class_id, financial_class_name, payer_id, 
                         reimbursement_rate, active, effective_date) VALUES 
                        (:facility_id, :financial_class_id, :financial_class_name, :payer_id,
                         :reimbursement_rate, TRUE, :effective_date)
                        ON CONFLICT (facility_id, financial_class_id) DO NOTHING
                    """), {
                        'facility_id': facility_id,
                        'financial_class_id': financial_class_id,
                        'financial_class_name': financial_class_name,
                        'payer_id': int(payer_id),
                        'reimbursement_rate': reimbursement_rate,
                        'effective_date': '2024-01-01'
                    })
            
            # Load place of service data
            print("  - Loading place of service data...")
            pos_data = [
                ('11', 'Office'),
                ('21', 'Inpatient Hospital'),
                ('22', 'Outpatient Hospital'),
                ('23', 'Emergency Room'),
                ('81', 'Independent Laboratory')
            ]
            
            for facility_id, _, _, _, _ in FACILITIES:
                for pos_code, pos_name in pos_data:
                    conn.execute(text("""
                        INSERT INTO facility_place_of_service 
                        (facility_id, place_of_service, place_of_service_name, active) VALUES 
                        (:facility_id, :pos_code, :pos_name, TRUE)
                        ON CONFLICT (facility_id, place_of_service) DO NOTHING
                    """), {
                        'facility_id': facility_id,
                        'pos_code': pos_code,
                        'pos_name': pos_name
                    })
            
            # Load physicians
            print("  - Loading physicians...")
            for i in range(50):  # Create 50 physicians
                provider_id = f"PROV{i+1:03d}"
                npi = f"{1234567890 + i}"
                
                conn.execute(text("""
                    INSERT INTO physicians 
                    (rendering_provider_id, first_name, last_name, npi, specialty_code, active) VALUES 
                    (:provider_id, :first_name, :last_name, :npi, :specialty_code, TRUE)
                    ON CONFLICT (rendering_provider_id) DO NOTHING
                """), {
                    'provider_id': provider_id,
                    'first_name': fake.first_name(),
                    'last_name': fake.last_name(),
                    'npi': npi,
                    'specialty_code': random.choice(['01', '02', '03', '04', '05'])
                })
            
            conn.commit()
            
    def load_rvu_data(self):
        """Load RVU data into both databases."""
        print("Loading RVU data...")
        
        # Load into staging database
        if self.load_staging:
            with self.staging_engine.connect() as conn:
                print("  - Loading RVU data into smart_claims_staging...")
                for code, description, work_rvu, pe_rvu, mp_rvu, total_rvu in CPT_CODES:
                    conn.execute(text("""
                        INSERT INTO rvu_data 
                        (procedure_code, description, work_rvu, practice_expense_rvu, 
                         malpractice_rvu, total_rvu, status) VALUES 
                        (:code, :description, :work_rvu, :pe_rvu, :mp_rvu, :total_rvu, 'ACTIVE')
                        ON CONFLICT (procedure_code) DO NOTHING
                    """), {
                        'code': code,
                        'description': description,
                        'work_rvu': work_rvu,
                        'pe_rvu': pe_rvu,
                        'mp_rvu': mp_rvu,
                        'total_rvu': total_rvu
                    })
                conn.commit()
        
        # Load into production database
        if self.load_production:
            with self.production_engine.connect() as conn:
                print("  - Loading RVU data into smart_pro_claims...")
                for code, description, work_rvu, pe_rvu, mp_rvu, total_rvu in CPT_CODES:
                    conn.execute(text("""
                        INSERT INTO rvu_data 
                        (procedure_code, description, work_rvu, practice_expense_rvu, 
                         malpractice_rvu, total_rvu, status, effective_date) VALUES 
                        (:code, :description, :work_rvu, :pe_rvu, :mp_rvu, :total_rvu, 'ACTIVE', :effective_date)
                        ON CONFLICT (procedure_code) DO NOTHING
                    """), {
                        'code': code,
                        'description': description,
                        'work_rvu': work_rvu,
                        'pe_rvu': pe_rvu,
                        'mp_rvu': mp_rvu,
                        'total_rvu': total_rvu,
                        'effective_date': '2024-01-01'
                    })
                conn.commit()

    def load_validation_rules(self):
        """Load validation rules into staging database."""
        if not self.load_staging:
            return
            
        print("Loading validation rules into smart_claims_staging...")
        
        validation_rules = [
            {
                'rule_name': 'patient_demographics_required',
                'rule_type': 'demographics',
                'rule_condition': {'required_fields': ['patient_first_name', 'patient_last_name', 'patient_date_of_birth']},
                'error_message': 'Patient demographics are required',
                'severity': 'error'
            },
            {
                'rule_name': 'service_date_range_validation',
                'rule_type': 'date_validation',
                'rule_condition': {'max_days_back': 365, 'max_days_forward': 30},
                'error_message': 'Service date must be within valid range',
                'severity': 'error'
            },
            {
                'rule_name': 'procedure_code_format',
                'rule_type': 'code_validation',
                'rule_condition': {'pattern': '^[0-9]{5}$', 'field': 'procedure_code'},
                'error_message': 'Procedure code must be 5 digits',
                'severity': 'warning'
            }
        ]
        
        with self.staging_engine.connect() as conn:
            for rule in validation_rules:
                conn.execute(text("""
                    INSERT INTO validation_rules 
                    (rule_name, rule_type, rule_condition, error_message, severity, is_active) VALUES 
                    (:rule_name, :rule_type, :rule_condition, :error_message, :severity, TRUE)
                    ON CONFLICT (rule_name) DO NOTHING
                """), {
                    'rule_name': rule['rule_name'],
                    'rule_type': rule['rule_type'],
                    'rule_condition': json.dumps(rule['rule_condition']),
                    'error_message': rule['error_message'],
                    'severity': rule['severity']
                })
            conn.commit()

    def generate_claims(self, num_claims: int = 100000):
        """Generate sample claims data."""
        if not self.load_staging:
            return
            
        print(f"Generating {num_claims:,} sample claims into smart_claims_staging...")
        
        # Create a batch
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with self.staging_engine.connect() as conn:
            # Create batch metadata
            conn.execute(text("""
                INSERT INTO batch_metadata 
                (batch_id, facility_id, source_system, total_claims, submitted_by, priority) VALUES 
                (:batch_id, :facility_id, :source_system, :total_claims, :submitted_by, 'medium')
            """), {
                'batch_id': batch_id,
                'facility_id': FACILITIES[0][0],  # Use first facility
                'source_system': 'Sample Data Generator',
                'total_claims': num_claims,
                'submitted_by': 'system'
            })
            
            # Get the batch ID
            result = conn.execute(text("SELECT id FROM batch_metadata WHERE batch_id = :batch_id"), 
                                {'batch_id': batch_id})
            batch_pk = result.fetchone()[0]
            
            # Generate claims in batches
            batch_size = 1000
            
            for batch_start in range(0, num_claims, batch_size):
                batch_end = min(batch_start + batch_size, num_claims)
                print(f"  - Processing claims {batch_start + 1:,} to {batch_end:,}")
                
                claims_data = []
                line_items_data = []
                
                for i in range(batch_start, batch_end):
                    claim_id = f"CLM{i+1:06d}"
                    facility_id = random.choice(FACILITIES)[0]
                    patient_account = f"PAT{i+1:08d}"
                    
                    # Generate realistic dates
                    service_date = fake.date_between(start_date='-90d', end_date='today')
                    admission_date = service_date - timedelta(days=random.randint(0, 5))
                    discharge_date = service_date + timedelta(days=random.randint(0, 7))
                    
                    # Select financial class and insurance
                    financial_class_id, financial_class_name, payer_id = random.choice(FINANCIAL_CLASSES)
                    
                    # Generate patient demographics
                    first_name = fake.first_name()
                    last_name = fake.last_name()
                    dob = fake.date_of_birth(minimum_age=18, maximum_age=85)
                    
                    # Select primary diagnosis
                    primary_dx_code, primary_dx_desc = random.choice(DIAGNOSIS_CODES)
                    
                    # Generate diagnosis codes list (1-5 additional diagnoses)
                    num_additional_dx = random.randint(0, 4)
                    additional_dx = random.sample(DIAGNOSIS_CODES, min(num_additional_dx, len(DIAGNOSIS_CODES)))
                    diagnosis_codes = [{'code': dx[0], 'description': dx[1]} for dx in additional_dx]
                    
                    # Generate provider information
                    provider_npi = f"{1234567890 + random.randint(0, 49)}"
                    provider_name = f"Dr. {fake.first_name()} {fake.last_name()}"
                    
                    # Calculate total charges (will be sum of line items)
                    num_line_items = random.randint(1, 5)
                    total_charges = 0
                    
                    # Generate line items first to calculate total
                    claim_line_items = []
                    for line_num in range(1, num_line_items + 1):
                        procedure_code, proc_desc, work_rvu, pe_rvu, mp_rvu, total_rvu = random.choice(CPT_CODES)
                        units = random.randint(1, 3)
                        
                        # Calculate charge based on RVU and a conversion factor
                        conversion_factor = random.uniform(35.0, 45.0)  # Typical Medicare conversion factor range
                        charge_amount = round(total_rvu * conversion_factor * units, 2)
                        total_charges += charge_amount
                        
                        # Generate diagnosis pointers (1-4 pointers)
                        max_pointers = min(4, len(diagnosis_codes) + 1)  # +1 for primary
                        num_pointers = random.randint(1, max_pointers)
                        diagnosis_pointers = list(range(1, num_pointers + 1))
                        
                        claim_line_items.append({
                            'line_number': line_num,
                            'service_date': service_date,
                            'procedure_code': procedure_code,
                            'procedure_description': proc_desc,
                            'units': units,
                            'charge_amount': charge_amount,
                            'rendering_provider_npi': provider_npi,
                            'rendering_provider_name': provider_name,
                            'diagnosis_pointers': diagnosis_pointers,
                            'rvu_work': work_rvu,
                            'rvu_practice_expense': pe_rvu,
                            'rvu_malpractice': mp_rvu,
                            'rvu_total': total_rvu
                        })
                    
                    # Calculate expected reimbursement (75-90% of charges)
                    reimbursement_rate = random.uniform(0.75, 0.90)
                    expected_reimbursement = round(total_charges * reimbursement_rate, 2)
                    
                    # Create claim record
                    claim_data = {
                        'claim_id': claim_id,
                        'facility_id': facility_id,
                        'patient_account_number': patient_account,
                        'medical_record_number': f"MRN{i+1:08d}",
                        'patient_first_name': first_name,
                        'patient_last_name': last_name,
                        'patient_middle_name': fake.first_name() if random.random() < 0.3 else None,
                        'patient_date_of_birth': dob,
                        'admission_date': admission_date,
                        'discharge_date': discharge_date,
                        'service_from_date': service_date,
                        'service_to_date': service_date,
                        'financial_class': financial_class_name,
                        'total_charges': total_charges,
                        'expected_reimbursement': expected_reimbursement,
                        'insurance_type': financial_class_name,
                        'insurance_plan_id': f"PLAN{random.randint(1000, 9999)}",
                        'subscriber_id': f"SUB{random.randint(100000, 999999)}",
                        'billing_provider_npi': provider_npi,
                        'billing_provider_name': provider_name,
                        'attending_provider_npi': provider_npi,
                        'attending_provider_name': provider_name,
                        'primary_diagnosis_code': primary_dx_code,
                        'diagnosis_codes': json.dumps(diagnosis_codes),
                        'batch_id': batch_pk,
                        'priority': random.choice(['low', 'medium', 'high'])
                    }
                    
                    claims_data.append(claim_data)
                    
                    # Add line items with claim reference
                    for item in claim_line_items:
                        item_data = item.copy()
                        item_data['claim_reference'] = len(claims_data)  # Will be updated after insert
                        line_items_data.append(item_data)
                
                # Insert claims batch
                for claim in claims_data:
                    conn.execute(text("""
                        INSERT INTO claims (
                            claim_id, facility_id, patient_account_number, medical_record_number,
                            patient_first_name, patient_last_name, patient_middle_name, patient_date_of_birth,
                            admission_date, discharge_date, service_from_date, service_to_date,
                            financial_class, total_charges, expected_reimbursement,
                            insurance_type, insurance_plan_id, subscriber_id,
                            billing_provider_npi, billing_provider_name,
                            attending_provider_npi, attending_provider_name,
                            primary_diagnosis_code, diagnosis_codes, batch_id, priority
                        ) VALUES (
                            :claim_id, :facility_id, :patient_account_number, :medical_record_number,
                            :patient_first_name, :patient_last_name, :patient_middle_name, :patient_date_of_birth,
                            :admission_date, :discharge_date, :service_from_date, :service_to_date,
                            :financial_class, :total_charges, :expected_reimbursement,
                            :insurance_type, :insurance_plan_id, :subscriber_id,
                            :billing_provider_npi, :billing_provider_name,
                            :attending_provider_npi, :attending_provider_name,
                            :primary_diagnosis_code, :diagnosis_codes, :batch_id, CAST(:priority AS claim_priority)
                        )
                    """), claim)
                
                # Get claim IDs for line items
                claim_ids = {}
                for claim in claims_data:
                    result = conn.execute(text("""
                        SELECT id FROM claims WHERE claim_id = :claim_id AND facility_id = :facility_id
                    """), {
                        'claim_id': claim['claim_id'],
                        'facility_id': claim['facility_id']
                    })
                    claim_ids[claim['claim_id']] = result.fetchone()[0]
                
                # Insert line items
                claim_idx = 0
                for item in line_items_data:
                    if claim_idx < len(claims_data):
                        claim_id = claims_data[claim_idx]['claim_id']
                        item['claim_id'] = claim_ids[claim_id]
                        
                        conn.execute(text("""
                            INSERT INTO claim_line_items (
                                claim_id, line_number, service_date, procedure_code, procedure_description,
                                units, charge_amount, rendering_provider_npi, rendering_provider_name,
                                diagnosis_pointers, rvu_work, rvu_practice_expense, rvu_malpractice, rvu_total
                            ) VALUES (
                                :claim_id, :line_number, :service_date, :procedure_code, :procedure_description,
                                :units, :charge_amount, :rendering_provider_npi, :rendering_provider_name,
                                :diagnosis_pointers, :rvu_work, :rvu_practice_expense, :rvu_malpractice, :rvu_total
                            )
                        """), {
                            'claim_id': item['claim_id'],
                            'line_number': item['line_number'],
                            'service_date': item['service_date'],
                            'procedure_code': item['procedure_code'],
                            'procedure_description': item['procedure_description'],
                            'units': item['units'],
                            'charge_amount': item['charge_amount'],
                            'rendering_provider_npi': item['rendering_provider_npi'],
                            'rendering_provider_name': item['rendering_provider_name'],
                            'diagnosis_pointers': json.dumps(item['diagnosis_pointers']),
                            'rvu_work': item['rvu_work'],
                            'rvu_practice_expense': item['rvu_practice_expense'],
                            'rvu_malpractice': item['rvu_malpractice'],
                            'rvu_total': item['rvu_total']
                        })
                        
                        if item['line_number'] == 1:  # Move to next claim after first line item
                            claim_idx += 1
                
                conn.commit()
            
            # Update batch metadata
            conn.execute(text("""
                UPDATE batch_metadata 
                SET status = 'pending', total_amount = (
                    SELECT SUM(total_charges) FROM claims WHERE batch_id = :batch_id
                )
                WHERE id = :batch_id
            """), {'batch_id': batch_pk})
            
            conn.commit()
            print(f"Successfully loaded {num_claims:,} claims into smart_claims_staging")

    def show_summary(self):
        """Show loading summary."""
        print("\n" + "="*60)
        print("SAMPLE DATA LOADING SUMMARY")
        print("="*60)
        
        if self.load_staging:
            with self.staging_engine.connect() as conn:
                # Claims summary
                result = conn.execute(text("SELECT COUNT(*) FROM claims"))
                claims_count = result.fetchone()[0]
                
                result = conn.execute(text("SELECT COUNT(*) FROM claim_line_items"))
                line_items_count = result.fetchone()[0]
                
                result = conn.execute(text("SELECT COUNT(*) FROM rvu_data"))
                rvu_count = result.fetchone()[0]
                
                result = conn.execute(text("SELECT COUNT(*) FROM validation_rules"))
                rules_count = result.fetchone()[0]
                
                print(f"STAGING DATABASE (smart_claims_staging):")
                print(f"  - Claims: {claims_count:,}")
                print(f"  - Line Items: {line_items_count:,}")
                print(f"  - RVU Codes: {rvu_count:,}")
                print(f"  - Validation Rules: {rules_count:,}")
        
        if self.load_production:
            with self.production_engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM facilities"))
                facilities_count = result.fetchone()[0]
                
                result = conn.execute(text("SELECT COUNT(*) FROM physicians"))
                physicians_count = result.fetchone()[0]
                
                result = conn.execute(text("SELECT COUNT(*) FROM core_standard_payers"))
                payers_count = result.fetchone()[0]
                
                result = conn.execute(text("SELECT COUNT(*) FROM rvu_data"))
                rvu_count = result.fetchone()[0]
                
                print(f"\nPRODUCTION DATABASE (smart_pro_claims):")
                print(f"  - Facilities: {facilities_count:,}")
                print(f"  - Physicians: {physicians_count:,}")
                print(f"  - Payers: {payers_count:,}")
                print(f"  - RVU Codes: {rvu_count:,}")
        
        print("\n" + "="*60)
        print("SUCCESS: Sample data loading completed successfully!")
        print("SUCCESS: System ready for high-performance claims processing")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description="Load sample data into PostgreSQL databases",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--skip-claims', 
        action='store_true', 
        help='Skip loading claims data (configuration only)'
    )
    parser.add_argument(
        '--staging-only', 
        action='store_true', 
        help='Load data only into smart_claims_staging database'
    )
    parser.add_argument(
        '--production-only', 
        action='store_true', 
        help='Load data only into smart_pro_claims database'
    )
    parser.add_argument(
        '--claims-count', 
        type=int, 
        default=100000,
        help='Number of claims to generate (default: 100000)'
    )
    
    args = parser.parse_args()
    
    if args.staging_only and args.production_only:
        parser.error("Cannot specify both --staging-only and --production-only")
    
    # Get database connections
    try:
        staging_url, production_url = get_database_connections()
    except Exception as e:
        print(f"Error getting database connections: {e}")
        sys.exit(1)
    
    # Determine which databases to load
    load_staging = not args.production_only
    load_production = not args.staging_only
    
    print("PostgreSQL Sample Data Loader")
    print("="*40)
    print(f"Staging Database: {'YES' if load_staging else 'NO'}")
    print(f"Production Database: {'YES' if load_production else 'NO'}")
    print(f"Claims Count: {args.claims_count:,}" if not args.skip_claims and load_staging else "Claims: SKIPPED")
    print("="*40)
    
    # Initialize loader
    loader = PostgreSQLDataLoader(staging_url, production_url, load_staging, load_production)
    
    try:
        # Load reference data into production database
        if load_production:
            loader.load_core_reference_data()
        
        # Load RVU data into both databases
        loader.load_rvu_data()
        
        # Load validation rules into staging database
        if load_staging:
            loader.load_validation_rules()
        
        # Load claims data into staging database
        if not args.skip_claims and load_staging:
            loader.generate_claims(args.claims_count)
        
        # Show summary
        loader.show_summary()
        
    except Exception as e:
        print(f"Error loading sample data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
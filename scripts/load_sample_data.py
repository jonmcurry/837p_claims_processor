#!/usr/bin/env python3
"""
Sample Data Loader for PostgreSQL Claims Staging Database

This script loads comprehensive sample data including:
- Facilities
- Providers
- RVU data
- Validation rules
- 100,000 sample claims with realistic healthcare data

Usage:
    python scripts/load_sample_data.py --connection-string "postgresql://claims_user:password@localhost:5432/claims_staging"
"""

import argparse
import random
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any

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
    # Common ICD-10 codes with descriptions
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
    # Common CPT codes with RVU values
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
    ('45378', 'Colonoscopy flexible; diagnostic', 4.43, 3.32, 0.22, 7.97),
    ('43239', 'Upper endoscopy; with biopsy', 3.00, 2.50, 0.16, 5.66),
    ('12001', 'Simple repair of superficial wounds; 2.5 cm or less', 1.09, 1.39, 0.07, 2.55),
    ('11042', 'Debridement, subcutaneous tissue; first 20 sq cm or less', 1.33, 1.61, 0.08, 3.02),
    ('29125', 'Application of short arm splint; static', 0.57, 1.02, 0.03, 1.62),
    ('90471', 'Immunization administration; 1 vaccine', 0.17, 0.28, 0.01, 0.46),
    ('90715', 'Tetanus, diphtheria toxoids and pertussis vaccine', 0.0, 0.32, 0.00, 0.32),
    ('J3420', 'Injection, vitamin B-12 cyanocobalamin', 0.10, 0.23, 0.01, 0.34)
]

PLACE_OF_SERVICE_CODES = [
    ('11', 'Office', 1),
    ('12', 'Home', 1),
    ('21', 'Inpatient Hospital', 1),
    ('22', 'Outpatient Hospital', 1),
    ('23', 'Emergency Room - Hospital', 1),
    ('24', 'Ambulatory Surgical Center', 1),
    ('25', 'Birthing Center', 1),
    ('26', 'Military Treatment Facility', 1),
    ('31', 'Skilled Nursing Facility', 1),
    ('32', 'Nursing Facility', 1),
    ('33', 'Custodial Care Facility', 1),
    ('34', 'Hospice', 1),
    ('41', 'Ambulance - Land', 1),
    ('42', 'Ambulance - Air or Water', 1),
    ('49', 'Independent Clinic', 1),
    ('50', 'Federally Qualified Health Center', 1),
    ('51', 'Inpatient Psychiatric Facility', 1),
    ('52', 'Psychiatric Facility-Partial Hospitalization', 1),
    ('53', 'Community Mental Health Center', 1),
    ('54', 'Intermediate Care Facility/Mentally Retarded', 1),
    ('55', 'Residential Substance Abuse Treatment Facility', 1),
    ('56', 'Psychiatric Residential Treatment Center', 1),
    ('57', 'Non-residential Substance Abuse Treatment Facility', 1),
    ('60', 'Mass Immunization Center', 1),
    ('61', 'Comprehensive Inpatient Rehabilitation Facility', 1),
    ('62', 'Comprehensive Outpatient Rehabilitation Facility', 1),
    ('65', 'End-Stage Renal Disease Treatment Facility', 1),
    ('71', 'Public Health Clinic', 1),
    ('72', 'Rural Health Clinic', 1),
    ('81', 'Independent Laboratory', 1),
    ('99', 'Other Place of Service', 1)
]

DEPARTMENTS = [
    'Emergency Medicine',
    'Internal Medicine',
    'Family Medicine',
    'Cardiology',
    'Orthopedics',
    'Neurology',
    'Psychiatry',
    'Radiology',
    'Laboratory',
    'Surgery',
    'Obstetrics and Gynecology',
    'Pediatrics',
    'Dermatology',
    'Ophthalmology',
    'Urology',
    'Gastroenterology',
    'Endocrinology',
    'Pulmonology',
    'Nephrology',
    'Oncology',
    'Anesthesiology',
    'Pathology',
    'Physical Therapy',
    'Occupational Therapy',
    'Pharmacy'
]

PHYSICIAN_SPECIALTIES = [
    ('01', 'General Practice'),
    ('02', 'General Surgery'),
    ('03', 'Allergy/Immunology'),
    ('04', 'Otolaryngology'),
    ('05', 'Anesthesiology'),
    ('06', 'Cardiology'),
    ('07', 'Dermatology'),
    ('08', 'Family Medicine'),
    ('09', 'Interventional Pain Management'),
    ('10', 'Gastroenterology'),
    ('11', 'Internal Medicine'),
    ('12', 'Osteopathic Medicine'),
    ('13', 'Neurology'),
    ('14', 'Neurosurgery'),
    ('16', 'Obstetrics/Gynecology'),
    ('18', 'Ophthalmology'),
    ('19', 'Oral Surgery'),
    ('20', 'Orthopedic Surgery'),
    ('22', 'Pathology'),
    ('23', 'Psychiatry'),
    ('24', 'Pulmonary Disease'),
    ('25', 'Radiology'),
    ('26', 'Physical Medicine and Rehabilitation'),
    ('27', 'Plastic and Reconstructive Surgery'),
    ('28', 'Colorectal Surgery'),
    ('29', 'Pulmonary Disease'),
    ('30', 'Diagnostic Radiology'),
    ('33', 'Thoracic Surgery'),
    ('34', 'Urology'),
    ('36', 'Nuclear Medicine'),
    ('37', 'Pediatric Medicine'),
    ('38', 'Geriatric Medicine'),
    ('39', 'Nephrology'),
    ('40', 'Hand Surgery'),
    ('41', 'Optometry'),
    ('44', 'Infectious Disease'),
    ('46', 'Pulmonary Disease'),
    ('48', 'Geriatric Psychiatry'),
    ('50', 'Nurse Practitioner'),
    ('89', 'Certified Clinical Nurse Specialist'),
    ('97', 'Physician Assistant')
]

def create_database_engine(connection_string: str):
    """Create SQLAlchemy engine for PostgreSQL."""
    try:
        engine = create_engine(connection_string, echo=False)
        return engine
    except Exception as e:
        print(f"Failed to create database engine: {e}")
        sys.exit(1)

def load_facilities(session):
    """Load sample facilities."""
    print("Loading facilities...")
    
    facilities = [
        ('FAC001', 'Springfield General Hospital', '1234567890', '11-1234567', '123 Main St', 'Springfield', 'IL', '62701'),
        ('FAC002', 'Metropolis Regional Medical Center', '2345678901', '22-2345678', '456 Oak Ave', 'Metropolis', 'IL', '62960'),
        ('FAC003', 'Chicago Downtown Clinic', '3456789012', '33-3456789', '789 State St', 'Chicago', 'IL', '60601'),
        ('FAC004', 'Naperville West Campus', '4567890123', '44-4567890', '321 West St', 'Naperville', 'IL', '60540'),
        ('FAC005', 'Evanston North Specialty Center', '5678901234', '55-5678901', '654 North Ave', 'Evanston', 'IL', '60201')
    ]
    
    for facility_id, name, npi, tax_id, addr1, city, state, zip_code in facilities:
        session.execute(text("""
            INSERT INTO facilities 
            (facility_id, facility_name, npi, tax_id, address_line_1, city, state, zip_code, active)
            VALUES (:facility_id, :name, :npi, :tax_id, :addr1, :city, :state, :zip_code, true)
            ON CONFLICT (facility_id) DO UPDATE SET
                facility_name = EXCLUDED.facility_name,
                npi = EXCLUDED.npi,
                tax_id = EXCLUDED.tax_id,
                address_line_1 = EXCLUDED.address_line_1,
                city = EXCLUDED.city,
                state = EXCLUDED.state,
                zip_code = EXCLUDED.zip_code,
                updated_at = CURRENT_TIMESTAMP
        """), {
            "facility_id": facility_id,
            "name": name,
            "npi": npi,
            "tax_id": tax_id,
            "addr1": addr1,
            "city": city,
            "state": state,
            "zip_code": zip_code
        })
    
    session.commit()
    print(">> Facilities loaded successfully")
    return ['FAC001', 'FAC002', 'FAC003', 'FAC004', 'FAC005']

def load_providers(session):
    """Load sample providers."""
    print("Loading providers...")
    
    provider_ids = []
    for i in range(200):  # 200 providers
        provider_id = f"PRV{i+1:06d}"
        first_name = fake.first_name()
        last_name = fake.last_name()
        npi = f"{fake.random_number(digits=10)}"
        specialty_code, specialty_name = random.choice(PHYSICIAN_SPECIALTIES)
        
        session.execute(text("""
            INSERT INTO providers 
            (provider_id, first_name, last_name, npi, specialty_code, specialty_name, active)
            VALUES (:provider_id, :first_name, :last_name, :npi, :specialty_code, :specialty_name, true)
            ON CONFLICT (provider_id) DO UPDATE SET
                first_name = EXCLUDED.first_name,
                last_name = EXCLUDED.last_name,
                npi = EXCLUDED.npi,
                specialty_code = EXCLUDED.specialty_code,
                specialty_name = EXCLUDED.specialty_name,
                updated_at = CURRENT_TIMESTAMP
        """), {
            "provider_id": provider_id,
            "first_name": first_name,
            "last_name": last_name,
            "npi": npi,
            "specialty_code": specialty_code,
            "specialty_name": specialty_name
        })
        
        provider_ids.append(provider_id)
    
    session.commit()
    print(">> Providers loaded successfully")
    return provider_ids

def load_validation_rules(session):
    """Load sample validation rules."""
    print("Loading validation rules...")
    
    validation_rules = [
        ('REQUIRED_FIELDS', 'Required fields validation', 'Validates all required claim fields are present', 'CRITICAL', True),
        ('NPI_FORMAT', 'NPI format validation', 'Validates NPI is 10 digits', 'HIGH', True),
        ('DATE_SEQUENCE', 'Date sequence validation', 'Validates service dates are logical', 'HIGH', True),
        ('PROCEDURE_CODE', 'Procedure code validation', 'Validates CPT codes exist in RVU table', 'HIGH', True),
        ('DIAGNOSIS_CODE', 'Diagnosis code validation', 'Validates ICD-10 format', 'HIGH', True),
        ('CHARGE_AMOUNT', 'Charge amount validation', 'Validates charges are positive numbers', 'MEDIUM', True),
        ('DUPLICATE_CLAIM', 'Duplicate claim check', 'Checks for duplicate claims', 'MEDIUM', True),
        ('PATIENT_AGE', 'Patient age validation', 'Validates patient age for procedure', 'LOW', True)
    ]
    
    for rule_name, display_name, description, severity, active in validation_rules:
        session.execute(text("""
            INSERT INTO validation_rules 
            (rule_name, display_name, description, severity, active)
            VALUES (:rule_name, :display_name, :description, :severity::validation_severity, :active)
            ON CONFLICT (rule_name) DO UPDATE SET
                display_name = EXCLUDED.display_name,
                description = EXCLUDED.description,
                severity = EXCLUDED.severity,
                active = EXCLUDED.active,
                updated_at = CURRENT_TIMESTAMP
        """), {
            "rule_name": rule_name,
            "display_name": display_name,
            "description": description,
            "severity": severity,
            "active": active
        })
    
    session.commit()
    print(">> Validation rules loaded successfully")

def load_rvu_data(session):
    """Load RVU data for CPT codes."""
    print("Loading RVU data...")
    
    for cpt_code, description, work_rvu, pe_rvu, mp_rvu, total_rvu in CPT_CODES:
        session.execute(text("""
            INSERT INTO rvu_data 
            (procedure_code, description, work_rvu, practice_expense_rvu, 
             malpractice_rvu, total_rvu, effective_date, active)
            VALUES (:code, :desc, :work_rvu, :pe_rvu, :mp_rvu, :total_rvu, 
                    CURRENT_DATE - INTERVAL '1 year', true)
            ON CONFLICT (procedure_code) DO UPDATE SET
                description = EXCLUDED.description,
                work_rvu = EXCLUDED.work_rvu,
                practice_expense_rvu = EXCLUDED.practice_expense_rvu,
                malpractice_rvu = EXCLUDED.malpractice_rvu,
                total_rvu = EXCLUDED.total_rvu,
                updated_at = CURRENT_TIMESTAMP
        """), {
            "code": cpt_code,
            "desc": description,
            "work_rvu": work_rvu,
            "pe_rvu": pe_rvu,
            "mp_rvu": mp_rvu,
            "total_rvu": total_rvu
        })
    
    session.commit()
    print(">> RVU data loaded successfully")

def create_batch_metadata(session, batch_id: str, facility_id: str, total_claims: int):
    """Create batch metadata for the claims load."""
    session.execute(text("""
        INSERT INTO batch_metadata 
        (batch_id, facility_id, file_name, total_claims, status, created_at)
        VALUES (:batch_id, :facility_id, :file_name, :total_claims, 'IN_PROGRESS'::processing_status, CURRENT_TIMESTAMP)
    """), {
        "batch_id": batch_id,
        "facility_id": facility_id,
        "file_name": "sample_data_load.py",
        "total_claims": total_claims
    })
    session.commit()

def load_claims_data(session, facility_ids: List[str], provider_ids: List[str]):
    """Load 100,000 sample claims with line items."""
    print("Loading 100,000 sample claims...")
    
    batch_id = f"BATCH_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    total_claims = 100000
    
    # Create batch metadata for each facility
    for facility_id in facility_ids:
        create_batch_metadata(session, f"{batch_id}_{facility_id}", facility_id, total_claims // len(facility_ids))
    
    # Payer types for claims
    payer_types = [
        ('Medicare', 'Government'),
        ('Medicaid', 'Government'),
        ('Blue Cross Blue Shield', 'Commercial'),
        ('Aetna', 'Commercial'),
        ('United Healthcare', 'Commercial'),
        ('Cigna', 'Commercial'),
        ('Self Pay', 'Self Pay')
    ]
    
    batch_size = 1000
    
    for batch_start in range(0, total_claims, batch_size):
        batch_end = min(batch_start + batch_size, total_claims)
        claims_data = []
        line_items_data = []
        
        for i in range(batch_start, batch_end):
            # Generate claim data
            facility_id = random.choice(facility_ids)
            claim_id = f"CLM{i+1:08d}"
            patient_account = f"PAT{i+1:08d}"
            
            # Patient demographics
            first_name = fake.first_name()
            last_name = fake.last_name()
            gender = random.choice(['M', 'F'])
            
            # Age distribution
            age = random.randint(1, 95)
            dob = fake.date_of_birth(minimum_age=age, maximum_age=age)
            
            # Insurance information
            payer_name, payer_type = random.choice(payer_types)
            subscriber_id = f"SUB{fake.random_number(digits=9)}"
            
            # Service dates
            service_date = fake.date_between(start_date='-2y', end_date='today')
            admission_date = service_date
            discharge_date = service_date + timedelta(days=random.randint(0, 7))
            
            # Claim type
            claim_type = random.choice(['PROFESSIONAL', 'INSTITUTIONAL'])
            billing_type = '837P' if claim_type == 'PROFESSIONAL' else '837I'
            
            # Diagnosis codes
            num_diagnoses = random.randint(1, 4)
            selected_diagnoses = random.sample(DIAGNOSIS_CODES, num_diagnoses)
            diagnosis_codes = [dx[0] for dx in selected_diagnoses]
            
            # Provider
            rendering_provider = random.choice(provider_ids)
            
            # Financial information
            total_charges = Decimal('0')
            
            # Generate line items (1-6 per claim)
            num_line_items = random.randint(1, 6)
            selected_cpts = random.sample(CPT_CODES, min(num_line_items, len(CPT_CODES)))
            
            for line_num, (cpt_code, cpt_desc, work_rvu, pe_rvu, mp_rvu, total_rvu) in enumerate(selected_cpts, 1):
                units = random.randint(1, 3)
                
                # Calculate charges
                conversion_factor = Decimal('34.61')
                charge_amount = round(Decimal(str(total_rvu)) * conversion_factor * units * Decimal(str(random.uniform(0.8, 1.2))), 2)
                total_charges += charge_amount
                
                line_items_data.append({
                    "claim_id": claim_id,
                    "facility_id": facility_id,
                    "line_number": line_num,
                    "procedure_code": cpt_code,
                    "units": units,
                    "charge_amount": charge_amount,
                    "place_of_service": random.choice(['11', '22', '23', '21', '24']),
                    "service_date": service_date,
                    "rendering_provider_id": rendering_provider,
                    "diagnosis_pointers": ','.join(str(j) for j in range(1, min(len(diagnosis_codes) + 1, 5)))
                })
            
            # Create claim record
            claims_data.append({
                "claim_id": claim_id,
                "facility_id": facility_id,
                "batch_id": f"{batch_id}_{facility_id}",
                "patient_account_number": patient_account,
                "patient_first_name": first_name,
                "patient_last_name": last_name,
                "patient_dob": dob,
                "patient_gender": gender,
                "subscriber_id": subscriber_id,
                "payer_name": payer_name,
                "payer_type": payer_type,
                "claim_type": claim_type,
                "billing_type": billing_type,
                "total_charges": total_charges,
                "service_from_date": service_date,
                "service_to_date": discharge_date if claim_type == 'INSTITUTIONAL' else service_date,
                "admission_date": admission_date if claim_type == 'INSTITUTIONAL' else None,
                "discharge_date": discharge_date if claim_type == 'INSTITUTIONAL' else None,
                "primary_diagnosis": diagnosis_codes[0],
                "diagnosis_codes": diagnosis_codes,
                "rendering_provider_id": rendering_provider,
                "place_of_service": random.choice(['11', '22', '23', '21', '24']),
                "raw_claim_data": {"source": "sample_data_generator", "version": "1.0"}
            })
        
        # Insert claims batch
        if claims_data:
            for claim in claims_data:
                session.execute(text("""
                    INSERT INTO claims 
                    (claim_id, facility_id, batch_id, patient_account_number,
                     patient_first_name, patient_last_name, patient_dob, patient_gender,
                     subscriber_id, payer_name, payer_type, claim_type, billing_type,
                     total_charges, service_from_date, service_to_date,
                     admission_date, discharge_date, primary_diagnosis, diagnosis_codes,
                     rendering_provider_id, place_of_service, raw_claim_data)
                    VALUES 
                    (:claim_id, :facility_id, :batch_id, :patient_account_number,
                     :patient_first_name, :patient_last_name, :patient_dob, :patient_gender,
                     :subscriber_id, :payer_name, :payer_type, :claim_type::claim_type, :billing_type,
                     :total_charges, :service_from_date, :service_to_date,
                     :admission_date, :discharge_date, :primary_diagnosis, :diagnosis_codes,
                     :rendering_provider_id, :place_of_service, :raw_claim_data)
                """), claim)
        
        # Insert line items batch
        if line_items_data:
            for line_item in line_items_data:
                session.execute(text("""
                    INSERT INTO claim_line_items 
                    (claim_id, facility_id, line_number, procedure_code, units,
                     charge_amount, place_of_service, service_date,
                     rendering_provider_id, diagnosis_pointers)
                    VALUES 
                    (:claim_id, :facility_id, :line_number, :procedure_code, :units,
                     :charge_amount, :place_of_service, :service_date,
                     :rendering_provider_id, :diagnosis_pointers)
                """), line_item)
        
        session.commit()
        
        # Progress indicator
        progress = ((batch_end) / total_claims) * 100
        print(f"  Progress: {progress:.1f}% ({batch_end:,}/{total_claims:,} claims)")
    
    # Update batch metadata to completed
    for facility_id in facility_ids:
        session.execute(text("""
            UPDATE batch_metadata 
            SET status = 'COMPLETED'::processing_status,
                processed_claims = total_claims,
                successful_claims = total_claims,
                end_time = CURRENT_TIMESTAMP
            WHERE batch_id = :batch_id
        """), {"batch_id": f"{batch_id}_{facility_id}"})
    
    session.commit()
    print(">> Claims data loaded successfully")

def main():
    """Main function to load all sample data."""
    parser = argparse.ArgumentParser(description='Load sample data for PostgreSQL claims staging database')
    parser.add_argument(
        '--connection-string',
        required=True,
        help='PostgreSQL connection string (e.g., postgresql://user:pass@host:port/database)'
    )
    parser.add_argument(
        '--skip-claims',
        action='store_true',
        help='Skip loading claims data (for faster testing)'
    )
    
    args = parser.parse_args()
    
    print("Starting PostgreSQL Claims Staging sample data loading...")
    print(f"Connection: {args.connection_string.split('@')[1] if '@' in args.connection_string else 'Local'}")
    
    try:
        # Create database engine and session
        engine = create_database_engine(args.connection_string)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Load data in order
        start_time = datetime.now()
        
        # 1. Load facilities
        facility_ids = load_facilities(session)
        
        # 2. Load providers
        provider_ids = load_providers(session)
        
        # 3. Load RVU data
        load_rvu_data(session)
        
        # 4. Load validation rules
        load_validation_rules(session)
        
        # 5. Load claims data (if not skipped)
        if not args.skip_claims:
            load_claims_data(session, facility_ids, provider_ids)
        else:
            print(">> Skipping claims data loading")
        
        # Final commit and summary
        session.commit()
        session.close()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n>> Sample data loading completed successfully!")
        print(f"Total time: {duration}")
        print(f"\nData Summary:")
        print(f"   - 5 Facilities")
        print(f"   - 200 Providers")
        print(f"   - {len(CPT_CODES)} CPT/RVU codes")
        print(f"   - 8 Validation rules")
        if not args.skip_claims:
            print(f"   - 100,000 Claims with line items")
        print(f"\n>> Ready for claims processing workflow!")
        
    except Exception as e:
        print(f"Error loading sample data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
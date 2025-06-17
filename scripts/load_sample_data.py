#!/usr/bin/env python3
"""
Sample Data Loader for Smart Pro Claims Database

This script loads comprehensive sample data including:
- Organizational hierarchy (organizations, regions, facilities)
- Facility configuration (financial classes, departments, coders, etc.)
- Physicians and providers
- 100,000 sample claims with realistic healthcare data

Usage:
    python scripts/load_sample_data.py --connection-string "mssql+pyodbc://user:pass@server/smart_pro_claims?driver=ODBC+Driver+17+for+SQL+Server"
"""

import argparse
import random
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any

import pyodbc
from faker import Faker
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

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
    """Create SQLAlchemy engine for SQL Server."""
    try:
        engine = create_engine(connection_string, echo=False)
        return engine
    except Exception as e:
        print(f"Failed to create database engine: {e}")
        sys.exit(1)

def load_organizational_hierarchy(session):
    """Load organizations, regions, and facilities."""
    print("Loading organizational hierarchy...")
    
    # Organization 1 with regions
    session.execute(text("""
        INSERT INTO dbo.facility_organization (org_name, installed_date, active)
        VALUES ('Regional Health System', GETDATE(), 1)
    """))
    
    session.execute(text("""
        INSERT INTO dbo.facility_organization (org_name, installed_date, active)
        VALUES ('Metro Medical Group', GETDATE(), 1)
    """))
    
    # Get organization IDs
    org1_result = session.execute(text("SELECT org_id FROM dbo.facility_organization WHERE org_name = 'Regional Health System'"))
    org1_id = org1_result.fetchone()[0]
    
    org2_result = session.execute(text("SELECT org_id FROM dbo.facility_organization WHERE org_name = 'Metro Medical Group'"))
    org2_id = org2_result.fetchone()[0]
    
    # Regions for Organization 1
    session.execute(text("""
        INSERT INTO dbo.facility_region (region_name, org_id, active)
        VALUES ('North Region', :org_id, 1)
    """), {"org_id": org1_id})
    
    session.execute(text("""
        INSERT INTO dbo.facility_region (region_name, org_id, active)
        VALUES ('South Region', :org_id, 1)
    """), {"org_id": org1_id})
    
    # Get region IDs
    region1_result = session.execute(text("SELECT region_id FROM dbo.facility_region WHERE region_name = 'North Region'"))
    region1_id = region1_result.fetchone()[0]
    
    region2_result = session.execute(text("SELECT region_id FROM dbo.facility_region WHERE region_name = 'South Region'"))
    region2_id = region2_result.fetchone()[0]
    
    # Facilities
    facilities = [
        ('FAC001', 'Facility A - North General Hospital', 250, 'Springfield', 'IL', region1_id, org1_id, 1),
        ('FAC002', 'Facility B - South Regional Medical Center', 180, 'Metropolis', 'IL', region2_id, org1_id, 7),
        ('FAC003', 'Facility C - Metro Downtown Clinic', 0, 'Chicago', 'IL', None, org2_id, 3),
        ('FAC004', 'Facility D - Metro West Campus', 120, 'Naperville', 'IL', None, org2_id, 6),
        ('FAC005', 'Facility E - Metro North Specialty Center', 80, 'Evanston', 'IL', None, org2_id, 9)
    ]
    
    for facility_id, name, beds, city, state, region_id, org_id, fiscal_month in facilities:
        session.execute(text("""
            INSERT INTO dbo.facilities 
            (facility_id, facility_name, installed_date, beds, city, state, updated_date, 
             region_id, fiscal_month, org_id, active)
            VALUES (:facility_id, :name, GETDATE(), :beds, :city, :state, GETDATE(),
                    :region_id, :fiscal_month, :org_id, 1)
        """), {
            "facility_id": facility_id,
            "name": name,
            "beds": beds,
            "city": city,
            "state": state,
            "region_id": region_id,
            "fiscal_month": fiscal_month,
            "org_id": org_id
        })
    
    session.commit()
    print(">> Organizational hierarchy loaded successfully")
    return ['FAC001', 'FAC002', 'FAC003', 'FAC004', 'FAC005']

def load_facility_financial_classes(session, facility_ids: List[str]):
    """Load financial classes for each facility."""
    print("Loading facility financial classes...")
    
    # Get payer IDs
    payers_result = session.execute(text("SELECT payer_id, payer_code FROM dbo.core_standard_payers"))
    payers = {row[1]: row[0] for row in payers_result.fetchall()}
    
    if not payers:
        raise Exception("No payers found in core_standard_payers table. Database schema may not have loaded correctly.")
    
    financial_classes = [
        ('A', 'Medicare A', '1', 0.8500, 'HIGH', True, 'A01'),
        ('B', 'Medicare B', '1', 0.8000, 'HIGH', True, 'B02'),
        ('MA', 'Medicaid', '2', 0.6500, 'MEDIUM', False, 'C03'),
        ('BC', 'Blue Cross', '3', 0.9000, 'HIGH', True, None),
        ('HM', 'HMO Plan', '6', 0.8500, 'HIGH', True, None),
        ('CO', 'Commercial Insurance', '8', 0.8700, 'HIGH', True, None),
        ('SP', 'Self Pay', '5', 0.2000, 'LOW', False, None),
        ('WC', 'Workers Compensation', '9', 0.9500, 'HIGH', True, None),
        ('TR', 'Tricare', '7', 0.8200, 'HIGH', True, None),
        ('OT', 'Other Insurance', '4', 0.7500, 'MEDIUM', False, None)
    ]
    
    for facility_id in facility_ids:
        for fc_id, fc_name, payer_code, rate, priority, auto_post, hcc in financial_classes:
            session.execute(text("""
                INSERT INTO dbo.facility_financial_classes 
                (facility_id, financial_class_id, financial_class_name, payer_id, 
                 reimbursement_rate, processing_priority, auto_posting_enabled, 
                 active, effective_date, HCC)
                VALUES (:facility_id, :fc_id, :fc_name, :payer_id, :rate, :priority, 
                        :auto_post, 1, DATEADD(YEAR, -1, GETDATE()), :hcc)
            """), {
                "facility_id": facility_id,
                "fc_id": fc_id,
                "fc_name": fc_name,
                "payer_id": payers[payer_code],
                "rate": rate,
                "priority": priority,
                "auto_post": auto_post,
                "hcc": hcc
            })
    
    session.commit()
    print(">> Facility financial classes loaded successfully")

def load_facility_place_of_service(session, facility_ids: List[str]):
    """Load place of service codes for each facility."""
    print("Loading facility place of service codes...")
    
    for facility_id in facility_ids:
        for pos_code, pos_name, origin in PLACE_OF_SERVICE_CODES:
            session.execute(text("""
                INSERT INTO dbo.facility_place_of_service 
                (facility_id, place_of_service, place_of_service_name, origin, active)
                VALUES (:facility_id, :pos_code, :pos_name, :origin, 1)
            """), {
                "facility_id": facility_id,
                "pos_code": pos_code,
                "pos_name": pos_name,
                "origin": origin
            })
    
    session.commit()
    print(">> Facility place of service codes loaded successfully")

def load_facility_departments(session, facility_ids: List[str]):
    """Load departments for each facility."""
    print("Loading facility departments...")
    
    for facility_id in facility_ids:
        for i, dept_name in enumerate(DEPARTMENTS, 1):
            dept_code = f"DEPT{i:03d}"
            session.execute(text("""
                INSERT INTO dbo.facility_departments 
                (department_code, department_name, facility_id, active)
                VALUES (:dept_code, :dept_name, :facility_id, 1)
            """), {
                "dept_code": dept_code,
                "dept_name": dept_name,
                "facility_id": facility_id
            })
    
    session.commit()
    print(">> Facility departments loaded successfully")

def load_facility_coders(session, facility_ids: List[str]):
    """Load coders for each facility."""
    print("Loading facility coders...")
    
    for facility_id in facility_ids:
        # 3-5 coders per facility
        num_coders = random.randint(3, 5)
        for i in range(num_coders):
            coder_id = f"COD{i+1:03d}"
            first_name = fake.first_name()
            last_name = fake.last_name()
            
            session.execute(text("""
                INSERT INTO dbo.facility_coders 
                (facility_id, coder_id, coder_first_name, coder_last_name, active)
                VALUES (:facility_id, :coder_id, :first_name, :last_name, 1)
            """), {
                "facility_id": facility_id,
                "coder_id": coder_id,
                "first_name": first_name,
                "last_name": last_name
            })
    
    session.commit()
    print(">> Facility coders loaded successfully")

def load_physicians(session):
    """Load physicians/providers."""
    print("Loading physicians...")
    
    physicians = []
    for i in range(200):  # 200 physicians
        provider_id = f"PRV{i+1:06d}"
        first_name = fake.first_name()
        last_name = fake.last_name()
        npi = f"{fake.random_number(digits=10)}"
        specialty_code, specialty_name = random.choice(PHYSICIAN_SPECIALTIES)
        
        session.execute(text("""
            INSERT INTO dbo.physicians 
            (rendering_provider_id, last_name, first_name, npi, specialty_code, active)
            VALUES (:provider_id, :last_name, :first_name, :npi, :specialty_code, 1)
        """), {
            "provider_id": provider_id,
            "last_name": last_name,
            "first_name": first_name,
            "npi": npi,
            "specialty_code": specialty_code
        })
        
        physicians.append(provider_id)
    
    session.commit()
    print(">> Physicians loaded successfully")
    return physicians

def load_rvu_data(session):
    """Load RVU data for CPT codes."""
    print("Loading RVU data...")
    
    for cpt_code, description, work_rvu, pe_rvu, mp_rvu, total_rvu in CPT_CODES:
        # Determine category based on CPT code
        if cpt_code.startswith('99'):
            category = 'Evaluation and Management'
            subcategory = 'Office Visits'
        elif cpt_code.startswith('93'):
            category = 'Medicine'
            subcategory = 'Cardiovascular'
        elif cpt_code.startswith('36'):
            category = 'Surgery'
            subcategory = 'Venipuncture'
        elif cpt_code.startswith('8'):
            category = 'Pathology and Laboratory'
            subcategory = 'Chemistry'
        elif cpt_code.startswith('7'):
            category = 'Radiology'
            if 'CT' in description:
                subcategory = 'Computed Tomography'
            elif 'MRI' in description:
                subcategory = 'Magnetic Resonance Imaging'
            elif 'ultrasound' in description.lower():
                subcategory = 'Ultrasound'
            else:
                subcategory = 'Diagnostic Radiology'
        elif cpt_code.startswith('4'):
            category = 'Surgery'
            subcategory = 'Endoscopy'
        elif cpt_code.startswith('1') or cpt_code.startswith('2'):
            category = 'Surgery'
            subcategory = 'Integumentary System'
        elif cpt_code.startswith('9'):
            category = 'Medicine'
            subcategory = 'Immunizations'
        elif cpt_code.startswith('J'):
            category = 'Medicine'
            subcategory = 'Injections'
        else:
            category = 'Medicine'
            subcategory = 'Other'
        
        session.execute(text("""
            INSERT INTO dbo.rvu_data 
            (procedure_code, description, category, subcategory, work_rvu, 
             practice_expense_rvu, malpractice_rvu, total_rvu, conversion_factor,
             non_facility_pe_rvu, facility_pe_rvu, effective_date, status,
             global_period, professional_component, technical_component, bilateral_surgery)
            VALUES (:code, :desc, :category, :subcategory, :work_rvu, :pe_rvu, :mp_rvu, 
                    :total_rvu, 34.6062, :pe_rvu, :pe_rvu, DATEADD(YEAR, -1, GETDATE()), 
                    'Active', '000', 1, 0, 0)
        """), {
            "code": cpt_code,
            "desc": description,
            "category": category,
            "subcategory": subcategory,
            "work_rvu": work_rvu,
            "pe_rvu": pe_rvu,
            "mp_rvu": mp_rvu,
            "total_rvu": total_rvu
        })
    
    session.commit()
    print(">> RVU data loaded successfully")

def load_claims_data(session, facility_ids: List[str], physician_ids: List[str]):
    """Load 100,000 sample claims with line items and diagnoses."""
    print("Loading 100,000 sample claims...")
    
    # Get financial classes for each facility
    fc_result = session.execute(text("""
        SELECT facility_id, financial_class_id 
        FROM dbo.facility_financial_classes 
        WHERE active = 1
    """))
    facility_financial_classes = {}
    for row in fc_result.fetchall():
        if row[0] not in facility_financial_classes:
            facility_financial_classes[row[0]] = []
        facility_financial_classes[row[0]].append(row[1])
    
    batch_size = 1000
    total_claims = 100000
    
    for batch_start in range(0, total_claims, batch_size):
        batch_end = min(batch_start + batch_size, total_claims)
        batch_claims = []
        batch_diagnoses = []
        batch_line_items = []
        
        for i in range(batch_start, batch_end):
            # Generate claim data
            facility_id = random.choice(facility_ids)
            patient_account = f"PAT{i+1:08d}"
            mrn = f"MRN{fake.random_number(digits=8)}"
            
            # Patient demographics
            gender = random.choice(['M', 'F'])
            if gender == 'M':
                first_name = fake.first_name_male()
            else:
                first_name = fake.first_name_female()
            last_name = fake.last_name()
            
            # Age distribution weighted toward working age adults
            age_weights = [0.1, 0.3, 0.4, 0.2]  # 0-17, 18-44, 45-64, 65+
            age_group = random.choices([0, 1, 2, 3], weights=age_weights)[0]
            if age_group == 0:
                age = random.randint(1, 17)
            elif age_group == 1:
                age = random.randint(18, 44)
            elif age_group == 2:
                age = random.randint(45, 64)
            else:
                age = random.randint(65, 95)
            
            dob = fake.date_of_birth(minimum_age=age, maximum_age=age)
            
            # Financial class
            financial_class = random.choice(facility_financial_classes[facility_id])
            
            # Create timestamp within last 2 years
            created_at = fake.date_time_between(start_date='-2y', end_date='now')
            
            batch_claims.append({
                "facility_id": facility_id,
                "patient_account_number": patient_account,
                "medical_record_number": mrn,
                "first_name": first_name,
                "last_name": last_name,
                "date_of_birth": dob,
                "gender": gender,
                "financial_class_id": financial_class,
                "created_at": created_at
            })
            
            # Generate diagnoses (1-5 per claim)
            num_diagnoses = random.choices([1, 2, 3, 4, 5], weights=[0.4, 0.3, 0.2, 0.07, 0.03])[0]
            selected_diagnoses = random.sample(DIAGNOSIS_CODES, num_diagnoses)
            
            for seq, (dx_code, dx_desc) in enumerate(selected_diagnoses, 1):
                dx_type = 'PRIMARY' if seq == 1 else 'SECONDARY'
                batch_diagnoses.append({
                    "facility_id": facility_id,
                    "patient_account_number": patient_account,
                    "diagnosis_sequence": seq,
                    "diagnosis_code": dx_code,
                    "diagnosis_description": dx_desc,
                    "diagnosis_type": dx_type,
                    "created_at": created_at
                })
            
            # Generate line items (1-8 per claim)
            num_line_items = random.choices([1, 2, 3, 4, 5, 6, 7, 8], 
                                          weights=[0.3, 0.25, 0.2, 0.1, 0.08, 0.04, 0.02, 0.01])[0]
            selected_cpts = random.sample(CPT_CODES, min(num_line_items, len(CPT_CODES)))
            
            for line_num, (cpt_code, cpt_desc, work_rvu, pe_rvu, mp_rvu, total_rvu) in enumerate(selected_cpts, 1):
                units = random.choices([1, 2, 3, 4], weights=[0.8, 0.15, 0.04, 0.01])[0]
                
                # Calculate charges based on RVU and conversion factor
                conversion_factor = Decimal('34.61')
                rvu_amount = Decimal(str(total_rvu)) * conversion_factor * units
                
                # Add random variation to charges (±20%)
                variation = Decimal(str(random.uniform(0.8, 1.2)))
                charge_amount = round(rvu_amount * variation, 2)
                
                # Reimbursement based on financial class rate
                fc_rates = {
                    'A': 0.85, 'B': 0.80, 'MA': 0.65, 'BC': 0.90, 'HM': 0.85,
                    'CO': 0.87, 'SP': 0.20, 'WC': 0.95, 'TR': 0.82, 'OT': 0.75
                }
                reimbursement_rate = fc_rates.get(financial_class, 0.80)
                reimbursement_amount = round(charge_amount * Decimal(str(reimbursement_rate)), 2)
                
                # Service dates (usually same day, sometimes span)
                service_from = created_at.date()
                if random.random() < 0.1:  # 10% chance of multi-day service
                    service_to = service_from + timedelta(days=random.randint(1, 7))
                else:
                    service_to = service_from
                
                # Place of service
                place_of_service = random.choice(['11', '22', '23', '21', '24'])
                
                # Diagnosis pointer
                diagnosis_pointer = ','.join([str(j) for j in range(1, min(num_diagnoses + 1, 5))])
                
                # Provider
                rendering_provider = random.choice(physician_ids)
                
                batch_line_items.append({
                    "facility_id": facility_id,
                    "patient_account_number": patient_account,
                    "line_number": line_num,
                    "procedure_code": cpt_code,
                    "units": units,
                    "charge_amount": charge_amount,
                    "service_from_date": service_from,
                    "service_to_date": service_to,
                    "diagnosis_pointer": diagnosis_pointer,
                    "place_of_service": place_of_service,
                    "rvu_value": Decimal(str(total_rvu)),
                    "reimbursement_amount": reimbursement_amount,
                    "rendering_provider_id": rendering_provider,
                    "created_at": created_at
                })
        
        # Insert batch data
        # Claims
        if batch_claims:
            claims_values = []
            for claim in batch_claims:
                claims_values.append(f"('{claim['facility_id']}', '{claim['patient_account_number']}', "
                                   f"'{claim['medical_record_number']}', '{claim['first_name']}', "
                                   f"'{claim['last_name']}', '{claim['date_of_birth']}', '{claim['gender']}', "
                                   f"'{claim['financial_class_id']}', '{claim['created_at']}')")
            
            claims_sql = f"""
                INSERT INTO dbo.claims 
                (facility_id, patient_account_number, medical_record_number, first_name, 
                 last_name, date_of_birth, gender, financial_class_id, created_at)
                VALUES {', '.join(claims_values)}
            """
            session.execute(text(claims_sql))
        
        # Diagnoses
        if batch_diagnoses:
            dx_values = []
            for dx in batch_diagnoses:
                dx_values.append(f"('{dx['facility_id']}', '{dx['patient_account_number']}', "
                               f"{dx['diagnosis_sequence']}, '{dx['diagnosis_code']}', "
                               f"'{dx['diagnosis_description']}', '{dx['diagnosis_type']}', "
                               f"'{dx['created_at']}')")
            
            dx_sql = f"""
                INSERT INTO dbo.claims_diagnosis 
                (facility_id, patient_account_number, diagnosis_sequence, diagnosis_code,
                 diagnosis_description, diagnosis_type, created_at)
                VALUES {', '.join(dx_values)}
            """
            session.execute(text(dx_sql))
        
        # Line items
        if batch_line_items:
            li_values = []
            for li in batch_line_items:
                li_values.append(f"('{li['facility_id']}', '{li['patient_account_number']}', "
                               f"{li['line_number']}, '{li['procedure_code']}', {li['units']}, "
                               f"{li['charge_amount']}, '{li['service_from_date']}', '{li['service_to_date']}', "
                               f"'{li['diagnosis_pointer']}', '{li['place_of_service']}', "
                               f"{li['rvu_value']}, {li['reimbursement_amount']}, "
                               f"'{li['rendering_provider_id']}', '{li['created_at']}')")
            
            li_sql = f"""
                INSERT INTO dbo.claims_line_items 
                (facility_id, patient_account_number, line_number, procedure_code, units,
                 charge_amount, service_from_date, service_to_date, diagnosis_pointer,
                 place_of_service, rvu_value, reimbursement_amount, rendering_provider_id, created_at)
                VALUES {', '.join(li_values)}
            """
            session.execute(text(li_sql))
        
        session.commit()
        
        # Progress indicator
        progress = ((batch_end) / total_claims) * 100
        print(f"  Progress: {progress:.1f}% ({batch_end:,}/{total_claims:,} claims)")
    
    print(">> Claims data loaded successfully")

def main():
    """Main function to load all sample data."""
    parser = argparse.ArgumentParser(description='Load sample data for Smart Pro Claims database')
    parser.add_argument(
        '--connection-string',
        required=True,
        help='SQL Server connection string'
    )
    parser.add_argument(
        '--skip-claims',
        action='store_true',
        help='Skip loading claims data (for faster testing)'
    )
    
    args = parser.parse_args()
    
    print("Starting Smart Pro Claims sample data loading...")
    print(f"Connection: {args.connection_string.split('@')[1] if '@' in args.connection_string else 'Local'}")
    
    try:
        # Create database engine and session
        engine = create_database_engine(args.connection_string)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Load data in order
        start_time = datetime.now()
        
        # 1. Load organizational hierarchy
        facility_ids = load_organizational_hierarchy(session)
        
        # 2. Load facility configuration
        load_facility_financial_classes(session, facility_ids)
        load_facility_place_of_service(session, facility_ids)
        load_facility_departments(session, facility_ids)
        load_facility_coders(session, facility_ids)
        
        # 3. Load physicians
        physician_ids = load_physicians(session)
        
        # 4. Load RVU data
        load_rvu_data(session)
        
        # 5. Load claims data (if not skipped)
        if not args.skip_claims:
            load_claims_data(session, facility_ids, physician_ids)
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
        print(f"   - 2 Organizations")
        print(f"   - 2 Regions")
        print(f"   - 5 Facilities")
        print(f"   - 200 Physicians")
        print(f"   - {len(CPT_CODES)} CPT/RVU codes")
        print(f"   - {len(DIAGNOSIS_CODES)} Diagnosis codes")
        if not args.skip_claims:
            print(f"   - 100,000 Claims with line items and diagnoses")
        print(f"\n>> Ready for testing and analytics!")
        
    except Exception as e:
        print(f"Error loading sample data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
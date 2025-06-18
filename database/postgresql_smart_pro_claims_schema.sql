-- Smart Pro Claims Database Schema for PostgreSQL
-- High-Performance analytics and reporting database
-- Complete migration from SQL Server to PostgreSQL
-- Includes partitioning, advanced indexes, and optimized for healthcare claims processing

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "tablefunc";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create custom types
CREATE TYPE processing_status AS ENUM (
    'pending', 'processing', 'validated', 'failed', 'completed', 'reprocessing'
);

CREATE TYPE failure_category AS ENUM (
    'validation_error', 'missing_data', 'duplicate_claim', 'invalid_facility',
    'invalid_provider', 'invalid_procedure', 'invalid_diagnosis', 
    'date_range_error', 'financial_error', 'ml_rejection', 'system_error'
);

CREATE TYPE claim_priority AS ENUM ('low', 'medium', 'high', 'critical');

CREATE TYPE payer_type_enum AS ENUM ('Government', 'Commercial', 'Self-Pay', 'Workers Compensation');

CREATE TYPE gender_enum AS ENUM ('M', 'F', 'U');

CREATE TYPE resolution_status_enum AS ENUM ('PENDING', 'IN_PROGRESS', 'RESOLVED', 'REJECTED', 'ESCALATED');

-- =============================================
-- CORE REFERENCE TABLES
-- =============================================

-- Core Standard Payers
CREATE TABLE core_standard_payers (
    payer_id SERIAL PRIMARY KEY,
    payer_name VARCHAR(200) NOT NULL,
    payer_code CHAR(10) NOT NULL UNIQUE,
    payer_type payer_type_enum,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- RVU Data
CREATE TABLE rvu_data (
    procedure_code VARCHAR(10) NOT NULL PRIMARY KEY,
    description VARCHAR(500),
    category VARCHAR(50),
    subcategory VARCHAR(50),
    work_rvu DECIMAL(8, 4),
    practice_expense_rvu DECIMAL(8, 4),
    malpractice_rvu DECIMAL(8, 4),
    total_rvu DECIMAL(8, 4),
    conversion_factor DECIMAL(8, 2),
    non_facility_pe_rvu DECIMAL(8, 4),
    facility_pe_rvu DECIMAL(8, 4),
    effective_date DATE,
    end_date DATE,
    status VARCHAR(20),
    global_period VARCHAR(10),
    professional_component BOOLEAN,
    technical_component BOOLEAN,
    bilateral_surgery BOOLEAN,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- =============================================
-- ORGANIZATIONAL HIERARCHY
-- =============================================

-- Facility Organization (Top Level)
CREATE TABLE facility_organization (
    org_id SERIAL PRIMARY KEY,
    org_name VARCHAR(100) NOT NULL,
    installed_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_by INTEGER,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Facility Region (Optional Middle Level)
CREATE TABLE facility_region (
    region_id SERIAL PRIMARY KEY,
    region_name VARCHAR(100) NOT NULL,
    org_id INTEGER NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_facility_region_org FOREIGN KEY (org_id) 
        REFERENCES facility_organization(org_id)
);

-- Facilities (Bottom Level)
CREATE TABLE facilities (
    facility_id VARCHAR(20) PRIMARY KEY,
    facility_name VARCHAR(100) NOT NULL,
    installed_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    beds INTEGER,
    city VARCHAR(24),
    state CHAR(2),
    updated_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_by INTEGER,
    region_id INTEGER,
    fiscal_month INTEGER CHECK (fiscal_month BETWEEN 1 AND 12),
    org_id INTEGER NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    
    CONSTRAINT fk_facilities_region FOREIGN KEY (region_id) 
        REFERENCES facility_region(region_id),
    CONSTRAINT fk_facilities_org FOREIGN KEY (org_id) 
        REFERENCES facility_organization(org_id)
);

-- =============================================
-- FACILITY CONFIGURATION TABLES
-- =============================================

-- Facility Financial Classes
CREATE TABLE facility_financial_classes (
    facility_id VARCHAR(20) NOT NULL,
    financial_class_id VARCHAR(10) NOT NULL,
    financial_class_name VARCHAR(100) NOT NULL,
    payer_id INTEGER NOT NULL,
    reimbursement_rate DECIMAL(5,4),
    processing_priority VARCHAR(10),
    auto_posting_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    effective_date DATE NOT NULL,
    end_date DATE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    hcc CHAR(3),
    
    PRIMARY KEY (facility_id, financial_class_id),
    CONSTRAINT fk_facility_financial_classes_facility FOREIGN KEY (facility_id) 
        REFERENCES facilities(facility_id),
    CONSTRAINT fk_facility_financial_classes_payer FOREIGN KEY (payer_id) 
        REFERENCES core_standard_payers(payer_id)
);

-- Facility Place of Service
CREATE TABLE facility_place_of_service (
    facility_id VARCHAR(20) NOT NULL,
    place_of_service VARCHAR(2) NOT NULL,
    place_of_service_name VARCHAR(60) NOT NULL,
    origin INTEGER,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (facility_id, place_of_service),
    CONSTRAINT fk_facility_place_of_service_facility FOREIGN KEY (facility_id) 
        REFERENCES facilities(facility_id)
);

-- Facility Departments
CREATE TABLE facility_departments (
    department_code VARCHAR(10) NOT NULL,
    department_name VARCHAR(50) NOT NULL,
    facility_id VARCHAR(20) NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (facility_id, department_code),
    CONSTRAINT fk_facility_departments_facility FOREIGN KEY (facility_id) 
        REFERENCES facilities(facility_id)
);

-- Facility Coders
CREATE TABLE facility_coders (
    facility_id VARCHAR(20) NOT NULL,
    coder_id VARCHAR(50) NOT NULL,
    coder_first_name VARCHAR(50) NOT NULL,
    coder_last_name VARCHAR(50) NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (facility_id, coder_id),
    CONSTRAINT fk_facility_coders_facility FOREIGN KEY (facility_id) 
        REFERENCES facilities(facility_id)
);

-- Physicians
CREATE TABLE physicians (
    rendering_provider_id VARCHAR(50) PRIMARY KEY,
    last_name VARCHAR(50) NOT NULL,
    first_name VARCHAR(50) NOT NULL,
    npi VARCHAR(10),
    specialty_code VARCHAR(10),
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- =============================================
-- CLAIMS PROCESSING TABLES (Updated from staging)
-- =============================================

-- Business validation rules
CREATE TABLE validation_rules (
    id SERIAL PRIMARY KEY,
    rule_name VARCHAR(100) NOT NULL UNIQUE,
    rule_type VARCHAR(50) NOT NULL,
    rule_condition JSONB NOT NULL,
    error_message TEXT NOT NULL,
    severity VARCHAR(20) DEFAULT 'error',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Batch metadata for processing tracking
CREATE TABLE batch_metadata (
    id BIGSERIAL PRIMARY KEY,
    batch_id VARCHAR(100) UNIQUE NOT NULL,
    facility_id VARCHAR(20) NOT NULL REFERENCES facilities(facility_id),
    source_system VARCHAR(50) NOT NULL,
    file_name VARCHAR(500),
    status processing_status DEFAULT 'pending' NOT NULL,
    priority claim_priority DEFAULT 'medium' NOT NULL,
    total_claims INTEGER DEFAULT 0 NOT NULL,
    processed_claims INTEGER DEFAULT 0 NOT NULL,
    failed_claims INTEGER DEFAULT 0 NOT NULL,
    total_amount DECIMAL(14,2),
    submitted_by VARCHAR(100) NOT NULL,
    approved_by VARCHAR(100),
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    processing_time_seconds DECIMAL(10,3),
    throughput_per_second DECIMAL(10,3),
    error_summary JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create sequence for claims table
CREATE SEQUENCE claims_id_seq;

-- Main claims table (production table - partitioned by created_at)
CREATE TABLE claims (
    id BIGINT NOT NULL DEFAULT nextval('claims_id_seq'),
    claim_id VARCHAR(50) NOT NULL,
    facility_id VARCHAR(20) NOT NULL,
    patient_account_number VARCHAR(50) NOT NULL,
    medical_record_number VARCHAR(50),
    
    -- Patient demographics
    patient_first_name VARCHAR(100) NOT NULL,
    patient_last_name VARCHAR(100) NOT NULL,
    patient_middle_name VARCHAR(100),
    patient_date_of_birth DATE NOT NULL,
    patient_ssn_encrypted BYTEA, -- Encrypted SSN
    
    -- Service information  
    admission_date DATE NOT NULL,
    discharge_date DATE NOT NULL,
    service_from_date DATE NOT NULL,
    service_to_date DATE NOT NULL,
    
    -- Financial information
    financial_class VARCHAR(50) NOT NULL,
    total_charges DECIMAL(12,2) NOT NULL,
    expected_reimbursement DECIMAL(12,2),
    
    -- Insurance information
    insurance_type VARCHAR(50) NOT NULL,
    insurance_plan_id VARCHAR(50),
    subscriber_id VARCHAR(50),
    
    -- Provider information
    billing_provider_npi VARCHAR(10) NOT NULL,
    billing_provider_name VARCHAR(200) NOT NULL,
    attending_provider_npi VARCHAR(10),
    attending_provider_name VARCHAR(200),
    
    -- Diagnosis codes
    primary_diagnosis_code VARCHAR(10) NOT NULL,
    diagnosis_codes JSONB DEFAULT '[]',
    
    -- Processing metadata
    processing_status processing_status DEFAULT 'pending' NOT NULL,
    priority claim_priority DEFAULT 'medium' NOT NULL,
    correlation_id VARCHAR(100),
    batch_id BIGINT REFERENCES batch_metadata(id),
    
    -- ML prediction results
    ml_prediction_score DECIMAL(5,4),
    ml_prediction_result VARCHAR(50),
    
    -- Validation results
    validation_errors JSONB,
    validation_warnings JSONB,
    
    -- Timestamps (partition key)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    processed_at TIMESTAMP,
    
    -- Retry information
    retry_count INTEGER DEFAULT 0 NOT NULL,
    last_retry_at TIMESTAMP,
    
    -- Additional fields from SQL Server
    patient_name VARCHAR(100) GENERATED ALWAYS AS (
        TRIM(patient_first_name || ' ' || COALESCE(patient_middle_name, '') || ' ' || patient_last_name)
    ) STORED,
    gender gender_enum DEFAULT 'U',
    secondary_insurance VARCHAR(10),
    active BOOLEAN NOT NULL DEFAULT TRUE,
    
    PRIMARY KEY (id, created_at),
    CONSTRAINT fk_claims_facility FOREIGN KEY (facility_id) 
        REFERENCES facilities(facility_id),
    CONSTRAINT uq_claim_facility UNIQUE (claim_id, facility_id, created_at)
) PARTITION BY RANGE (created_at);

-- Create sequence for claims_diagnosis table
CREATE SEQUENCE claims_diagnosis_id_seq;

-- Claims Diagnosis
CREATE TABLE claims_diagnosis (
    id BIGINT NOT NULL DEFAULT nextval('claims_diagnosis_id_seq'),
    facility_id VARCHAR(20) NOT NULL,
    patient_account_number VARCHAR(50) NOT NULL,
    diagnosis_sequence INTEGER NOT NULL,
    diagnosis_code VARCHAR(20) NOT NULL,
    diagnosis_description VARCHAR(255),
    diagnosis_type VARCHAR(10),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    claim_id BIGINT, -- Reference to claims table
    
    PRIMARY KEY (id, created_at),
    CONSTRAINT fk_claims_diagnosis_facility FOREIGN KEY (facility_id) 
        REFERENCES facilities(facility_id)
) PARTITION BY RANGE (created_at);

-- Create sequence for claim_line_items table
CREATE SEQUENCE claim_line_items_id_seq;

-- Claim line items table (enhanced from staging)
CREATE TABLE claim_line_items (
    id BIGINT NOT NULL DEFAULT nextval('claim_line_items_id_seq'),
    claim_id BIGINT NOT NULL,
    facility_id VARCHAR(20) NOT NULL,
    patient_account_number VARCHAR(50) NOT NULL,
    line_number INTEGER NOT NULL,
    
    -- Service information
    service_date DATE NOT NULL,
    procedure_code VARCHAR(10) NOT NULL,
    procedure_description VARCHAR(500),
    modifier_codes JSONB DEFAULT '[]',
    
    -- Additional modifiers from SQL Server
    modifier1 VARCHAR(2),
    modifier2 VARCHAR(2),
    modifier3 VARCHAR(2),
    modifier4 VARCHAR(2),
    
    -- Quantity and charges
    units INTEGER NOT NULL DEFAULT 1,
    charge_amount DECIMAL(10,2) NOT NULL,
    
    -- Service dates
    service_from_date DATE,
    service_to_date DATE,
    
    -- Additional fields from SQL Server
    diagnosis_pointer VARCHAR(10),
    place_of_service VARCHAR(2),
    revenue_code VARCHAR(4),
    
    -- Provider information
    rendering_provider_npi VARCHAR(10),
    rendering_provider_name VARCHAR(200),
    rendering_provider_id VARCHAR(50), -- For compatibility
    
    -- RVU information
    rvu_work DECIMAL(8,4),
    rvu_practice_expense DECIMAL(8,4),
    rvu_malpractice DECIMAL(8,4),
    rvu_total DECIMAL(8,4),
    rvu_value DECIMAL(8,4), -- Alias for compatibility
    expected_reimbursement DECIMAL(10,2),
    reimbursement_amount DECIMAL(10,2), -- Alias for compatibility
    
    -- Diagnosis pointers (from staging)
    diagnosis_pointers JSONB,
    
    -- Timestamps (partition key)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (id, created_at),
    CONSTRAINT fk_claim_line_items_facility FOREIGN KEY (facility_id) 
        REFERENCES facilities(facility_id),
    CONSTRAINT fk_claim_line_items_place_of_service FOREIGN KEY (facility_id, place_of_service) 
        REFERENCES facility_place_of_service(facility_id, place_of_service),
    CONSTRAINT fk_claim_line_items_provider FOREIGN KEY (rendering_provider_id) 
        REFERENCES physicians(rendering_provider_id),
    CONSTRAINT fk_claim_line_items_rvu FOREIGN KEY (procedure_code) 
        REFERENCES rvu_data(procedure_code),
    CONSTRAINT uq_claim_line_number UNIQUE (claim_id, line_number, created_at)
) PARTITION BY RANGE (created_at);

-- =============================================
-- FAILED CLAIMS MANAGEMENT
-- =============================================

-- Failed Claims Patterns
CREATE TABLE failed_claims_patterns (
    pattern_id VARCHAR(50) PRIMARY KEY,
    pattern_name VARCHAR(200) NOT NULL,
    pattern_description TEXT,
    failure_category failure_category,
    severity_level VARCHAR(20),
    frequency_score INTEGER,
    pattern_rules JSONB,
    auto_repair_possible BOOLEAN DEFAULT FALSE,
    repair_template JSONB,
    occurrence_count INTEGER DEFAULT 0,
    resolution_rate DECIMAL(5, 4),
    average_resolution_time_hours DECIMAL(8, 2),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Failed Claims (enhanced from staging)
CREATE TABLE failed_claims (
    id BIGSERIAL,
    claim_id VARCHAR(50),
    batch_id VARCHAR(50),
    facility_id VARCHAR(20),
    patient_account_number VARCHAR(50),
    original_claim_id BIGINT, -- Reference to original claim
    claim_reference VARCHAR(50) NOT NULL,
    
    -- Failure information
    original_data JSONB,
    failure_reason TEXT NOT NULL,
    failure_category failure_category NOT NULL,
    processing_stage VARCHAR(50) NOT NULL,
    failed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    failure_details JSONB NOT NULL,
    
    -- Resolution tracking
    repair_suggestions JSONB,
    resolution_status resolution_status_enum DEFAULT 'PENDING',
    assigned_to VARCHAR(100),
    resolved_at TIMESTAMP,
    resolution_notes TEXT,
    resolution_action VARCHAR(50),
    resolved_by VARCHAR(100), -- From staging
    
    -- Pattern matching
    error_pattern_id VARCHAR(50),
    
    -- Priority and impact
    priority_level VARCHAR(10) DEFAULT 'MEDIUM',
    impact_level VARCHAR(10) DEFAULT 'MEDIUM',
    potential_revenue_loss DECIMAL(12, 2),
    
    -- Reprocessing information (from staging)
    can_reprocess BOOLEAN DEFAULT TRUE NOT NULL,
    reprocess_count INTEGER DEFAULT 0 NOT NULL,
    last_reprocess_at TIMESTAMP,
    
    -- Financial impact
    charge_amount DECIMAL(12,2) NOT NULL,
    expected_reimbursement DECIMAL(12,2),
    
    -- Timestamps (partition key)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    coder_id VARCHAR(50),
    
    PRIMARY KEY (id, failed_at),
    CONSTRAINT fk_failed_claims_facility FOREIGN KEY (facility_id) 
        REFERENCES facilities(facility_id),
    CONSTRAINT fk_failed_claims_pattern FOREIGN KEY (error_pattern_id) 
        REFERENCES failed_claims_patterns(pattern_id),
    CONSTRAINT fk_failed_claims_coder FOREIGN KEY (facility_id, coder_id) 
        REFERENCES facility_coders(facility_id, coder_id)
) PARTITION BY RANGE (failed_at);

-- =============================================
-- AUDIT AND LOGGING TABLES
-- =============================================

-- Audit Log (enhanced from staging)
-- Create sequence for audit_logs table
CREATE SEQUENCE audit_logs_id_seq;

CREATE TABLE audit_logs (
    id BIGINT NOT NULL DEFAULT nextval('audit_logs_id_seq'),
    audit_id BIGINT GENERATED ALWAYS AS IDENTITY, -- For compatibility
    
    -- Action information
    table_name VARCHAR(100) NOT NULL,
    record_id VARCHAR(50) NOT NULL,
    operation VARCHAR(20) NOT NULL,
    action_type VARCHAR(50) NOT NULL,
    action_description TEXT NOT NULL,
    
    -- User information
    user_id VARCHAR(100) NOT NULL,
    user_name VARCHAR(200) NOT NULL,
    user_role VARCHAR(50) NOT NULL,
    session_id VARCHAR(100),
    ip_address INET NOT NULL,
    user_agent VARCHAR(500),
    
    -- Data changes
    old_values JSONB,
    new_values JSONB,
    changed_columns TEXT,
    
    -- Resource information
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(100),
    
    -- PHI access tracking
    accessed_phi BOOLEAN DEFAULT FALSE NOT NULL,
    phi_fields_accessed JSONB,
    business_justification TEXT,
    
    -- Approval workflow
    reason VARCHAR(500),
    approval_required BOOLEAN DEFAULT FALSE,
    approved_by VARCHAR(100),
    approved_at TIMESTAMP,
    
    -- Request information
    request_id VARCHAR(100) NOT NULL,
    
    -- Additional context
    additional_context JSONB,
    
    -- Timestamp (partition key)
    operation_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Data Access Log
-- Create sequence for data_access_log table
CREATE SEQUENCE data_access_log_id_seq;

CREATE TABLE data_access_log (
    id BIGINT NOT NULL DEFAULT nextval('data_access_log_id_seq'),
    access_id BIGINT GENERATED ALWAYS AS IDENTITY, -- For compatibility
    access_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    user_id VARCHAR(100) NOT NULL,
    user_role VARCHAR(50),
    department VARCHAR(100),
    table_name VARCHAR(100) NOT NULL,
    record_id VARCHAR(50),
    access_type VARCHAR(20),
    data_classification VARCHAR(20),
    business_justification VARCHAR(500),
    patient_account_number VARCHAR(50),
    facility_id VARCHAR(20),
    ip_address INET,
    application_name VARCHAR(100),
    query_executed TEXT,
    
    PRIMARY KEY (id, access_timestamp)
) PARTITION BY RANGE (access_timestamp);

-- =============================================
-- PERFORMANCE AND REPORTING TABLES
-- =============================================

-- Daily Processing Summary
CREATE TABLE daily_processing_summary (
    summary_date DATE NOT NULL,
    facility_id VARCHAR(20) NOT NULL,
    total_claims_processed INTEGER,
    total_claims_failed INTEGER,
    total_line_items INTEGER,
    total_charge_amount DECIMAL(15, 2),
    total_reimbursement_amount DECIMAL(15, 2),
    average_reimbursement_rate DECIMAL(5, 4),
    average_processing_time_seconds DECIMAL(8, 2),
    throughput_claims_per_hour DECIMAL(10, 2),
    error_rate_percentage DECIMAL(5, 2),
    ml_accuracy_percentage DECIMAL(5, 2),
    validation_pass_rate DECIMAL(5, 2),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (summary_date, facility_id),
    CONSTRAINT fk_daily_processing_summary_facility FOREIGN KEY (facility_id) 
        REFERENCES facilities(facility_id)
);

-- Performance Metrics (enhanced from staging)
CREATE TABLE performance_metrics (
    id BIGSERIAL,
    metric_id BIGINT GENERATED ALWAYS AS IDENTITY, -- For compatibility
    metric_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    metric_type VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(20,6) NOT NULL,
    unit VARCHAR(20) NOT NULL,
    
    -- Legacy fields for compatibility
    facility_id VARCHAR(20),
    claims_per_second DECIMAL(10, 4),
    records_per_minute DECIMAL(10, 2),
    cpu_usage_percent DECIMAL(5, 2),
    memory_usage_mb INTEGER,
    database_response_time_ms DECIMAL(8, 2),
    queue_depth INTEGER,
    error_rate DECIMAL(5, 4),
    processing_accuracy DECIMAL(5, 4),
    revenue_per_claim DECIMAL(10, 2),
    additional_metrics JSONB,
    
    -- Enhanced fields from staging
    batch_id VARCHAR(100),
    service_name VARCHAR(50) NOT NULL,
    tags JSONB,
    
    -- Timestamp (partition key)
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    
    PRIMARY KEY (id, recorded_at),
    CONSTRAINT fk_performance_metrics_facility FOREIGN KEY (facility_id) 
        REFERENCES facilities(facility_id)
) PARTITION BY RANGE (recorded_at);

-- =============================================
-- PARTITION MANAGEMENT
-- =============================================

-- Create monthly partitions for current year (2025)
-- Claims partitions
CREATE TABLE claims_2025_01 PARTITION OF claims
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE claims_2025_02 PARTITION OF claims
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
CREATE TABLE claims_2025_03 PARTITION OF claims
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');
CREATE TABLE claims_2025_04 PARTITION OF claims
    FOR VALUES FROM ('2025-04-01') TO ('2025-05-01');
CREATE TABLE claims_2025_05 PARTITION OF claims
    FOR VALUES FROM ('2025-05-01') TO ('2025-06-01');
CREATE TABLE claims_2025_06 PARTITION OF claims
    FOR VALUES FROM ('2025-06-01') TO ('2025-07-01');
CREATE TABLE claims_2025_07 PARTITION OF claims
    FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');
CREATE TABLE claims_2025_08 PARTITION OF claims
    FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE claims_2025_09 PARTITION OF claims
    FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE claims_2025_10 PARTITION OF claims
    FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE claims_2025_11 PARTITION OF claims
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE claims_2025_12 PARTITION OF claims
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

-- Claims diagnosis partitions
CREATE TABLE claims_diagnosis_2025_01 PARTITION OF claims_diagnosis
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE claims_diagnosis_2025_02 PARTITION OF claims_diagnosis
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
CREATE TABLE claims_diagnosis_2025_03 PARTITION OF claims_diagnosis
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');
CREATE TABLE claims_diagnosis_2025_04 PARTITION OF claims_diagnosis
    FOR VALUES FROM ('2025-04-01') TO ('2025-05-01');
CREATE TABLE claims_diagnosis_2025_05 PARTITION OF claims_diagnosis
    FOR VALUES FROM ('2025-05-01') TO ('2025-06-01');
CREATE TABLE claims_diagnosis_2025_06 PARTITION OF claims_diagnosis
    FOR VALUES FROM ('2025-06-01') TO ('2025-07-01');
CREATE TABLE claims_diagnosis_2025_07 PARTITION OF claims_diagnosis
    FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');
CREATE TABLE claims_diagnosis_2025_08 PARTITION OF claims_diagnosis
    FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE claims_diagnosis_2025_09 PARTITION OF claims_diagnosis
    FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE claims_diagnosis_2025_10 PARTITION OF claims_diagnosis
    FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE claims_diagnosis_2025_11 PARTITION OF claims_diagnosis
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE claims_diagnosis_2025_12 PARTITION OF claims_diagnosis
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

-- Claim line items partitions
CREATE TABLE claim_line_items_2025_01 PARTITION OF claim_line_items
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE claim_line_items_2025_02 PARTITION OF claim_line_items
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
CREATE TABLE claim_line_items_2025_03 PARTITION OF claim_line_items
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');
CREATE TABLE claim_line_items_2025_04 PARTITION OF claim_line_items
    FOR VALUES FROM ('2025-04-01') TO ('2025-05-01');
CREATE TABLE claim_line_items_2025_05 PARTITION OF claim_line_items
    FOR VALUES FROM ('2025-05-01') TO ('2025-06-01');
CREATE TABLE claim_line_items_2025_06 PARTITION OF claim_line_items
    FOR VALUES FROM ('2025-06-01') TO ('2025-07-01');
CREATE TABLE claim_line_items_2025_07 PARTITION OF claim_line_items
    FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');
CREATE TABLE claim_line_items_2025_08 PARTITION OF claim_line_items
    FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE claim_line_items_2025_09 PARTITION OF claim_line_items
    FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE claim_line_items_2025_10 PARTITION OF claim_line_items
    FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE claim_line_items_2025_11 PARTITION OF claim_line_items
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE claim_line_items_2025_12 PARTITION OF claim_line_items
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

-- Failed claims partitions
CREATE TABLE failed_claims_2025_01 PARTITION OF failed_claims
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE failed_claims_2025_02 PARTITION OF failed_claims
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
CREATE TABLE failed_claims_2025_03 PARTITION OF failed_claims
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');
CREATE TABLE failed_claims_2025_04 PARTITION OF failed_claims
    FOR VALUES FROM ('2025-04-01') TO ('2025-05-01');
CREATE TABLE failed_claims_2025_05 PARTITION OF failed_claims
    FOR VALUES FROM ('2025-05-01') TO ('2025-06-01');
CREATE TABLE failed_claims_2025_06 PARTITION OF failed_claims
    FOR VALUES FROM ('2025-06-01') TO ('2025-07-01');
CREATE TABLE failed_claims_2025_07 PARTITION OF failed_claims
    FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');
CREATE TABLE failed_claims_2025_08 PARTITION OF failed_claims
    FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE failed_claims_2025_09 PARTITION OF failed_claims
    FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE failed_claims_2025_10 PARTITION OF failed_claims
    FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE failed_claims_2025_11 PARTITION OF failed_claims
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE failed_claims_2025_12 PARTITION OF failed_claims
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

-- Audit logs partitions
CREATE TABLE audit_logs_2025_01 PARTITION OF audit_logs
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE audit_logs_2025_02 PARTITION OF audit_logs
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
CREATE TABLE audit_logs_2025_03 PARTITION OF audit_logs
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');
CREATE TABLE audit_logs_2025_04 PARTITION OF audit_logs
    FOR VALUES FROM ('2025-04-01') TO ('2025-05-01');
CREATE TABLE audit_logs_2025_05 PARTITION OF audit_logs
    FOR VALUES FROM ('2025-05-01') TO ('2025-06-01');
CREATE TABLE audit_logs_2025_06 PARTITION OF audit_logs
    FOR VALUES FROM ('2025-06-01') TO ('2025-07-01');
CREATE TABLE audit_logs_2025_07 PARTITION OF audit_logs
    FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');
CREATE TABLE audit_logs_2025_08 PARTITION OF audit_logs
    FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE audit_logs_2025_09 PARTITION OF audit_logs
    FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE audit_logs_2025_10 PARTITION OF audit_logs
    FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE audit_logs_2025_11 PARTITION OF audit_logs
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE audit_logs_2025_12 PARTITION OF audit_logs
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

-- Data access log partitions
CREATE TABLE data_access_log_2025_01 PARTITION OF data_access_log
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE data_access_log_2025_02 PARTITION OF data_access_log
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
CREATE TABLE data_access_log_2025_03 PARTITION OF data_access_log
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');
CREATE TABLE data_access_log_2025_04 PARTITION OF data_access_log
    FOR VALUES FROM ('2025-04-01') TO ('2025-05-01');
CREATE TABLE data_access_log_2025_05 PARTITION OF data_access_log
    FOR VALUES FROM ('2025-05-01') TO ('2025-06-01');
CREATE TABLE data_access_log_2025_06 PARTITION OF data_access_log
    FOR VALUES FROM ('2025-06-01') TO ('2025-07-01');
CREATE TABLE data_access_log_2025_07 PARTITION OF data_access_log
    FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');
CREATE TABLE data_access_log_2025_08 PARTITION OF data_access_log
    FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE data_access_log_2025_09 PARTITION OF data_access_log
    FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE data_access_log_2025_10 PARTITION OF data_access_log
    FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE data_access_log_2025_11 PARTITION OF data_access_log
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE data_access_log_2025_12 PARTITION OF data_access_log
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

-- Performance metrics partitions
CREATE TABLE performance_metrics_2025_01 PARTITION OF performance_metrics
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE performance_metrics_2025_02 PARTITION OF performance_metrics
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
CREATE TABLE performance_metrics_2025_03 PARTITION OF performance_metrics
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');
CREATE TABLE performance_metrics_2025_04 PARTITION OF performance_metrics
    FOR VALUES FROM ('2025-04-01') TO ('2025-05-01');
CREATE TABLE performance_metrics_2025_05 PARTITION OF performance_metrics
    FOR VALUES FROM ('2025-05-01') TO ('2025-06-01');
CREATE TABLE performance_metrics_2025_06 PARTITION OF performance_metrics
    FOR VALUES FROM ('2025-06-01') TO ('2025-07-01');
CREATE TABLE performance_metrics_2025_07 PARTITION OF performance_metrics
    FOR VALUES FROM ('2025-07-01') TO ('2025-08-01');
CREATE TABLE performance_metrics_2025_08 PARTITION OF performance_metrics
    FOR VALUES FROM ('2025-08-01') TO ('2025-09-01');
CREATE TABLE performance_metrics_2025_09 PARTITION OF performance_metrics
    FOR VALUES FROM ('2025-09-01') TO ('2025-10-01');
CREATE TABLE performance_metrics_2025_10 PARTITION OF performance_metrics
    FOR VALUES FROM ('2025-10-01') TO ('2025-11-01');
CREATE TABLE performance_metrics_2025_11 PARTITION OF performance_metrics
    FOR VALUES FROM ('2025-11-01') TO ('2025-12-01');
CREATE TABLE performance_metrics_2025_12 PARTITION OF performance_metrics
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');

-- =============================================
-- INDEXES FOR PERFORMANCE OPTIMIZATION
-- =============================================

-- Core Standard Payers Indexes
CREATE INDEX CONCURRENTLY idx_core_standard_payers_code 
ON core_standard_payers(payer_code);
CREATE INDEX CONCURRENTLY idx_core_standard_payers_name 
ON core_standard_payers(payer_name);

-- RVU Data Indexes
CREATE INDEX CONCURRENTLY idx_rvu_data_category 
ON rvu_data(category, subcategory);
CREATE INDEX CONCURRENTLY idx_rvu_data_effective_date 
ON rvu_data(effective_date, end_date);
CREATE INDEX CONCURRENTLY idx_rvu_data_status ON rvu_data(status, procedure_code);

-- Facilities Indexes
CREATE INDEX CONCURRENTLY idx_facilities_org_region 
ON facilities(org_id, region_id);
CREATE INDEX CONCURRENTLY idx_facilities_state_city 
ON facilities(state, city);

-- Financial Classes Indexes
CREATE INDEX CONCURRENTLY idx_facility_financial_classes_payer 
ON facility_financial_classes(payer_id);
CREATE INDEX CONCURRENTLY idx_facility_financial_classes_effective 
ON facility_financial_classes(effective_date, end_date);

-- Claims Indexes (optimized for high-performance processing)
CREATE INDEX idx_claims_facility_status ON claims(facility_id, processing_status);
CREATE INDEX idx_claims_correlation_id ON claims(correlation_id);
CREATE INDEX idx_claims_batch_id ON claims(batch_id);
CREATE INDEX idx_claims_patient_name ON claims(patient_last_name, patient_first_name);
CREATE INDEX idx_claims_dob ON claims(patient_date_of_birth);
CREATE INDEX idx_claims_financial_class ON claims(facility_id, financial_class);

-- Partial indexes for active processing
CREATE INDEX idx_claims_active_processing 
    ON claims(facility_id, created_at) 
    WHERE processing_status IN ('pending', 'processing');

CREATE INDEX idx_claims_failed_processing 
    ON claims(facility_id, processing_status, created_at) 
    WHERE processing_status = 'failed';

-- Claims Diagnosis Indexes
CREATE INDEX idx_claims_diagnosis_code 
ON claims_diagnosis(diagnosis_code);

-- Claim Line Items Indexes
CREATE INDEX idx_claim_line_items_claim_procedure ON claim_line_items(claim_id, procedure_code);
CREATE INDEX idx_claim_line_items_service_date ON claim_line_items(service_date);
CREATE INDEX idx_claim_line_items_provider ON claim_line_items(rendering_provider_npi);
CREATE INDEX idx_claim_line_items_procedure ON claim_line_items(procedure_code);
CREATE INDEX idx_claim_line_items_provider_legacy ON claim_line_items(rendering_provider_id);

-- Batch processing indexes
CREATE INDEX idx_batch_metadata_status_priority ON batch_metadata(status, priority, submitted_at);
CREATE INDEX idx_batch_metadata_facility_status ON batch_metadata(facility_id, status);

-- Failed Claims Indexes
CREATE INDEX idx_failed_claims_facility_status 
ON failed_claims(facility_id, resolution_status);
CREATE INDEX idx_failed_claims_category 
ON failed_claims(failure_category, processing_stage);
CREATE INDEX idx_failed_claims_resolution ON failed_claims(resolution_status, assigned_to);
CREATE INDEX idx_failed_claims_category_facility ON failed_claims(failure_category, facility_id);

-- Audit Log Indexes
CREATE INDEX idx_audit_logs_table_operation 
ON audit_logs(table_name, operation);
CREATE INDEX idx_audit_logs_user_timestamp 
ON audit_logs(user_id, operation_timestamp);
CREATE INDEX idx_audit_logs_user_time ON audit_logs(user_id, created_at);
CREATE INDEX idx_audit_logs_phi_access ON audit_logs(accessed_phi, created_at);

-- Data Access Log Indexes
CREATE INDEX idx_data_access_log_user_timestamp 
ON data_access_log(user_id, access_timestamp);
CREATE INDEX idx_data_access_log_table 
ON data_access_log(table_name, access_type);

-- Performance Metrics Indexes
CREATE INDEX idx_performance_metrics_facility_date 
ON performance_metrics(facility_id, metric_date);
CREATE INDEX idx_performance_metrics_type 
ON performance_metrics(metric_type, metric_date);
CREATE INDEX idx_performance_metrics_type_time ON performance_metrics(metric_type, recorded_at);
CREATE INDEX idx_performance_metrics_service ON performance_metrics(service_name, recorded_at);

-- Daily Processing Summary Indexes
CREATE INDEX idx_daily_processing_summary_facility 
ON daily_processing_summary(facility_id, summary_date);

-- Validation rules indexes
CREATE INDEX idx_validation_rules_active ON validation_rules(is_active, rule_type);
CREATE INDEX idx_validation_rules_severity ON validation_rules(severity, is_active);

-- Composite indexes for complex queries
CREATE INDEX idx_claims_composite_processing 
    ON claims(facility_id, processing_status, priority, created_at);

CREATE INDEX idx_line_items_composite_rvu 
    ON claim_line_items(procedure_code, service_date, rvu_total);

-- =============================================
-- TRIGGERS AND FUNCTIONS
-- =============================================

-- Update timestamp triggers
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update triggers to all tables with updated_at columns
CREATE TRIGGER update_core_standard_payers_updated_at BEFORE UPDATE ON core_standard_payers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_rvu_data_updated_at BEFORE UPDATE ON rvu_data
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_facility_departments_updated_at BEFORE UPDATE ON facility_departments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_validation_rules_updated_at BEFORE UPDATE ON validation_rules
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_batch_metadata_updated_at BEFORE UPDATE ON batch_metadata
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_claims_updated_at BEFORE UPDATE ON claims
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_claim_line_items_updated_at BEFORE UPDATE ON claim_line_items
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_failed_claims_updated_at BEFORE UPDATE ON failed_claims
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_failed_claims_patterns_updated_at BEFORE UPDATE ON failed_claims_patterns
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to calculate reimbursement rate
CREATE OR REPLACE FUNCTION calculate_reimbursement_rate(
    charge_amount DECIMAL(10,2),
    reimbursement_amount DECIMAL(10,2)
)
RETURNS DECIMAL(5,4) AS $$
DECLARE
    rate DECIMAL(5,4);
BEGIN
    IF charge_amount = 0 OR charge_amount IS NULL THEN
        rate := 0;
    ELSE
        rate := reimbursement_amount / charge_amount;
    END IF;
    
    RETURN rate;
END;
$$ LANGUAGE plpgsql;

-- =============================================
-- INITIAL DATA LOAD
-- =============================================

-- Load Standard Payers
INSERT INTO core_standard_payers (payer_name, payer_code, payer_type, active) VALUES
('Medicare', '1', 'Government', TRUE),
('Medicaid', '2', 'Government', TRUE),
('BlueCross', '3', 'Commercial', TRUE),
('Others', '4', 'Commercial', TRUE),
('Self Payer', '5', 'Self-Pay', TRUE),
('HMO', '6', 'Commercial', TRUE),
('Tricare', '7', 'Government', TRUE),
('Commercial', '8', 'Commercial', TRUE),
('Workers Comp', '9', 'Workers Compensation', TRUE),
('MC Advantage', '10', 'Government', TRUE);

-- =============================================
-- SEQUENCE OWNERSHIP
-- =============================================

-- Set sequence ownership for partitioned tables
ALTER SEQUENCE claims_id_seq OWNED BY claims.id;
ALTER SEQUENCE claims_diagnosis_id_seq OWNED BY claims_diagnosis.id;
ALTER SEQUENCE claim_line_items_id_seq OWNED BY claim_line_items.id;
ALTER SEQUENCE audit_logs_id_seq OWNED BY audit_logs.id;
ALTER SEQUENCE data_access_log_id_seq OWNED BY data_access_log.id;

-- =============================================
-- PERFORMANCE OPTIMIZATION SETTINGS
-- =============================================

-- Connection pooling optimization settings
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '2GB';
ALTER SYSTEM SET effective_cache_size = '6GB';
ALTER SYSTEM SET work_mem = '32MB';
ALTER SYSTEM SET maintenance_work_mem = '512MB';
ALTER SYSTEM SET max_wal_size = '2GB';
ALTER SYSTEM SET wal_buffers = '32MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;

-- Enable query optimization
ALTER SYSTEM SET enable_partitionwise_join = on;
ALTER SYSTEM SET enable_partitionwise_aggregate = on;
ALTER SYSTEM SET enable_parallel_append = on;
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;
ALTER SYSTEM SET max_parallel_workers = 8;

-- Enable JIT compilation for complex queries
ALTER SYSTEM SET jit = on;
ALTER SYSTEM SET jit_above_cost = 100000;

-- Security settings
ALTER SYSTEM SET ssl = off;
ALTER SYSTEM SET log_connections = on;
ALTER SYSTEM SET log_disconnections = on;
ALTER SYSTEM SET log_statement = 'mod';

-- Performance monitoring
ALTER SYSTEM SET track_activities = on;
ALTER SYSTEM SET track_counts = on;
ALTER SYSTEM SET track_io_timing = on;

COMMENT ON DATABASE postgres IS 'Smart Pro Claims - Complete PostgreSQL claims processing and analytics database';

SELECT 'Smart Pro Claims PostgreSQL database schema created successfully.' AS status;
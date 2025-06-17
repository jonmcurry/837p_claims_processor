-- High-Performance PostgreSQL Schema for Claims Processing Staging Database
-- Optimized for 100,000+ claims/15 seconds processing target
-- HIPAA-compliant with encryption and audit logging

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

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

-- Facilities lookup table
CREATE TABLE facilities (
    id SERIAL PRIMARY KEY,
    facility_id VARCHAR(20) UNIQUE NOT NULL,
    facility_name VARCHAR(200) NOT NULL,
    npi VARCHAR(10) NOT NULL,
    tax_id VARCHAR(20) NOT NULL,
    address_line1 VARCHAR(200) NOT NULL,
    address_line2 VARCHAR(200),
    city VARCHAR(100) NOT NULL,
    state VARCHAR(2) NOT NULL,
    zip_code VARCHAR(10) NOT NULL,
    phone VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Providers lookup table
CREATE TABLE providers (
    id SERIAL PRIMARY KEY,
    npi VARCHAR(10) UNIQUE NOT NULL,
    provider_name VARCHAR(200) NOT NULL,
    specialty_code VARCHAR(10),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- RVU lookup table for procedure codes
CREATE TABLE rvu_data (
    id SERIAL PRIMARY KEY,
    procedure_code VARCHAR(10) NOT NULL,
    modifier VARCHAR(2),
    year INTEGER NOT NULL,
    work_rvu DECIMAL(8,4) NOT NULL,
    practice_expense_rvu DECIMAL(8,4) NOT NULL,
    malpractice_rvu DECIMAL(8,4) NOT NULL,
    total_rvu DECIMAL(8,4) NOT NULL,
    conversion_factor DECIMAL(8,4) DEFAULT 36.04,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Business validation rules
CREATE TABLE validation_rules (
    id SERIAL PRIMARY KEY,
    rule_name VARCHAR(100) NOT NULL,
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
    facility_id VARCHAR(20) REFERENCES facilities(facility_id),
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

-- Main claims table (staging)
CREATE TABLE claims (
    id BIGSERIAL PRIMARY KEY,
    claim_id VARCHAR(50) NOT NULL,
    facility_id VARCHAR(20) NOT NULL REFERENCES facilities(facility_id),
    patient_account_number VARCHAR(50) NOT NULL,
    medical_record_number VARCHAR(50),
    
    -- Patient demographics (encrypted fields marked with _encrypted suffix)
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
    billing_provider_npi VARCHAR(10) NOT NULL REFERENCES providers(npi),
    billing_provider_name VARCHAR(200) NOT NULL,
    attending_provider_npi VARCHAR(10) REFERENCES providers(npi),
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
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    processed_at TIMESTAMP,
    
    -- Retry information
    retry_count INTEGER DEFAULT 0 NOT NULL,
    last_retry_at TIMESTAMP,
    
    CONSTRAINT uq_claim_facility UNIQUE (claim_id, facility_id)
);

-- Claim line items table
CREATE TABLE claim_line_items (
    id BIGSERIAL PRIMARY KEY,
    claim_id BIGINT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    line_number INTEGER NOT NULL,
    
    -- Service information
    service_date DATE NOT NULL,
    procedure_code VARCHAR(10) NOT NULL,
    procedure_description VARCHAR(500),
    modifier_codes JSONB DEFAULT '[]',
    
    -- Quantity and charges
    units INTEGER NOT NULL,
    charge_amount DECIMAL(10,2) NOT NULL,
    
    -- Provider information
    rendering_provider_npi VARCHAR(10) REFERENCES providers(npi),
    rendering_provider_name VARCHAR(200),
    
    -- RVU information
    rvu_work DECIMAL(8,4),
    rvu_practice_expense DECIMAL(8,4),
    rvu_malpractice DECIMAL(8,4),
    rvu_total DECIMAL(8,4),
    expected_reimbursement DECIMAL(10,2),
    
    -- Diagnosis pointers
    diagnosis_pointers JSONB,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT uq_claim_line_number UNIQUE (claim_id, line_number)
);

-- Failed claims tracking
CREATE TABLE failed_claims (
    id BIGSERIAL PRIMARY KEY,
    original_claim_id BIGINT,
    claim_reference VARCHAR(50) NOT NULL,
    facility_id VARCHAR(20) NOT NULL REFERENCES facilities(facility_id),
    
    -- Failure information
    failure_category failure_category NOT NULL,
    failure_reason TEXT NOT NULL,
    failure_details JSONB NOT NULL,
    
    -- Original claim data
    claim_data JSONB NOT NULL,
    
    -- Resolution tracking
    resolution_status VARCHAR(50) DEFAULT 'pending' NOT NULL,
    assigned_to VARCHAR(100),
    resolution_notes TEXT,
    resolved_by VARCHAR(100),
    resolved_at TIMESTAMP,
    
    -- Reprocessing information
    can_reprocess BOOLEAN DEFAULT TRUE NOT NULL,
    reprocess_count INTEGER DEFAULT 0 NOT NULL,
    last_reprocess_at TIMESTAMP,
    
    -- Financial impact
    charge_amount DECIMAL(12,2) NOT NULL,
    expected_reimbursement DECIMAL(12,2),
    
    -- Timestamps
    failed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Audit logging for HIPAA compliance
CREATE TABLE audit_logs (
    id BIGSERIAL,
    
    -- Action information
    action_type VARCHAR(50) NOT NULL,
    action_description TEXT NOT NULL,
    
    -- User information
    user_id VARCHAR(100) NOT NULL,
    user_name VARCHAR(200) NOT NULL,
    user_role VARCHAR(50) NOT NULL,
    ip_address INET NOT NULL,
    user_agent VARCHAR(500),
    
    -- Resource information
    resource_type VARCHAR(50) NOT NULL,
    resource_id VARCHAR(100),
    
    -- PHI access tracking
    accessed_phi BOOLEAN DEFAULT FALSE NOT NULL,
    phi_fields_accessed JSONB,
    business_justification TEXT,
    
    -- Request information
    request_id VARCHAR(100) NOT NULL,
    session_id VARCHAR(100),
    
    -- Additional context
    additional_context JSONB,
    
    -- Timestamp (partitioned by month for performance)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
) PARTITION BY RANGE (created_at);

-- Performance metrics tracking
CREATE TABLE performance_metrics (
    id BIGSERIAL,
    
    -- Metric information
    metric_type VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(20,6) NOT NULL,
    unit VARCHAR(20) NOT NULL,
    
    -- Context
    facility_id VARCHAR(20),
    batch_id VARCHAR(100),
    service_name VARCHAR(50) NOT NULL,
    
    -- Tags for filtering
    tags JSONB,
    
    -- Timestamp (partitioned by month for performance)
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    
    -- Primary key must include partition key for partitioned tables
    PRIMARY KEY (id, recorded_at)
) PARTITION BY RANGE (recorded_at);

-- Create monthly partitions for audit_logs (example for 2025)
CREATE TABLE audit_logs_2025_01 PARTITION OF audit_logs
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE audit_logs_2025_02 PARTITION OF audit_logs
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
CREATE TABLE audit_logs_2025_03 PARTITION OF audit_logs
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');

-- Create monthly partitions for performance_metrics
CREATE TABLE performance_metrics_2025_01 PARTITION OF performance_metrics
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE performance_metrics_2025_02 PARTITION OF performance_metrics
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
CREATE TABLE performance_metrics_2025_03 PARTITION OF performance_metrics
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');

-- High-performance indexes for claims processing
-- Primary processing indexes
CREATE INDEX CONCURRENTLY idx_claims_facility_status ON claims(facility_id, processing_status);
CREATE INDEX CONCURRENTLY idx_claims_correlation_id ON claims(correlation_id);
CREATE INDEX CONCURRENTLY idx_claims_created_at_btree ON claims USING BTREE(created_at);
CREATE INDEX CONCURRENTLY idx_claims_batch_id ON claims(batch_id);

-- Partial indexes for active processing
CREATE INDEX CONCURRENTLY idx_claims_active_processing 
    ON claims(facility_id, created_at) 
    WHERE processing_status IN ('pending', 'processing');

CREATE INDEX CONCURRENTLY idx_claims_failed_processing 
    ON claims(facility_id, processing_status, created_at) 
    WHERE processing_status = 'failed';

-- Line items indexes
CREATE INDEX CONCURRENTLY idx_line_items_claim_procedure ON claim_line_items(claim_id, procedure_code);
CREATE INDEX CONCURRENTLY idx_line_items_service_date ON claim_line_items(service_date);
CREATE INDEX CONCURRENTLY idx_line_items_provider ON claim_line_items(rendering_provider_npi);

-- Batch processing indexes
CREATE INDEX CONCURRENTLY idx_batch_metadata_status_priority ON batch_metadata(status, priority, submitted_at);
CREATE INDEX CONCURRENTLY idx_batch_metadata_facility_status ON batch_metadata(facility_id, status);

-- Failed claims indexes
CREATE INDEX CONCURRENTLY idx_failed_claims_resolution ON failed_claims(resolution_status, assigned_to);
CREATE INDEX CONCURRENTLY idx_failed_claims_category_facility ON failed_claims(failure_category, facility_id);
CREATE INDEX CONCURRENTLY idx_failed_claims_failed_at ON failed_claims(failed_at);

-- Audit and metrics indexes
CREATE INDEX CONCURRENTLY idx_audit_logs_user_time ON audit_logs(user_id, created_at);
CREATE INDEX CONCURRENTLY idx_audit_logs_phi_access ON audit_logs(accessed_phi, created_at);
CREATE INDEX CONCURRENTLY idx_performance_metrics_type_time ON performance_metrics(metric_type, recorded_at);
CREATE INDEX CONCURRENTLY idx_performance_metrics_service ON performance_metrics(service_name, recorded_at);

-- Lookup table indexes
CREATE INDEX CONCURRENTLY idx_facilities_facility_id ON facilities(facility_id);
CREATE INDEX CONCURRENTLY idx_providers_npi ON providers(npi);
CREATE INDEX CONCURRENTLY idx_rvu_data_procedure_year ON rvu_data(procedure_code, year);

-- Composite indexes for complex queries
CREATE INDEX CONCURRENTLY idx_claims_composite_processing 
    ON claims(facility_id, processing_status, priority, created_at);

CREATE INDEX CONCURRENTLY idx_line_items_composite_rvu 
    ON claim_line_items(procedure_code, service_date, rvu_total);

-- Update timestamp triggers
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update triggers to all tables with updated_at columns
CREATE TRIGGER update_facilities_updated_at BEFORE UPDATE ON facilities
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_providers_updated_at BEFORE UPDATE ON providers
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

-- Connection pooling optimization settings
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '1GB';
ALTER SYSTEM SET effective_cache_size = '3GB';
ALTER SYSTEM SET work_mem = '16MB';
ALTER SYSTEM SET maintenance_work_mem = '256MB';
ALTER SYSTEM SET checkpoint_segments = 64;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;

-- Enable query optimization
ALTER SYSTEM SET enable_partitionwise_join = on;
ALTER SYSTEM SET enable_partitionwise_aggregate = on;
ALTER SYSTEM SET enable_parallel_append = on;
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;

-- Security settings
ALTER SYSTEM SET ssl = on;
ALTER SYSTEM SET log_connections = on;
ALTER SYSTEM SET log_disconnections = on;
ALTER SYSTEM SET log_statement = 'mod';

COMMENT ON DATABASE postgres IS 'High-performance claims processing staging database optimized for 100k+ claims/15s throughput';
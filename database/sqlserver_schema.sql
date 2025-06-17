-- Smart Pro Claims Database Schema for SQL Server
-- High-Performance analytics and reporting database
-- Includes partitioning, columnstore indexes, and optimized for healthcare claims processing

-- Create the database
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'smart_pro_claims')
BEGIN
    CREATE DATABASE smart_pro_claims;
END
GO

USE smart_pro_claims;
GO

-- Create filegroups for partitioning
ALTER DATABASE smart_pro_claims 
ADD FILEGROUP Claims_FG1;
ALTER DATABASE smart_pro_claims 
ADD FILEGROUP Claims_FG2;
ALTER DATABASE smart_pro_claims 
ADD FILEGROUP Claims_FG3;
ALTER DATABASE smart_pro_claims 
ADD FILEGROUP Claims_FG4;
GO

-- Add files to filegroups (adjust paths as needed)
ALTER DATABASE smart_pro_claims 
ADD FILE (
    NAME = 'Claims_FG1_Data',
    FILENAME = 'C:\SQLData\Claims_FG1_Data.ndf',
    SIZE = 1GB,
    MAXSIZE = 50GB,
    FILEGROWTH = 256MB
) TO FILEGROUP Claims_FG1;

ALTER DATABASE smart_pro_claims 
ADD FILE (
    NAME = 'Claims_FG2_Data',
    FILENAME = 'C:\SQLData\Claims_FG2_Data.ndf', 
    SIZE = 1GB,
    MAXSIZE = 50GB,
    FILEGROWTH = 256MB
) TO FILEGROUP Claims_FG2;

ALTER DATABASE smart_pro_claims 
ADD FILE (
    NAME = 'Claims_FG3_Data',
    FILENAME = 'C:\SQLData\Claims_FG3_Data.ndf',
    SIZE = 1GB, 
    MAXSIZE = 50GB,
    FILEGROWTH = 256MB
) TO FILEGROUP Claims_FG3;

ALTER DATABASE smart_pro_claims 
ADD FILE (
    NAME = 'Claims_FG4_Data',
    FILENAME = 'C:\SQLData\Claims_FG4_Data.ndf',
    SIZE = 1GB,
    MAXSIZE = 50GB, 
    FILEGROWTH = 256MB
) TO FILEGROUP Claims_FG4;
GO

-- Create partition function for date-based partitioning
CREATE PARTITION FUNCTION ClaimsDatePartitionFunction (DATETIME2)
AS RANGE RIGHT FOR VALUES (
    '2024-01-01', '2024-04-01', '2024-07-01', '2024-10-01',
    '2025-01-01', '2025-04-01', '2025-07-01', '2025-10-01'
);
GO

-- Create partition scheme
CREATE PARTITION SCHEME ClaimsDatePartitionScheme
AS PARTITION ClaimsDatePartitionFunction
TO (Claims_FG1, Claims_FG2, Claims_FG3, Claims_FG4, 
    Claims_FG1, Claims_FG2, Claims_FG3, Claims_FG4);
GO

-- =============================================
-- CORE REFERENCE TABLES
-- =============================================

-- Core Standard Payers
CREATE TABLE dbo.core_standard_payers (
    payer_id INT IDENTITY(1,1) PRIMARY KEY,
    payer_name VARCHAR(200) NOT NULL,
    payer_code CHAR(10) NOT NULL UNIQUE,
    payer_type VARCHAR(50) NULL,
    active BIT NOT NULL DEFAULT 1,
    created_at DATETIME2(7) NOT NULL DEFAULT GETUTCDATE(),
    updated_at DATETIME2(7) NOT NULL DEFAULT GETUTCDATE()
);
GO

-- RVU Data
CREATE TABLE dbo.rvu_data (
    procedure_code VARCHAR(10) NOT NULL PRIMARY KEY,
    description VARCHAR(500) NULL,
    category VARCHAR(50) NULL,
    subcategory VARCHAR(50) NULL,
    work_rvu DECIMAL(8, 4) NULL,
    practice_expense_rvu DECIMAL(8, 4) NULL,
    malpractice_rvu DECIMAL(8, 4) NULL,
    total_rvu DECIMAL(8, 4) NULL,
    conversion_factor DECIMAL(8, 2) NULL,
    non_facility_pe_rvu DECIMAL(8, 4) NULL,
    facility_pe_rvu DECIMAL(8, 4) NULL,
    effective_date DATE NULL,
    end_date DATE NULL,
    status VARCHAR(20) NULL,
    global_period VARCHAR(10) NULL,
    professional_component BIT NULL,
    technical_component BIT NULL,
    bilateral_surgery BIT NULL,
    created_at DATETIME2(7) NOT NULL DEFAULT GETUTCDATE(),
    updated_at DATETIME2(7) NOT NULL DEFAULT GETUTCDATE()
);
GO

-- =============================================
-- ORGANIZATIONAL HIERARCHY
-- =============================================

-- Facility Organization (Top Level)
CREATE TABLE dbo.facility_organization (
    org_id INT IDENTITY(1,1) PRIMARY KEY,
    org_name VARCHAR(100) NOT NULL,
    installed_date DATETIME NOT NULL DEFAULT GETDATE(),
    updated_by INT NULL,
    active BIT NOT NULL DEFAULT 1,
    created_at DATETIME2(7) NOT NULL DEFAULT GETUTCDATE()
);
GO

-- Facility Region (Optional Middle Level)
CREATE TABLE dbo.facility_region (
    region_id INT IDENTITY(1,1) PRIMARY KEY,
    region_name VARCHAR(100) NOT NULL,
    org_id INT NOT NULL,
    active BIT NOT NULL DEFAULT 1,
    created_at DATETIME2(7) NOT NULL DEFAULT GETUTCDATE(),
    
    CONSTRAINT FK_facility_region_org FOREIGN KEY (org_id) 
        REFERENCES dbo.facility_organization(org_id)
);
GO

-- Facilities (Bottom Level)
CREATE TABLE dbo.facilities (
    facility_id VARCHAR(20) PRIMARY KEY,
    facility_name VARCHAR(100) NOT NULL,
    installed_date DATETIME NOT NULL DEFAULT GETDATE(),
    beds INT NULL,
    city VARCHAR(24) NULL,
    state CHAR(2) NULL,
    updated_date DATETIME NOT NULL DEFAULT GETDATE(),
    updated_by INT NULL,
    region_id INT NULL,
    fiscal_month INT NULL CHECK (fiscal_month BETWEEN 1 AND 12),
    org_id INT NOT NULL,
    active BIT NOT NULL DEFAULT 1,
    
    CONSTRAINT FK_facilities_region FOREIGN KEY (region_id) 
        REFERENCES dbo.facility_region(region_id),
    CONSTRAINT FK_facilities_org FOREIGN KEY (org_id) 
        REFERENCES dbo.facility_organization(org_id)
);
GO

-- =============================================
-- FACILITY CONFIGURATION TABLES
-- =============================================

-- Facility Financial Classes
CREATE TABLE dbo.facility_financial_classes (
    facility_id VARCHAR(20) NOT NULL,
    financial_class_id VARCHAR(10) NOT NULL,
    financial_class_name VARCHAR(100) NOT NULL,
    payer_id INT NOT NULL,
    reimbursement_rate DECIMAL(5,4) NULL,
    processing_priority VARCHAR(10) NULL,
    auto_posting_enabled BIT NOT NULL DEFAULT 0,
    active BIT NOT NULL DEFAULT 1,
    effective_date DATE NOT NULL,
    end_date DATE NULL,
    created_at DATETIME2(7) NOT NULL DEFAULT GETUTCDATE(),
    HCC CHAR(3) NULL,
    
    PRIMARY KEY (facility_id, financial_class_id),
    CONSTRAINT FK_facility_financial_classes_facility FOREIGN KEY (facility_id) 
        REFERENCES dbo.facilities(facility_id),
    CONSTRAINT FK_facility_financial_classes_payer FOREIGN KEY (payer_id) 
        REFERENCES dbo.core_standard_payers(payer_id)
);
GO

-- Facility Place of Service
CREATE TABLE dbo.facility_place_of_service (
    facility_id VARCHAR(20) NOT NULL,
    place_of_service VARCHAR(2) NOT NULL,
    place_of_service_name VARCHAR(30) NOT NULL,
    origin INT NULL,
    active BIT NOT NULL DEFAULT 1,
    created_at DATETIME2(7) NOT NULL DEFAULT GETUTCDATE(),
    
    PRIMARY KEY (facility_id, place_of_service),
    CONSTRAINT FK_facility_place_of_service_facility FOREIGN KEY (facility_id) 
        REFERENCES dbo.facilities(facility_id)
);
GO

-- Facility Departments
CREATE TABLE dbo.facility_departments (
    department_code VARCHAR(10) NOT NULL,
    department_name VARCHAR(50) NOT NULL,
    facility_id VARCHAR(20) NOT NULL,
    active BIT NOT NULL DEFAULT 1,
    created_at DATETIME2(7) NOT NULL DEFAULT GETUTCDATE(),
    updated_at DATETIME2(7) NOT NULL DEFAULT GETUTCDATE(),
    
    PRIMARY KEY (facility_id, department_code),
    CONSTRAINT FK_facility_departments_facility FOREIGN KEY (facility_id) 
        REFERENCES dbo.facilities(facility_id)
);
GO

-- Facility Coders
CREATE TABLE dbo.facility_coders (
    facility_id VARCHAR(20) NOT NULL,
    coder_id VARCHAR(50) NOT NULL,
    coder_first_name VARCHAR(50) NOT NULL,
    coder_last_name VARCHAR(50) NOT NULL,
    active BIT NOT NULL DEFAULT 1,
    created_at DATETIME2(7) NOT NULL DEFAULT GETUTCDATE(),
    
    PRIMARY KEY (facility_id, coder_id),
    CONSTRAINT FK_facility_coders_facility FOREIGN KEY (facility_id) 
        REFERENCES dbo.facilities(facility_id)
);
GO

-- Physicians
CREATE TABLE dbo.physicians (
    rendering_provider_id VARCHAR(50) PRIMARY KEY,
    last_name VARCHAR(50) NOT NULL,
    first_name VARCHAR(50) NOT NULL,
    npi VARCHAR(10) NULL,
    specialty_code VARCHAR(10) NULL,
    active BIT NOT NULL DEFAULT 1,
    created_at DATETIME2(7) NOT NULL DEFAULT GETUTCDATE()
);
GO

-- =============================================
-- CLAIMS PROCESSING TABLES
-- =============================================

-- Claims (Main claims table)
CREATE TABLE dbo.claims (
    facility_id VARCHAR(20) NOT NULL,
    patient_account_number VARCHAR(50) NOT NULL,
    medical_record_number VARCHAR(50) NULL,
    patient_name VARCHAR(100) NULL,
    first_name VARCHAR(50) NULL,
    last_name VARCHAR(50) NULL,
    date_of_birth DATE NULL,
    gender VARCHAR(1) NULL CHECK (gender IN ('M', 'F', 'U')),
    financial_class_id VARCHAR(10) NULL,
    secondary_insurance VARCHAR(10) NULL,
    active BIT NOT NULL DEFAULT 1,
    created_at DATETIME2(7) NOT NULL DEFAULT GETUTCDATE(),
    updated_at DATETIME2(7) NOT NULL DEFAULT GETUTCDATE(),
    
    PRIMARY KEY (facility_id, patient_account_number),
    CONSTRAINT FK_claims_facility FOREIGN KEY (facility_id) 
        REFERENCES dbo.facilities(facility_id),
    CONSTRAINT FK_claims_financial_class FOREIGN KEY (facility_id, financial_class_id) 
        REFERENCES dbo.facility_financial_classes(facility_id, financial_class_id),
    CONSTRAINT FK_claims_secondary_insurance FOREIGN KEY (facility_id, secondary_insurance) 
        REFERENCES dbo.facility_financial_classes(facility_id, financial_class_id)
) ON ClaimsDatePartitionScheme(created_at);
GO

-- Claims Diagnosis
CREATE TABLE dbo.claims_diagnosis (
    facility_id VARCHAR(20) NOT NULL,
    patient_account_number VARCHAR(50) NOT NULL,
    diagnosis_sequence INT NOT NULL,
    diagnosis_code VARCHAR(20) NOT NULL,
    diagnosis_description VARCHAR(255) NULL,
    diagnosis_type VARCHAR(10) NULL,
    created_at DATETIME2(7) NOT NULL DEFAULT GETUTCDATE(),
    
    PRIMARY KEY (facility_id, patient_account_number, diagnosis_sequence),
    CONSTRAINT FK_claims_diagnosis_claims FOREIGN KEY (facility_id, patient_account_number) 
        REFERENCES dbo.claims(facility_id, patient_account_number)
) ON ClaimsDatePartitionScheme(created_at);
GO

-- Claims Line Items
CREATE TABLE dbo.claims_line_items (
    facility_id VARCHAR(20) NOT NULL,
    patient_account_number VARCHAR(50) NOT NULL,
    line_number INT NOT NULL,
    procedure_code VARCHAR(10) NOT NULL,
    modifier1 VARCHAR(2) NULL,
    modifier2 VARCHAR(2) NULL,
    modifier3 VARCHAR(2) NULL,
    modifier4 VARCHAR(2) NULL,
    units INT NOT NULL DEFAULT 1,
    charge_amount NUMERIC(10,2) NOT NULL,
    service_from_date DATE NULL,
    service_to_date DATE NULL,
    diagnosis_pointer VARCHAR(4) NULL,
    place_of_service VARCHAR(2) NULL,
    revenue_code VARCHAR(4) NULL,
    created_at DATETIME2(7) NOT NULL DEFAULT GETUTCDATE(),
    rvu_value NUMERIC(8,4) NULL,
    reimbursement_amount NUMERIC(10,2) NULL,
    rendering_provider_id VARCHAR(50) NULL,
    
    PRIMARY KEY (facility_id, patient_account_number, line_number),
    CONSTRAINT FK_claims_line_items_claims FOREIGN KEY (facility_id, patient_account_number) 
        REFERENCES dbo.claims(facility_id, patient_account_number),
    CONSTRAINT FK_claims_line_items_place_of_service FOREIGN KEY (facility_id, place_of_service) 
        REFERENCES dbo.facility_place_of_service(facility_id, place_of_service),
    CONSTRAINT FK_claims_line_items_provider FOREIGN KEY (rendering_provider_id) 
        REFERENCES dbo.physicians(rendering_provider_id),
    CONSTRAINT FK_claims_line_items_rvu FOREIGN KEY (procedure_code) 
        REFERENCES dbo.rvu_data(procedure_code)
) ON ClaimsDatePartitionScheme(created_at);
GO

-- =============================================
-- FAILED CLAIMS MANAGEMENT
-- =============================================

-- Failed Claims Patterns
CREATE TABLE dbo.failed_claims_patterns (
    pattern_id VARCHAR(50) PRIMARY KEY,
    pattern_name VARCHAR(200) NOT NULL,
    pattern_description NVARCHAR(1000) NULL,
    failure_category VARCHAR(50) NULL,
    severity_level VARCHAR(20) NULL,
    frequency_score INT NULL,
    pattern_rules NVARCHAR(MAX) NULL,
    auto_repair_possible BIT NULL DEFAULT 0,
    repair_template NVARCHAR(MAX) NULL,
    occurrence_count INT NULL DEFAULT 0,
    resolution_rate DECIMAL(5, 4) NULL,
    average_resolution_time_hours DECIMAL(8, 2) NULL,
    created_at DATETIME2(7) NOT NULL DEFAULT GETUTCDATE(),
    updated_at DATETIME2(7) NOT NULL DEFAULT GETUTCDATE()
);
GO

-- Failed Claims
CREATE TABLE dbo.failed_claims (
    claim_id VARCHAR(50) PRIMARY KEY,
    batch_id VARCHAR(50) NULL,
    facility_id VARCHAR(20) NULL,
    patient_account_number VARCHAR(50) NULL,
    original_data NVARCHAR(MAX) NULL,
    failure_reason NVARCHAR(1000) NOT NULL,
    failure_category VARCHAR(50) NOT NULL,
    processing_stage VARCHAR(50) NOT NULL,
    failed_at DATETIME2(7) NOT NULL DEFAULT GETUTCDATE(),
    repair_suggestions NVARCHAR(MAX) NULL,
    resolution_status VARCHAR(20) NULL DEFAULT 'PENDING',
    assigned_to VARCHAR(100) NULL,
    resolved_at DATETIME2(7) NULL,
    resolution_notes NVARCHAR(2000) NULL,
    resolution_action VARCHAR(50) NULL,
    error_pattern_id VARCHAR(50) NULL,
    priority_level VARCHAR(10) NULL DEFAULT 'MEDIUM',
    impact_level VARCHAR(10) NULL DEFAULT 'MEDIUM',
    potential_revenue_loss DECIMAL(12, 2) NULL,
    created_at DATETIME2(7) NOT NULL DEFAULT GETUTCDATE(),
    updated_at DATETIME2(7) NOT NULL DEFAULT GETUTCDATE(),
    coder_id VARCHAR(50) NULL,
    
    CONSTRAINT FK_failed_claims_facility FOREIGN KEY (facility_id) 
        REFERENCES dbo.facilities(facility_id),
    CONSTRAINT FK_failed_claims_pattern FOREIGN KEY (error_pattern_id) 
        REFERENCES dbo.failed_claims_patterns(pattern_id),
    CONSTRAINT FK_failed_claims_coder FOREIGN KEY (facility_id, coder_id) 
        REFERENCES dbo.facility_coders(facility_id, coder_id)
) ON ClaimsDatePartitionScheme(failed_at);
GO

-- =============================================
-- AUDIT AND LOGGING TABLES
-- =============================================

-- Audit Log
CREATE TABLE dbo.audit_log (
    audit_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    record_id VARCHAR(50) NOT NULL,
    operation VARCHAR(20) NOT NULL,
    user_id VARCHAR(100) NULL,
    session_id VARCHAR(100) NULL,
    ip_address VARCHAR(45) NULL,
    user_agent VARCHAR(500) NULL,
    old_values NVARCHAR(MAX) NULL,
    new_values NVARCHAR(MAX) NULL,
    changed_columns NVARCHAR(500) NULL,
    operation_timestamp DATETIME2(7) NOT NULL DEFAULT GETUTCDATE(),
    reason VARCHAR(500) NULL,
    approval_required BIT NULL DEFAULT 0,
    approved_by VARCHAR(100) NULL,
    approved_at DATETIME2(7) NULL
) ON ClaimsDatePartitionScheme(operation_timestamp);
GO

-- Data Access Log
CREATE TABLE dbo.data_access_log (
    access_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    access_timestamp DATETIME2(7) NOT NULL DEFAULT GETUTCDATE(),
    user_id VARCHAR(100) NOT NULL,
    user_role VARCHAR(50) NULL,
    department VARCHAR(100) NULL,
    table_name VARCHAR(100) NOT NULL,
    record_id VARCHAR(50) NULL,
    access_type VARCHAR(20) NULL,
    data_classification VARCHAR(20) NULL,
    business_justification VARCHAR(500) NULL,
    patient_account_number VARCHAR(50) NULL,
    facility_id VARCHAR(20) NULL,
    ip_address VARCHAR(45) NULL,
    application_name VARCHAR(100) NULL,
    query_executed NVARCHAR(MAX) NULL
) ON ClaimsDatePartitionScheme(access_timestamp);
GO

-- =============================================
-- PERFORMANCE AND REPORTING TABLES
-- =============================================

-- Daily Processing Summary
CREATE TABLE dbo.daily_processing_summary (
    summary_date DATE NOT NULL,
    facility_id VARCHAR(20) NOT NULL,
    total_claims_processed INT NULL,
    total_claims_failed INT NULL,
    total_line_items INT NULL,
    total_charge_amount DECIMAL(15, 2) NULL,
    total_reimbursement_amount DECIMAL(15, 2) NULL,
    average_reimbursement_rate DECIMAL(5, 4) NULL,
    average_processing_time_seconds DECIMAL(8, 2) NULL,
    throughput_claims_per_hour DECIMAL(10, 2) NULL,
    error_rate_percentage DECIMAL(5, 2) NULL,
    ml_accuracy_percentage DECIMAL(5, 2) NULL,
    validation_pass_rate DECIMAL(5, 2) NULL,
    created_at DATETIME2(7) NOT NULL DEFAULT GETUTCDATE(),
    
    PRIMARY KEY (summary_date, facility_id),
    CONSTRAINT FK_daily_processing_summary_facility FOREIGN KEY (facility_id) 
        REFERENCES dbo.facilities(facility_id)
);
GO

-- Performance Metrics
CREATE TABLE dbo.performance_metrics (
    metric_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    metric_date DATETIME2(7) NOT NULL DEFAULT GETUTCDATE(),
    metric_type VARCHAR(50) NOT NULL,
    facility_id VARCHAR(20) NULL,
    claims_per_second DECIMAL(10, 4) NULL,
    records_per_minute DECIMAL(10, 2) NULL,
    cpu_usage_percent DECIMAL(5, 2) NULL,
    memory_usage_mb INT NULL,
    database_response_time_ms DECIMAL(8, 2) NULL,
    queue_depth INT NULL,
    error_rate DECIMAL(5, 4) NULL,
    processing_accuracy DECIMAL(5, 4) NULL,
    revenue_per_claim DECIMAL(10, 2) NULL,
    additional_metrics NVARCHAR(MAX) NULL,
    
    CONSTRAINT FK_performance_metrics_facility FOREIGN KEY (facility_id) 
        REFERENCES dbo.facilities(facility_id)
) ON ClaimsDatePartitionScheme(metric_date);
GO

-- =============================================
-- INDEXES FOR PERFORMANCE OPTIMIZATION
-- =============================================

-- Core Standard Payers Indexes
CREATE NONCLUSTERED INDEX IX_core_standard_payers_code 
ON dbo.core_standard_payers(payer_code);
CREATE NONCLUSTERED INDEX IX_core_standard_payers_name 
ON dbo.core_standard_payers(payer_name);

-- RVU Data Indexes
CREATE NONCLUSTERED INDEX IX_rvu_data_category 
ON dbo.rvu_data(category, subcategory);
CREATE NONCLUSTERED INDEX IX_rvu_data_effective_date 
ON dbo.rvu_data(effective_date, end_date);

-- Facilities Indexes
CREATE NONCLUSTERED INDEX IX_facilities_org_region 
ON dbo.facilities(org_id, region_id);
CREATE NONCLUSTERED INDEX IX_facilities_state_city 
ON dbo.facilities(state, city);

-- Financial Classes Indexes
CREATE NONCLUSTERED INDEX IX_facility_financial_classes_payer 
ON dbo.facility_financial_classes(payer_id);
CREATE NONCLUSTERED INDEX IX_facility_financial_classes_effective 
ON dbo.facility_financial_classes(effective_date, end_date);

-- Claims Indexes
CREATE NONCLUSTERED INDEX IX_claims_patient_name 
ON dbo.claims(last_name, first_name);
CREATE NONCLUSTERED INDEX IX_claims_dob 
ON dbo.claims(date_of_birth);
CREATE NONCLUSTERED INDEX IX_claims_financial_class 
ON dbo.claims(facility_id, financial_class_id);

-- Claims Diagnosis Indexes
CREATE NONCLUSTERED INDEX IX_claims_diagnosis_code 
ON dbo.claims_diagnosis(diagnosis_code);

-- Claims Line Items Indexes
CREATE NONCLUSTERED INDEX IX_claims_line_items_procedure 
ON dbo.claims_line_items(procedure_code);
CREATE NONCLUSTERED INDEX IX_claims_line_items_service_date 
ON dbo.claims_line_items(service_from_date, service_to_date);
CREATE NONCLUSTERED INDEX IX_claims_line_items_provider 
ON dbo.claims_line_items(rendering_provider_id);

-- Failed Claims Indexes
CREATE NONCLUSTERED INDEX IX_failed_claims_facility_status 
ON dbo.failed_claims(facility_id, resolution_status);
CREATE NONCLUSTERED INDEX IX_failed_claims_category 
ON dbo.failed_claims(failure_category, processing_stage);
CREATE NONCLUSTERED INDEX IX_failed_claims_failed_at 
ON dbo.failed_claims(failed_at);

-- Audit Log Indexes
CREATE NONCLUSTERED INDEX IX_audit_log_table_operation 
ON dbo.audit_log(table_name, operation);
CREATE NONCLUSTERED INDEX IX_audit_log_user_timestamp 
ON dbo.audit_log(user_id, operation_timestamp);

-- Data Access Log Indexes
CREATE NONCLUSTERED INDEX IX_data_access_log_user_timestamp 
ON dbo.data_access_log(user_id, access_timestamp);
CREATE NONCLUSTERED INDEX IX_data_access_log_table 
ON dbo.data_access_log(table_name, access_type);

-- Performance Metrics Indexes
CREATE NONCLUSTERED INDEX IX_performance_metrics_facility_date 
ON dbo.performance_metrics(facility_id, metric_date);
CREATE NONCLUSTERED INDEX IX_performance_metrics_type 
ON dbo.performance_metrics(metric_type, metric_date);

-- Daily Processing Summary Indexes
CREATE NONCLUSTERED INDEX IX_daily_processing_summary_facility 
ON dbo.daily_processing_summary(facility_id, summary_date);

-- =============================================
-- COLUMNSTORE INDEXES FOR ANALYTICS
-- =============================================

-- Claims analytics columnstore index
CREATE NONCLUSTERED COLUMNSTORE INDEX NCCI_claims_analytics 
ON dbo.claims (facility_id, patient_account_number, financial_class_id, 
               date_of_birth, gender, created_at, updated_at);

-- Claims line items analytics columnstore index
CREATE NONCLUSTERED COLUMNSTORE INDEX NCCI_claims_line_items_analytics 
ON dbo.claims_line_items (facility_id, patient_account_number, procedure_code, 
                         charge_amount, reimbursement_amount, service_from_date, 
                         place_of_service, rendering_provider_id, rvu_value);

-- Failed claims analytics columnstore index
CREATE NONCLUSTERED COLUMNSTORE INDEX NCCI_failed_claims_analytics 
ON dbo.failed_claims (facility_id, failure_category, processing_stage, 
                     failed_at, resolution_status, potential_revenue_loss);

-- Performance metrics analytics columnstore index
CREATE NONCLUSTERED COLUMNSTORE INDEX NCCI_performance_metrics_analytics 
ON dbo.performance_metrics (facility_id, metric_type, metric_date, 
                           claims_per_second, error_rate, processing_accuracy);

-- =============================================
-- STORED PROCEDURES AND FUNCTIONS
-- =============================================

-- Procedure to refresh all statistics
CREATE PROCEDURE dbo.RefreshAllStatistics
AS
BEGIN
    SET NOCOUNT ON;
    
    UPDATE STATISTICS dbo.claims WITH FULLSCAN;
    UPDATE STATISTICS dbo.claims_line_items WITH FULLSCAN;
    UPDATE STATISTICS dbo.failed_claims WITH FULLSCAN;
    UPDATE STATISTICS dbo.performance_metrics WITH FULLSCAN;
    UPDATE STATISTICS dbo.daily_processing_summary WITH FULLSCAN;
    UPDATE STATISTICS dbo.audit_log WITH FULLSCAN;
    UPDATE STATISTICS dbo.data_access_log WITH FULLSCAN;
END;
GO

-- Function to calculate reimbursement rate
CREATE FUNCTION dbo.CalculateReimbursementRate(
    @ChargeAmount DECIMAL(10,2),
    @ReimbursementAmount DECIMAL(10,2)
)
RETURNS DECIMAL(5,4)
AS
BEGIN
    DECLARE @Rate DECIMAL(5,4);
    
    IF @ChargeAmount = 0 OR @ChargeAmount IS NULL
        SET @Rate = 0;
    ELSE
        SET @Rate = @ReimbursementAmount / @ChargeAmount;
    
    RETURN @Rate;
END;
GO

-- =============================================
-- INITIAL DATA LOAD
-- =============================================

-- Load Standard Payers
INSERT INTO dbo.core_standard_payers (payer_name, payer_code, payer_type, active) VALUES
('Medicare', '1', 'Government', 1),
('Medicaid', '2', 'Government', 1),
('BlueCross', '3', 'Commercial', 1),
('Others', '4', 'Commercial', 1),
('Self Payer', '5', 'Self-Pay', 1),
('HMO', '6', 'Commercial', 1),
('Tricare', '7', 'Government', 1),
('Commercial', '8', 'Commercial', 1),
('Workers Comp', '9', 'Workers Compensation', 1),
('MC Advantage', '10', 'Government', 1);

PRINT 'Standard payers data loaded successfully.';

PRINT 'Smart Pro Claims database schema created successfully.';
GO
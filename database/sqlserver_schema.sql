-- High-Performance SQL Server Production Schema for Claims Processing
-- Optimized for analytics, reporting, and high-volume data processing
-- Includes partitioning, columnstore indexes, and in-memory OLTP tables

USE ClaimsProcessingProduction;
GO

-- Create filegroups for partitioning
ALTER DATABASE ClaimsProcessingProduction 
ADD FILEGROUP Claims_FG1;
ALTER DATABASE ClaimsProcessingProduction 
ADD FILEGROUP Claims_FG2;
ALTER DATABASE ClaimsProcessingProduction 
ADD FILEGROUP Claims_FG3;
ALTER DATABASE ClaimsProcessingProduction 
ADD FILEGROUP Claims_FG4;
GO

-- Add files to filegroups (adjust paths as needed)
ALTER DATABASE ClaimsProcessingProduction 
ADD FILE (
    NAME = 'Claims_FG1_Data',
    FILENAME = 'C:\Data\Claims_FG1_Data.ndf',
    SIZE = 1GB,
    MAXSIZE = 50GB,
    FILEGROWTH = 256MB
) TO FILEGROUP Claims_FG1;

ALTER DATABASE ClaimsProcessingProduction 
ADD FILE (
    NAME = 'Claims_FG2_Data',
    FILENAME = 'C:\Data\Claims_FG2_Data.ndf', 
    SIZE = 1GB,
    MAXSIZE = 50GB,
    FILEGROWTH = 256MB
) TO FILEGROUP Claims_FG2;

ALTER DATABASE ClaimsProcessingProduction 
ADD FILE (
    NAME = 'Claims_FG3_Data',
    FILENAME = 'C:\Data\Claims_FG3_Data.ndf',
    SIZE = 1GB, 
    MAXSIZE = 50GB,
    FILEGROWTH = 256MB
) TO FILEGROUP Claims_FG3;

ALTER DATABASE ClaimsProcessingProduction 
ADD FILE (
    NAME = 'Claims_FG4_Data',
    FILENAME = 'C:\Data\Claims_FG4_Data.ndf',
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

-- Facilities master table
CREATE TABLE dbo.Facilities (
    FacilityID INT IDENTITY(1,1) PRIMARY KEY,
    FacilityCode NVARCHAR(20) NOT NULL UNIQUE,
    FacilityName NVARCHAR(200) NOT NULL,
    NPI NVARCHAR(10) NOT NULL,
    TaxID NVARCHAR(20) NOT NULL,
    AddressLine1 NVARCHAR(200) NOT NULL,
    AddressLine2 NVARCHAR(200) NULL,
    City NVARCHAR(100) NOT NULL,
    State NCHAR(2) NOT NULL,
    ZipCode NVARCHAR(10) NOT NULL,
    Phone NVARCHAR(20) NULL,
    IsActive BIT NOT NULL DEFAULT 1,
    CreatedDate DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    ModifiedDate DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    
    INDEX IX_Facilities_FacilityCode NONCLUSTERED (FacilityCode),
    INDEX IX_Facilities_NPI NONCLUSTERED (NPI),
    INDEX IX_Facilities_IsActive NONCLUSTERED (IsActive)
);
GO

-- Providers master table
CREATE TABLE dbo.Providers (
    ProviderID INT IDENTITY(1,1) PRIMARY KEY,
    NPI NVARCHAR(10) NOT NULL UNIQUE,
    ProviderName NVARCHAR(200) NOT NULL,
    SpecialtyCode NVARCHAR(10) NULL,
    IsActive BIT NOT NULL DEFAULT 1,
    CreatedDate DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    ModifiedDate DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    
    INDEX IX_Providers_NPI NONCLUSTERED (NPI),
    INDEX IX_Providers_SpecialtyCode NONCLUSTERED (SpecialtyCode),
    INDEX IX_Providers_IsActive NONCLUSTERED (IsActive)
);
GO

-- Payers master table
CREATE TABLE dbo.Payers (
    PayerID INT IDENTITY(1,1) PRIMARY KEY,
    PayerCode NVARCHAR(20) NOT NULL UNIQUE,
    PayerName NVARCHAR(200) NOT NULL,
    PayerType NVARCHAR(50) NOT NULL, -- Medicare, Medicaid, Commercial, etc.
    ReimbursementRate DECIMAL(5,4) NULL,
    IsActive BIT NOT NULL DEFAULT 1,
    CreatedDate DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    ModifiedDate DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    
    INDEX IX_Payers_PayerCode NONCLUSTERED (PayerCode),
    INDEX IX_Payers_PayerType NONCLUSTERED (PayerType),
    INDEX IX_Payers_IsActive NONCLUSTERED (IsActive)
);
GO

-- ICD-10 diagnosis codes lookup
CREATE TABLE dbo.DiagnosisCodes (
    DiagnosisCodeID INT IDENTITY(1,1) PRIMARY KEY,
    ICD10Code NVARCHAR(10) NOT NULL UNIQUE,
    Description NVARCHAR(500) NOT NULL,
    Category NVARCHAR(100) NULL,
    IsActive BIT NOT NULL DEFAULT 1,
    EffectiveDate DATE NOT NULL,
    ExpirationDate DATE NULL,
    
    INDEX IX_DiagnosisCodes_ICD10Code NONCLUSTERED (ICD10Code),
    INDEX IX_DiagnosisCodes_Category NONCLUSTERED (Category),
    INDEX IX_DiagnosisCodes_IsActive NONCLUSTERED (IsActive)
);
GO

-- CPT procedure codes lookup
CREATE TABLE dbo.ProcedureCodes (
    ProcedureCodeID INT IDENTITY(1,1) PRIMARY KEY,
    CPTCode NVARCHAR(10) NOT NULL UNIQUE,
    Description NVARCHAR(500) NOT NULL,
    Category NVARCHAR(100) NULL,
    IsActive BIT NOT NULL DEFAULT 1,
    EffectiveDate DATE NOT NULL,
    ExpirationDate DATE NULL,
    
    INDEX IX_ProcedureCodes_CPTCode NONCLUSTERED (CPTCode),
    INDEX IX_ProcedureCodes_Category NONCLUSTERED (Category),
    INDEX IX_ProcedureCodes_IsActive NONCLUSTERED (IsActive)
);
GO

-- RVU data table for reimbursement calculations
CREATE TABLE dbo.RVUData (
    RVUID INT IDENTITY(1,1) PRIMARY KEY,
    CPTCode NVARCHAR(10) NOT NULL,
    Modifier NVARCHAR(2) NULL,
    Year INT NOT NULL,
    WorkRVU DECIMAL(8,4) NOT NULL,
    PracticeExpenseRVU DECIMAL(8,4) NOT NULL,
    MalpracticeRVU DECIMAL(8,4) NOT NULL,
    TotalRVU DECIMAL(8,4) NOT NULL,
    ConversionFactor DECIMAL(8,4) NOT NULL DEFAULT 36.04,
    IsActive BIT NOT NULL DEFAULT 1,
    CreatedDate DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    
    INDEX IX_RVUData_CPTCode_Year NONCLUSTERED (CPTCode, Year),
    INDEX IX_RVUData_IsActive NONCLUSTERED (IsActive)
);
GO

-- Main claims table (partitioned by service date)
CREATE TABLE dbo.Claims (
    ClaimID BIGINT IDENTITY(1,1) NOT NULL,
    ClaimNumber NVARCHAR(50) NOT NULL,
    FacilityID INT NOT NULL,
    PatientAccountNumber NVARCHAR(50) NOT NULL,
    MedicalRecordNumber NVARCHAR(50) NULL,
    
    -- Patient demographics (encrypted in production)
    PatientFirstName NVARCHAR(100) NOT NULL,
    PatientLastName NVARCHAR(100) NOT NULL,
    PatientMiddleName NVARCHAR(100) NULL,
    PatientDateOfBirth DATE NOT NULL,
    PatientSSN VARBINARY(128) NULL, -- Encrypted
    PatientGender NCHAR(1) NULL,
    
    -- Service information
    AdmissionDate DATE NOT NULL,
    DischargeDate DATE NOT NULL,
    ServiceFromDate DATE NOT NULL,
    ServiceToDate DATE NOT NULL,
    
    -- Financial information
    FinancialClass NVARCHAR(50) NOT NULL,
    TotalCharges DECIMAL(12,2) NOT NULL,
    TotalRVU DECIMAL(8,4) NULL,
    ExpectedReimbursement DECIMAL(12,2) NULL,
    ActualReimbursement DECIMAL(12,2) NULL,
    
    -- Insurance information
    PayerID INT NULL,
    InsuranceType NVARCHAR(50) NOT NULL,
    InsurancePlanID NVARCHAR(50) NULL,
    SubscriberID NVARCHAR(50) NULL,
    
    -- Provider information
    BillingProviderID INT NOT NULL,
    BillingProviderName NVARCHAR(200) NOT NULL,
    AttendingProviderID INT NULL,
    AttendingProviderName NVARCHAR(200) NULL,
    
    -- Diagnosis information
    PrimaryDiagnosisCodeID INT NOT NULL,
    SecondaryDiagnosisCount INT DEFAULT 0,
    
    -- Processing metadata
    ProcessingStatus NVARCHAR(20) NOT NULL DEFAULT 'Processed',
    BatchID NVARCHAR(100) NULL,
    ProcessedDate DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    
    -- ML prediction results
    MLPredictionScore DECIMAL(5,4) NULL,
    MLPredictionResult NVARCHAR(50) NULL,
    
    -- Audit fields
    CreatedDate DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    CreatedBy NVARCHAR(100) NOT NULL,
    ModifiedDate DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    ModifiedBy NVARCHAR(100) NOT NULL,
    
    CONSTRAINT PK_Claims PRIMARY KEY CLUSTERED (ClaimID, ServiceFromDate),
    CONSTRAINT FK_Claims_Facilities FOREIGN KEY (FacilityID) REFERENCES dbo.Facilities(FacilityID),
    CONSTRAINT FK_Claims_Payers FOREIGN KEY (PayerID) REFERENCES dbo.Payers(PayerID),
    CONSTRAINT FK_Claims_BillingProvider FOREIGN KEY (BillingProviderID) REFERENCES dbo.Providers(ProviderID),
    CONSTRAINT FK_Claims_AttendingProvider FOREIGN KEY (AttendingProviderID) REFERENCES dbo.Providers(ProviderID),
    CONSTRAINT FK_Claims_PrimaryDiagnosis FOREIGN KEY (PrimaryDiagnosisCodeID) REFERENCES dbo.DiagnosisCodes(DiagnosisCodeID)
) ON ClaimsDatePartitionScheme(ServiceFromDate);
GO

-- Claim line items table (partitioned)
CREATE TABLE dbo.ClaimsLineItems (
    LineItemID BIGINT IDENTITY(1,1) NOT NULL,
    ClaimID BIGINT NOT NULL,
    LineNumber INT NOT NULL,
    ServiceFromDate DATE NOT NULL, -- Partition key
    
    -- Service information
    ServiceDate DATE NOT NULL,
    ProcedureCodeID INT NOT NULL,
    ProcedureDescription NVARCHAR(500) NULL,
    ModifierCodes NVARCHAR(20) NULL,
    
    -- Quantity and charges
    Units INT NOT NULL,
    ChargeAmount DECIMAL(10,2) NOT NULL,
    
    -- Provider information
    RenderingProviderID INT NULL,
    RenderingProviderName NVARCHAR(200) NULL,
    
    -- RVU and reimbursement information
    WorkRVU DECIMAL(8,4) NULL,
    PracticeExpenseRVU DECIMAL(8,4) NULL,
    MalpracticeRVU DECIMAL(8,4) NULL,
    TotalRVU DECIMAL(8,4) NULL,
    ExpectedReimbursement DECIMAL(10,2) NULL,
    ActualReimbursement DECIMAL(10,2) NULL,
    
    -- Diagnosis pointers
    DiagnosisPointers NVARCHAR(50) NULL,
    
    -- Audit fields
    CreatedDate DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    CreatedBy NVARCHAR(100) NOT NULL,
    ModifiedDate DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    ModifiedBy NVARCHAR(100) NOT NULL,
    
    CONSTRAINT PK_ClaimsLineItems PRIMARY KEY CLUSTERED (LineItemID, ServiceFromDate),
    CONSTRAINT FK_ClaimsLineItems_Claims FOREIGN KEY (ClaimID, ServiceFromDate) REFERENCES dbo.Claims(ClaimID, ServiceFromDate),
    CONSTRAINT FK_ClaimsLineItems_ProcedureCode FOREIGN KEY (ProcedureCodeID) REFERENCES dbo.ProcedureCodes(ProcedureCodeID),
    CONSTRAINT FK_ClaimsLineItems_RenderingProvider FOREIGN KEY (RenderingProviderID) REFERENCES dbo.Providers(ProviderID)
) ON ClaimsDatePartitionScheme(ServiceFromDate);
GO

-- Claim diagnosis codes junction table
CREATE TABLE dbo.ClaimDiagnosisCodes (
    ClaimDiagnosisID BIGINT IDENTITY(1,1) PRIMARY KEY,
    ClaimID BIGINT NOT NULL,
    ServiceFromDate DATE NOT NULL,
    DiagnosisCodeID INT NOT NULL,
    DiagnosisSequence INT NOT NULL,
    IsPrimary BIT NOT NULL DEFAULT 0,
    
    CONSTRAINT FK_ClaimDiagnosisCodes_Claims FOREIGN KEY (ClaimID, ServiceFromDate) REFERENCES dbo.Claims(ClaimID, ServiceFromDate),
    CONSTRAINT FK_ClaimDiagnosisCodes_DiagnosisCode FOREIGN KEY (DiagnosisCodeID) REFERENCES dbo.DiagnosisCodes(DiagnosisCodeID),
    CONSTRAINT UQ_ClaimDiagnosisCodes_Sequence UNIQUE (ClaimID, DiagnosisSequence)
) ON ClaimsDatePartitionScheme(ServiceFromDate);
GO

-- Performance metrics table (in-memory OLTP for high-speed inserts)
CREATE TABLE dbo.PerformanceMetrics (
    MetricID BIGINT IDENTITY(1,1) PRIMARY KEY NONCLUSTERED,
    
    -- Metric information
    MetricType NVARCHAR(50) NOT NULL,
    MetricName NVARCHAR(100) NOT NULL,
    MetricValue DECIMAL(20,6) NOT NULL,
    Unit NVARCHAR(20) NOT NULL,
    
    -- Context
    FacilityID INT NULL,
    BatchID NVARCHAR(100) NULL,
    ServiceName NVARCHAR(50) NOT NULL,
    
    -- Tags (JSON format)
    Tags NVARCHAR(MAX) NULL,
    
    -- Timestamp
    RecordedDate DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    
    INDEX IX_PerformanceMetrics_MetricType_Date CLUSTERED (MetricType, RecordedDate),
    INDEX IX_PerformanceMetrics_ServiceName NONCLUSTERED (ServiceName, RecordedDate),
    INDEX IX_PerformanceMetrics_FacilityID NONCLUSTERED (FacilityID, RecordedDate)
) WITH (MEMORY_OPTIMIZED = ON, DURABILITY = SCHEMA_AND_DATA);
GO

-- Analytics aggregation tables (columnstore for fast analytics)
CREATE TABLE dbo.ClaimsAnalyticsSummary (
    SummaryID BIGINT IDENTITY(1,1) NOT NULL,
    
    -- Dimensions
    FacilityID INT NOT NULL,
    PayerID INT NULL,
    ServiceYear INT NOT NULL,
    ServiceMonth INT NOT NULL,
    ServiceDay INT NOT NULL,
    
    -- Measures
    ClaimCount INT NOT NULL,
    TotalCharges DECIMAL(14,2) NOT NULL,
    TotalRVU DECIMAL(10,4) NOT NULL,
    ExpectedReimbursement DECIMAL(14,2) NOT NULL,
    ActualReimbursement DECIMAL(14,2) NULL,
    
    -- Processing metrics
    AverageProcessingTime DECIMAL(8,2) NULL,
    SuccessRate DECIMAL(5,2) NULL,
    
    -- Timestamp
    CreatedDate DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    
    CONSTRAINT PK_ClaimsAnalyticsSummary PRIMARY KEY NONCLUSTERED (SummaryID)
);

-- Create clustered columnstore index for analytics
CREATE CLUSTERED COLUMNSTORE INDEX CCI_ClaimsAnalyticsSummary 
ON dbo.ClaimsAnalyticsSummary;
GO

-- RVU analytics table (columnstore)
CREATE TABLE dbo.RVUAnalytics (
    RVUAnalyticsID BIGINT IDENTITY(1,1) NOT NULL,
    
    -- Dimensions
    FacilityID INT NOT NULL,
    ProviderID INT NOT NULL,
    ProcedureCodeID INT NOT NULL,
    ServiceYear INT NOT NULL,
    ServiceMonth INT NOT NULL,
    
    -- Measures
    ServiceCount INT NOT NULL,
    TotalUnits INT NOT NULL,
    TotalWorkRVU DECIMAL(12,4) NOT NULL,
    TotalPracticeExpenseRVU DECIMAL(12,4) NOT NULL,
    TotalMalpracticeRVU DECIMAL(12,4) NOT NULL,
    TotalRVU DECIMAL(12,4) NOT NULL,
    TotalReimbursement DECIMAL(14,2) NOT NULL,
    
    -- Performance metrics
    AverageRVUPerService DECIMAL(8,4) NOT NULL,
    AverageReimbursementPerRVU DECIMAL(8,2) NOT NULL,
    
    -- Timestamp
    CreatedDate DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    
    CONSTRAINT PK_RVUAnalytics PRIMARY KEY NONCLUSTERED (RVUAnalyticsID)
);

-- Create clustered columnstore index
CREATE CLUSTERED COLUMNSTORE INDEX CCI_RVUAnalytics 
ON dbo.RVUAnalytics;
GO

-- Diagnosis code analytics table (columnstore)
CREATE TABLE dbo.DiagnosisAnalytics (
    DiagnosisAnalyticsID BIGINT IDENTITY(1,1) NOT NULL,
    
    -- Dimensions
    FacilityID INT NOT NULL,
    DiagnosisCodeID INT NOT NULL,
    ServiceYear INT NOT NULL,
    ServiceMonth INT NOT NULL,
    
    -- Measures
    ClaimCount INT NOT NULL,
    LineItemCount INT NOT NULL,
    TotalCharges DECIMAL(14,2) NOT NULL,
    TotalRVU DECIMAL(12,4) NOT NULL,
    TotalReimbursement DECIMAL(14,2) NOT NULL,
    
    -- Performance metrics
    AverageChargePerClaim DECIMAL(10,2) NOT NULL,
    AverageRVUPerClaim DECIMAL(8,4) NOT NULL,
    AverageReimbursementPerClaim DECIMAL(10,2) NOT NULL,
    
    -- Timestamp
    CreatedDate DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    
    CONSTRAINT PK_DiagnosisAnalytics PRIMARY KEY NONCLUSTERED (DiagnosisAnalyticsID)
);

-- Create clustered columnstore index
CREATE CLUSTERED COLUMNSTORE INDEX CCI_DiagnosisAnalytics 
ON dbo.DiagnosisAnalytics;
GO

-- Payer analytics table (columnstore)
CREATE TABLE dbo.PayerAnalytics (
    PayerAnalyticsID BIGINT IDENTITY(1,1) NOT NULL,
    
    -- Dimensions
    PayerID INT NOT NULL,
    FacilityID INT NOT NULL,
    ServiceYear INT NOT NULL,
    ServiceMonth INT NOT NULL,
    
    -- Measures
    ClaimCount INT NOT NULL,
    TotalCharges DECIMAL(14,2) NOT NULL,
    TotalRVU DECIMAL(12,4) NOT NULL,
    ExpectedReimbursement DECIMAL(14,2) NOT NULL,
    ActualReimbursement DECIMAL(14,2) NULL,
    ReimbursementVariance DECIMAL(14,2) NULL,
    
    -- Performance metrics
    PaymentAccuracy DECIMAL(5,2) NULL,
    AveragePaymentDelay INT NULL,
    DenialRate DECIMAL(5,2) NULL,
    
    -- Timestamp
    CreatedDate DATETIME2 NOT NULL DEFAULT GETUTCDATE(),
    
    CONSTRAINT PK_PayerAnalytics PRIMARY KEY NONCLUSTERED (PayerAnalyticsID)
);

-- Create clustered columnstore index
CREATE CLUSTERED COLUMNSTORE INDEX CCI_PayerAnalytics 
ON dbo.PayerAnalytics;
GO

-- Create non-clustered indexes for optimal query performance
-- Claims table indexes
CREATE NONCLUSTERED INDEX IX_Claims_FacilityID_ServiceDate 
ON dbo.Claims (FacilityID, ServiceFromDate, ServiceToDate);

CREATE NONCLUSTERED INDEX IX_Claims_PayerID_ServiceDate 
ON dbo.Claims (PayerID, ServiceFromDate) 
WHERE PayerID IS NOT NULL;

CREATE NONCLUSTERED INDEX IX_Claims_ProcessedDate 
ON dbo.Claims (ProcessedDate);

CREATE NONCLUSTERED INDEX IX_Claims_ClaimNumber 
ON dbo.Claims (ClaimNumber);

CREATE NONCLUSTERED INDEX IX_Claims_PatientAccount 
ON dbo.Claims (PatientAccountNumber, FacilityID);

-- Line items indexes
CREATE NONCLUSTERED INDEX IX_ClaimsLineItems_ClaimID 
ON dbo.ClaimsLineItems (ClaimID, ServiceFromDate);

CREATE NONCLUSTERED INDEX IX_ClaimsLineItems_ProcedureCode_ServiceDate 
ON dbo.ClaimsLineItems (ProcedureCodeID, ServiceDate);

CREATE NONCLUSTERED INDEX IX_ClaimsLineItems_RenderingProvider 
ON dbo.ClaimsLineItems (RenderingProviderID, ServiceDate) 
WHERE RenderingProviderID IS NOT NULL;

-- Performance optimization views
CREATE VIEW dbo.vw_ClaimsWithMetrics
AS
SELECT 
    c.ClaimID,
    c.ClaimNumber,
    f.FacilityName,
    p.PayerName,
    c.ServiceFromDate,
    c.ServiceToDate,
    c.TotalCharges,
    c.TotalRVU,
    c.ExpectedReimbursement,
    c.ActualReimbursement,
    c.ProcessedDate,
    -- Performance metrics
    DATEDIFF(DAY, c.ServiceFromDate, c.ProcessedDate) AS ProcessingDaysFromService,
    CASE 
        WHEN c.ActualReimbursement IS NOT NULL 
        THEN ((c.ActualReimbursement - c.ExpectedReimbursement) / c.ExpectedReimbursement) * 100 
        ELSE NULL 
    END AS ReimbursementVariancePercent,
    -- Line item count
    (SELECT COUNT(*) FROM dbo.ClaimsLineItems cli WHERE cli.ClaimID = c.ClaimID AND cli.ServiceFromDate = c.ServiceFromDate) AS LineItemCount
FROM dbo.Claims c
INNER JOIN dbo.Facilities f ON c.FacilityID = f.FacilityID
LEFT JOIN dbo.Payers p ON c.PayerID = p.PayerID;
GO

-- Create indexed views for common analytics queries
CREATE VIEW dbo.vw_MonthlyClaimsSummary
WITH SCHEMABINDING
AS
SELECT 
    c.FacilityID,
    c.PayerID,
    YEAR(c.ServiceFromDate) AS ServiceYear,
    MONTH(c.ServiceFromDate) AS ServiceMonth,
    COUNT_BIG(*) AS ClaimCount,
    SUM(c.TotalCharges) AS TotalCharges,
    SUM(c.TotalRVU) AS TotalRVU,
    SUM(c.ExpectedReimbursement) AS ExpectedReimbursement,
    SUM(c.ActualReimbursement) AS ActualReimbursement
FROM dbo.Claims c
GROUP BY c.FacilityID, c.PayerID, YEAR(c.ServiceFromDate), MONTH(c.ServiceFromDate);
GO

CREATE UNIQUE CLUSTERED INDEX IX_vw_MonthlyClaimsSummary 
ON dbo.vw_MonthlyClaimsSummary (FacilityID, PayerID, ServiceYear, ServiceMonth);
GO

-- Stored procedures for data processing
CREATE PROCEDURE dbo.usp_RefreshAnalyticsSummaries
    @StartDate DATE = NULL,
    @EndDate DATE = NULL
AS
BEGIN
    SET NOCOUNT ON;
    
    -- Default to current month if no dates provided
    IF @StartDate IS NULL OR @EndDate IS NULL
    BEGIN
        SET @StartDate = DATEFROMPARTS(YEAR(GETDATE()), MONTH(GETDATE()), 1);
        SET @EndDate = EOMONTH(@StartDate);
    END
    
    -- Refresh claims analytics summary
    MERGE dbo.ClaimsAnalyticsSummary AS target
    USING (
        SELECT 
            c.FacilityID,
            c.PayerID,
            YEAR(c.ServiceFromDate) AS ServiceYear,
            MONTH(c.ServiceFromDate) AS ServiceMonth,
            DAY(c.ServiceFromDate) AS ServiceDay,
            COUNT(*) AS ClaimCount,
            SUM(c.TotalCharges) AS TotalCharges,
            SUM(c.TotalRVU) AS TotalRVU,
            SUM(c.ExpectedReimbursement) AS ExpectedReimbursement,
            SUM(c.ActualReimbursement) AS ActualReimbursement
        FROM dbo.Claims c
        WHERE c.ServiceFromDate BETWEEN @StartDate AND @EndDate
        GROUP BY c.FacilityID, c.PayerID, YEAR(c.ServiceFromDate), MONTH(c.ServiceFromDate), DAY(c.ServiceFromDate)
    ) AS source ON (
        target.FacilityID = source.FacilityID 
        AND ISNULL(target.PayerID, 0) = ISNULL(source.PayerID, 0)
        AND target.ServiceYear = source.ServiceYear
        AND target.ServiceMonth = source.ServiceMonth
        AND target.ServiceDay = source.ServiceDay
    )
    WHEN MATCHED THEN
        UPDATE SET 
            ClaimCount = source.ClaimCount,
            TotalCharges = source.TotalCharges,
            TotalRVU = source.TotalRVU,
            ExpectedReimbursement = source.ExpectedReimbursement,
            ActualReimbursement = source.ActualReimbursement
    WHEN NOT MATCHED THEN
        INSERT (FacilityID, PayerID, ServiceYear, ServiceMonth, ServiceDay, 
                ClaimCount, TotalCharges, TotalRVU, ExpectedReimbursement, ActualReimbursement)
        VALUES (source.FacilityID, source.PayerID, source.ServiceYear, source.ServiceMonth, source.ServiceDay,
                source.ClaimCount, source.TotalCharges, source.TotalRVU, source.ExpectedReimbursement, source.ActualReimbursement);
END
GO

-- Database optimization settings
ALTER DATABASE ClaimsProcessingProduction SET COMPATIBILITY_LEVEL = 150;
ALTER DATABASE ClaimsProcessingProduction SET AUTO_CREATE_STATISTICS ON;
ALTER DATABASE ClaimsProcessingProduction SET AUTO_UPDATE_STATISTICS ON;
ALTER DATABASE ClaimsProcessingProduction SET AUTO_UPDATE_STATISTICS_ASYNC ON;
ALTER DATABASE ClaimsProcessingProduction SET PARAMETERIZATION FORCED;
ALTER DATABASE ClaimsProcessingProduction SET QUERY_STORE = ON;
GO

-- Enable In-Memory OLTP
ALTER DATABASE ClaimsProcessingProduction 
ADD FILEGROUP ClaimsProcessingProduction_mod CONTAINS MEMORY_OPTIMIZED_DATA;

ALTER DATABASE ClaimsProcessingProduction 
ADD FILE (
    NAME = 'ClaimsProcessingProduction_mod',
    FILENAME = 'C:\Data\ClaimsProcessingProduction_mod.ndf'
) TO FILEGROUP ClaimsProcessingProduction_mod;
GO

PRINT 'SQL Server production schema created successfully with partitioning, columnstore indexes, and analytics optimization.';
# Smart Pro Claims - Sample Data Loader

This directory contains scripts to load comprehensive sample data into the Smart Pro Claims database for testing, development, and demonstration purposes.

## Overview

The sample data loader creates a realistic healthcare claims processing environment with:

### Organizational Structure
- **2 Organizations**:
  - Regional Health System (with regions)
  - Metro Medical Group (no regions)
- **2 Regions** under Regional Health System:
  - North Region 
  - South Region
- **5 Facilities**:
  - FAC001: Facility A - North General Hospital (250 beds, North Region)
  - FAC002: Facility B - South Regional Medical Center (180 beds, South Region)  
  - FAC003: Facility C - Metro Downtown Clinic (0 beds, Metro Medical Group)
  - FAC004: Facility D - Metro West Campus (120 beds, Metro Medical Group)
  - FAC005: Facility E - Metro North Specialty Center (80 beds, Metro Medical Group)

### Configuration Data
- **10 Standard Payers** (Medicare, Medicaid, BlueCross, etc.)
- **Financial Classes** for each facility with realistic reimbursement rates
- **Place of Service codes** (Office, Hospital, ER, etc.)
- **Departments** (Emergency, Internal Medicine, Cardiology, etc.)
- **Facility Coders** (3-5 per facility)
- **200 Physicians** with specialties and NPI numbers

### Clinical Data
- **30 Common CPT Codes** with accurate RVU values
- **30 Common ICD-10 Diagnosis Codes**
- **100,000 Sample Claims** with:
  - Realistic patient demographics
  - 1-5 diagnoses per claim
  - 1-8 line items per claim
  - Accurate charge and reimbursement calculations
  - Date ranges spanning 2 years

## Files

### Core Scripts
- **`load_sample_data.py`** - Main Python script that loads all sample data
- **`load_sample_data.ps1`** - PowerShell wrapper for easy Windows execution
- **`requirements_sample_data.txt`** - Python package requirements

### Supporting Files
- **`README_SAMPLE_DATA.md`** - This documentation file

## Prerequisites

### Software Requirements
- **Python 3.9+** with pip
- **SQL Server** with the smart_pro_claims database schema already applied
- **ODBC Driver 17 for SQL Server** (or newer)

### Database Requirements
- The `smart_pro_claims` database must exist with the complete schema
- User must have db_owner permissions or equivalent
- Standard payers should already be loaded (included in schema script)

## Usage

### Option 1: PowerShell Script (Recommended for Windows)

#### Using Integrated Authentication
```powershell
# Navigate to project root
cd C:\Claims_Processor\app

# Run with integrated authentication
.\scripts\load_sample_data.ps1 -ServerName "localhost" -IntegratedAuth

# Skip claims data for faster loading (testing configuration only)
.\scripts\load_sample_data.ps1 -ServerName "localhost" -IntegratedAuth -SkipClaims
```

#### Using SQL Server Authentication
```powershell
# Run with SQL Server authentication
.\scripts\load_sample_data.ps1 -ServerName "localhost" -Username "claims_analytics_user" -Password "YourPassword"

# Specify custom database name and port
.\scripts\load_sample_data.ps1 -ServerName "myserver" -DatabaseName "smart_pro_claims" -Username "claims_user" -Password "MyPassword" -Port "1433"
```

### Option 2: Direct Python Execution

#### Install Requirements
```bash
pip install -r scripts/requirements_sample_data.txt
```

#### Run Script
```bash
# With integrated authentication
python scripts/load_sample_data.py --connection-string "mssql+pyodbc://localhost/smart_pro_claims?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"

# With SQL Server authentication
python scripts/load_sample_data.py --connection-string "mssql+pyodbc://user:password@localhost/smart_pro_claims?driver=ODBC+Driver+17+for+SQL+Server"

# Skip claims data for faster loading
python scripts/load_sample_data.py --connection-string "..." --skip-claims
```

## Data Generation Details

### Claims Data Characteristics
- **Patient Demographics**: Realistic age and gender distribution
- **Financial Classes**: Weighted toward common payers (Medicare, Commercial)
- **Service Dates**: Distributed over 2-year period
- **Charges**: Calculated from RVU values with realistic variation
- **Reimbursement**: Based on financial class contracted rates
- **Diagnoses**: Weighted toward common conditions
- **Procedures**: Mix of E&M, diagnostics, and procedures

### Performance Considerations
- **Batch Processing**: Claims loaded in 1,000-record batches
- **Memory Efficient**: Uses bulk inserts, not ORM objects
- **Progress Tracking**: Real-time progress indicators
- **Realistic Volume**: 100,000 claims represents ~2 months for a mid-size facility

## Expected Runtime

| Component | Records | Estimated Time |
|-----------|---------|----------------|
| Organizations & Facilities | 9 | < 1 second |
| Financial Classes | 50 | < 1 second |
| Place of Service | 155 | < 5 seconds |
| Departments | 125 | < 5 seconds |
| Coders | 20 | < 1 second |
| Physicians | 200 | < 10 seconds |
| RVU Data | 30 | < 1 second |
| Claims Data | 100,000 | 5-15 minutes |
| **Total** | **~100,600** | **5-20 minutes** |

*Runtime varies based on hardware, SQL Server configuration, and network latency*

## Verification Queries

After loading, verify the data with these SQL queries:

```sql
-- Check organizational hierarchy
SELECT 
    o.org_name,
    r.region_name,
    f.facility_name,
    f.beds,
    f.city + ', ' + f.state as location
FROM facility_organization o
LEFT JOIN facility_region r ON o.org_id = r.org_id
LEFT JOIN facilities f ON o.org_id = f.org_id
ORDER BY o.org_name, r.region_name, f.facility_name;

-- Check claims summary by facility
SELECT 
    f.facility_name,
    COUNT(c.patient_account_number) as total_claims,
    COUNT(DISTINCT c.patient_account_number) as unique_patients,
    SUM(li.charge_amount) as total_charges,
    SUM(li.reimbursement_amount) as total_reimbursement,
    AVG(li.reimbursement_amount / li.charge_amount) as avg_reimbursement_rate
FROM facilities f
LEFT JOIN claims c ON f.facility_id = c.facility_id
LEFT JOIN claims_line_items li ON c.facility_id = li.facility_id 
    AND c.patient_account_number = li.patient_account_number
GROUP BY f.facility_id, f.facility_name
ORDER BY total_claims DESC;

-- Check diagnosis distribution
SELECT 
    cd.diagnosis_code,
    cd.diagnosis_description,
    COUNT(*) as frequency,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
FROM claims_diagnosis cd
GROUP BY cd.diagnosis_code, cd.diagnosis_description
ORDER BY frequency DESC;

-- Check procedure distribution
SELECT 
    li.procedure_code,
    rd.description,
    COUNT(*) as frequency,
    SUM(li.charge_amount) as total_charges,
    AVG(li.rvu_value) as avg_rvu
FROM claims_line_items li
JOIN rvu_data rd ON li.procedure_code = rd.procedure_code
GROUP BY li.procedure_code, rd.description
ORDER BY frequency DESC;
```

## Troubleshooting

### Common Issues

#### 1. "Python not found"
**Solution**: Install Python 3.9+ and ensure it's in your PATH
```powershell
# Check Python installation
python --version
```

#### 2. "ODBC Driver not found"
**Solution**: Install ODBC Driver 17 for SQL Server
```powershell
# Download from Microsoft website or use chocolatey
choco install sql-server-odbc-driver
```

#### 3. "Database connection failed"
**Solutions**:
- Verify SQL Server is running: `Get-Service MSSQLSERVER`
- Check database name: Ensure "smart_pro_claims" database exists
- Verify credentials: Test with SQL Server Management Studio
- Check firewall: Ensure port 1433 is open

#### 4. "Permission denied"
**Solution**: Grant appropriate permissions
```sql
-- Grant permissions to user
USE smart_pro_claims;
ALTER ROLE db_owner ADD MEMBER claims_analytics_user;
```

#### 5. "Memory errors during claims loading"
**Solutions**:
- Use `--skip-claims` flag to load configuration only
- Increase SQL Server memory allocation
- Run during off-peak hours

#### 6. "Slow performance"
**Solutions**:
- Ensure database files are on SSD storage
- Increase SQL Server memory allocation
- Temporarily disable antivirus scanning of data directories
- Use `--skip-claims` for configuration testing

### Performance Tuning

For faster loading on production systems:

```sql
-- Temporarily disable constraints (run before loading)
ALTER TABLE claims_line_items NOCHECK CONSTRAINT ALL;
ALTER TABLE claims_diagnosis NOCHECK CONSTRAINT ALL;

-- Re-enable after loading
ALTER TABLE claims_line_items CHECK CONSTRAINT ALL;
ALTER TABLE claims_diagnosis CHECK CONSTRAINT ALL;

-- Update statistics after loading
UPDATE STATISTICS claims;
UPDATE STATISTICS claims_line_items;
UPDATE STATISTICS claims_diagnosis;
```

## Data Usage

This sample data enables testing of:

### Analytics Features
- Facility performance comparison
- Payer mix analysis  
- RVU productivity reporting
- Diagnosis trending
- Financial performance metrics

### Operational Features
- Claims processing workflows
- Validation rule testing
- Billing and reimbursement calculations
- Provider productivity analysis
- Quality reporting

### Security Features
- Role-based access testing
- Audit trail generation
- PHI access logging
- Data masking validation

## Cleanup

To remove all sample data:

```sql
-- WARNING: This will delete ALL data in these tables
DELETE FROM claims_line_items;
DELETE FROM claims_diagnosis;
DELETE FROM claims;
DELETE FROM facility_coders;
DELETE FROM facility_departments;
DELETE FROM facility_place_of_service;
DELETE FROM facility_financial_classes;
DELETE FROM physicians;
DELETE FROM rvu_data WHERE procedure_code IN (SELECT procedure_code FROM rvu_data); -- Keep if needed
DELETE FROM facilities;
DELETE FROM facility_region;
DELETE FROM facility_organization;
```

## Support

For issues with the sample data loader:

1. Check this README for common solutions
2. Verify all prerequisites are met
3. Test database connectivity separately
4. Review SQL Server error logs
5. Use `--skip-claims` flag to isolate issues

---

**Note**: This sample data is for development and testing purposes only. Do not use in production environments with real patient data.
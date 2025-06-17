# Smart Pro Claims - Database Setup

This directory contains scripts to automatically setup PostgreSQL and/or SQL Server databases with complete schema and sample data for the Smart Pro Claims system.

## Overview

The database setup scripts provide a complete automation solution that:

1. **Checks database connectivity** and verifies required tools are available
2. **Creates databases** if they don't exist (PostgreSQL and/or SQL Server)
3. **Loads database schemas** with all tables, indexes, and constraints
4. **Loads sample data** including 100,000 realistic claims with healthcare data
5. **Provides comprehensive error handling** and progress reporting

## Files

### Core Scripts
- **`setup_database.py`** - Main Python script for database setup automation
- **`setup_database.ps1`** - PowerShell wrapper for easy Windows execution
- **`load_sample_data.py`** - Sample data generation (called automatically)

### Supporting Files
- **`README_DATABASE_SETUP.md`** - This documentation file
- **`requirements.txt`** - Python package requirements

## Prerequisites

### Software Requirements
- **Python 3.11+** with pip
- **PostgreSQL 12+** (if using PostgreSQL)
- **SQL Server 2019+** (if using SQL Server)
- **Command line tools**:
  - `psql` (PostgreSQL client) - for PostgreSQL setup
  - `sqlcmd` (SQL Server client) - for SQL Server setup

### Database Requirements
- **Database server** must be running and accessible
- **User permissions** to create databases and objects
- **Network connectivity** to database servers

### Python Dependencies
```bash
pip install -r requirements.txt
```

Key packages:
- `psycopg2-binary` (PostgreSQL connectivity)
- `pyodbc` (SQL Server connectivity)
- `sqlalchemy` (Database abstraction)
- `faker` (Sample data generation)
- `rich` (Enhanced console output)

## Usage

### Option 1: Python Script (Cross-platform)

#### PostgreSQL Only
```bash
python scripts/setup_database.py \
  --postgres-host localhost \
  --postgres-user postgres \
  --postgres-password mypassword
```

#### SQL Server Only (Integrated Auth)
```bash
python scripts/setup_database.py \
  --sqlserver-host localhost \
  --integrated-auth
```

#### SQL Server Only (SQL Auth)
```bash
python scripts/setup_database.py \
  --sqlserver-host localhost \
  --sqlserver-user sa \
  --sqlserver-password mypassword
```

#### Both Databases
```bash
python scripts/setup_database.py \
  --postgres-host localhost \
  --postgres-user postgres \
  --postgres-password pg_pass \
  --sqlserver-host localhost \
  --sqlserver-user sa \
  --sqlserver-password sql_pass
```

#### Custom Database Names
```bash
python scripts/setup_database.py \
  --postgres-host localhost \
  --postgres-user postgres \
  --postgres-password mypass \
  --postgres-database my_claims_db \
  --sqlserver-host localhost \
  --integrated-auth \
  --sqlserver-database my_claims_db
```

### Option 2: PowerShell Script (Windows)

#### PostgreSQL Only
```powershell
.\scripts\setup_database.ps1 -PostgresHost "localhost" -PostgresUser "postgres" -PostgresPassword "mypassword"
```

#### SQL Server with Integrated Auth
```powershell
.\scripts\setup_database.ps1 -SqlServerHost "localhost" -IntegratedAuth
```

#### Both Databases
```powershell
.\scripts\setup_database.ps1 `
  -PostgresHost "localhost" `
  -PostgresUser "postgres" `
  -PostgresPassword "pg_pass" `
  -SqlServerHost "localhost" `
  -SqlServerUser "sa" `
  -SqlServerPassword "sql_pass"
```

## Command Line Options

### PostgreSQL Options
| Option | Description | Default |
|--------|-------------|---------|
| `--postgres-host` | PostgreSQL server hostname | Required |
| `--postgres-port` | PostgreSQL server port | 5432 |
| `--postgres-user` | PostgreSQL username | Required |
| `--postgres-password` | PostgreSQL password | Required |
| `--postgres-database` | Database name to create/use | smart_pro_claims |

### SQL Server Options
| Option | Description | Default |
|--------|-------------|---------|
| `--sqlserver-host` | SQL Server hostname | Required |
| `--sqlserver-user` | SQL Server username | Required (unless --integrated-auth) |
| `--sqlserver-password` | SQL Server password | Required (unless --integrated-auth) |
| `--sqlserver-database` | Database name to create/use | smart_pro_claims |
| `--integrated-auth` | Use Windows integrated authentication | False |

### General Options
| Option | Description | Default |
|--------|-------------|---------|
| `--skip-sample-data` | Skip loading sample data entirely | False |
| `--skip-claims-data` | Load configuration only (no claims) | False |

## What Gets Created

### Database Objects

#### PostgreSQL Schema
- **Core Tables**: Organizations, facilities, financial classes, providers
- **Claims Tables**: Claims, line items, diagnoses (partitioned for performance)
- **Reference Data**: RVU data, standard payers, procedure codes
- **Analytics Views**: Materialized views for common queries
- **Indexes**: B-tree and GIN indexes for optimal query performance

#### SQL Server Schema  
- **Core Tables**: Same structure as PostgreSQL
- **Partitioned Tables**: Claims data partitioned by date for performance
- **Columnstore Indexes**: For analytics workloads
- **Stored Procedures**: Performance optimization routines
- **Audit Tables**: Comprehensive audit trail and access logging

### Sample Data

#### Organizational Structure
- **2 Organizations**:
  - Regional Health System (with regions)
  - Metro Medical Group (no regions)
- **2 Regions**: North Region, South Region
- **5 Facilities**: Mix of hospitals and clinics with realistic bed counts

#### Configuration Data
- **10 Standard Payers** (Medicare, Medicaid, BlueCross, etc.)
- **Financial Classes** for each facility with realistic reimbursement rates
- **Place of Service codes** (Office, Hospital, ER, etc.)
- **Departments** (Emergency, Internal Medicine, Cardiology, etc.)
- **20 Facility Coders** (3-5 per facility)
- **200 Physicians** with specialties and NPI numbers

#### Clinical Data
- **30 Common CPT Codes** with accurate RVU values
- **30 Common ICD-10 Diagnosis Codes**
- **100,000 Sample Claims** with:
  - Realistic patient demographics (age/gender distribution)
  - 1-5 diagnoses per claim
  - 1-8 line items per claim
  - Accurate charge and reimbursement calculations
  - Service dates spanning 2 years

## Expected Runtime

| Component | Records | Estimated Time |
|-----------|---------|----------------|
| Database Creation | 1-2 DBs | 10-30 seconds |
| Schema Loading | ~100 objects | 1-2 minutes |
| Sample Data Configuration | ~600 records | 10-30 seconds |
| Claims Data | 100,000 claims | 5-15 minutes |
| **Total** | **~100,700** | **7-20 minutes** |

*Runtime varies based on hardware, database configuration, and network latency*

## Verification

After setup completes, verify your installation:

### Quick Check
```sql
-- Check database objects
SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'; -- PostgreSQL
SELECT name FROM sys.tables; -- SQL Server

-- Check sample data counts
SELECT 'Facilities' as table_name, COUNT(*) as record_count FROM facilities
UNION ALL
SELECT 'Claims', COUNT(*) FROM claims
UNION ALL
SELECT 'Physicians', COUNT(*) FROM physicians;
```

### Detailed Verification
```sql
-- Check organizational hierarchy
SELECT 
    o.org_name,
    r.region_name,
    f.facility_name,
    f.beds,
    f.city || ', ' || f.state as location  -- PostgreSQL
    -- f.city + ', ' + f.state as location -- SQL Server
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
```

## Troubleshooting

### Common Issues

#### 1. "Database connection failed"
**Causes**:
- Database server not running
- Incorrect credentials
- Network connectivity issues
- Firewall blocking connections

**Solutions**:
```bash
# Check PostgreSQL service
sudo systemctl status postgresql  # Linux
Get-Service postgresql*           # Windows

# Check SQL Server service  
sudo systemctl status mssql-server  # Linux
Get-Service MSSQLSERVER            # Windows

# Test connectivity
psql -h localhost -U postgres -c "SELECT version();"
sqlcmd -S localhost -E -Q "SELECT @@VERSION"
```

#### 2. "Command not found: psql/sqlcmd"
**Solutions**:
```bash
# Install PostgreSQL client tools
sudo apt-get install postgresql-client  # Ubuntu/Debian
brew install postgresql                 # macOS
# Download from PostgreSQL website      # Windows

# Install SQL Server command line tools
# Download from Microsoft website or use package manager
```

#### 3. "Permission denied creating database"
**Solutions**:
```sql
-- PostgreSQL: Grant createdb permission
ALTER USER postgres CREATEDB;

-- SQL Server: Ensure user has dbcreator role
ALTER SERVER ROLE dbcreator ADD MEMBER [username];
```

#### 4. "Schema loading failed"
**Causes**:
- SQL syntax differences between PostgreSQL/SQL Server
- Missing dependencies
- Insufficient permissions

**Solutions**:
- Check database logs for specific errors
- Ensure user has DDL permissions
- Verify schema file exists and is readable

#### 5. "Sample data loading timeout"
**Solutions**:
- Use `--skip-claims-data` for faster setup (configuration only)
- Increase database memory allocation
- Run during off-peak hours
- Check available disk space

### Performance Tuning

For faster setup on production systems:

```sql
-- PostgreSQL: Temporarily adjust settings
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET checkpoint_segments = 32;
SELECT pg_reload_conf();

-- SQL Server: Temporarily disable constraints
ALTER TABLE claims_line_items NOCHECK CONSTRAINT ALL;
ALTER TABLE claims_diagnosis NOCHECK CONSTRAINT ALL;

-- Remember to re-enable after loading
ALTER TABLE claims_line_items CHECK CONSTRAINT ALL;
ALTER TABLE claims_diagnosis CHECK CONSTRAINT ALL;
```

## Security Considerations

### Production Deployment
- **Use strong passwords** for database accounts
- **Create dedicated service accounts** instead of using admin accounts
- **Configure SSL/TLS** for database connections
- **Implement network security** (VPNs, private networks)
- **Enable database auditing** and monitoring
- **Regular security updates** for database software

### HIPAA Compliance
- **Enable encryption at rest** for database files
- **Configure audit logging** for all data access
- **Implement access controls** and role-based security
- **Set up data masking** for non-production environments
- **Regular backup testing** and disaster recovery procedures

## Integration

### CI/CD Integration
```yaml
# Example GitHub Actions workflow
- name: Setup Database
  run: |
    python scripts/setup_database.py \
      --postgres-host ${{ secrets.POSTGRES_HOST }} \
      --postgres-user ${{ secrets.POSTGRES_USER }} \
      --postgres-password ${{ secrets.POSTGRES_PASSWORD }} \
      --skip-claims-data
```

### Docker Integration
```dockerfile
# Example Dockerfile usage
COPY scripts/ /app/scripts/
RUN python scripts/setup_database.py --postgres-host db --postgres-user postgres --postgres-password $POSTGRES_PASSWORD
```

## Support

For issues with database setup:

1. **Check Prerequisites**: Ensure all required software is installed
2. **Verify Connectivity**: Test database connections independently
3. **Review Logs**: Check database server logs for errors
4. **Use Minimal Setup**: Try `--skip-claims-data` to isolate issues
5. **Check Documentation**: Review database-specific documentation

---

**Note**: This setup creates a complete development/testing environment. For production use, implement additional security measures and follow your organization's database deployment procedures.
# PostgreSQL Claims Processing Architecture

## Overview

The PostgreSQL database is now focused solely on **claims processing**, not reference data storage. This eliminates the schema mismatch issues and creates a clean separation of concerns.

## Database Roles

### PostgreSQL (`claims_staging`)
- **Purpose**: Claims processing workflow
- **Contains**: Claims, line items, batch metadata, processing rules, RVU data
- **References**: Facility and provider data stored in SQL Server
- **Schema**: `postgresql_claims_processing_schema.sql`

### SQL Server (`smart_pro_claims`)  
- **Purpose**: Master data and analytics
- **Contains**: Facilities, providers, organizations, regions, configuration
- **Schema**: `sqlserver_schema.sql`

## Key Changes Made

### 1. Removed Foreign Key Dependencies
- PostgreSQL claims table no longer has FK constraints to facilities/providers
- Uses string references (`facility_id`, `billing_provider_npi`) instead
- Eliminates schema sync issues between databases

### 2. Streamlined PostgreSQL Schema
- **Removed**: `facilities` and `providers` tables
- **Kept**: `claims`, `claim_line_items`, `batch_metadata`, `rvu_data` (matched to SQL Server schema), `validation_rules`
- **Focus**: Processing-specific tables only

### 3. Updated Sample Data Loading
- **PostgreSQL**: Uses hardcoded facility/provider references from SQL Server
- **SQL Server**: Continues to load full reference data
- **No Duplication**: Reference data stays in single source of truth

## Sample Data Strategy

### PostgreSQL Loading
```sql
-- Uses hardcoded references to SQL Server data
facility_ids = ['FAC001', 'FAC002', 'FAC003', 'FAC004', 'FAC005']
provider_npis = [('1234567890', 'Dr. John Smith'), ...]

-- Claims reference these IDs but don't require local tables
INSERT INTO claims (facility_id, billing_provider_npi, ...)
VALUES ('FAC001', '1234567890', ...)
```

### SQL Server Loading  
```sql
-- Full reference data loading
INSERT INTO facilities (facility_id, facility_name, ...)
INSERT INTO providers (npi, provider_name, ...)
INSERT INTO claims (facility_id, ...)  -- With FK constraints
```

## RVU Data Strategy

**Why RVU data is in both databases:**
- **PostgreSQL**: Needed for real-time claims processing calculations
- **SQL Server**: Needed for analytics and reporting
- **Schema**: Identical structure between both databases to maintain consistency

RVU data is essential for claims processing as it's used to:
- Calculate expected reimbursements
- Validate procedure code values
- Perform real-time pricing calculations

## Benefits

1. **No Schema Conflicts**: Different schemas for different purposes (except RVU which matches)
2. **Single Source of Truth**: Facilities/providers only in SQL Server
3. **Processing Performance**: RVU data local to PostgreSQL for fast calculations
4. **Clear Separation**: Processing vs analytics databases
5. **Better Performance**: PostgreSQL optimized for claims processing workflow

## Usage

### Setup PostgreSQL for Claims Processing
```bash
python3 scripts/setup_postgres_for_claims.py
# OR
python3 scripts/setup_database.py --postgres-host localhost --postgres-user claims_user --postgres-password password
```

### Load Claims into PostgreSQL
```bash
python3 scripts/load_sample_data.py --connection-string "postgresql://claims_user:password@localhost:5432/claims_staging"
```

### Query Claims with Reference Data
When you need facility/provider details, join across databases or use application-level lookups to SQL Server.

## Files Changed

1. **`database/postgresql_claims_processing_schema.sql`** - New processing-focused schema
2. **`scripts/load_sample_data.py`** - Updated with `load_claims_data_postgresql_only()`
3. **`scripts/setup_database.py`** - Points to new schema file
4. **`docs/POSTGRES_CLAIMS_ARCHITECTURE.md`** - This documentation

## Migration Notes

- Existing PostgreSQL databases should be recreated with the new schema
- Claims processing workflows should use PostgreSQL connection strings
- Reference data queries should use SQL Server connection strings
- Analytics/reporting can use SQL Server for full dataset access
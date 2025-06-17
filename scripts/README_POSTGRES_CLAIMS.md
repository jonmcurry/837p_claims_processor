# Loading Claims into PostgreSQL Staging Database

**IMPORTANT:** To load claims into PostgreSQL instead of SQL Server, you MUST use a PostgreSQL connection string starting with `postgresql://` or `postgres://`.

The script automatically detects the database type from the connection string:
- `postgresql://...` → Loads into PostgreSQL `public.claims` table  
- `mssql+pyodbc://...` → Loads into SQL Server `dbo.claims` table

## Solutions to Load Claims into PostgreSQL:

## Option 0: Using the Updated setup_database.py (EASIEST)

The setup script now prioritizes PostgreSQL when both databases are configured:

```bash
# Quick interactive setup for PostgreSQL claims processing
python3 scripts/setup_postgres_for_claims.py

# OR setup PostgreSQL directly
python3 scripts/setup_database.py \
    --postgres-host localhost \
    --postgres-user claims_user \
    --postgres-password your_password \
    --postgres-database claims_staging
```

This will:
1. Create the PostgreSQL database
2. Load the claims processing schema  
3. **Automatically load claims into PostgreSQL public.claims table**

## Option 1: Using the Updated PowerShell Script

```powershell
# Load claims into PostgreSQL
.\scripts\load_sample_data.ps1 `
    -ServerName "localhost" `
    -DatabaseName "claims_staging" `
    -Username "claims_user" `
    -Password "your_password" `
    -DatabaseType "postgresql" `
    -Port "5432"
```

## Option 2: Using the Bash Script (Linux/WSL/Mac)

```bash
# Using default connection
./scripts/load_postgres_claims.sh

# Using custom connection string
./scripts/load_postgres_claims.sh "postgresql://claims_user:password@localhost:5432/claims_staging"
```

## Option 3: Direct Python Command

```bash
python scripts/load_sample_data.py --connection-string "postgresql://claims_user:password@localhost:5432/claims_staging"
```

## Important Notes

1. **Database Type Detection**: The script automatically detects PostgreSQL vs SQL Server based on the connection string
2. **Claims Processing**: Claims will only load into PostgreSQL `public.claims` table when using a PostgreSQL connection string
3. **Schema Match**: Claims data will match the PostgreSQL schema exactly with proper foreign keys and data types
4. **Requirements**: Make sure you have `psycopg2-binary` installed for PostgreSQL connections

## Verifying Claims Were Loaded

After running the script, you should see:
- Claims loaded into `public.claims` table
- Line items in `public.claim_line_items` table
- Batch metadata in `public.batch_metadata` table
- Message: "Claims data loaded successfully into PostgreSQL public.claims"

## Why Claims Were Loading into SQL Server

The issue was that the script defaults to SQL Server when it can't detect the database type. Common causes:

1. **Using SQL Server connection string:** `mssql+pyodbc://...` loads into SQL Server
2. **Using PowerShell script without -DatabaseType parameter:** Defaults to SQL Server  
3. **Invalid connection string:** Falls back to SQL Server

## Troubleshooting

**If claims are still loading into SQL Server:**

1. ✅ **Verify connection string:** Must start with `postgresql://` or `postgres://`
2. ✅ **Check script output:** Should show "Detected database type: postgresql"
3. ✅ **Look for this message:** ">>> CLAIMS WILL BE LOADED INTO POSTGRESQL public.claims TABLE"
4. ✅ **Confirm success message:** "Claims data loaded successfully into PostgreSQL public.claims"

**Example of correct vs incorrect:**
```bash
# ✅ CORRECT - Loads into PostgreSQL
postgresql://claims_user:password@localhost:5432/claims_staging

# ❌ INCORRECT - Loads into SQL Server  
mssql+pyodbc://user:pass@localhost:1433/smart_pro_claims?driver=ODBC+Driver+17+for+SQL+Server
```
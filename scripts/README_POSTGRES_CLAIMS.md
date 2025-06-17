# Loading Claims into PostgreSQL

To load claims into PostgreSQL for the claims processing workflow, you have two options:

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

## Troubleshooting

If claims are still loading into SQL Server:
1. Verify your connection string starts with `postgresql://`
2. Check that you're not accidentally using a SQL Server connection string
3. Confirm the script shows "Detected database type: postgresql"
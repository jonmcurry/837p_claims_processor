# Claims Processing Pipeline

## Quick Start

The complete claims processing pipeline is now available in a single script that reads configuration from `config.json`.

### 1. Setup Configuration

The script uses `config.json` for database connections and processing settings:

```json
{
  "databases": {
    "postgresql": {
      "host": "localhost",
      "port": 5432,
      "database": "claims_staging",
      "username": "claims_user",
      "password": "password"
    },
    "sqlserver": {
      "driver": "ODBC Driver 17 for SQL Server",
      "server": "localhost",
      "database": "smart_pro_claims",
      "username": "sa",
      "password": "your_password"
    }
  },
  "processing": {
    "batch_size": 1000,
    "conversion_factor": 38.87
  }
}
```

### 2. Run Processing

```bash
# Using default config.json
python process_claims_complete.py

# Using custom config file
python process_claims_complete.py --config production_config.json

# Override config with command line arguments
python process_claims_complete.py \
  --pg-conn "postgresql://user:pass@host:5432/db" \
  --ss-conn "DRIVER={ODBC Driver 17 for SQL Server};SERVER=server;DATABASE=db;UID=user;PWD=pass"
```

## What It Does

1. **Reads** pending claims from PostgreSQL staging database
2. **Validates** claims against business rules:
   - Service date validations
   - Financial amount checks
   - Provider NPI format validation
   - Diagnosis code requirements
   - Custom rules from `validation_rules` table

3. **Calculates** RVU-based reimbursements:
   - Looks up RVU values for procedure codes
   - Applies units of service
   - Uses Medicare conversion factor

4. **Transfers** to SQL Server:
   - Successful claims → `dbo.claims` and `dbo.claim_line_items`
   - Failed claims → PostgreSQL `failed_claims` table

5. **Updates** metrics:
   - Batch metadata with processing counts
   - Processing metrics for monitoring
   - Validation failure statistics

## Output Example

```
📋 Configuration loaded from: config.json
🔄 Batch size: 1000
💵 Conversion factor: $38.87
🔄 Starting Complete Claims Processing Pipeline...
📋 Loaded 15 validation rules
📊 Processing 250 pending claims...
   ✓ Processed 10/250 claims...
   ✓ Processed 20/250 claims...

============================================================
✅ CLAIMS PROCESSING COMPLETE
============================================================

📊 Processing Statistics:
   • Total Claims: 250
   • Successfully Processed: 235
   • Failed: 15
   • Success Rate: 94.0%
   • Processing Time: 12.45 seconds
   • Throughput: 20.1 claims/second

❌ Validation Failures by Type:
   • invalid_provider: 8
   • date_range_error: 4
   • financial_error: 3
```

## Additional Scripts

- **Reset claims**: `python transfer_claims_to_sqlserver.py --reset`
- **Check status**: `python process_claims_simple.py --check-only`
- **Load sample data**: `python scripts/load_sample_data.py`

## Monitoring

View results in the Claims UI:
```bash
cd claims_ui
python run.py
```

Access at http://localhost:8050
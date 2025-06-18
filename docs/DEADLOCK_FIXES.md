# SQL Server Deadlock Fixes

## Issues Resolved

The application was experiencing SQL Server deadlocks (error 1205) when multiple concurrent processes tried to insert claims data using MERGE statements.

## Implemented Solutions

### 1. Retry Logic with Exponential Backoff
- Added automatic retry mechanism (up to 3 attempts) for deadlock errors
- Implements exponential backoff with jitter to prevent retry storms
- Located in: `src/core/database/batch_operations.py`

### 2. Smaller Batch Sizes
- Reduced batch sizes to minimize lock contention:
  - Claims: 100 → 50 records per batch
  - Line items: 50 → 25 records per batch

### 3. Lock Hints and Timeout Settings
- Added `WITH (UPDLOCK, ROWLOCK)` hints to MERGE statements
- Set `LOCK_TIMEOUT 5000` (5 seconds) to prevent indefinite blocking
- Ensures `READ COMMITTED` isolation level

### 4. Connection Pool Improvements
- Added `pool_pre_ping=True` to test connections before use
- Set `pool_recycle=3600` to refresh connections hourly
- Configured proper isolation level in connection args

### 5. Database Indexes
- Created covering indexes for MERGE operation lookup columns
- Enables SQL Server to acquire locks more efficiently
- Run the script: `scripts/sql_server_deadlock_fix.sql`

## How to Apply the Fixes

1. **Update the code**: The batch_operations.py file has been updated with retry logic and optimized queries.

2. **Run the SQL script** on your SQL Server database:
   ```bash
   sqlcmd -S your_server -d smart_pro_claims -i scripts/sql_server_deadlock_fix.sql
   ```

3. **Monitor deadlocks**: The SQL script includes a query to view recent deadlocks for monitoring.

## Expected Results

- Significantly reduced deadlock occurrences
- Automatic recovery from transient deadlocks
- Better performance due to optimized indexes
- More stable concurrent processing

## Additional Recommendations

1. Consider implementing queue-based processing to further reduce concurrent writes
2. Monitor deadlock frequency using the provided SQL query
3. Adjust batch sizes based on your specific workload if needed
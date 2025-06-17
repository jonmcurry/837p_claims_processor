# Pipeline Runner Script

The `run_pipeline.py` script provides a simple command-line interface to run the claims processing pipeline directly.

## Overview

This script allows you to process batches of claims from PostgreSQL (staging) to SQL Server (production) using the ultra high-performance batch processing pipeline.

## Prerequisites

1. **Environment Setup**: Make sure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **Database Configuration**: Ensure your `.env` file contains the correct database connection settings:
   - PostgreSQL connection settings (pg_host, pg_user, pg_password, etc.)
   - SQL Server connection settings (sql_host, sql_user, sql_password, etc.)

3. **Batch Data**: Ensure your batch exists in the PostgreSQL `batch_metadata` table with pending claims.

## Usage

### Basic Usage
```bash
# Run from project root directory
python scripts/run_pipeline.py BATCH_001
```

### Alternative Syntax
```bash
python scripts/run_pipeline.py --batch-id BATCH_001
```

### Options

- `--dry-run`: Validate claims but don't actually process them
- `--verbose` or `-v`: Enable debug-level logging
- `--config-check`: Check configuration and exit
- `--help`: Show help message

### Examples

```bash
# Process a batch with verbose logging
python scripts/run_pipeline.py --batch-id BATCH_001 --verbose

# Dry run to validate without processing
python scripts/run_pipeline.py --batch-id BATCH_001 --dry-run

# Check configuration
python scripts/run_pipeline.py --config-check
```

## Features

### Error Handling
- Comprehensive error logging with structured output
- Failed claims are logged to the database
- Graceful handling of interruptions (Ctrl+C)

### Performance Monitoring
- Real-time throughput calculation
- Performance target validation
- Detailed processing metrics

### Logging
- Console output with timestamps
- Log file creation (`pipeline_runner.log`)
- Structured JSON logging for monitoring systems

## Pipeline Process

The script executes the following stages:

1. **Data Fetch**: Retrieves pending claims from PostgreSQL
2. **Validation**: Validates claims using business rules
3. **ML Prediction**: Applies machine learning models (if enabled)
4. **Calculation**: Calculates RVU and reimbursement amounts
5. **Data Transfer**: Transfers processed claims to SQL Server

## Performance Targets

The pipeline is optimized for ultra high-performance processing:
- Target: 6,667 claims/second (400k+ claims in 60 seconds)
- Uses parallel processing with dynamic worker allocation
- Optimized SQL queries and connection pooling

## Exit Codes

- `0`: Success
- `1`: General error or processing failure
- `130`: Interrupted by user (Ctrl+C)

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the project root and dependencies are installed
2. **Database Connection**: Check your `.env` file configuration
3. **Batch Not Found**: Verify the batch_id exists in the database
4. **Performance Issues**: Check system resources and database connection pools

### Log Files

- `pipeline_runner.log`: Detailed processing logs
- Check console output for real-time status

## Integration

This script can be integrated into:
- Cron jobs for scheduled processing
- CI/CD pipelines for automated testing
- Monitoring systems for operational oversight
- Docker containers for containerized deployments

## Example Output

```
2024-01-15T10:30:00 - INFO - Starting pipeline runner - batch_id=BATCH_001
2024-01-15T10:30:01 - INFO - Initializing claims processing pipeline
2024-01-15T10:30:02 - INFO - Starting batch processing - batch_id=BATCH_001
2024-01-15T10:30:15 - INFO - ✅ Performance target met! - target=6667 claims/sec - actual=7234.5 claims/sec
2024-01-15T10:30:15 - INFO - ✅ Pipeline execution completed successfully
```
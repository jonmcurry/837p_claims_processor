# 837P Claims Processor - Logging System

## Overview

This document describes the comprehensive file-based logging system implemented for the 837P Claims Processor. The system provides centralized logging with file rotation, structured logging, and error tracking across all components.

## Architecture

### Log Directory Structure

```
logs/
├── claims/          # Claims processing logs
├── api/            # API request/response logs
├── frontend/       # Frontend error logs
├── system/         # System-level logs (database, cache)
├── analytics/      # Analytics and reporting logs
├── security/       # Security and audit logs
└── archive/        # Archived log files
```

### Core Components

1. **Logger Configuration** (`src/core/logging/logger_config.py`)
   - Centralized configuration for all loggers
   - File rotation management
   - Structured logging support
   - Error aggregation

2. **Frontend Logger** (`frontend/src/utils/logger.ts`)
   - Client-side error capture
   - Automatic log batching and transmission
   - User action tracking
   - Performance monitoring

3. **Log Management** (`scripts/log_management.py`)
   - Automated log rotation
   - Log cleanup and archiving
   - Log statistics and monitoring

## Usage

### Python/Backend Logging

```python
from src.core.logging import get_logger, log_error

# Get a logger for your module
logger = get_logger(__name__, "claims", structured=True)

# Log different levels
logger.info("Processing started", batch_id="BATCH123")
logger.warning("High memory usage detected", memory_usage="85%")
logger.error("Processing failed", error="Database timeout")

# Log errors with context
try:
    # Your code here
    pass
except Exception as e:
    log_error(__name__, e, {"batch_id": batch_id, "operation": "process_claims"})
```

### Frontend/TypeScript Logging

```typescript
import { logger, logError, logUserAction } from '@/utils/logger';

// Log user actions
logUserAction('claim_submitted', { claimId: 'CLM123', facility: 'FAC001' });

// Log errors
try {
    // Your code here
} catch (error) {
    logError('Failed to submit claim', error, { claimId: 'CLM123' });
}

// Log API errors
logger.logApiError('/api/claims', error, requestData);
```

### Security/Audit Logging

```python
from src.core.logging import get_audit_logger

audit_logger = get_audit_logger()

# Log security events
audit_logger.info(
    "User login successful",
    extra={
        "user": "john.doe",
        "action": "login",
        "resource": "claims_system",
        "result": "success"
    }
)
```

## Log Types and Levels

### Log Levels
- **DEBUG**: Detailed information for debugging
- **INFO**: General information about system operation
- **WARNING**: Warning messages for potential issues
- **ERROR**: Error messages for failures that don't stop execution
- **CRITICAL**: Critical errors that may stop system operation

### Log Categories
- **Claims**: Claims processing, validation, ML predictions
- **API**: HTTP requests, responses, middleware operations
- **Frontend**: Client-side errors, user actions, performance
- **System**: Database, cache, infrastructure operations
- **Analytics**: Reporting, data analysis operations
- **Security**: Authentication, authorization, audit trails

## Configuration

### Log File Rotation

- **Claims logs**: 100MB max, 10 backups
- **API logs**: 50MB max, 10 backups
- **Frontend logs**: 50MB max, 5 backups
- **System logs**: 100MB max, 10 backups
- **Security logs**: 100MB max, 20 backups (kept longer for compliance)

### Log Formats

**Structured JSON Format** (Python):
```json
{
  "timestamp": "2025-01-17T10:30:00.000Z",
  "level": "info",
  "logger": "claims.processor",
  "message": "Claim processed successfully",
  "batch_id": "BATCH123",
  "claim_id": "CLM456",
  "processing_time": 1.23
}
```

**Standard Format** (General):
```
2025-01-17 10:30:00 - claims.processor - INFO - Claim processed successfully
```

## Management Commands

### Manual Log Operations

```bash
# Rotate large log files
python scripts/log_management.py --rotate

# Clean up logs older than 30 days
python scripts/log_management.py --cleanup 30

# Archive logs older than 7 days
python scripts/log_management.py --archive 7

# View log statistics
python scripts/log_management.py --stats

# Generate comprehensive report
python scripts/log_management.py --report

# Monitor log growth
python scripts/log_management.py --monitor
```

### Automated Maintenance

Set up cron jobs for automated log management:

```bash
# Daily log rotation (if needed)
0 2 * * * /path/to/python /path/to/scripts/log_management.py --rotate

# Weekly cleanup of old logs
0 3 * * 0 /path/to/python /path/to/scripts/log_management.py --cleanup 30

# Monthly archiving
0 4 1 * * /path/to/python /path/to/scripts/log_management.py --archive 7
```

## Monitoring and Alerting

### Key Metrics to Monitor

1. **Log Volume**: Track log file sizes and growth rates
2. **Error Rates**: Monitor error/warning frequency
3. **Disk Usage**: Ensure sufficient space for logs
4. **Performance Impact**: Monitor logging overhead

### Alert Conditions

- Log directory exceeds 1GB
- Error rate exceeds 5% of total logs
- Critical errors detected
- Log rotation failures
- Frontend error spikes

## Security Considerations

### Data Protection

- **PII/PHI Redaction**: Sensitive data is automatically filtered from logs
- **Access Control**: Log files have restricted file permissions
- **Audit Trail**: All security events are logged with detailed context
- **Encryption**: Consider encrypting archived logs for compliance

### Compliance

- **HIPAA**: Security logs maintained for audit requirements
- **Data Retention**: Configurable retention periods
- **Access Logging**: All log access is tracked

## Performance Impact

### Optimizations

- **Async Logging**: Non-blocking log operations
- **Batching**: Frontend logs are batched for efficiency
- **Compression**: Rotated logs are automatically compressed
- **Caching**: Logger instances are cached to reduce overhead

### Monitoring

- Log writing latency: < 1ms average
- Disk I/O impact: < 5% of system capacity
- Memory usage: < 100MB for log buffers

## Troubleshooting

### Common Issues

1. **Log Files Not Created**
   - Check directory permissions
   - Verify logger configuration
   - Check disk space

2. **High Log Volume**
   - Review log levels (reduce DEBUG in production)
   - Implement more aggressive rotation
   - Consider log sampling for high-frequency events

3. **Missing Frontend Logs**
   - Check network connectivity to log endpoint
   - Verify CORS configuration
   - Check browser console for errors

4. **Performance Issues**
   - Review log volume and frequency
   - Consider async logging
   - Check disk I/O performance

### Debug Commands

```bash
# Check log directory status
ls -la logs/

# Monitor real-time logs
tail -f logs/claims/claims_processing.log

# Check log file sizes
du -sh logs/*

# Search for specific errors
grep -r "ERROR" logs/

# Count log entries by level
grep -c "ERROR\|WARNING\|INFO" logs/system/system.log
```

## Development Guidelines

### Adding Logging to New Modules

1. Import the logging utilities:
   ```python
   from src.core.logging import get_logger, log_error
   ```

2. Create a logger instance:
   ```python
   logger = get_logger(__name__, "category", structured=True)
   ```

3. Add appropriate logging throughout your code:
   - Start/end of major operations
   - Error conditions with context
   - Important state changes
   - Performance metrics

### Best Practices

- **Use appropriate log levels**: Don't log everything as ERROR
- **Include context**: Add relevant data to help with debugging
- **Avoid sensitive data**: Never log passwords, keys, or PHI
- **Performance aware**: Don't log in tight loops without sampling
- **Error context**: Always include operation context with errors

## Future Enhancements

### Planned Features

1. **Centralized Logging**: Integration with ELK stack or similar
2. **Real-time Monitoring**: Dashboard for log metrics
3. **Automated Alerting**: Integration with monitoring systems
4. **Log Analytics**: Advanced search and analysis capabilities
5. **Distributed Tracing**: Request tracing across services

### Integration Opportunities

- **Prometheus**: Export log metrics
- **Grafana**: Visualization dashboards
- **Elasticsearch**: Advanced log search
- **Slack/Teams**: Alert notifications
- **SIEM**: Security information management
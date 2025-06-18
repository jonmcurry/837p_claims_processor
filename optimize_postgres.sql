-- PostgreSQL High-Performance Configuration for Claims Processing
-- Execute these commands as a superuser to optimize for bulk insert throughput

-- Connection and Memory Settings (require restart)
ALTER SYSTEM SET max_connections = 300;
ALTER SYSTEM SET shared_buffers = '1GB';
ALTER SYSTEM SET effective_cache_size = '4GB';

-- Write Performance Optimizations (most require restart)
ALTER SYSTEM SET wal_buffers = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET checkpoint_timeout = '15min';
ALTER SYSTEM SET max_wal_size = '8GB';
ALTER SYSTEM SET min_wal_size = '2GB';

-- Parallel Processing (require restart)
ALTER SYSTEM SET max_parallel_workers = 8;
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;

-- I/O Performance (can be changed without restart)
ALTER SYSTEM SET random_page_cost = 1.1;  -- For SSD storage
ALTER SYSTEM SET seq_page_cost = 1.0;
ALTER SYSTEM SET effective_io_concurrency = 200;  -- For SSD

-- Session-level defaults (these will be overridden per session)
ALTER SYSTEM SET work_mem = '256MB';  -- Default, sessions will override
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET synchronous_commit = off;  -- Async commits for speed

-- Apply changes that don't require restart
SELECT pg_reload_conf();

-- Show current settings
SHOW shared_buffers;
SHOW max_connections;
SHOW checkpoint_completion_target;
SHOW synchronous_commit;

-- Note: Most performance settings require a PostgreSQL restart to take effect
-- Run: sudo systemctl restart postgresql (Linux) or restart PostgreSQL service (Windows)
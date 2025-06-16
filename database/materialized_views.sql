-- Materialized Views for High-Performance Analytics Queries
-- Optimized for claims processing system analytics and reporting

-- =============================================================================
-- CLAIMS PROCESSING ANALYTICS MATERIALIZED VIEWS
-- =============================================================================

-- Daily Claims Processing Summary
CREATE MATERIALIZED VIEW mv_daily_claims_summary AS
SELECT 
    DATE(created_at) as processing_date,
    facility_id,
    COUNT(*) as total_claims,
    COUNT(CASE WHEN processing_status = 'completed' THEN 1 END) as completed_claims,
    COUNT(CASE WHEN processing_status = 'failed' THEN 1 END) as failed_claims,
    COUNT(CASE WHEN processing_status = 'processing' THEN 1 END) as processing_claims,
    SUM(total_charges) as total_charges_amount,
    AVG(total_charges) as avg_claim_amount,
    MAX(total_charges) as max_claim_amount,
    MIN(total_charges) as min_claim_amount,
    COUNT(DISTINCT patient_account_number) as unique_patients,
    COUNT(DISTINCT billing_provider_npi) as unique_providers,
    AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) as avg_processing_time_seconds
FROM claims
WHERE created_at >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY DATE(created_at), facility_id;

-- Index for fast date range queries
CREATE UNIQUE INDEX idx_mv_daily_claims_summary_pk 
ON mv_daily_claims_summary (processing_date, facility_id);

CREATE INDEX idx_mv_daily_claims_summary_date 
ON mv_daily_claims_summary (processing_date);

CREATE INDEX idx_mv_daily_claims_summary_facility 
ON mv_daily_claims_summary (facility_id);

-- =============================================================================

-- Hourly Claims Throughput (for real-time monitoring)
CREATE MATERIALIZED VIEW mv_hourly_claims_throughput AS
SELECT 
    DATE_TRUNC('hour', created_at) as processing_hour,
    facility_id,
    COUNT(*) as claims_count,
    COUNT(*) / 3600.0 as claims_per_second,
    SUM(total_charges) as total_revenue,
    COUNT(CASE WHEN processing_status = 'failed' THEN 1 END) as failed_count,
    (COUNT(CASE WHEN processing_status = 'failed' THEN 1 END)::float / COUNT(*)) * 100 as failure_rate_percent
FROM claims
WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '7 days'
GROUP BY DATE_TRUNC('hour', created_at), facility_id;

-- Index for real-time dashboard queries
CREATE UNIQUE INDEX idx_mv_hourly_throughput_pk 
ON mv_hourly_claims_throughput (processing_hour, facility_id);

CREATE INDEX idx_mv_hourly_throughput_recent 
ON mv_hourly_claims_throughput (processing_hour DESC);

-- =============================================================================

-- Failed Claims Analysis Summary
CREATE MATERIALIZED VIEW mv_failed_claims_analysis AS
SELECT 
    DATE(failed_at) as failure_date,
    facility_id,
    failure_category,
    failure_reason,
    COUNT(*) as failure_count,
    SUM(charge_amount) as total_failed_amount,
    AVG(charge_amount) as avg_failed_amount,
    COUNT(CASE WHEN resolution_status = 'resolved' THEN 1 END) as resolved_count,
    COUNT(CASE WHEN resolution_status = 'pending' THEN 1 END) as pending_count,
    AVG(CASE 
        WHEN resolution_status = 'resolved' AND resolved_at IS NOT NULL 
        THEN EXTRACT(EPOCH FROM (resolved_at - failed_at)) / 3600.0 
    END) as avg_resolution_time_hours
FROM failed_claims
WHERE failed_at >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY DATE(failed_at), facility_id, failure_category, failure_reason;

-- Index for failure analysis queries
CREATE INDEX idx_mv_failed_claims_analysis_date 
ON mv_failed_claims_analysis (failure_date);

CREATE INDEX idx_mv_failed_claims_analysis_category 
ON mv_failed_claims_analysis (failure_category);

CREATE INDEX idx_mv_failed_claims_analysis_facility 
ON mv_failed_claims_analysis (facility_id);

-- =============================================================================

-- Provider Performance Analytics
CREATE MATERIALIZED VIEW mv_provider_performance AS
SELECT 
    billing_provider_npi,
    billing_provider_name,
    facility_id,
    DATE_TRUNC('month', created_at) as performance_month,
    COUNT(*) as total_claims,
    COUNT(CASE WHEN processing_status = 'completed' THEN 1 END) as successful_claims,
    COUNT(CASE WHEN processing_status = 'failed' THEN 1 END) as failed_claims,
    (COUNT(CASE WHEN processing_status = 'completed' THEN 1 END)::float / COUNT(*)) * 100 as success_rate_percent,
    SUM(total_charges) as total_charges,
    AVG(total_charges) as avg_claim_amount,
    COUNT(DISTINCT patient_account_number) as unique_patients,
    COUNT(DISTINCT primary_diagnosis_code) as unique_diagnoses,
    AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) as avg_processing_time_seconds
FROM claims
WHERE created_at >= CURRENT_DATE - INTERVAL '12 months'
    AND billing_provider_npi IS NOT NULL
GROUP BY billing_provider_npi, billing_provider_name, facility_id, DATE_TRUNC('month', created_at);

-- Index for provider analytics
CREATE INDEX idx_mv_provider_performance_npi 
ON mv_provider_performance (billing_provider_npi);

CREATE INDEX idx_mv_provider_performance_month 
ON mv_provider_performance (performance_month);

CREATE INDEX idx_mv_provider_performance_facility 
ON mv_provider_performance (facility_id);

-- =============================================================================

-- Diagnosis Code Analytics (RVU Analysis)
CREATE MATERIALIZED VIEW mv_diagnosis_analytics AS
SELECT 
    primary_diagnosis_code,
    DATE_TRUNC('month', c.created_at) as analytics_month,
    COUNT(*) as claim_count,
    SUM(c.total_charges) as total_charges,
    AVG(c.total_charges) as avg_charges_per_claim,
    COUNT(DISTINCT c.patient_account_number) as unique_patients,
    COUNT(DISTINCT c.billing_provider_npi) as unique_providers,
    COUNT(DISTINCT c.facility_id) as unique_facilities,
    -- Line item analytics
    SUM(cli.total_line_items) as total_line_items,
    AVG(cli.total_line_items) as avg_line_items_per_claim,
    SUM(cli.total_units) as total_units,
    AVG(cli.total_units) as avg_units_per_claim,
    SUM(cli.total_line_charges) as total_line_charges,
    -- Processing status
    COUNT(CASE WHEN c.processing_status = 'completed' THEN 1 END) as completed_claims,
    COUNT(CASE WHEN c.processing_status = 'failed' THEN 1 END) as failed_claims,
    (COUNT(CASE WHEN c.processing_status = 'completed' THEN 1 END)::float / COUNT(*)) * 100 as completion_rate_percent
FROM claims c
LEFT JOIN (
    SELECT 
        claim_id,
        COUNT(*) as total_line_items,
        SUM(units) as total_units,
        SUM(charge_amount) as total_line_charges
    FROM claim_line_items
    GROUP BY claim_id
) cli ON c.claim_id = cli.claim_id
WHERE c.created_at >= CURRENT_DATE - INTERVAL '12 months'
    AND c.primary_diagnosis_code IS NOT NULL
GROUP BY primary_diagnosis_code, DATE_TRUNC('month', c.created_at);

-- Index for diagnosis analytics
CREATE INDEX idx_mv_diagnosis_analytics_code 
ON mv_diagnosis_analytics (primary_diagnosis_code);

CREATE INDEX idx_mv_diagnosis_analytics_month 
ON mv_diagnosis_analytics (analytics_month);

-- =============================================================================

-- Procedure Code Performance (RVU and Payer Analysis)
CREATE MATERIALIZED VIEW mv_procedure_performance AS
SELECT 
    cli.procedure_code,
    cli.procedure_description,
    c.insurance_type as payer_type,
    DATE_TRUNC('month', c.created_at) as performance_month,
    COUNT(*) as procedure_count,
    SUM(cli.charge_amount) as total_charges,
    AVG(cli.charge_amount) as avg_charge_per_procedure,
    SUM(cli.units) as total_units,
    AVG(cli.units) as avg_units_per_procedure,
    COUNT(DISTINCT c.claim_id) as unique_claims,
    COUNT(DISTINCT c.patient_account_number) as unique_patients,
    COUNT(DISTINCT c.billing_provider_npi) as unique_providers,
    -- Performance metrics
    COUNT(CASE WHEN c.processing_status = 'completed' THEN 1 END) as successful_procedures,
    COUNT(CASE WHEN c.processing_status = 'failed' THEN 1 END) as failed_procedures,
    (COUNT(CASE WHEN c.processing_status = 'completed' THEN 1 END)::float / COUNT(*)) * 100 as success_rate_percent
FROM claim_line_items cli
INNER JOIN claims c ON cli.claim_id = c.claim_id
WHERE c.created_at >= CURRENT_DATE - INTERVAL '12 months'
    AND cli.procedure_code IS NOT NULL
GROUP BY cli.procedure_code, cli.procedure_description, c.insurance_type, DATE_TRUNC('month', c.created_at);

-- Index for procedure analytics
CREATE INDEX idx_mv_procedure_performance_code 
ON mv_procedure_performance (procedure_code);

CREATE INDEX idx_mv_procedure_performance_payer 
ON mv_procedure_performance (payer_type);

CREATE INDEX idx_mv_procedure_performance_month 
ON mv_procedure_performance (performance_month);

-- =============================================================================

-- Facility Performance Dashboard
CREATE MATERIALIZED VIEW mv_facility_dashboard AS
SELECT 
    f.facility_id,
    f.facility_name,
    DATE_TRUNC('day', c.created_at) as performance_date,
    -- Volume metrics
    COUNT(*) as daily_claims,
    COUNT(DISTINCT c.patient_account_number) as unique_patients,
    COUNT(DISTINCT c.billing_provider_npi) as unique_providers,
    -- Financial metrics
    SUM(c.total_charges) as daily_revenue,
    AVG(c.total_charges) as avg_claim_amount,
    -- Processing metrics
    COUNT(CASE WHEN c.processing_status = 'completed' THEN 1 END) as completed_claims,
    COUNT(CASE WHEN c.processing_status = 'failed' THEN 1 END) as failed_claims,
    COUNT(CASE WHEN c.processing_status = 'processing' THEN 1 END) as processing_claims,
    (COUNT(CASE WHEN c.processing_status = 'completed' THEN 1 END)::float / COUNT(*)) * 100 as completion_rate_percent,
    AVG(EXTRACT(EPOCH FROM (c.updated_at - c.created_at))) as avg_processing_time_seconds,
    -- Failed claims analysis
    COALESCE(fc.failed_amount, 0) as failed_claims_amount,
    COALESCE(fc.failure_categories, 0) as unique_failure_categories
FROM facilities f
INNER JOIN claims c ON f.facility_id = c.facility_id
LEFT JOIN (
    SELECT 
        facility_id,
        DATE(failed_at) as failure_date,
        SUM(charge_amount) as failed_amount,
        COUNT(DISTINCT failure_category) as failure_categories
    FROM failed_claims
    WHERE failed_at >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY facility_id, DATE(failed_at)
) fc ON f.facility_id = fc.facility_id AND DATE(c.created_at) = fc.failure_date
WHERE c.created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY f.facility_id, f.facility_name, DATE_TRUNC('day', c.created_at), fc.failed_amount, fc.failure_categories;

-- Index for facility dashboard
CREATE INDEX idx_mv_facility_dashboard_facility 
ON mv_facility_dashboard (facility_id);

CREATE INDEX idx_mv_facility_dashboard_date 
ON mv_facility_dashboard (performance_date);

-- =============================================================================

-- Real-time Performance Metrics (5-minute intervals)
CREATE MATERIALIZED VIEW mv_realtime_metrics AS
SELECT 
    DATE_TRUNC('minute', created_at) - 
    (EXTRACT(MINUTE FROM created_at)::integer % 5) * INTERVAL '1 minute' as metric_interval,
    facility_id,
    COUNT(*) as claims_processed,
    COUNT(*) / 300.0 as claims_per_second,  -- 5 minutes = 300 seconds
    COUNT(CASE WHEN processing_status = 'completed' THEN 1 END) as successful_claims,
    COUNT(CASE WHEN processing_status = 'failed' THEN 1 END) as failed_claims,
    SUM(total_charges) as total_revenue,
    AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) as avg_processing_time_seconds,
    MIN(created_at) as interval_start,
    MAX(created_at) as interval_end
FROM claims
WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
GROUP BY 
    DATE_TRUNC('minute', created_at) - 
    (EXTRACT(MINUTE FROM created_at)::integer % 5) * INTERVAL '1 minute',
    facility_id;

-- Index for real-time queries
CREATE UNIQUE INDEX idx_mv_realtime_metrics_pk 
ON mv_realtime_metrics (metric_interval, facility_id);

CREATE INDEX idx_mv_realtime_metrics_recent 
ON mv_realtime_metrics (metric_interval DESC);

-- =============================================================================

-- Batch Processing Performance Analytics
CREATE MATERIALIZED VIEW mv_batch_performance AS
SELECT 
    batch_id,
    facility_id,
    source_system,
    DATE_TRUNC('hour', created_at) as processing_hour,
    total_claims,
    processed_claims,
    failed_claims,
    (processed_claims::float / total_claims) * 100 as success_rate_percent,
    EXTRACT(EPOCH FROM (completed_at - started_at)) as processing_time_seconds,
    processed_claims / GREATEST(EXTRACT(EPOCH FROM (completed_at - started_at)), 1) as throughput_claims_per_second,
    submitted_by,
    priority_level
FROM batch_metadata
WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
    AND status = 'completed';

-- Index for batch analytics
CREATE INDEX idx_mv_batch_performance_hour 
ON mv_batch_performance (processing_hour);

CREATE INDEX idx_mv_batch_performance_facility 
ON mv_batch_performance (facility_id);

CREATE INDEX idx_mv_batch_performance_throughput 
ON mv_batch_performance (throughput_claims_per_second DESC);

-- =============================================================================
-- REFRESH PROCEDURES AND SCHEDULING
-- =============================================================================

-- Function to refresh all materialized views
CREATE OR REPLACE FUNCTION refresh_analytics_views() 
RETURNS void AS $$
BEGIN
    -- Refresh in dependency order
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_daily_claims_summary;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_hourly_claims_throughput;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_failed_claims_analysis;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_provider_performance;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_diagnosis_analytics;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_procedure_performance;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_facility_dashboard;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_realtime_metrics;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_batch_performance;
    
    -- Log refresh completion
    INSERT INTO system_logs (log_level, message, details, created_at)
    VALUES ('INFO', 'Analytics materialized views refreshed', 
            jsonb_build_object('refreshed_at', NOW()), NOW());
END;
$$ LANGUAGE plpgsql;

-- Function to refresh only real-time views (frequent updates)
CREATE OR REPLACE FUNCTION refresh_realtime_views() 
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_hourly_claims_throughput;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_realtime_metrics;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_facility_dashboard;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- PERFORMANCE OPTIMIZATION QUERIES
-- =============================================================================

-- Query to analyze view performance and usage
CREATE VIEW view_performance_stats AS
SELECT 
    schemaname,
    matviewname as view_name,
    n_tup_ins as rows_inserted,
    n_tup_upd as rows_updated,
    n_tup_del as rows_deleted,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||matviewname)) as size,
    last_refresh_time
FROM pg_stat_user_tables t
JOIN (
    SELECT 
        schemaname,
        matviewname,
        CASE 
            WHEN ispopulated THEN 'populated'
            ELSE 'not_populated'
        END as status,
        pg_stat_get_last_analyze_time(oid) as last_refresh_time
    FROM pg_matviews
) mv ON t.schemaname = mv.schemaname AND t.relname = mv.matviewname
WHERE t.schemaname = 'public' 
    AND t.relname LIKE 'mv_%'
ORDER BY pg_total_relation_size(schemaname||'.'||relname) DESC;

-- Query to identify slow-performing views
CREATE VIEW slow_view_queries AS
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows,
    100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
FROM pg_stat_statements
WHERE query ILIKE '%mv_%'
    AND calls > 10
ORDER BY mean_time DESC
LIMIT 20;

-- =============================================================================
-- AUTOMATED REFRESH SCHEDULING (PostgreSQL with pg_cron extension)
-- =============================================================================

-- Schedule real-time view refresh every 5 minutes
-- SELECT cron.schedule('realtime-views-refresh', '*/5 * * * *', 'SELECT refresh_realtime_views();');

-- Schedule full analytics refresh every hour
-- SELECT cron.schedule('analytics-views-refresh', '0 * * * *', 'SELECT refresh_analytics_views();');

-- Schedule daily view optimization and statistics update
-- SELECT cron.schedule('daily-view-maintenance', '0 2 * * *', 
--     'ANALYZE; VACUUM ANALYZE; SELECT refresh_analytics_views();');

-- =============================================================================
-- INDEXES FOR OPTIMAL QUERY PERFORMANCE
-- =============================================================================

-- Additional indexes on base tables for materialized view performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_claims_created_at_facility 
ON claims (created_at, facility_id) WHERE created_at >= CURRENT_DATE - INTERVAL '90 days';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_claims_status_created_at 
ON claims (processing_status, created_at) WHERE created_at >= CURRENT_DATE - INTERVAL '30 days';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_failed_claims_failed_at_category 
ON failed_claims (failed_at, failure_category) WHERE failed_at >= CURRENT_DATE - INTERVAL '90 days';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_line_items_procedure_claim 
ON claim_line_items (procedure_code, claim_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_batch_metadata_created_status 
ON batch_metadata (created_at, status) WHERE created_at >= CURRENT_DATE - INTERVAL '7 days';

-- Partial indexes for recent data (better performance)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_claims_recent_processing 
ON claims (created_at, processing_status, facility_id) 
WHERE created_at >= CURRENT_DATE - INTERVAL '7 days';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_failed_claims_recent 
ON failed_claims (failed_at, failure_category, resolution_status) 
WHERE failed_at >= CURRENT_DATE - INTERVAL '30 days';

-- =============================================================================
-- MONITORING AND ALERTING VIEWS
-- =============================================================================

-- View for monitoring materialized view freshness
CREATE VIEW mv_freshness_monitor AS
SELECT 
    'mv_daily_claims_summary' as view_name,
    EXTRACT(EPOCH FROM (NOW() - MAX(processing_date))) / 3600 as hours_since_last_data,
    CASE 
        WHEN EXTRACT(EPOCH FROM (NOW() - MAX(processing_date))) / 3600 > 25 THEN 'STALE'
        WHEN EXTRACT(EPOCH FROM (NOW() - MAX(processing_date))) / 3600 > 2 THEN 'WARNING'
        ELSE 'FRESH'
    END as freshness_status
FROM mv_daily_claims_summary
UNION ALL
SELECT 
    'mv_realtime_metrics' as view_name,
    EXTRACT(EPOCH FROM (NOW() - MAX(metric_interval))) / 60 as minutes_since_last_data,
    CASE 
        WHEN EXTRACT(EPOCH FROM (NOW() - MAX(metric_interval))) / 60 > 10 THEN 'STALE'
        WHEN EXTRACT(EPOCH FROM (NOW() - MAX(metric_interval))) / 60 > 6 THEN 'WARNING'
        ELSE 'FRESH'
    END as freshness_status
FROM mv_realtime_metrics;

-- Performance threshold monitoring
CREATE VIEW performance_alerts AS
SELECT 
    'throughput' as metric_type,
    facility_id,
    metric_interval,
    claims_per_second as current_value,
    6667 as target_value,
    CASE 
        WHEN claims_per_second < 4000 THEN 'CRITICAL'
        WHEN claims_per_second < 5000 THEN 'WARNING'
        ELSE 'OK'
    END as alert_level
FROM mv_realtime_metrics
WHERE metric_interval >= NOW() - INTERVAL '1 hour'
    AND claims_per_second > 0
UNION ALL
SELECT 
    'failure_rate' as metric_type,
    facility_id,
    processing_date::timestamp as metric_interval,
    (failed_claims::float / total_claims * 100) as current_value,
    5.0 as target_value,
    CASE 
        WHEN (failed_claims::float / total_claims * 100) > 10 THEN 'CRITICAL'
        WHEN (failed_claims::float / total_claims * 100) > 5 THEN 'WARNING'
        ELSE 'OK'
    END as alert_level
FROM mv_daily_claims_summary
WHERE processing_date >= CURRENT_DATE - INTERVAL '1 day'
    AND total_claims > 0;
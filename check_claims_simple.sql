-- Check claim status distribution
-- Run: psql -U postgres -d claims_staging -f check_claims_simple.sql

\echo 'Claim Status Distribution:'
\echo '========================='

SELECT 
    processing_status, 
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
FROM claims 
GROUP BY processing_status 
ORDER BY count DESC;

\echo ''
\echo 'Summary:'
SELECT 
    COUNT(*) FILTER (WHERE processing_status = 'pending') as pending_claims,
    COUNT(*) FILTER (WHERE processing_status = 'completed') as completed_claims,
    COUNT(*) FILTER (WHERE processing_status NOT IN ('pending', 'completed')) as other_status,
    COUNT(*) as total_claims
FROM claims;

\echo ''
\echo 'Recent Processing Activity (last 10 processed):'
SELECT 
    claim_id,
    processing_status,
    processed_at,
    expected_reimbursement
FROM claims 
WHERE processed_at IS NOT NULL
ORDER BY processed_at DESC 
LIMIT 10;
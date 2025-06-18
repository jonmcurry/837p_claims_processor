-- Reset all completed claims back to pending status
-- Run this if you want to process all 100,000 claims again

-- Check current status distribution
SELECT processing_status, COUNT(*) as count 
FROM claims 
GROUP BY processing_status 
ORDER BY count DESC;

-- Reset completed claims to pending
UPDATE claims 
SET processing_status = 'pending'::processing_status,
    processed_at = NULL,
    updated_at = NOW()
WHERE processing_status = 'completed';

-- Verify the update
SELECT processing_status, COUNT(*) as count 
FROM claims 
GROUP BY processing_status 
ORDER BY count DESC;
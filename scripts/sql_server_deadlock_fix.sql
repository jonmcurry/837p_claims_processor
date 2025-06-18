-- SQL Server Deadlock Prevention Indexes
-- These indexes help prevent deadlocks during concurrent MERGE operations

-- Add non-clustered indexes for the MERGE operation lookup columns
-- This helps SQL Server acquire locks more efficiently
use smart_pro_claims
-- Index for claims table MERGE operation
IF NOT EXISTS (
    SELECT * FROM sys.indexes 
    WHERE name = 'IX_claims_facility_patient' 
    AND object_id = OBJECT_ID('dbo.claims')
)
BEGIN
    CREATE NONCLUSTERED INDEX IX_claims_facility_patient
    ON dbo.claims (facility_id, patient_account_number)
    INCLUDE (medical_record_number, patient_name, first_name, last_name, 
             date_of_birth, gender, financial_class_id, secondary_insurance)
    WITH (FILLFACTOR = 90, PAD_INDEX = ON);
END
GO

-- Index for claims_line_items table MERGE operation
IF NOT EXISTS (
    SELECT * FROM sys.indexes 
    WHERE name = 'IX_claims_line_items_facility_patient_line' 
    AND object_id = OBJECT_ID('dbo.claims_line_items')
)
BEGIN
    CREATE NONCLUSTERED INDEX IX_claims_line_items_facility_patient_line
    ON dbo.claims_line_items (facility_id, patient_account_number, line_number)
    INCLUDE (procedure_code, units, charge_amount, service_from_date, 
             service_to_date, diagnosis_pointer, rendering_provider_id)
    WITH (FILLFACTOR = 90, PAD_INDEX = ON);
END
GO

-- Update statistics to ensure query optimizer uses the new indexes
UPDATE STATISTICS dbo.claims WITH FULLSCAN;
UPDATE STATISTICS dbo.claims_line_items WITH FULLSCAN;
GO

-- Set database options to reduce blocking
ALTER DATABASE CURRENT SET READ_COMMITTED_SNAPSHOT ON;
GO

-- Optional: Add monitoring for deadlock detection
-- This query shows recent deadlocks
SELECT 
    xed.value('@timestamp', 'datetime') as timestamp,
    xed.query('.') as deadlock_graph
FROM (
    SELECT CAST(target_data AS XML) AS target_data
    FROM sys.dm_xe_session_targets st
    JOIN sys.dm_xe_sessions s ON s.address = st.event_session_address
    WHERE s.name = 'system_health'
    AND st.target_name = 'ring_buffer'
) AS data
CROSS APPLY target_data.nodes('RingBufferTarget/event[@name="xml_deadlock_report"]') AS xEventData(xed)
WHERE xed.value('@timestamp', 'datetime') > DATEADD(hour, -24, GETDATE())
ORDER BY timestamp DESC;
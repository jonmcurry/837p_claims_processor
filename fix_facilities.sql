-- Add missing facilities to SQL Server to fix foreign key constraint error
-- Run this against the smart_pro_claims database

USE smart_pro_claims;
GO

-- Check current facilities
SELECT 'Before' as Status, COUNT(*) as FacilityCount FROM dbo.facilities;

-- Insert the required facilities if they don't exist
IF NOT EXISTS (SELECT 1 FROM dbo.facilities WHERE facility_id = 'FAC001')
BEGIN
    INSERT INTO dbo.facilities (
        facility_id, facility_name, npi, tax_id, 
        address_line1, city, state, zip_code,
        phone, created_at, active
    ) VALUES (
        'FAC001', 'Springfield General Hospital', '1234567890', '11-1234567',
        '123 Main St', 'Springfield', 'IL', '62701',
        '555-123-4567', GETDATE(), 1
    );
    PRINT 'Added facility FAC001 - Springfield General Hospital';
END

IF NOT EXISTS (SELECT 1 FROM dbo.facilities WHERE facility_id = 'FAC002')
BEGIN
    INSERT INTO dbo.facilities (
        facility_id, facility_name, npi, tax_id, 
        address_line1, city, state, zip_code,
        phone, created_at, active
    ) VALUES (
        'FAC002', 'Metropolis Regional Medical Center', '2345678901', '22-2345678',
        '456 Oak Ave', 'Metropolis', 'IL', '62701',
        '555-234-5678', GETDATE(), 1
    );
    PRINT 'Added facility FAC002 - Metropolis Regional Medical Center';
END

IF NOT EXISTS (SELECT 1 FROM dbo.facilities WHERE facility_id = 'FAC003')
BEGIN
    INSERT INTO dbo.facilities (
        facility_id, facility_name, npi, tax_id, 
        address_line1, city, state, zip_code,
        phone, created_at, active
    ) VALUES (
        'FAC003', 'Gotham City Medical Center', '3456789012', '33-3456789',
        '789 Pine St', 'Gotham', 'IL', '62701',
        '555-345-6789', GETDATE(), 1
    );
    PRINT 'Added facility FAC003 - Gotham City Medical Center';
END

IF NOT EXISTS (SELECT 1 FROM dbo.facilities WHERE facility_id = 'FAC004')
BEGIN
    INSERT INTO dbo.facilities (
        facility_id, facility_name, npi, tax_id, 
        address_line1, city, state, zip_code,
        phone, created_at, active
    ) VALUES (
        'FAC004', 'Central City Hospital', '4567890123', '44-4567890',
        '321 Elm St', 'Central City', 'IL', '62701',
        '555-456-7890', GETDATE(), 1
    );
    PRINT 'Added facility FAC004 - Central City Hospital';
END

IF NOT EXISTS (SELECT 1 FROM dbo.facilities WHERE facility_id = 'FAC005')
BEGIN
    INSERT INTO dbo.facilities (
        facility_id, facility_name, npi, tax_id, 
        address_line1, city, state, zip_code,
        phone, created_at, active
    ) VALUES (
        'FAC005', 'Star City General Hospital', '5678901234', '55-5678901',
        '654 Maple Dr', 'Star City', 'IL', '62701',
        '555-567-8901', GETDATE(), 1
    );
    PRINT 'Added facility FAC005 - Star City General Hospital';
END

-- Verify the facilities were added
SELECT 'After' as Status, COUNT(*) as FacilityCount FROM dbo.facilities;

-- Show all facilities
SELECT facility_id, facility_name, npi, city, state, active 
FROM dbo.facilities 
ORDER BY facility_id;

PRINT 'Facility setup complete. Claims processing should now work without foreign key errors.';
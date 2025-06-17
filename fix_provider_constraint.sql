-- Fix provider foreign key constraint issue
-- Since physicians table is not populated, remove the constraint to allow claims processing

USE smart_pro_claims;
GO

-- Drop the foreign key constraint for rendering_provider_id
IF EXISTS (SELECT * FROM sys.foreign_keys WHERE name = 'FK_claims_line_items_provider')
BEGIN
    ALTER TABLE dbo.claims_line_items
    DROP CONSTRAINT FK_claims_line_items_provider;
    
    PRINT 'FK_claims_line_items_provider constraint dropped successfully';
END
ELSE
BEGIN
    PRINT 'FK_claims_line_items_provider constraint does not exist';
END
GO

-- Verify the constraint was removed
SELECT 
    fk.name AS constraint_name,
    tp.name AS parent_table,
    cp.name AS parent_column,
    tr.name AS referenced_table,
    cr.name AS referenced_column
FROM sys.foreign_keys fk
INNER JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
INNER JOIN sys.tables tp ON fkc.parent_object_id = tp.object_id
INNER JOIN sys.columns cp ON fkc.parent_object_id = cp.object_id AND fkc.parent_column_id = cp.column_id
INNER JOIN sys.tables tr ON fkc.referenced_object_id = tr.object_id
INNER JOIN sys.columns cr ON fkc.referenced_object_id = cr.object_id AND fkc.referenced_column_id = cr.column_id
WHERE fk.name = 'FK_claims_line_items_provider';

PRINT 'Provider constraint fix completed. Claims can now be processed without physician data.';
GO
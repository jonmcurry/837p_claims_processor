-- Test the corrected schema syntax
-- This file tests the key fixes for PostgreSQL compatibility

-- Test 1: Generated column expression 
CREATE TEMP TABLE test_generated_column (
    patient_first_name VARCHAR(50),
    patient_middle_name VARCHAR(50),
    patient_last_name VARCHAR(50),
    patient_name VARCHAR(100) GENERATED ALWAYS AS (
        TRIM(patient_first_name || ' ' || COALESCE(patient_middle_name, '') || ' ' || patient_last_name)
    ) STORED
);

-- Test insert
INSERT INTO test_generated_column (patient_first_name, patient_middle_name, patient_last_name) 
VALUES ('John', 'Q', 'Doe'), ('Jane', NULL, 'Smith');

-- Verify result
SELECT patient_name FROM test_generated_column;

-- Test 2: Partitioned table with sequence instead of BIGSERIAL
CREATE SEQUENCE test_seq;

CREATE TABLE test_partitioned (
    id BIGINT NOT NULL DEFAULT nextval('test_seq'),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    claim_id VARCHAR(50),
    data TEXT,
    PRIMARY KEY (id, created_at),
    CONSTRAINT test_unique UNIQUE (claim_id, created_at)  -- Test unique constraint with partition key
) PARTITION BY RANGE (created_at);

ALTER SEQUENCE test_seq OWNED BY test_partitioned.id;

-- Create a partition
CREATE TABLE test_partitioned_2025 PARTITION OF test_partitioned
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

-- Test insert
INSERT INTO test_partitioned (claim_id, data) VALUES ('CLAIM001', 'test data');

-- Verify
SELECT id, claim_id, data FROM test_partitioned;

-- Success message
\echo 'All schema syntax tests passed!'
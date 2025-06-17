#!/usr/bin/env python3
"""
Transfer Claims from PostgreSQL to SQL Server

This script actually transfers processed claims from PostgreSQL to SQL Server.
It reads from PostgreSQL and inserts into the SQL Server production tables.

Usage:
    python transfer_claims_to_sqlserver.py \
        --pg-conn "postgresql://claims_user:password@localhost:5432/claims_staging" \
        --ss-conn "DRIVER={ODBC Driver 17 for SQL Server};SERVER=localhost;DATABASE=smart_pro_claims;UID=sa;PWD=password"
"""

import argparse
import psycopg2
import pyodbc
from datetime import datetime
import json
import decimal

def get_sqlserver_connection(conn_string):
    """Create SQL Server connection."""
    return pyodbc.connect(conn_string)

def transfer_claims(pg_conn_string, ss_conn_string, limit=None):
    """Transfer claims from PostgreSQL to SQL Server."""
    pg_conn = None
    ss_conn = None
    
    try:
        # Connect to both databases
        print("🔄 Connecting to databases...")
        pg_conn = psycopg2.connect(pg_conn_string)
        pg_cursor = pg_conn.cursor()
        
        ss_conn = get_sqlserver_connection(ss_conn_string)
        ss_cursor = ss_conn.cursor()
        
        print("✅ Connected to both PostgreSQL and SQL Server")
        
        # Get claims marked as completed but not yet transferred
        query = """
            SELECT 
                c.id, c.claim_id, c.facility_id, c.patient_account_number,
                c.patient_first_name, c.patient_last_name, c.patient_middle_name,
                c.patient_date_of_birth, c.admission_date, c.discharge_date,
                c.service_from_date, c.service_to_date, c.financial_class,
                c.total_charges, c.expected_reimbursement, c.insurance_type,
                c.insurance_plan_id, c.subscriber_id, c.billing_provider_npi,
                c.billing_provider_name, c.attending_provider_npi, c.attending_provider_name,
                c.primary_diagnosis_code, c.diagnosis_codes, c.payer_name, c.payer_code,
                c.ml_prediction_score, c.processed_at, c.batch_id
            FROM claims c
            WHERE c.processing_status = 'completed'
            AND NOT EXISTS (
                SELECT 1 FROM claim_transfers ct 
                WHERE ct.claim_id = c.id
            )
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        pg_cursor.execute(query)
        claims = pg_cursor.fetchall()
        
        if not claims:
            print("❌ No completed claims found to transfer")
            
            # Check if claims were already transferred
            pg_cursor.execute("SELECT COUNT(*) FROM claims WHERE processing_status = 'completed'")
            completed_count = pg_cursor.fetchone()[0]
            print(f"   Total completed claims: {completed_count}")
            
            return
        
        print(f"📋 Found {len(claims)} claims to transfer to SQL Server")
        
        transferred = 0
        failed = 0
        
        # Create transfers tracking table if it doesn't exist
        pg_cursor.execute("""
            CREATE TABLE IF NOT EXISTS claim_transfers (
                id SERIAL PRIMARY KEY,
                claim_id BIGINT REFERENCES claims(id),
                transferred_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                sqlserver_claim_id VARCHAR(50)
            )
        """)
        pg_conn.commit()
        
        for claim in claims:
            pg_claim_id = claim[0]
            claim_id = claim[1]
            
            try:
                # Insert into SQL Server claims table
                ss_cursor.execute("""
                    INSERT INTO dbo.claims (
                        claim_id, facility_id, patient_account_number,
                        patient_first_name, patient_last_name, patient_middle_name,
                        patient_date_of_birth, admission_date, discharge_date,
                        service_from_date, service_to_date, financial_class,
                        total_charges, expected_reimbursement, claim_type,
                        insurance_plan_id, subscriber_id, billing_provider_npi,
                        billing_provider_name, attending_provider_npi, attending_provider_name,
                        primary_diagnosis_code, additional_diagnosis_codes,
                        payer_name, payer_code, ml_prediction_score,
                        processing_date, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, GETDATE(), GETDATE())
                """, (
                    claim_id, claim[2], claim[3],  # claim_id, facility_id, patient_account
                    claim[4], claim[5], claim[6],  # patient names
                    claim[7], claim[8], claim[9],  # dates
                    claim[10], claim[11], claim[12],  # service dates, financial_class
                    float(claim[13]) if claim[13] else 0,  # total_charges
                    float(claim[14]) if claim[14] else 0,  # expected_reimbursement
                    'P',  # claim_type (P for Professional)
                    claim[16], claim[17], claim[18],  # insurance info
                    claim[19], claim[20], claim[21],  # provider info
                    claim[22],  # primary_diagnosis_code
                    json.dumps(claim[23]) if claim[23] else '[]',  # diagnosis_codes as JSON string
                    claim[24], claim[25],  # payer info
                    float(claim[26]) if claim[26] else None,  # ml_prediction_score
                    claim[27] or datetime.now()  # processed_at
                ))
                
                # Also get and transfer claim line items
                pg_cursor.execute("""
                    SELECT 
                        line_number, service_date, procedure_code, procedure_description,
                        units, charge_amount, rendering_provider_npi, rendering_provider_name,
                        rvu_work, rvu_practice_expense, rvu_malpractice, rvu_total,
                        expected_reimbursement, diagnosis_pointers
                    FROM claim_line_items
                    WHERE claim_id = %s
                    ORDER BY line_number
                """, (pg_claim_id,))
                
                line_items = pg_cursor.fetchall()
                
                # Insert line items into SQL Server
                for line in line_items:
                    ss_cursor.execute("""
                        INSERT INTO dbo.claim_line_items (
                            claim_id, line_number, service_date, procedure_code,
                            procedure_description, units, charge_amount,
                            rendering_provider_npi, rendering_provider_name,
                            work_rvu, practice_expense_rvu, malpractice_rvu, total_rvu,
                            expected_reimbursement, diagnosis_pointers,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, GETDATE(), GETDATE())
                    """, (
                        claim_id, line[0], line[1], line[2], line[3],
                        line[4], float(line[5]) if line[5] else 0,
                        line[6], line[7],
                        float(line[8]) if line[8] else 0,
                        float(line[9]) if line[9] else 0,
                        float(line[10]) if line[10] else 0,
                        float(line[11]) if line[11] else 0,
                        float(line[12]) if line[12] else 0,
                        json.dumps(line[13]) if line[13] else '[]'
                    ))
                
                # Record successful transfer
                pg_cursor.execute("""
                    INSERT INTO claim_transfers (claim_id, sqlserver_claim_id)
                    VALUES (%s, %s)
                """, (pg_claim_id, claim_id))
                
                transferred += 1
                
                if transferred % 10 == 0:
                    print(f"   ✓ Transferred {transferred} claims...")
                    # Commit in batches
                    ss_conn.commit()
                    pg_conn.commit()
                    
            except Exception as e:
                print(f"   ❌ Error transferring claim {claim_id}: {e}")
                failed += 1
                # Rollback this claim
                ss_conn.rollback()
                continue
        
        # Final commit
        ss_conn.commit()
        pg_conn.commit()
        
        # Update batch metadata if any claims had batch_ids
        pg_cursor.execute("""
            UPDATE batch_metadata bm
            SET 
                processed_claims = (
                    SELECT COUNT(*) 
                    FROM claims c 
                    JOIN claim_transfers ct ON c.id = ct.claim_id
                    WHERE c.batch_id = bm.id
                ),
                status = 'completed'::processing_status,
                completed_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE EXISTS (
                SELECT 1 FROM claims c 
                JOIN claim_transfers ct ON c.id = ct.claim_id
                WHERE c.batch_id = bm.id
            )
        """)
        pg_conn.commit()
        
        print(f"\n✅ Transfer complete!")
        print(f"   • Transferred: {transferred} claims")
        print(f"   • Failed: {failed} claims")
        
        # Verify in SQL Server
        ss_cursor.execute("SELECT COUNT(*) FROM dbo.claims")
        ss_count = ss_cursor.fetchone()[0]
        print(f"\n📊 SQL Server now has {ss_count} total claims")
        
        # Show sample transferred claims
        ss_cursor.execute("""
            SELECT TOP 5 claim_id, patient_account_number, total_charges, created_at
            FROM dbo.claims
            ORDER BY created_at DESC
        """)
        
        print("\n📄 Recently transferred claims in SQL Server:")
        for row in ss_cursor.fetchall():
            print(f"   • {row[0]} - Account: {row[1]}, Charges: ${row[2]:,.2f}, Created: {row[3]}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if pg_conn:
            pg_conn.rollback()
        if ss_conn:
            ss_conn.rollback()
    finally:
        if pg_cursor:
            pg_cursor.close()
        if pg_conn:
            pg_conn.close()
        if ss_cursor:
            ss_cursor.close()
        if ss_conn:
            ss_conn.close()

def reset_claims_for_reprocessing(pg_conn_string):
    """Reset claims back to pending status for reprocessing."""
    try:
        conn = psycopg2.connect(pg_conn_string)
        cursor = conn.cursor()
        
        print("🔄 Resetting claims to pending status...")
        
        cursor.execute("""
            UPDATE claims 
            SET processing_status = 'pending'::processing_status,
                processed_at = NULL,
                ml_prediction_score = NULL,
                ml_prediction_result = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE processing_status = 'completed'
        """)
        
        updated = cursor.rowcount
        
        # Also drop the transfers table so claims can be reprocessed
        cursor.execute("DROP TABLE IF EXISTS claim_transfers")
        
        conn.commit()
        
        print(f"✅ Reset {updated} claims back to pending status")
        print("   Claims are ready to be processed again")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Transfer claims from PostgreSQL to SQL Server")
    parser.add_argument(
        "--pg-conn",
        required=True,
        help="PostgreSQL connection string"
    )
    parser.add_argument(
        "--ss-conn",
        required=True,
        help="SQL Server connection string (ODBC format)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of claims to transfer (for testing)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset claims back to pending status"
    )
    
    args = parser.parse_args()
    
    if args.reset:
        reset_claims_for_reprocessing(args.pg_conn)
    else:
        transfer_claims(args.pg_conn, args.ss_conn, args.limit)

if __name__ == "__main__":
    main()
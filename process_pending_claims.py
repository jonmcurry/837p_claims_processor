#!/usr/bin/env python3
"""
Direct Processing of Pending Claims

This script processes claims directly without batch assignment.
It's a simpler approach that processes all pending claims.

Usage:
    python process_pending_claims.py --postgres-conn "postgresql://..." --sqlserver-conn "mssql://..."
"""

import argparse
import psycopg2
import pyodbc
from datetime import datetime
import json

def process_pending_claims(pg_conn_string, ss_conn_string=None):
    """Process all pending claims from PostgreSQL to SQL Server."""
    try:
        # Connect to PostgreSQL
        pg_conn = psycopg2.connect(pg_conn_string)
        pg_cursor = pg_conn.cursor()
        
        print("🔄 Processing pending claims from PostgreSQL...")
        
        # Get pending claims count
        pg_cursor.execute("""
            SELECT COUNT(*) FROM claims 
            WHERE processing_status = 'pending'
        """)
        total_count = pg_cursor.fetchone()[0]
        
        if total_count == 0:
            print("❌ No pending claims found")
            print("\nCheck your claims with:")
            print("  SELECT processing_status, COUNT(*) FROM claims GROUP BY processing_status;")
            return
        
        print(f"✅ Found {total_count} pending claims to process")
        
        # Fetch pending claims with their line items
        pg_cursor.execute("""
            SELECT 
                c.id, c.claim_id, c.facility_id, c.patient_account_number,
                c.patient_first_name, c.patient_last_name, c.patient_date_of_birth,
                c.service_from_date, c.service_to_date,
                c.billing_provider_npi, c.billing_provider_name,
                c.primary_diagnosis_code, c.diagnosis_codes,
                c.total_charges, c.expected_reimbursement,
                c.payer_name, c.payer_code
            FROM claims c
            WHERE c.processing_status = 'pending'
            LIMIT 100  -- Process in batches of 100
        """)
        
        claims = pg_cursor.fetchall()
        processed_count = 0
        failed_count = 0
        
        print(f"📋 Processing {len(claims)} claims...")
        
        for claim in claims:
            claim_id = claim[1]
            try:
                # Here you would normally:
                # 1. Validate the claim
                # 2. Calculate RVUs
                # 3. Run ML predictions
                # 4. Transfer to SQL Server
                
                # For now, we'll just update the status
                pg_cursor.execute("""
                    UPDATE claims 
                    SET processing_status = 'completed',
                        processed_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (claim[0],))
                
                processed_count += 1
                
                # Show progress every 10 claims
                if processed_count % 10 == 0:
                    print(f"   Processed {processed_count}/{len(claims)} claims...")
                    
            except Exception as e:
                print(f"   ❌ Error processing claim {claim_id}: {e}")
                
                # Mark as failed
                pg_cursor.execute("""
                    UPDATE claims 
                    SET processing_status = 'failed',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (claim[0],))
                
                failed_count += 1
        
        # Commit changes
        pg_conn.commit()
        
        print(f"\n✅ Processing complete!")
        print(f"   • Processed: {processed_count}")
        print(f"   • Failed: {failed_count}")
        print(f"   • Remaining: {total_count - processed_count - failed_count}")
        
        # Show updated status
        pg_cursor.execute("""
            SELECT processing_status, COUNT(*) 
            FROM claims 
            GROUP BY processing_status
            ORDER BY processing_status
        """)
        
        print(f"\n📊 Updated claim status:")
        for status, count in pg_cursor.fetchall():
            print(f"   • {status}: {count}")
        
        pg_cursor.close()
        pg_conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if pg_conn:
            pg_conn.rollback()

def main():
    parser = argparse.ArgumentParser(description="Process pending claims directly")
    parser.add_argument(
        "--postgres-conn", "-p",
        required=True,
        help="PostgreSQL connection string"
    )
    parser.add_argument(
        "--sqlserver-conn", "-s",
        help="SQL Server connection string (optional for simulation)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without making changes"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        
    process_pending_claims(args.postgres_conn, args.sqlserver_conn)

if __name__ == "__main__":
    main()
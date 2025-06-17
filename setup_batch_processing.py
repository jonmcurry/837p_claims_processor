#!/usr/bin/env python3
"""
Setup Batch Processing for Claims

This script:
1. Creates a batch in the batch_metadata table
2. Associates pending claims with the batch
3. Prepares claims for processing

Usage:
    python setup_batch_processing.py --connection-string "postgresql://..." [--batch-id BATCH_001] [--facility FAC001]
"""

import argparse
import psycopg2
from datetime import datetime
import sys

def create_batch_and_assign_claims(conn_string, batch_id="BATCH_001", facility_id=None):
    """Create a batch and assign pending claims to it."""
    try:
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor()
        
        print(f"🔄 Setting up batch processing for: {batch_id}")
        
        # First, check if there are any pending claims
        cursor.execute("""
            SELECT COUNT(*), MIN(facility_id) as sample_facility
            FROM claims 
            WHERE processing_status = 'pending' 
            AND batch_id IS NULL
        """)
        count, sample_facility = cursor.fetchone()
        
        if count == 0:
            print("❌ No pending claims found without a batch_id")
            print("\nTo check your claims:")
            print("  SELECT processing_status, COUNT(*) FROM claims GROUP BY processing_status;")
            return False
        
        print(f"✅ Found {count} pending claims without batch assignment")
        
        # Use provided facility_id or take from claims
        if not facility_id and sample_facility:
            facility_id = sample_facility
            print(f"📍 Using facility_id from claims: {facility_id}")
        elif not facility_id:
            facility_id = "FAC001"  # Default
            print(f"📍 Using default facility_id: {facility_id}")
        
        # Check if batch already exists
        cursor.execute("SELECT id, status, total_claims FROM batch_metadata WHERE batch_id = %s", (batch_id,))
        existing_batch = cursor.fetchone()
        
        if existing_batch:
            batch_db_id, status, total = existing_batch
            print(f"⚠️  Batch {batch_id} already exists (status: {status}, claims: {total})")
            
            # If batch is completed or failed, we might want to create a new one
            if status in ['completed', 'failed']:
                print("   Creating new batch with suffix...")
                batch_id = f"{batch_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            else:
                print("   Reusing existing batch")
        
        # Create new batch if needed
        if not existing_batch or status in ['completed', 'failed']:
            cursor.execute("""
                INSERT INTO batch_metadata (
                    batch_id, facility_id, source_system, 
                    status, priority, total_claims, 
                    submitted_by, submitted_at
                ) VALUES (
                    %s, %s, 'manual_processing', 
                    'pending', 'medium', %s,
                    'setup_script', %s
                ) RETURNING id
            """, (batch_id, facility_id, count, datetime.now()))
            
            batch_db_id = cursor.fetchone()[0]
            print(f"✅ Created batch {batch_id} (ID: {batch_db_id})")
        
        # Assign claims to batch
        print(f"🔗 Assigning claims to batch {batch_id}...")
        
        # Update claims with batch_id
        cursor.execute("""
            UPDATE claims 
            SET batch_id = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE processing_status = 'pending' 
            AND batch_id IS NULL
            AND facility_id = %s
            RETURNING claim_id
        """, (batch_db_id, facility_id))
        
        updated_claims = cursor.fetchall()
        updated_count = len(updated_claims)
        
        if updated_count > 0:
            # Update batch metadata with actual count
            cursor.execute("""
                UPDATE batch_metadata 
                SET total_claims = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (updated_count, batch_db_id))
            
            conn.commit()
            print(f"✅ Successfully assigned {updated_count} claims to batch {batch_id}")
            print(f"\n📊 Batch Summary:")
            print(f"   • Batch ID: {batch_id}")
            print(f"   • Database ID: {batch_db_id}")
            print(f"   • Facility: {facility_id}")
            print(f"   • Total Claims: {updated_count}")
            print(f"   • Status: pending")
            
            # Show sample claims
            print(f"\n📋 Sample claims in batch:")
            cursor.execute("""
                SELECT claim_id, patient_account_number, service_from_date, total_charges
                FROM claims 
                WHERE batch_id = %s
                LIMIT 5
            """, (batch_db_id,))
            
            for claim in cursor.fetchall():
                print(f"   • {claim[0]} - Account: {claim[1]}, Date: {claim[2]}, Charges: ${claim[3]:,.2f}")
            
            print(f"\n🚀 Ready to process! Run:")
            print(f"   python run_claims_processing.py {batch_id}")
            
            return True
        else:
            print(f"⚠️  No claims were updated. Check facility_id: {facility_id}")
            conn.rollback()
            return False
            
    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        if conn:
            conn.rollback()
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def check_batch_status(conn_string, batch_id):
    """Check the status of a batch."""
    try:
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                bm.batch_id, bm.status, bm.total_claims, 
                bm.processed_claims, bm.failed_claims,
                bm.facility_id, bm.submitted_at,
                COUNT(c.id) as actual_claims
            FROM batch_metadata bm
            LEFT JOIN claims c ON c.batch_id = bm.id
            WHERE bm.batch_id = %s
            GROUP BY bm.batch_id, bm.status, bm.total_claims, 
                     bm.processed_claims, bm.failed_claims,
                     bm.facility_id, bm.submitted_at
        """, (batch_id,))
        
        result = cursor.fetchone()
        if result:
            print(f"\n📊 Batch Status for {batch_id}:")
            print(f"   • Status: {result[1]}")
            print(f"   • Total Claims: {result[2]}")
            print(f"   • Processed: {result[3] or 0}")
            print(f"   • Failed: {result[4] or 0}")
            print(f"   • Facility: {result[5]}")
            print(f"   • Submitted: {result[6]}")
            print(f"   • Actual Claims in DB: {result[7]}")
        else:
            print(f"❌ Batch {batch_id} not found")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error checking batch: {e}")

def main():
    parser = argparse.ArgumentParser(description="Setup batch processing for claims")
    parser.add_argument(
        "--connection-string", "-c",
        required=True,
        help="PostgreSQL connection string"
    )
    parser.add_argument(
        "--batch-id", "-b",
        default="BATCH_001",
        help="Batch ID to create (default: BATCH_001)"
    )
    parser.add_argument(
        "--facility", "-f",
        help="Facility ID to filter claims (optional)"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check batch status, don't create"
    )
    
    args = parser.parse_args()
    
    if args.check_only:
        check_batch_status(args.connection_string, args.batch_id)
    else:
        success = create_batch_and_assign_claims(
            args.connection_string,
            args.batch_id,
            args.facility
        )
        if success:
            check_batch_status(args.connection_string, args.batch_id)

if __name__ == "__main__":
    main()
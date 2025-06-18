#!/usr/bin/env python3
"""Check claim status distribution in PostgreSQL."""

import psycopg2
from src.core.config.settings import settings

def check_claim_distribution():
    """Check the distribution of claims by processing status."""
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=settings.pg_host,
            port=settings.pg_port,
            database=settings.pg_database,
            user=settings.pg_user,
            password=settings.pg_password.get_secret_value()
        )
        
        with conn.cursor() as cursor:
            # Get claim status distribution
            cursor.execute("""
                SELECT processing_status, COUNT(*) as count 
                FROM claims 
                GROUP BY processing_status 
                ORDER BY count DESC
            """)
            
            print("\nClaim Status Distribution in PostgreSQL:")
            print("-" * 50)
            total = 0
            results = cursor.fetchall()
            for status, count in results:
                print(f"{status:20}: {count:,}")
                total += count
            print("-" * 50)
            print(f"{'TOTAL':20}: {total:,}")
            
            # Check for specific statuses
            cursor.execute("SELECT COUNT(*) FROM claims WHERE processing_status = 'completed'")
            completed = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM claims WHERE processing_status = 'pending'")
            pending = cursor.fetchone()[0]
            
            print(f"\nAnalysis:")
            print(f"- Pending claims ready to process: {pending:,}")
            print(f"- Completed claims (already processed): {completed:,}")
            print(f"- Total claims in database: {total:,}")
            
            if completed > 0:
                print(f"\nNote: {completed:,} claims were already processed in previous runs.")
                print("To reprocess all claims, run: psql -d claims_staging -f reset_claims.sql")
            
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nAlternative: Run this SQL directly in PostgreSQL:")
        print("SELECT processing_status, COUNT(*) FROM claims GROUP BY processing_status;")

if __name__ == "__main__":
    check_claim_distribution()
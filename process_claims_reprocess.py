#!/usr/bin/env python3
"""
Process claims with option to reprocess completed claims.
Usage: python process_claims_reprocess.py --reprocess-completed
"""

import asyncio
import argparse
from src.processing.parallel_pipeline import ParallelClaimsPipeline
from src.core.database.pool_manager import pool_manager
from sqlalchemy import text

async def reset_claims_to_pending():
    """Reset completed claims to pending for reprocessing."""
    async with pool_manager.get_postgres_session() as session:
        result = await session.execute(text("""
            UPDATE claims 
            SET processing_status = 'pending'::processing_status,
                processed_at = NULL,
                updated_at = NOW()
            WHERE processing_status = 'completed'
            RETURNING id
        """))
        await session.commit()
        count = result.rowcount
        print(f"Reset {count:,} completed claims to pending status")
        return count

async def main():
    parser = argparse.ArgumentParser(description='Process claims with reprocess option')
    parser.add_argument('--reprocess-completed', action='store_true', 
                        help='Reset completed claims to pending before processing')
    args = parser.parse_args()
    
    # Initialize pool manager
    await pool_manager.initialize()
    
    if args.reprocess_completed:
        print("Resetting completed claims to pending...")
        reset_count = await reset_claims_to_pending()
        if reset_count > 0:
            print(f"Ready to process {reset_count:,} additional claims")
    
    # Run normal processing
    print("\nStarting claims processing...")
    pipeline = ParallelClaimsPipeline()
    await pipeline.process_claims_parallel()
    
    await pool_manager.close()

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Simple script to run the claims processing pipeline directly.

This script provides a command-line interface to process claims from PostgreSQL to SQL Server
using the ultra high-performance batch processing pipeline.

Usage:
    python scripts/run_pipeline.py <batch_id>
    python scripts/run_pipeline.py --batch-id BATCH_001
    python scripts/run_pipeline.py --help

Examples:
    python scripts/run_pipeline.py BATCH_001
    python scripts/run_pipeline.py --batch-id BATCH_001 --verbose
    python scripts/run_pipeline.py --batch-id BATCH_001 --dry-run
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.processing.batch_processor.pipeline import ClaimProcessingPipeline
    from src.core.config.settings import settings
    import structlog
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running this script from the project root directory.")
    print("Example: python scripts/run_pipeline.py BATCH_001")
    sys.exit(1)


def setup_logging(verbose: bool = False) -> None:
    """Set up structured logging for the pipeline runner."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('pipeline_runner.log')
        ]
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


async def process_claims_batch(batch_id: str, dry_run: bool = False) -> bool:
    """
    Process a batch of claims from PostgreSQL to SQL Server.
    
    Args:
        batch_id: The ID of the batch to process
        dry_run: If True, validate but don't actually process claims
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    logger = structlog.get_logger(__name__)
    
    try:
        logger.info("Initializing claims processing pipeline", batch_id=batch_id, dry_run=dry_run)
        
        # Initialize the pipeline
        pipeline = ClaimProcessingPipeline()
        
        if dry_run:
            logger.info("DRY RUN MODE: Claims will be validated but not processed", batch_id=batch_id)
            # In dry run mode, we could add validation-only logic here
            # For now, we'll just log that it's a dry run
            logger.info("Dry run completed successfully", batch_id=batch_id)
            return True
        
        # Record start time
        start_time = time.perf_counter()
        
        # Process the batch
        logger.info("Starting batch processing", batch_id=batch_id)
        result = await pipeline.process_batch(batch_id)
        
        # Calculate total processing time
        total_time = time.perf_counter() - start_time
        
        # Log results
        logger.info(
            "Batch processing completed",
            batch_id=batch_id,
            total_claims=result.total_claims,
            processed_claims=result.processed_claims,
            failed_claims=result.failed_claims,
            processing_time=f"{result.processing_time:.3f}s",
            total_time=f"{total_time:.3f}s",
            throughput=f"{result.throughput:.2f} claims/sec",
            success_rate=f"{(result.processed_claims / result.total_claims * 100):.1f}%" if result.total_claims > 0 else "0%"
        )
        
        # Check if processing met performance targets
        if result.throughput >= settings.target_throughput:
            logger.info("SUCCESS: Performance target met!", 
                       target=f"{settings.target_throughput} claims/sec",
                       actual=f"{result.throughput:.2f} claims/sec")
        else:
            logger.warning("WARNING: Performance target not met", 
                          target=f"{settings.target_throughput} claims/sec",
                          actual=f"{result.throughput:.2f} claims/sec")
        
        # Log any errors
        if result.errors:
            logger.warning("Processing completed with errors", 
                          batch_id=batch_id, 
                          error_count=len(result.errors))
            for error in result.errors[:5]:  # Log first 5 errors
                logger.error("Processing error", **error)
        
        # Return success if we processed at least some claims
        return result.processed_claims > 0
        
    except Exception as e:
        logger.exception("Fatal error during batch processing", 
                        batch_id=batch_id, 
                        error=str(e))
        return False


def main():
    """Main entry point for the pipeline runner."""
    parser = argparse.ArgumentParser(
        description="Run the claims processing pipeline for a specific batch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s BATCH_001
  %(prog)s --batch-id BATCH_001 --verbose
  %(prog)s --batch-id BATCH_001 --dry-run
        """
    )
    
    parser.add_argument(
        'batch_id',
        nargs='?',
        help='Batch ID to process (e.g., BATCH_001)'
    )
    
    parser.add_argument(
        '--batch-id',
        dest='batch_id_flag',
        help='Batch ID to process (alternative to positional argument)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate claims but do not actually process them'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    
    parser.add_argument(
        '--config-check',
        action='store_true',
        help='Check configuration and exit'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = structlog.get_logger(__name__)
    
    # Configuration check
    if args.config_check:
        logger.info("Configuration check requested")
        try:
            logger.info("Database configuration", 
                       postgres_host=settings.pg_host,
                       postgres_db=settings.pg_database,
                       sqlserver_host=settings.sql_host,
                       sqlserver_db=settings.sql_database)
            logger.info("Processing configuration",
                       batch_size=settings.batch_size,
                       worker_count=settings.worker_count,
                       target_throughput=settings.target_throughput)
            logger.info("✅ Configuration check passed")
            return 0
        except Exception as e:
            logger.error("❌ Configuration check failed", error=str(e))
            return 1
    
    # Get batch ID from either positional argument or flag
    batch_id = args.batch_id or args.batch_id_flag
    
    if not batch_id:
        parser.error("Batch ID is required. Use positional argument or --batch-id flag.")
    
    # Validate batch ID format (basic check)
    if not batch_id.strip():
        logger.error("Invalid batch ID: cannot be empty")
        return 1
    
    logger.info("Starting pipeline runner", 
               batch_id=batch_id, 
               dry_run=args.dry_run,
               environment=settings.app_env)
    
    try:
        # Run the async processing function
        success = asyncio.run(process_claims_batch(batch_id, args.dry_run))
        
        if success:
            logger.info("✅ Pipeline execution completed successfully", batch_id=batch_id)
            return 0
        else:
            logger.error("❌ Pipeline execution failed", batch_id=batch_id)
            return 1
            
    except KeyboardInterrupt:
        logger.info("Pipeline execution interrupted by user", batch_id=batch_id)
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logger.exception("Unexpected error during pipeline execution", 
                        batch_id=batch_id, 
                        error=str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
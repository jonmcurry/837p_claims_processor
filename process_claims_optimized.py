#!/usr/bin/env python3
"""
Ultra High-Performance Claims Processing Pipeline

Optimized for 100,000 claims in 15 seconds (6,667+ claims/second)

Features:
- Optimized connection pooling with warm-up
- RVU cache with preloading  
- Batch database operations
- Parallel processing with async/await
- Performance monitoring

Usage:
    python process_claims_optimized.py                     # Process all pending claims
    python process_claims_optimized.py --batch-id BATCH123 # Process specific batch
    python process_claims_optimized.py --limit 10000       # Process limited number
    python process_claims_optimized.py --env config/.env   # Specify env file
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from decimal import Decimal
from typing import Dict, Optional

import structlog

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.config.settings import settings
from src.core.database.pool_manager import pool_manager, initialize_pools, close_pools
from src.core.cache.rvu_cache import rvu_cache
from src.processing.parallel_pipeline import parallel_processor
from src.core.database.batch_operations import batch_ops

# Configure structured logging
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

logger = structlog.get_logger(__name__)


class OptimizedClaimsProcessor:
    """Ultra high-performance claims processor targeting 6,667+ claims/second."""
    
    def __init__(self):
        self.start_time = time.time()
        self.stats = {
            'total': 0,
            'processed': 0,
            'failed': 0,
            'processing_time': 0,
            'throughput': 0,
            'stage_times': {},
            'target_met': False
        }
        
    async def process_claims(self, batch_id: str = None, limit: int = None) -> Dict:
        """Main processing function using optimized parallel pipeline."""
        logger.info("Starting Ultra High-Performance Claims Processing",
                   target="100,000 claims in 15 seconds (6,667 claims/sec)")
        
        try:
            # Initialize all systems
            await self._initialize_systems()
            
            # Execute parallel processing pipeline
            result = await parallel_processor.process_claims_parallel(batch_id, limit)
            
            # Update statistics
            self.stats.update({
                'total': result.total_claims,
                'processed': result.processed_claims,
                'failed': result.failed_claims,
                'processing_time': result.processing_time,
                'throughput': result.throughput,
                'stage_times': result.stage_times,
                'target_met': result.throughput >= 6667
            })
            
            # Show comprehensive results
            await self._show_results()
            
            return self.stats
            
        except Exception as e:
            logger.exception("Ultra high-performance processing failed", error=str(e))
            self.stats['processing_time'] = time.time() - self.start_time
            return self.stats
        finally:
            # Clean shutdown
            await self._cleanup()
            
    async def _initialize_systems(self):
        """Initialize all optimized systems."""
        logger.info("Initializing ultra high-performance systems...")
        init_start = time.time()
        
        # Initialize systems in parallel
        initialization_tasks = [
            pool_manager.initialize(),
            rvu_cache.initialize(),
        ]
        
        await asyncio.gather(*initialization_tasks, return_exceptions=True)
        
        init_time = time.time() - init_start
        logger.info(f"System initialization completed in {init_time:.2f}s")
        
        # Log system status
        await self._log_system_status()
        
    async def _log_system_status(self):
        """Log the status of all optimized systems."""
        try:
            # Pool statistics
            pool_stats = pool_manager.get_pool_stats()
            logger.info("Database pool status", pool_stats=pool_stats)
            
            # Cache statistics
            cache_stats = rvu_cache.get_cache_stats()
            logger.info("RVU cache status", cache_stats=cache_stats)
            
            # Database health check
            health_status = await pool_manager.health_check()
            logger.info("Database health check", health_status=health_status)
            
        except Exception as e:
            logger.warning("System status check failed", error=str(e))
            
    async def _show_results(self):
        """Display comprehensive processing results."""
        print("\n" + "="*80)
        print("ULTRA HIGH-PERFORMANCE CLAIMS PROCESSING COMPLETE")
        print("="*80)
        
        # Main statistics
        print(f"\n🚀 Performance Results:")
        print(f"   • Total Claims: {self.stats['total']:,}")
        print(f"   • Successfully Processed: {self.stats['processed']:,}")
        print(f"   • Failed: {self.stats['failed']:,}")
        print(f"   • Success Rate: {(self.stats['processed']/self.stats['total']*100):.1f}%" if self.stats['total'] > 0 else "   • Success Rate: 0%")
        print(f"   • Processing Time: {self.stats['processing_time']:.2f} seconds")
        print(f"   • Throughput: {self.stats['throughput']:.1f} claims/second")
        
        # Target assessment
        target_throughput = 6667
        if self.stats['target_met']:
            print(f"\n🎯 TARGET ACHIEVED! Exceeded {target_throughput:,} claims/second")
            improvement = self.stats['throughput'] - target_throughput
            print(f"   ⚡ Performance surplus: +{improvement:.0f} claims/sec")
        else:
            print(f"\n⚠️  Target not met. Required: {target_throughput:,} claims/second")
            shortfall = target_throughput - self.stats['throughput']
            print(f"   📈 Improvement needed: +{shortfall:.0f} claims/sec")
            
        # Stage-by-stage breakdown
        if self.stats['stage_times']:
            print(f"\n📊 Stage Performance Breakdown:")
            for stage, timing in self.stats['stage_times'].items():
                stage_throughput = self.stats['total'] / timing if timing > 0 else 0
                print(f"   • {stage.title()}: {timing:.2f}s ({stage_throughput:.0f} claims/sec)")
                
        # Performance analysis for 100k claims in 15 seconds
        print(f"\n🎯 Target Analysis (100,000 claims in 15 seconds):")
        if self.stats['total'] > 0:
            extrapolated_time = (100000 / self.stats['throughput']) if self.stats['throughput'] > 0 else float('inf')
            print(f"   • Extrapolated time for 100k claims: {extrapolated_time:.1f} seconds")
            
            if extrapolated_time <= 15:
                print(f"   ✅ CAN achieve 100k claims in 15 seconds!")
                margin = 15 - extrapolated_time
                print(f"   ⏱️  Time margin: {margin:.1f} seconds")
            else:
                shortage = extrapolated_time - 15
                print(f"   ❌ Would take {shortage:.1f} seconds longer than target")
                needed_improvement = (100000 / 15) / self.stats['throughput']
                print(f"   📈 Need {needed_improvement:.1f}x performance improvement")
        
        # System performance
        await self._show_system_performance()
        
        # Processing recommendations
        self._show_optimization_recommendations()
        
    async def _show_system_performance(self):
        """Show system-level performance metrics."""
        try:
            print(f"\n⚙️  System Performance:")
            
            # Database pool performance
            pool_stats = pool_manager.get_pool_stats()
            print(f"   • PostgreSQL Pool: {pool_stats.get('postgres', {}).get('total', 0)} connections")
            print(f"   • SQL Server Pool: {pool_stats.get('sqlserver', {}).get('total', 0)} connections")
            
            # Cache performance
            cache_stats = rvu_cache.get_cache_stats()
            print(f"   • RVU Cache Hit Rate: {cache_stats.get('hit_rate_percent', 0):.1f}%")
            print(f"   • RVU Cache Size: {cache_stats.get('local_cache_size', 0):,} procedure codes")
            
            # Database statistics
            db_stats = await batch_ops.get_processing_statistics()
            postgres_stats = db_stats.get('postgres', {})
            sqlserver_stats = db_stats.get('sqlserver', {})
            
            print(f"   • PostgreSQL Claims: {sum(postgres_stats.values()) if postgres_stats else 0:,}")
            print(f"   • SQL Server Claims: {sqlserver_stats.get('total_claims', 0):,}")
            
        except Exception as e:
            logger.warning("System performance display failed", error=str(e))
            
    def _show_optimization_recommendations(self):
        """Provide optimization recommendations based on performance."""
        print(f"\n💡 Optimization Recommendations:")
        
        if self.stats['throughput'] < 6667:
            print("   📈 Performance Improvements:")
            print("   • Increase database connection pool sizes")
            print("   • Add more parallel worker processes")
            print("   • Optimize RVU cache preloading")
            print("   • Consider database query optimization")
            print("   • Scale to multiple processing servers")
        else:
            print("   ✅ Performance is optimal!")
            print("   • Consider monitoring for sustained load")
            print("   • Plan for horizontal scaling as volume grows")
            
        # Resource recommendations
        print(f"\n🖥️  Resource Recommendations:")
        print("   • CPU: High core count (16+ cores) for parallel processing")
        print("   • Memory: 32GB+ for large batch processing")
        print("   • Network: High bandwidth (10Gbps+) for database I/O")
        print("   • Storage: NVMe SSDs for database and cache performance")
        
    async def _cleanup(self):
        """Clean shutdown of all systems."""
        logger.info("Shutting down ultra high-performance systems...")
        
        try:
            # Shutdown parallel processor (includes ML shutdown)
            await parallel_processor.shutdown()
            
            # Close RVU cache
            await rvu_cache.close()
            
            # Close database pools
            await close_pools()
            
            logger.info("System shutdown completed")
            
        except Exception as e:
            logger.warning("Cleanup failed", error=str(e))


async def main():
    """Main entry point for optimized claims processing."""
    parser = argparse.ArgumentParser(
        description="Ultra High-Performance Claims Processor (Target: 6,667+ claims/sec)"
    )
    parser.add_argument(
        "--batch-id", "-b",
        help="Process specific batch ID"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        help="Limit number of claims to process"
    )
    parser.add_argument(
        "--env", "-e",
        default="config/.env",
        help="Environment file path (default: config/.env)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)
    
    # Display startup banner
    print("="*80)
    print("🚀 ULTRA HIGH-PERFORMANCE CLAIMS PROCESSOR")
    print("   Target: 100,000 claims in 15 seconds (6,667+ claims/sec)")
    print("="*80)
    
    if args.batch_id:
        print(f"📦 Processing Batch: {args.batch_id}")
    if args.limit:
        print(f"🔢 Claim Limit: {args.limit:,}")
    print(f"⚙️  Environment: {args.env}")
    print()
    
    # Initialize and run processor
    processor = OptimizedClaimsProcessor()
    
    try:
        stats = await processor.process_claims(
            batch_id=args.batch_id,
            limit=args.limit
        )
        
        # Exit with appropriate code
        if stats['target_met']:
            print("\n🎯 SUCCESS: Performance target achieved!")
            sys.exit(0)
        else:
            print("\n⚠️  WARNING: Performance target not met")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
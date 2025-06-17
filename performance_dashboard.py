#!/usr/bin/env python3
"""
Real-time Performance Dashboard for Claims Processing

Displays live performance metrics and monitoring for the ultra high-performance
claims processing pipeline.

Usage:
    python performance_dashboard.py                    # Start dashboard
    python performance_dashboard.py --export metrics.csv  # Export metrics
    python performance_dashboard.py --report             # Generate report
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.monitoring.performance_monitor import performance_monitor
from src.core.database.pool_manager import pool_manager
from src.core.cache.rvu_cache import rvu_cache


class PerformanceDashboard:
    """Real-time performance dashboard for claims processing."""
    
    def __init__(self):
        self.running = False
        
    async def start_dashboard(self):
        """Start the real-time performance dashboard."""
        print("="*80)
        print("🚀 ULTRA HIGH-PERFORMANCE CLAIMS PROCESSING DASHBOARD")
        print("   Target: 6,667+ claims/second | 100,000 claims in 15 seconds")
        print("="*80)
        print()
        
        # Initialize systems
        await self._initialize_systems()
        
        # Start monitoring
        await performance_monitor.start_monitoring(interval_seconds=2.0)
        
        self.running = True
        
        try:
            while self.running:
                await self._display_dashboard()
                await asyncio.sleep(3.0)  # Update every 3 seconds
                
        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped by user")
        finally:
            await performance_monitor.stop_monitoring()
            await self._cleanup()
            
    async def _initialize_systems(self):
        """Initialize monitoring systems."""
        print("🔧 Initializing monitoring systems...")
        
        try:
            await pool_manager.initialize()
            await rvu_cache.initialize()
            print("✅ Systems initialized successfully")
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            
    async def _display_dashboard(self):
        """Display the real-time dashboard."""
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("="*80)
        print(f"🚀 REAL-TIME PERFORMANCE DASHBOARD - {current_time}")
        print("="*80)
        
        # Get current metrics
        metrics = performance_monitor.get_current_metrics()
        
        if metrics:
            await self._display_performance_metrics(metrics)
        else:
            print("⏳ Collecting performance data...")
            
        # Display system status
        await self._display_system_status()
        
        # Display recent summary
        await self._display_performance_summary()
        
        print("="*80)
        print("Press Ctrl+C to stop monitoring")
        
    async def _display_performance_metrics(self, metrics):
        """Display current performance metrics."""
        print(f"📊 CURRENT PERFORMANCE")
        print(f"   ⚡ Throughput: {metrics.throughput_claims_per_sec:,.0f} claims/sec")
        
        # Target indicator
        target_status = "🎯 TARGET MET" if metrics.target_throughput_met else "❌ BELOW TARGET"
        print(f"   🎯 Target Status: {target_status} (6,667 claims/sec)")
        
        print(f"   ⏱️  Latency P99: {metrics.latency_p99_ms:.1f}ms")
        print(f"   ⏱️  Latency Avg: {metrics.latency_avg_ms:.1f}ms")
        
        # Performance assessment
        if metrics.throughput_claims_per_sec >= 6667:
            assessment = "🟢 EXCELLENT"
        elif metrics.throughput_claims_per_sec >= 5000:
            assessment = "🟡 GOOD"
        elif metrics.throughput_claims_per_sec >= 3000:
            assessment = "🟠 FAIR"
        else:
            assessment = "🔴 POOR"
            
        print(f"   📈 Performance: {assessment}")
        print()
        
    async def _display_system_status(self):
        """Display system resource status."""
        metrics = performance_monitor.get_current_metrics()
        
        if not metrics:
            return
            
        print(f"🖥️  SYSTEM RESOURCES")
        
        # CPU status
        cpu_status = "🔴 HIGH" if metrics.cpu_usage_percent > 80 else "🟢 OK"
        print(f"   🔧 CPU Usage: {metrics.cpu_usage_percent:.1f}% {cpu_status}")
        
        # Memory status
        memory_status = "🔴 HIGH" if metrics.memory_usage_percent > 80 else "🟢 OK"
        print(f"   🧠 Memory: {metrics.memory_usage_percent:.1f}% ({metrics.memory_usage_gb:.1f}GB) {memory_status}")
        
        print()
        print(f"💾 DATABASE CONNECTIONS")
        
        # PostgreSQL pool
        pg_util = (metrics.postgres_active_connections / metrics.postgres_total_connections * 100) if metrics.postgres_total_connections > 0 else 0
        pg_status = "🔴 HIGH" if pg_util > 80 else "🟢 OK"
        print(f"   🐘 PostgreSQL: {metrics.postgres_active_connections}/{metrics.postgres_total_connections} ({pg_util:.0f}%) {pg_status}")
        
        # SQL Server pool
        ss_util = (metrics.sqlserver_active_connections / metrics.sqlserver_total_connections * 100) if metrics.sqlserver_total_connections > 0 else 0
        ss_status = "🔴 HIGH" if ss_util > 80 else "🟢 OK"
        print(f"   📊 SQL Server: {metrics.sqlserver_active_connections}/{metrics.sqlserver_total_connections} ({ss_util:.0f}%) {ss_status}")
        
        print()
        print(f"⚡ CACHE PERFORMANCE")
        
        # RVU Cache
        cache_status = "🔴 LOW" if metrics.rvu_cache_hit_rate < 80 else "🟢 EXCELLENT" if metrics.rvu_cache_hit_rate > 95 else "🟡 GOOD"
        print(f"   🏎️  RVU Cache: {metrics.rvu_cache_hit_rate:.1f}% hit rate ({metrics.rvu_cache_size:,} codes) {cache_status}")
        
        # ML Cache
        ml_cache_status = "🔴 LOW" if metrics.ml_cache_hit_rate < 70 else "🟢 EXCELLENT" if metrics.ml_cache_hit_rate > 90 else "🟡 GOOD"
        print(f"   🧠 ML Cache: {metrics.ml_cache_hit_rate:.1f}% hit rate {ml_cache_status}")
        print()
        
        print(f"🤖 ML PERFORMANCE")
        
        # ML Processing
        ml_throughput_status = "🟢 FAST" if metrics.ml_throughput_claims_per_sec > 5000 else "🟡 MODERATE" if metrics.ml_throughput_claims_per_sec > 2000 else "🔴 SLOW"
        print(f"   ⚡ ML Throughput: {metrics.ml_throughput_claims_per_sec:.0f} claims/sec {ml_throughput_status}")
        
        # ML Latency
        ml_latency_status = "🟢 FAST" if metrics.ml_prediction_time_ms < 10 else "🟡 MODERATE" if metrics.ml_prediction_time_ms < 50 else "🔴 SLOW"
        print(f"   ⏱️  ML Latency: {metrics.ml_prediction_time_ms:.1f}ms {ml_latency_status}")
        
        # Active ML Predictions
        print(f"   🔄 Active Predictions: {metrics.ml_active_predictions}")
        print()
        
    async def _display_performance_summary(self):
        """Display performance summary."""
        summary_5min = performance_monitor.get_metrics_summary(5)
        
        if not summary_5min:
            return
            
        print(f"📈 PERFORMANCE SUMMARY (Last 5 minutes)")
        print(f"   📊 Avg Throughput: {summary_5min.get('avg_throughput', 0):,.0f} claims/sec")
        print(f"   🚀 Peak Throughput: {summary_5min.get('max_throughput', 0):,.0f} claims/sec")
        print(f"   🎯 Target Met: {summary_5min.get('target_throughput_met_percent', 0):.0f}% of time")
        print(f"   🔧 Avg CPU: {summary_5min.get('avg_cpu_usage', 0):.1f}%")
        print(f"   🧠 Peak Memory: {summary_5min.get('max_memory_usage', 0):.1f}%")
        print(f"   ⚡ Cache Hit Rate: {summary_5min.get('avg_cache_hit_rate', 0):.1f}%")
        print()
        
        # Performance projection for 100k claims
        if summary_5min.get('avg_throughput', 0) > 0:
            time_for_100k = 100000 / summary_5min['avg_throughput']
            target_status = "✅ CAN ACHIEVE" if time_for_100k <= 15 else "❌ CANNOT ACHIEVE"
            print(f"🎯 100K CLAIMS PROJECTION")
            print(f"   ⏱️  Estimated Time: {time_for_100k:.1f} seconds")
            print(f"   🎯 Target (15s): {target_status}")
            print()
            
    async def _cleanup(self):
        """Cleanup dashboard resources."""
        try:
            await rvu_cache.close()
            await pool_manager.close()
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")
            
    async def export_metrics(self, filename: str):
        """Export performance metrics to file."""
        print(f"📤 Exporting metrics to {filename}...")
        
        try:
            performance_monitor.export_metrics_csv(filename)
            print(f"✅ Metrics exported successfully to {filename}")
        except Exception as e:
            print(f"❌ Export failed: {e}")
            
    async def generate_report(self):
        """Generate and display performance report."""
        print("📋 Generating performance report...")
        
        try:
            # Initialize systems briefly for report
            await self._initialize_systems()
            
            # Start monitoring briefly to collect some data
            await performance_monitor.start_monitoring(interval_seconds=1.0)
            await asyncio.sleep(5)  # Collect 5 seconds of data
            
            report = await performance_monitor.generate_performance_report()
            
            await performance_monitor.stop_monitoring()
            
            print("\n" + "="*80)
            print("📋 PERFORMANCE REPORT")
            print("="*80)
            
            print(f"\n🕐 Report Time: {datetime.fromtimestamp(report['report_timestamp'])}")
            print(f"⏱️  Monitoring Duration: {report['monitoring_duration_minutes']:.1f} minutes")
            print(f"📊 Measurements: {report['total_measurements']}")
            
            current = report.get('current_performance', {})
            print(f"\n🚀 CURRENT PERFORMANCE")
            print(f"   ⚡ Throughput: {current.get('throughput_claims_per_sec', 0):,.0f} claims/sec")
            print(f"   ⏱️  Latency P99: {current.get('latency_p99_ms', 0):.1f}ms")
            print(f"   🎯 Target Met: {'Yes' if current.get('target_throughput_met') else 'No'}")
            
            targets = report.get('targets', {})
            print(f"\n🎯 PERFORMANCE TARGETS")
            print(f"   ⚡ Throughput Target: {targets.get('throughput_target', 0):,} claims/sec")
            print(f"   ⏱️  Latency Target: {targets.get('latency_p99_target_ms', 0)}ms")
            
            capacity = report.get('system_capacity', {})
            print(f"\n💾 SYSTEM CAPACITY")
            print(f"   🐘 PostgreSQL Pool: {capacity.get('postgres_pool_utilization', 0):.0f}% utilized")
            print(f"   📊 SQL Server Pool: {capacity.get('sqlserver_pool_utilization', 0):.0f}% utilized")
            print(f"   ⚡ RVU Cache: {capacity.get('rvu_cache_hit_rate', 0):.1f}% hit rate")
            
            assessment = report.get('performance_assessment', 'UNKNOWN')
            print(f"\n📈 OVERALL ASSESSMENT: {assessment}")
            
            # Save report to file
            report_filename = f"performance_report_{int(time.time())}.json"
            with open(report_filename, 'w') as f:
                json.dump(report, f, indent=2)
                
            print(f"\n💾 Full report saved to: {report_filename}")
            
        except Exception as e:
            print(f"❌ Report generation failed: {e}")
        finally:
            await self._cleanup()


async def main():
    """Main entry point for performance dashboard."""
    parser = argparse.ArgumentParser(
        description="Performance Dashboard for Ultra High-Performance Claims Processing"
    )
    parser.add_argument(
        "--export", "-e",
        metavar="FILENAME",
        help="Export metrics to CSV file"
    )
    parser.add_argument(
        "--report", "-r",
        action="store_true",
        help="Generate performance report"
    )
    
    args = parser.parse_args()
    
    dashboard = PerformanceDashboard()
    
    try:
        if args.export:
            await dashboard.export_metrics(args.export)
        elif args.report:
            await dashboard.generate_report()
        else:
            await dashboard.start_dashboard()
            
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
    except Exception as e:
        print(f"💥 Dashboard error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
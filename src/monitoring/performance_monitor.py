"""Real-time performance monitoring and metrics collection for claims processing."""

import asyncio
import time
import psutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import structlog

from src.core.config.settings import settings
from src.core.database.pool_manager import pool_manager
from src.core.cache.rvu_cache import rvu_cache
from src.core.database.batch_operations import batch_ops
from src.processing.ml_pipeline.async_ml_manager import async_ml_manager

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics."""
    
    timestamp: float = field(default_factory=time.time)
    
    # Processing metrics
    throughput_claims_per_sec: float = 0.0
    latency_avg_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    
    # System metrics
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    memory_usage_gb: float = 0.0
    
    # Database metrics
    postgres_active_connections: int = 0
    postgres_total_connections: int = 0
    sqlserver_active_connections: int = 0
    sqlserver_total_connections: int = 0
    
    # Cache metrics
    rvu_cache_hit_rate: float = 0.0
    rvu_cache_size: int = 0
    
    # ML metrics
    ml_prediction_time_ms: float = 0.0
    ml_cache_hit_rate: float = 0.0
    ml_throughput_claims_per_sec: float = 0.0
    ml_active_predictions: int = 0
    
    # Processing quality metrics
    success_rate_percent: float = 0.0
    error_rate_percent: float = 0.0
    
    # Performance targets
    target_throughput_met: bool = False
    target_latency_met: bool = False


class PerformanceMonitor:
    """Real-time performance monitoring for ultra high-performance processing."""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.max_history = 1000  # Keep last 1000 measurements
        self.monitoring_active = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Performance targets
        self.target_throughput = settings.target_throughput  # 6667 claims/sec
        self.target_latency_p99 = settings.target_latency_p99  # 100ms
        
        # Latency tracking
        self.latency_measurements: List[float] = []
        self.max_latency_measurements = 10000
        
    async def start_monitoring(self, interval_seconds: float = 1.0):
        """Start real-time performance monitoring."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(
            self._monitoring_loop(interval_seconds)
        )
        
        logger.info("Performance monitoring started", 
                   interval=interval_seconds,
                   target_throughput=self.target_throughput)
        
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Performance monitoring stopped")
        
    async def _monitoring_loop(self, interval: float):
        """Main monitoring loop."""
        try:
            while self.monitoring_active:
                metrics = await self._collect_metrics()
                self._store_metrics(metrics)
                
                # Log critical performance issues
                await self._check_performance_alerts(metrics)
                
                await asyncio.sleep(interval)
                
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.exception("Monitoring loop failed", error=str(e))
            
    async def _collect_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive performance metrics."""
        metrics = PerformanceMetrics()
        
        try:
            # System metrics
            metrics.cpu_usage_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            metrics.memory_usage_percent = memory_info.percent
            metrics.memory_usage_gb = memory_info.used / (1024**3)
            
            # Database pool metrics
            pool_stats = pool_manager.get_pool_stats()
            
            pg_stats = pool_stats.get('postgres', {})
            metrics.postgres_active_connections = pg_stats.get('checked_out', 0)
            metrics.postgres_total_connections = pg_stats.get('total', 0)
            
            ss_stats = pool_stats.get('sqlserver', {})
            metrics.sqlserver_active_connections = ss_stats.get('checked_out', 0)
            metrics.sqlserver_total_connections = ss_stats.get('total', 0)
            
            # Cache metrics
            cache_stats = rvu_cache.get_cache_stats()
            metrics.rvu_cache_hit_rate = cache_stats.get('hit_rate_percent', 0)
            metrics.rvu_cache_size = cache_stats.get('local_cache_size', 0)
            
            # ML metrics
            try:
                ml_stats = async_ml_manager.get_processing_metrics()
                ml_manager_metrics = ml_stats.get('ml_manager_metrics', {})
                
                metrics.ml_prediction_time_ms = ml_manager_metrics.get('avg_prediction_time_ms', 0)
                metrics.ml_cache_hit_rate = ml_manager_metrics.get('cache_hit_rate_percent', 0)
                metrics.ml_throughput_claims_per_sec = ml_manager_metrics.get('avg_throughput_claims_per_sec', 0)
                metrics.ml_active_predictions = ml_manager_metrics.get('active_predictions', 0)
            except Exception as e:
                logger.warning("Failed to collect ML metrics", error=str(e))
            
            # Latency metrics
            if self.latency_measurements:
                sorted_latencies = sorted(self.latency_measurements)
                count = len(sorted_latencies)
                
                metrics.latency_avg_ms = sum(sorted_latencies) / count * 1000
                metrics.latency_p95_ms = sorted_latencies[int(count * 0.95)] * 1000
                metrics.latency_p99_ms = sorted_latencies[int(count * 0.99)] * 1000
                
            # Performance targets
            metrics.target_throughput_met = metrics.throughput_claims_per_sec >= self.target_throughput
            metrics.target_latency_met = metrics.latency_p99_ms <= self.target_latency_p99
            
        except Exception as e:
            logger.warning("Metrics collection failed", error=str(e))
            
        return metrics
        
    def _store_metrics(self, metrics: PerformanceMetrics):
        """Store metrics in history buffer."""
        self.metrics_history.append(metrics)
        
        # Trim history to max size
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history:]
            
    async def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check for performance issues and log alerts."""
        alerts = []
        
        # CPU usage alerts
        if metrics.cpu_usage_percent > 90:
            alerts.append(f"High CPU usage: {metrics.cpu_usage_percent:.1f}%")
        elif metrics.cpu_usage_percent < 10:
            alerts.append(f"Low CPU usage: {metrics.cpu_usage_percent:.1f}% (may indicate bottleneck)")
            
        # Memory alerts
        if metrics.memory_usage_percent > 85:
            alerts.append(f"High memory usage: {metrics.memory_usage_percent:.1f}%")
            
        # Database connection alerts
        if metrics.postgres_active_connections > metrics.postgres_total_connections * 0.9:
            alerts.append("PostgreSQL connection pool near capacity")
            
        if metrics.sqlserver_active_connections > metrics.sqlserver_total_connections * 0.9:
            alerts.append("SQL Server connection pool near capacity")
            
        # Cache performance alerts
        if metrics.rvu_cache_hit_rate < 80:
            alerts.append(f"Low RVU cache hit rate: {metrics.rvu_cache_hit_rate:.1f}%")
            
        # Performance target alerts
        if not metrics.target_throughput_met and metrics.throughput_claims_per_sec > 0:
            alerts.append(f"Throughput below target: {metrics.throughput_claims_per_sec:.0f} < {self.target_throughput}")
            
        if not metrics.target_latency_met and metrics.latency_p99_ms > 0:
            alerts.append(f"Latency above target: P99 {metrics.latency_p99_ms:.1f}ms > {self.target_latency_p99}ms")
            
        # Log alerts
        for alert in alerts:
            logger.warning("Performance alert", alert=alert, metrics=metrics.__dict__)
            
    def record_processing_latency(self, latency_seconds: float):
        """Record processing latency measurement."""
        self.latency_measurements.append(latency_seconds)
        
        # Trim measurements to max size
        if len(self.latency_measurements) > self.max_latency_measurements:
            self.latency_measurements = self.latency_measurements[-self.max_latency_measurements:]
            
    def update_throughput(self, claims_processed: int, time_elapsed: float):
        """Update throughput calculation."""
        if self.metrics_history:
            throughput = claims_processed / time_elapsed if time_elapsed > 0 else 0
            self.metrics_history[-1].throughput_claims_per_sec = throughput
            
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
        
    def get_metrics_summary(self, minutes: int = 5) -> Dict:
        """Get performance summary for the last N minutes."""
        if not self.metrics_history:
            return {}
            
        cutoff_time = time.time() - (minutes * 60)
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {}
            
        summary = {
            'time_window_minutes': minutes,
            'measurements_count': len(recent_metrics),
            'avg_throughput': sum(m.throughput_claims_per_sec for m in recent_metrics) / len(recent_metrics),
            'max_throughput': max(m.throughput_claims_per_sec for m in recent_metrics),
            'avg_cpu_usage': sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics),
            'max_memory_usage': max(m.memory_usage_percent for m in recent_metrics),
            'avg_cache_hit_rate': sum(m.rvu_cache_hit_rate for m in recent_metrics) / len(recent_metrics),
            'target_throughput_met_percent': sum(1 for m in recent_metrics if m.target_throughput_met) / len(recent_metrics) * 100,
            'target_latency_met_percent': sum(1 for m in recent_metrics if m.target_latency_met) / len(recent_metrics) * 100,
        }
        
        return summary
        
    def export_metrics_csv(self, filename: str):
        """Export metrics history to CSV file."""
        import csv
        
        if not self.metrics_history:
            logger.warning("No metrics to export")
            return
            
        try:
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = [
                    'timestamp', 'throughput_claims_per_sec', 'latency_avg_ms', 
                    'latency_p95_ms', 'latency_p99_ms', 'cpu_usage_percent',
                    'memory_usage_percent', 'memory_usage_gb', 
                    'postgres_active_connections', 'postgres_total_connections',
                    'sqlserver_active_connections', 'sqlserver_total_connections',
                    'rvu_cache_hit_rate', 'rvu_cache_size', 'success_rate_percent',
                    'error_rate_percent', 'target_throughput_met', 'target_latency_met'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for metrics in self.metrics_history:
                    row = {field: getattr(metrics, field, '') for field in fieldnames}
                    writer.writerow(row)
                    
            logger.info(f"Metrics exported to {filename}", count=len(self.metrics_history))
            
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            
    async def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        if not self.metrics_history:
            return {"error": "No metrics data available"}
            
        current = self.get_current_metrics()
        summary_5min = self.get_metrics_summary(5)
        summary_15min = self.get_metrics_summary(15)
        
        report = {
            'report_timestamp': time.time(),
            'monitoring_duration_minutes': (time.time() - self.metrics_history[0].timestamp) / 60,
            'total_measurements': len(self.metrics_history),
            
            'current_performance': {
                'throughput_claims_per_sec': current.throughput_claims_per_sec if current else 0,
                'latency_p99_ms': current.latency_p99_ms if current else 0,
                'cpu_usage_percent': current.cpu_usage_percent if current else 0,
                'memory_usage_percent': current.memory_usage_percent if current else 0,
                'target_throughput_met': current.target_throughput_met if current else False,
                'target_latency_met': current.target_latency_met if current else False,
            },
            
            'performance_last_5min': summary_5min,
            'performance_last_15min': summary_15min,
            
            'targets': {
                'throughput_target': self.target_throughput,
                'latency_p99_target_ms': self.target_latency_p99,
            },
            
            'system_capacity': {
                'postgres_pool_utilization': (current.postgres_active_connections / current.postgres_total_connections * 100) if current and current.postgres_total_connections > 0 else 0,
                'sqlserver_pool_utilization': (current.sqlserver_active_connections / current.sqlserver_total_connections * 100) if current and current.sqlserver_total_connections > 0 else 0,
                'rvu_cache_hit_rate': current.rvu_cache_hit_rate if current else 0,
                'rvu_cache_size': current.rvu_cache_size if current else 0,
            }
        }
        
        # Performance assessment
        if summary_5min:
            target_met_5min = summary_5min.get('target_throughput_met_percent', 0)
            if target_met_5min >= 90:
                report['performance_assessment'] = 'EXCELLENT'
            elif target_met_5min >= 70:
                report['performance_assessment'] = 'GOOD'
            elif target_met_5min >= 50:
                report['performance_assessment'] = 'FAIR'
            else:
                report['performance_assessment'] = 'POOR'
        else:
            report['performance_assessment'] = 'UNKNOWN'
            
        return report


# Global performance monitor instance
performance_monitor = PerformanceMonitor()
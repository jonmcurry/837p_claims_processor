"""Advanced resource monitoring for predictive scaling decisions."""

import asyncio
import time
import logging
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import threading
import json

import psutil
import docker
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from src.core.config import config
from src.monitoring.metrics.comprehensive_metrics import metrics_collector


logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ResourceThreshold(NamedTuple):
    """Resource threshold configuration."""
    warning_threshold: float
    critical_threshold: float
    duration_seconds: int
    recovery_threshold: float


@dataclass
class ResourceAlert:
    """Resource alert information."""
    alert_id: str
    resource_name: str
    severity: AlertSeverity
    message: str
    current_value: float
    threshold_value: float
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    duration_seconds: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContainerMetrics:
    """Container-specific metrics."""
    container_id: str
    container_name: str
    cpu_percent: float
    memory_usage_mb: float
    memory_limit_mb: float
    memory_percent: float
    network_rx_mb: float
    network_tx_mb: float
    disk_read_mb: float
    disk_write_mb: float
    restart_count: int
    status: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DatabaseMetrics:
    """Database performance metrics."""
    active_connections: int
    max_connections: int
    connection_percent: float
    queries_per_second: float
    slow_queries: int
    cache_hit_ratio: float
    replication_lag_ms: float
    disk_usage_percent: float
    index_usage_percent: float
    lock_waits: int
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class QueueMetrics:
    """Queue system metrics."""
    queue_name: str
    depth: int
    enqueue_rate: float
    dequeue_rate: float
    avg_wait_time_ms: float
    max_wait_time_ms: float
    consumer_count: int
    failed_messages: int
    timestamp: datetime = field(default_factory=datetime.now)


class AdvancedResourceMonitor:
    """Comprehensive resource monitoring system for scaling decisions."""
    
    def __init__(self,
                 docker_client: Optional[docker.DockerClient] = None,
                 redis_client: Optional[redis.Redis] = None,
                 db_session: Optional[AsyncSession] = None,
                 monitoring_interval_seconds: int = 30):
        
        self.docker_client = docker_client
        self.redis_client = redis_client
        self.db_session = db_session
        self.monitoring_interval_seconds = monitoring_interval_seconds
        
        # Resource thresholds
        self._resource_thresholds = self._initialize_thresholds()
        
        # Metrics storage
        self._system_metrics_history: deque = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self._container_metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=720))  # 12 hours
        self._database_metrics_history: deque = deque(maxlen=720)
        self._queue_metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=720))
        
        # Active alerts
        self._active_alerts: Dict[str, ResourceAlert] = {}
        self._alert_history: deque = deque(maxlen=1000)
        self._alert_lock = threading.RLock()
        
        # Performance baselines
        self._performance_baselines: Dict[str, float] = {}
        self._baseline_calculation_window = 24 * 60  # 24 hours
        
        # Anomaly detection
        self._anomaly_detection_enabled = True
        self._anomaly_thresholds = {
            'cpu_percent': 2.0,      # Standard deviations
            'memory_percent': 2.0,
            'queue_depth': 3.0,
            'response_time': 2.5
        }
        
        # Background monitoring tasks
        self._monitoring_tasks: List[asyncio.Task] = []
        self._start_monitoring_tasks()
    
    def _initialize_thresholds(self) -> Dict[str, ResourceThreshold]:
        """Initialize resource monitoring thresholds."""
        return {
            'cpu_percent': ResourceThreshold(70.0, 85.0, 300, 60.0),  # CPU usage %
            'memory_percent': ResourceThreshold(75.0, 90.0, 300, 65.0),  # Memory usage %
            'disk_usage_percent': ResourceThreshold(80.0, 95.0, 600, 70.0),  # Disk usage %
            'queue_depth': ResourceThreshold(1000.0, 5000.0, 180, 500.0),  # Queue depth
            'queue_wait_time_ms': ResourceThreshold(5000.0, 15000.0, 300, 3000.0),  # Queue wait time
            'database_connections_percent': ResourceThreshold(70.0, 90.0, 300, 60.0),  # DB connections %
            'database_cache_hit_ratio': ResourceThreshold(85.0, 70.0, 600, 90.0),  # Cache hit ratio (reverse)
            'response_time_p95_ms': ResourceThreshold(2000.0, 5000.0, 300, 1500.0),  # Response time
            'error_rate_percent': ResourceThreshold(2.0, 5.0, 180, 1.0),  # Error rate
            'network_utilization_percent': ResourceThreshold(70.0, 90.0, 300, 60.0),  # Network usage
            'container_restart_rate': ResourceThreshold(5.0, 10.0, 900, 2.0)  # Container restarts per hour
        }
    
    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics."""
        
        start_time = time.time()
        
        # Basic system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk_usage = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        # Load averages
        load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
        
        # Process information
        process_count = len(psutil.pids())
        
        # Network interface metrics
        network_interfaces = psutil.net_io_counters(pernic=True)
        total_network_bytes = sum(
            iface.bytes_sent + iface.bytes_recv 
            for iface in network_interfaces.values()
        )
        
        system_metrics = {
            'cpu_percent': cpu_percent,
            'cpu_count': psutil.cpu_count(),
            'load_avg_1m': load_avg[0],
            'load_avg_5m': load_avg[1],
            'load_avg_15m': load_avg[2],
            'memory_total_gb': memory.total / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'disk_total_gb': disk_usage.total / (1024**3),
            'disk_used_gb': disk_usage.used / (1024**3),
            'disk_free_gb': disk_usage.free / (1024**3),
            'disk_percent': (disk_usage.used / disk_usage.total) * 100,
            'network_bytes_sent': network.bytes_sent if network else 0,
            'network_bytes_recv': network.bytes_recv if network else 0,
            'network_total_gb': total_network_bytes / (1024**3),
            'process_count': process_count,
            'timestamp': datetime.now()
        }
        
        # Store in history
        self._system_metrics_history.append(system_metrics)
        
        # Check for threshold violations
        await self._check_system_thresholds(system_metrics)
        
        # Record collection time
        collection_time_ms = (time.time() - start_time) * 1000
        
        logger.debug(f"Collected system metrics in {collection_time_ms:.1f}ms")
        
        return system_metrics
    
    async def collect_container_metrics(self) -> Dict[str, ContainerMetrics]:
        """Collect Docker container metrics."""
        
        container_metrics = {}
        
        if not self.docker_client:
            return container_metrics
        
        try:
            containers = self.docker_client.containers.list()
            
            for container in containers:
                try:
                    # Get container stats
                    stats = container.stats(stream=False)
                    
                    # Calculate CPU percentage
                    cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                               stats['precpu_stats']['cpu_usage']['total_usage']
                    system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                                  stats['precpu_stats']['system_cpu_usage']
                    
                    cpu_percent = 0.0
                    if system_delta > 0:
                        cpu_percent = (cpu_delta / system_delta) * 100.0
                    
                    # Memory metrics
                    memory_usage = stats['memory_stats'].get('usage', 0)
                    memory_limit = stats['memory_stats'].get('limit', 0)
                    memory_percent = (memory_usage / memory_limit) * 100 if memory_limit > 0 else 0
                    
                    # Network metrics
                    networks = stats.get('networks', {})
                    network_rx = sum(net.get('rx_bytes', 0) for net in networks.values())
                    network_tx = sum(net.get('tx_bytes', 0) for net in networks.values())
                    
                    # Disk I/O metrics
                    blkio_stats = stats.get('blkio_stats', {})
                    disk_read = sum(
                        stat.get('value', 0) 
                        for stat in blkio_stats.get('io_service_bytes_recursive', [])
                        if stat.get('op') == 'Read'
                    )
                    disk_write = sum(
                        stat.get('value', 0) 
                        for stat in blkio_stats.get('io_service_bytes_recursive', [])
                        if stat.get('op') == 'Write'
                    )
                    
                    # Container info
                    container.reload()
                    restart_count = container.attrs.get('RestartCount', 0)
                    
                    metrics = ContainerMetrics(
                        container_id=container.id,
                        container_name=container.name,
                        cpu_percent=cpu_percent,
                        memory_usage_mb=memory_usage / (1024**2),
                        memory_limit_mb=memory_limit / (1024**2),
                        memory_percent=memory_percent,
                        network_rx_mb=network_rx / (1024**2),
                        network_tx_mb=network_tx / (1024**2),
                        disk_read_mb=disk_read / (1024**2),
                        disk_write_mb=disk_write / (1024**2),
                        restart_count=restart_count,
                        status=container.status
                    )
                    
                    container_metrics[container.name] = metrics
                    self._container_metrics_history[container.name].append(metrics)
                    
                    # Check container-specific thresholds
                    await self._check_container_thresholds(metrics)
                
                except Exception as e:
                    logger.warning(f"Failed to collect metrics for container {container.name}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to collect container metrics: {e}")
        
        return container_metrics
    
    async def collect_database_metrics(self) -> Optional[DatabaseMetrics]:
        """Collect database performance metrics."""
        
        if not self.db_session:
            return None
        
        try:
            # PostgreSQL-specific queries
            queries = {
                'connections': """
                    SELECT count(*) as active_connections,
                           setting::int as max_connections
                    FROM pg_stat_activity, pg_settings 
                    WHERE pg_settings.name = 'max_connections'
                    GROUP BY setting
                """,
                'activity': """
                    SELECT 
                        sum(calls) as total_calls,
                        sum(total_time) as total_time_ms,
                        count(*) filter (where total_time/calls > 1000) as slow_queries
                    FROM pg_stat_statements 
                    WHERE calls > 0
                """,
                'cache_stats': """
                    SELECT 
                        sum(heap_blks_hit)::float / 
                        (sum(heap_blks_hit) + sum(heap_blks_read))::float * 100 as cache_hit_ratio
                    FROM pg_statio_user_tables
                    WHERE heap_blks_hit + heap_blks_read > 0
                """,
                'locks': """
                    SELECT count(*) as lock_waits
                    FROM pg_stat_activity 
                    WHERE wait_event_type = 'Lock'
                """,
                'replication': """
                    SELECT COALESCE(
                        EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp())) * 1000, 0
                    ) as lag_ms
                """,
                'database_size': """
                    SELECT 
                        pg_database_size(current_database())::float / 
                        (1024*1024*1024) as size_gb
                """
            }
            
            results = {}
            for query_name, query in queries.items():
                try:
                    result = await self.db_session.execute(text(query))
                    row = result.fetchone()
                    results[query_name] = row._asdict() if row else {}
                except Exception as e:
                    logger.warning(f"Database query {query_name} failed: {e}")
                    results[query_name] = {}
            
            # Calculate metrics
            connections = results.get('connections', {})
            activity = results.get('activity', {})
            cache_stats = results.get('cache_stats', {})
            locks = results.get('locks', {})
            replication = results.get('replication', {})
            
            active_connections = connections.get('active_connections', 0)
            max_connections = connections.get('max_connections', 100)
            
            total_calls = activity.get('total_calls', 0)
            total_time_ms = activity.get('total_time_ms', 0)
            
            # Calculate QPS (queries per second) - approximate
            qps = total_calls / 60 if total_calls > 0 else 0  # Rough estimate
            
            metrics = DatabaseMetrics(
                active_connections=active_connections,
                max_connections=max_connections,
                connection_percent=(active_connections / max_connections) * 100,
                queries_per_second=qps,
                slow_queries=activity.get('slow_queries', 0),
                cache_hit_ratio=cache_stats.get('cache_hit_ratio', 0),
                replication_lag_ms=replication.get('lag_ms', 0),
                disk_usage_percent=0,  # Would need additional query
                index_usage_percent=0,  # Would need additional query
                lock_waits=locks.get('lock_waits', 0)
            )
            
            self._database_metrics_history.append(metrics)
            
            # Check database thresholds
            await self._check_database_thresholds(metrics)
            
            return metrics
        
        except Exception as e:
            logger.error(f"Failed to collect database metrics: {e}")
            return None
    
    async def collect_queue_metrics(self) -> Dict[str, QueueMetrics]:
        """Collect queue system metrics."""
        
        queue_metrics = {}
        
        if not self.redis_client:
            return queue_metrics
        
        try:
            # Get Redis info
            redis_info = await self.redis_client.info()
            
            # Common queue patterns to monitor
            queue_patterns = [
                'batch_processing:*',
                'ml_inference:*',
                'validation:*',
                'failed_claims:*'
            ]
            
            for pattern in queue_patterns:
                try:
                    queue_name = pattern.replace(':*', '')
                    
                    # Get queue length
                    queue_length = await self.redis_client.llen(f"queue:{queue_name}")
                    
                    # Get processing stats (would come from your queue system)
                    # For now, using mock calculations
                    enqueue_rate = 10.0  # messages per second
                    dequeue_rate = 8.0   # messages per second
                    avg_wait_time = max(0, queue_length * 100)  # Rough estimate
                    max_wait_time = avg_wait_time * 2
                    consumer_count = 2  # Active consumers
                    failed_messages = 0
                    
                    metrics = QueueMetrics(
                        queue_name=queue_name,
                        depth=queue_length,
                        enqueue_rate=enqueue_rate,
                        dequeue_rate=dequeue_rate,
                        avg_wait_time_ms=avg_wait_time,
                        max_wait_time_ms=max_wait_time,
                        consumer_count=consumer_count,
                        failed_messages=failed_messages
                    )
                    
                    queue_metrics[queue_name] = metrics
                    self._queue_metrics_history[queue_name].append(metrics)
                    
                    # Check queue thresholds
                    await self._check_queue_thresholds(metrics)
                
                except Exception as e:
                    logger.warning(f"Failed to collect metrics for queue {pattern}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to collect queue metrics: {e}")
        
        return queue_metrics
    
    async def _check_system_thresholds(self, metrics: Dict[str, Any]):
        """Check system metrics against thresholds and generate alerts."""
        
        checks = [
            ('cpu_percent', metrics['cpu_percent']),
            ('memory_percent', metrics['memory_percent']),
            ('disk_usage_percent', metrics['disk_percent'])
        ]
        
        for metric_name, value in checks:
            await self._evaluate_threshold(metric_name, value, 'system')
    
    async def _check_container_thresholds(self, metrics: ContainerMetrics):
        """Check container metrics against thresholds."""
        
        container_prefix = f"container_{metrics.container_name}"
        
        checks = [
            (f'{container_prefix}_cpu_percent', metrics.cpu_percent),
            (f'{container_prefix}_memory_percent', metrics.memory_percent),
            (f'{container_prefix}_restart_rate', metrics.restart_count)  # Simplified
        ]
        
        for metric_name, value in checks:
            await self._evaluate_threshold(metric_name, value, 'container', 
                                         {'container_name': metrics.container_name})
    
    async def _check_database_thresholds(self, metrics: DatabaseMetrics):
        """Check database metrics against thresholds."""
        
        checks = [
            ('database_connections_percent', metrics.connection_percent),
            ('database_cache_hit_ratio', metrics.cache_hit_ratio)
        ]
        
        for metric_name, value in checks:
            # For cache hit ratio, we want to alert when it's too LOW
            if metric_name == 'database_cache_hit_ratio':
                threshold = self._resource_thresholds.get(metric_name)
                if threshold:
                    # Reverse the logic for cache hit ratio
                    is_warning = value < threshold.warning_threshold
                    is_critical = value < threshold.critical_threshold
                    
                    if is_critical:
                        await self._generate_alert(metric_name, value, threshold.critical_threshold,
                                                 AlertSeverity.CRITICAL, 'database')
                    elif is_warning:
                        await self._generate_alert(metric_name, value, threshold.warning_threshold,
                                                 AlertSeverity.WARNING, 'database')
            else:
                await self._evaluate_threshold(metric_name, value, 'database')
    
    async def _check_queue_thresholds(self, metrics: QueueMetrics):
        """Check queue metrics against thresholds."""
        
        queue_prefix = f"queue_{metrics.queue_name}"
        
        checks = [
            (f'{queue_prefix}_depth', metrics.depth),
            (f'{queue_prefix}_wait_time_ms', metrics.avg_wait_time_ms)
        ]
        
        # Map to general threshold names
        threshold_mapping = {
            f'{queue_prefix}_depth': 'queue_depth',
            f'{queue_prefix}_wait_time_ms': 'queue_wait_time_ms'
        }
        
        for metric_name, value in checks:
            threshold_name = threshold_mapping.get(metric_name, metric_name)
            await self._evaluate_threshold(threshold_name, value, 'queue',
                                         {'queue_name': metrics.queue_name})
    
    async def _evaluate_threshold(self, metric_name: str, value: float, 
                                source: str, metadata: Optional[Dict] = None):
        """Evaluate metric against threshold and generate alerts if needed."""
        
        threshold = self._resource_thresholds.get(metric_name)
        if not threshold:
            return
        
        alert_id = f"{source}_{metric_name}"
        
        # Check if we should trigger an alert
        is_warning = value >= threshold.warning_threshold
        is_critical = value >= threshold.critical_threshold
        
        with self._alert_lock:
            existing_alert = self._active_alerts.get(alert_id)
            
            if is_critical:
                if not existing_alert or existing_alert.severity != AlertSeverity.CRITICAL:
                    await self._generate_alert(metric_name, value, threshold.critical_threshold,
                                             AlertSeverity.CRITICAL, source, metadata)
            
            elif is_warning:
                if not existing_alert or existing_alert.severity == AlertSeverity.INFO:
                    await self._generate_alert(metric_name, value, threshold.warning_threshold,
                                             AlertSeverity.WARNING, source, metadata)
            
            else:
                # Check for recovery
                if existing_alert and value <= threshold.recovery_threshold:
                    await self._resolve_alert(alert_id)
    
    async def _generate_alert(self, metric_name: str, current_value: float,
                            threshold_value: float, severity: AlertSeverity,
                            source: str, metadata: Optional[Dict] = None):
        """Generate a new alert."""
        
        alert_id = f"{source}_{metric_name}"
        
        message = (f"{source.title()} {metric_name} is {severity.value}: "
                  f"{current_value:.2f} (threshold: {threshold_value:.2f})")
        
        alert = ResourceAlert(
            alert_id=alert_id,
            resource_name=metric_name,
            severity=severity,
            message=message,
            current_value=current_value,
            threshold_value=threshold_value,
            triggered_at=datetime.now(),
            metadata=metadata or {}
        )
        
        with self._alert_lock:
            self._active_alerts[alert_id] = alert
            self._alert_history.append(alert)
        
        # Log alert
        if severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
            logger.error(f"ALERT [{severity.value.upper()}]: {message}")
        else:
            logger.warning(f"ALERT [{severity.value.upper()}]: {message}")
        
        # Record alert metric
        metrics_collector.record_resource_alert(
            resource_name=metric_name,
            severity=severity.value,
            current_value=current_value,
            threshold_value=threshold_value,
            source=source
        )
        
        # Send to external alerting systems (Slack, PagerDuty, etc.)
        await self._send_external_alert(alert)
    
    async def _resolve_alert(self, alert_id: str):
        """Resolve an active alert."""
        
        with self._alert_lock:
            alert = self._active_alerts.get(alert_id)
            if alert:
                alert.resolved_at = datetime.now()
                alert.duration_seconds = int(
                    (alert.resolved_at - alert.triggered_at).total_seconds()
                )
                
                del self._active_alerts[alert_id]
                
                logger.info(f"RESOLVED: {alert.message} "
                           f"(duration: {alert.duration_seconds}s)")
                
                # Record resolution
                metrics_collector.record_alert_resolution(
                    alert_id=alert_id,
                    duration_seconds=alert.duration_seconds
                )
    
    async def _send_external_alert(self, alert: ResourceAlert):
        """Send alert to external systems."""
        try:
            # This would integrate with your alerting systems
            # For example: Slack, PagerDuty, email, etc.
            
            if alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
                # Send to critical alert channels
                logger.info(f"Would send critical alert to external systems: {alert.message}")
            
        except Exception as e:
            logger.error(f"Failed to send external alert: {e}")
    
    async def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in metrics using statistical analysis."""
        
        anomalies = []
        
        if not self._anomaly_detection_enabled or len(self._system_metrics_history) < 30:
            return anomalies
        
        try:
            # Analyze recent system metrics
            recent_metrics = list(self._system_metrics_history)[-30:]  # Last 30 measurements
            
            for metric_name in ['cpu_percent', 'memory_percent']:
                values = [m[metric_name] for m in recent_metrics]
                
                if len(values) > 10:
                    mean_val = np.mean(values[:-1])  # Exclude current value
                    std_val = np.std(values[:-1])
                    current_val = values[-1]
                    
                    # Check for anomaly
                    threshold = self._anomaly_thresholds.get(metric_name, 2.0)
                    
                    if std_val > 0:
                        z_score = abs(current_val - mean_val) / std_val
                        
                        if z_score > threshold:
                            anomalies.append({
                                'metric_name': metric_name,
                                'current_value': current_val,
                                'mean_value': mean_val,
                                'std_deviation': std_val,
                                'z_score': z_score,
                                'threshold': threshold,
                                'severity': 'high' if z_score > threshold * 1.5 else 'medium'
                            })
            
            # Analyze queue depth anomalies
            for queue_name, queue_history in self._queue_metrics_history.items():
                if len(queue_history) > 10:
                    depths = [m.depth for m in list(queue_history)[-20:]]
                    
                    if len(depths) > 5:
                        mean_depth = np.mean(depths[:-1])
                        std_depth = np.std(depths[:-1])
                        current_depth = depths[-1]
                        
                        threshold = self._anomaly_thresholds.get('queue_depth', 3.0)
                        
                        if std_depth > 0:
                            z_score = abs(current_depth - mean_depth) / std_depth
                            
                            if z_score > threshold:
                                anomalies.append({
                                    'metric_name': f'queue_depth_{queue_name}',
                                    'current_value': current_depth,
                                    'mean_value': mean_depth,
                                    'std_deviation': std_depth,
                                    'z_score': z_score,
                                    'threshold': threshold,
                                    'severity': 'high' if z_score > threshold * 1.5 else 'medium'
                                })
        
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
        
        return anomalies
    
    def _start_monitoring_tasks(self):
        """Start background monitoring tasks."""
        
        async def system_monitor():
            """Monitor system metrics."""
            while True:
                try:
                    await self.collect_system_metrics()
                    await asyncio.sleep(self.monitoring_interval_seconds)
                except Exception as e:
                    logger.error(f"System monitoring error: {e}")
                    await asyncio.sleep(self.monitoring_interval_seconds)
        
        async def container_monitor():
            """Monitor container metrics."""
            while True:
                try:
                    await self.collect_container_metrics()
                    await asyncio.sleep(self.monitoring_interval_seconds)
                except Exception as e:
                    logger.error(f"Container monitoring error: {e}")
                    await asyncio.sleep(self.monitoring_interval_seconds)
        
        async def database_monitor():
            """Monitor database metrics."""
            while True:
                try:
                    await self.collect_database_metrics()
                    await asyncio.sleep(self.monitoring_interval_seconds * 2)  # Less frequent
                except Exception as e:
                    logger.error(f"Database monitoring error: {e}")
                    await asyncio.sleep(self.monitoring_interval_seconds * 2)
        
        async def queue_monitor():
            """Monitor queue metrics."""
            while True:
                try:
                    await self.collect_queue_metrics()
                    await asyncio.sleep(self.monitoring_interval_seconds)
                except Exception as e:
                    logger.error(f"Queue monitoring error: {e}")
                    await asyncio.sleep(self.monitoring_interval_seconds)
        
        async def anomaly_detector():
            """Run anomaly detection."""
            while True:
                try:
                    await asyncio.sleep(300)  # Run every 5 minutes
                    anomalies = await self.detect_anomalies()
                    
                    if anomalies:
                        logger.warning(f"Detected {len(anomalies)} metric anomalies")
                        for anomaly in anomalies:
                            logger.warning(f"Anomaly: {anomaly['metric_name']} = "
                                         f"{anomaly['current_value']:.2f} "
                                         f"(z-score: {anomaly['z_score']:.2f})")
                
                except Exception as e:
                    logger.error(f"Anomaly detection error: {e}")
        
        # Start all monitoring tasks
        self._monitoring_tasks = [
            asyncio.create_task(system_monitor()),
            asyncio.create_task(container_monitor()),
            asyncio.create_task(queue_monitor()),
            asyncio.create_task(anomaly_detector())
        ]
        
        # Add database monitor if available
        if self.db_session:
            self._monitoring_tasks.append(asyncio.create_task(database_monitor()))
        
        logger.info(f"Started {len(self._monitoring_tasks)} monitoring tasks")
    
    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status."""
        
        # Current metrics summary
        current_metrics = {}
        if self._system_metrics_history:
            latest_system = self._system_metrics_history[-1]
            current_metrics['system'] = {
                'cpu_percent': latest_system['cpu_percent'],
                'memory_percent': latest_system['memory_percent'],
                'disk_percent': latest_system['disk_percent'],
                'load_avg_1m': latest_system['load_avg_1m'],
                'timestamp': latest_system['timestamp'].isoformat()
            }
        
        # Active alerts
        with self._alert_lock:
            active_alerts = [
                {
                    'alert_id': alert.alert_id,
                    'resource_name': alert.resource_name,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'current_value': alert.current_value,
                    'threshold_value': alert.threshold_value,
                    'triggered_at': alert.triggered_at.isoformat(),
                    'duration_seconds': int((datetime.now() - alert.triggered_at).total_seconds())
                }
                for alert in self._active_alerts.values()
            ]
        
        # Recent anomalies
        recent_anomalies = await self.detect_anomalies()
        
        # Container status
        container_status = {}
        for container_name, history in self._container_metrics_history.items():
            if history:
                latest = history[-1]
                container_status[container_name] = {
                    'cpu_percent': latest.cpu_percent,
                    'memory_percent': latest.memory_percent,
                    'status': latest.status,
                    'restart_count': latest.restart_count
                }
        
        # Queue status
        queue_status = {}
        for queue_name, history in self._queue_metrics_history.items():
            if history:
                latest = history[-1]
                queue_status[queue_name] = {
                    'depth': latest.depth,
                    'avg_wait_time_ms': latest.avg_wait_time_ms,
                    'consumer_count': latest.consumer_count
                }
        
        # Database status
        database_status = None
        if self._database_metrics_history:
            latest_db = self._database_metrics_history[-1]
            database_status = {
                'connection_percent': latest_db.connection_percent,
                'cache_hit_ratio': latest_db.cache_hit_ratio,
                'queries_per_second': latest_db.queries_per_second,
                'slow_queries': latest_db.slow_queries,
                'replication_lag_ms': latest_db.replication_lag_ms
            }
        
        return {
            'monitoring_enabled': True,
            'monitoring_interval_seconds': self.monitoring_interval_seconds,
            'current_metrics': current_metrics,
            'active_alerts': active_alerts,
            'active_alert_count': len(active_alerts),
            'recent_anomalies': recent_anomalies,
            'container_status': container_status,
            'queue_status': queue_status,
            'database_status': database_status,
            'anomaly_detection_enabled': self._anomaly_detection_enabled,
            'thresholds_configured': len(self._resource_thresholds)
        }
    
    async def shutdown(self):
        """Gracefully shutdown the resource monitor."""
        
        # Cancel all monitoring tasks
        for task in self._monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
        
        logger.info("Advanced resource monitor shut down")


# Global resource monitor instance
advanced_resource_monitor: Optional[AdvancedResourceMonitor] = None

def initialize_resource_monitor(
    docker_client: Optional[docker.DockerClient] = None,
    redis_client: Optional[redis.Redis] = None,
    db_session: Optional[AsyncSession] = None
) -> AdvancedResourceMonitor:
    """Initialize the global resource monitor instance."""
    global advanced_resource_monitor
    advanced_resource_monitor = AdvancedResourceMonitor(
        docker_client=docker_client,
        redis_client=redis_client,
        db_session=db_session,
        monitoring_interval_seconds=config.get_monitoring_settings().get('monitoring_interval_seconds', 30)
    )
    return advanced_resource_monitor
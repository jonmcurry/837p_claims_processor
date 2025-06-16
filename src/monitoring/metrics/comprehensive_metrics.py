"""Comprehensive Prometheus metrics for production claims processing system."""

import time
import psutil
import asyncio
from contextlib import contextmanager
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from threading import Thread
import structlog

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    Summary,
    Enum,
    generate_latest,
    start_http_server,
    CollectorRegistry,
    multiprocess,
    values
)
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.database.base import get_postgres_session

logger = structlog.get_logger(__name__)


@dataclass
class MetricConfig:
    """Configuration for metric collection."""
    collection_interval: int = 30  # seconds
    retention_days: int = 30
    enable_detailed_metrics: bool = True
    enable_system_metrics: bool = True


class ComprehensiveMetricsCollector:
    """Comprehensive metrics collection for production monitoring."""

    def __init__(self, config: MetricConfig = None):
        """Initialize comprehensive metrics collector."""
        self.config = config or MetricConfig()
        self.registry = CollectorRegistry()
        self._setup_business_metrics()
        self._setup_infrastructure_metrics()
        self._setup_application_metrics()
        self._setup_security_metrics()
        self._setup_ml_metrics()
        
        # Start background metric collection
        self._start_background_collection()
        
        logger.info("Comprehensive metrics collector initialized",
                   collection_interval=self.config.collection_interval)

    def _setup_business_metrics(self):
        """Setup business-focused metrics."""
        
        # Claims Processing Metrics
        self.claims_processed_total = Counter(
            'claims_processed_total',
            'Total number of claims processed successfully',
            ['facility_id', 'financial_class', 'insurance_type', 'processing_stage'],
            registry=self.registry
        )
        
        self.claims_failed_total = Counter(
            'claims_failed_total', 
            'Total number of failed claims',
            ['facility_id', 'failure_category', 'failure_stage', 'severity'],
            registry=self.registry
        )
        
        self.claims_revenue_dollars = Counter(
            'claims_revenue_dollars_total',
            'Total revenue from processed claims in dollars',
            ['facility_id', 'insurance_type', 'service_type'],
            registry=self.registry
        )
        
        self.claims_processing_latency = Histogram(
            'claims_processing_latency_seconds',
            'Time taken to process individual claims',
            ['facility_id', 'complexity_level'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 300.0],
            registry=self.registry
        )
        
        # Batch Processing Metrics
        self.batch_processing_time = Histogram(
            'batch_processing_time_seconds',
            'Time taken to process entire batches',
            ['facility_id', 'batch_size_category'],
            buckets=[1, 5, 15, 30, 60, 120, 300, 600, 1800],
            registry=self.registry
        )
        
        self.batch_throughput_claims_per_second = Gauge(
            'batch_throughput_claims_per_second',
            'Current batch processing throughput',
            ['facility_id'],
            registry=self.registry
        )
        
        self.batch_size_distribution = Histogram(
            'batch_size_claims',
            'Distribution of batch sizes',
            buckets=[10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000],
            registry=self.registry
        )
        
        # Validation Metrics
        self.validation_rules_executed = Counter(
            'validation_rules_executed_total',
            'Total validation rules executed',
            ['rule_category', 'rule_severity'],
            registry=self.registry
        )
        
        self.validation_failures = Counter(
            'validation_failures_total',
            'Total validation rule failures',
            ['rule_id', 'rule_category', 'severity'],
            registry=self.registry
        )
        
        self.validation_pass_rate = Gauge(
            'validation_pass_rate_percent',
            'Percentage of claims passing validation',
            ['facility_id', 'rule_category'],
            registry=self.registry
        )
        
        # RVU and Financial Metrics
        self.rvu_calculated_total = Counter(
            'rvu_calculated_total',
            'Total RVUs calculated',
            ['facility_id', 'procedure_category'],
            registry=self.registry
        )
        
        self.reimbursement_variance = Histogram(
            'reimbursement_variance_percent',
            'Variance between expected and actual reimbursement',
            ['facility_id', 'payer_type'],
            buckets=[-50, -20, -10, -5, -1, 0, 1, 5, 10, 20, 50],
            registry=self.registry
        )
        
        # SLA Compliance Metrics
        self.sla_compliance_percentage = Gauge(
            'sla_compliance_percentage',
            'SLA compliance percentage',
            ['sla_type', 'facility_id'],
            registry=self.registry
        )
        
        self.processing_target_adherence = Gauge(
            'processing_target_adherence_percent',
            'Adherence to 100k claims/15s target',
            registry=self.registry
        )

    def _setup_infrastructure_metrics(self):
        """Setup infrastructure and system metrics."""
        
        # Database Performance
        self.database_query_duration = Histogram(
            'database_query_duration_seconds',
            'Database query execution time',
            ['database', 'query_type', 'table'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
            registry=self.registry
        )
        
        self.database_connections_active = Gauge(
            'database_connections_active',
            'Active database connections',
            ['database', 'connection_pool'],
            registry=self.registry
        )
        
        self.database_deadlocks_total = Counter(
            'database_deadlocks_total',
            'Total database deadlocks',
            ['database'],
            registry=self.registry
        )
        
        # Cache Performance
        self.cache_hits_total = Counter(
            'cache_hits_total',
            'Total cache hits',
            ['cache_type', 'cache_layer'],
            registry=self.registry
        )
        
        self.cache_misses_total = Counter(
            'cache_misses_total',
            'Total cache misses',
            ['cache_type', 'cache_layer'],
            registry=self.registry
        )
        
        self.cache_hit_ratio = Gauge(
            'cache_hit_ratio_percent',
            'Cache hit ratio percentage',
            ['cache_type'],
            registry=self.registry
        )
        
        # Queue Metrics
        self.queue_depth = Gauge(
            'queue_depth',
            'Current queue depth',
            ['queue_name', 'priority'],
            registry=self.registry
        )
        
        self.queue_processing_rate = Gauge(
            'queue_processing_rate_per_second',
            'Queue processing rate',
            ['queue_name'],
            registry=self.registry
        )
        
        # Network Metrics
        self.network_latency_ms = Histogram(
            'network_latency_milliseconds',
            'Network latency to external endpoints',
            ['endpoint', 'protocol'],
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500],
            registry=self.registry
        )

    def _setup_application_metrics(self):
        """Setup application-specific metrics."""
        
        # API Performance
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0],
            registry=self.registry
        )
        
        self.active_sessions = Gauge(
            'active_sessions_count',
            'Number of active user sessions',
            ['user_role'],
            registry=self.registry
        )
        
        # Application Health
        self.application_health_status = Enum(
            'application_health_status',
            'Overall application health status',
            states=['healthy', 'degraded', 'unhealthy'],
            registry=self.registry
        )
        
        self.service_availability = Gauge(
            'service_availability_percent',
            'Service availability percentage',
            ['service_name'],
            registry=self.registry
        )
        
        # Resource Utilization
        self.cpu_usage_percent = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage',
            ['service', 'host'],
            registry=self.registry
        )
        
        self.memory_usage_mb = Gauge(
            'memory_usage_megabytes',
            'Memory usage in megabytes',
            ['service', 'host'],
            registry=self.registry
        )
        
        self.disk_usage_percent = Gauge(
            'disk_usage_percent',
            'Disk usage percentage',
            ['mount_point', 'host'],
            registry=self.registry
        )

    def _setup_security_metrics(self):
        """Setup security and compliance metrics."""
        
        # Authentication Metrics
        self.login_attempts_total = Counter(
            'login_attempts_total',
            'Total login attempts',
            ['status', 'user_role', 'ip_subnet'],
            registry=self.registry
        )
        
        self.failed_login_attempts = Counter(
            'failed_login_attempts_total',
            'Failed login attempts',
            ['failure_reason', 'ip_address'],
            registry=self.registry
        )
        
        self.session_duration = Histogram(
            'session_duration_minutes',
            'User session duration',
            ['user_role'],
            buckets=[1, 5, 15, 30, 60, 120, 240, 480],
            registry=self.registry
        )
        
        # PHI Access Metrics
        self.phi_access_total = Counter(
            'phi_access_total',
            'Total PHI access attempts',
            ['user_role', 'access_type', 'status'],
            registry=self.registry
        )
        
        self.phi_fields_accessed = Counter(
            'phi_fields_accessed_total',
            'Total PHI fields accessed',
            ['field_type', 'user_role'],
            registry=self.registry
        )
        
        self.unauthorized_access_attempts = Counter(
            'unauthorized_access_attempts_total',
            'Unauthorized access attempts',
            ['resource_type', 'user_role', 'ip_address'],
            registry=self.registry
        )
        
        # Audit Metrics
        self.audit_events_total = Counter(
            'audit_events_total',
            'Total audit events logged',
            ['event_type', 'severity'],
            registry=self.registry
        )
        
        self.compliance_violations = Counter(
            'compliance_violations_total',
            'HIPAA compliance violations detected',
            ['violation_type', 'severity'],
            registry=self.registry
        )

    def _setup_ml_metrics(self):
        """Setup machine learning model metrics."""
        
        # Model Performance
        self.ml_prediction_accuracy = Gauge(
            'ml_prediction_accuracy_percent',
            'ML model prediction accuracy',
            ['model_name', 'model_version'],
            registry=self.registry
        )
        
        self.ml_prediction_latency = Histogram(
            'ml_prediction_latency_seconds',
            'ML model prediction latency',
            ['model_name', 'batch_size_category'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry
        )
        
        self.ml_model_inference_total = Counter(
            'ml_model_inference_total',
            'Total ML model inferences',
            ['model_name', 'prediction_result'],
            registry=self.registry
        )
        
        self.ml_model_errors = Counter(
            'ml_model_errors_total',
            'ML model errors',
            ['model_name', 'error_type'],
            registry=self.registry
        )
        
        # Feature Engineering
        self.feature_extraction_time = Histogram(
            'feature_extraction_time_seconds',
            'Time taken for feature extraction',
            ['feature_type'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
            registry=self.registry
        )

    def _start_background_collection(self):
        """Start background thread for metric collection."""
        def collect_system_metrics():
            while True:
                try:
                    if self.config.enable_system_metrics:
                        # CPU metrics
                        cpu_percent = psutil.cpu_percent(interval=1)
                        self.cpu_usage_percent.labels(
                            service='claims_processor',
                            host='local'
                        ).set(cpu_percent)
                        
                        # Memory metrics
                        memory = psutil.virtual_memory()
                        self.memory_usage_mb.labels(
                            service='claims_processor',
                            host='local'
                        ).set(memory.used / (1024 * 1024))
                        
                        # Disk metrics
                        disk = psutil.disk_usage('/')
                        self.disk_usage_percent.labels(
                            mount_point='/',
                            host='local'
                        ).set(disk.percent)
                    
                    time.sleep(self.config.collection_interval)
                    
                except Exception as e:
                    logger.error("Error collecting system metrics", error=str(e))
                    time.sleep(self.config.collection_interval)
        
        collector_thread = Thread(target=collect_system_metrics, daemon=True)
        collector_thread.start()

    async def collect_database_metrics(self):
        """Collect database performance metrics."""
        try:
            async with get_postgres_session() as session:
                # Query performance metrics
                query = text("""
                    SELECT 
                        query,
                        calls,
                        total_time,
                        mean_time,
                        rows
                    FROM pg_stat_statements 
                    ORDER BY total_time DESC 
                    LIMIT 10
                """)
                
                result = await session.execute(query)
                
                for row in result.fetchall():
                    self.database_query_duration.labels(
                        database='postgresql',
                        query_type='select',
                        table='various'
                    ).observe(row.mean_time / 1000)  # Convert to seconds
                
                # Connection metrics
                conn_query = text("""
                    SELECT state, count(*) as count
                    FROM pg_stat_activity 
                    WHERE datname = current_database()
                    GROUP BY state
                """)
                
                conn_result = await session.execute(conn_query)
                for row in conn_result.fetchall():
                    if row.state == 'active':
                        self.database_connections_active.labels(
                            database='postgresql',
                            connection_pool='main'
                        ).set(row.count)
                        
        except Exception as e:
            logger.error("Error collecting database metrics", error=str(e))

    def record_claim_processed(self, facility_id: str, financial_class: str, 
                             insurance_type: str, processing_time: float,
                             revenue: float = 0.0):
        """Record successful claim processing."""
        self.claims_processed_total.labels(
            facility_id=facility_id,
            financial_class=financial_class,
            insurance_type=insurance_type,
            processing_stage='completed'
        ).inc()
        
        self.claims_processing_latency.labels(
            facility_id=facility_id,
            complexity_level=self._categorize_complexity(processing_time)
        ).observe(processing_time)
        
        if revenue > 0:
            self.claims_revenue_dollars.labels(
                facility_id=facility_id,
                insurance_type=insurance_type,
                service_type='standard'
            ).inc(revenue)

    def record_claim_failed(self, facility_id: str, failure_category: str,
                          failure_stage: str, severity: str = 'error'):
        """Record failed claim processing."""
        self.claims_failed_total.labels(
            facility_id=facility_id,
            failure_category=failure_category,
            failure_stage=failure_stage,
            severity=severity
        ).inc()

    def record_batch_processing(self, facility_id: str, batch_size: int,
                              processing_time: float, throughput: float):
        """Record batch processing metrics."""
        self.batch_processing_time.labels(
            facility_id=facility_id,
            batch_size_category=self._categorize_batch_size(batch_size)
        ).observe(processing_time)
        
        self.batch_throughput_claims_per_second.labels(
            facility_id=facility_id
        ).set(throughput)
        
        self.batch_size_distribution.observe(batch_size)
        
        # Update target adherence (100k claims/15s = 6667 claims/sec)
        target_adherence = min(100, (throughput / 6667) * 100)
        self.processing_target_adherence.set(target_adherence)

    def record_validation_result(self, rule_id: str, rule_category: str,
                               severity: str, passed: bool):
        """Record validation rule execution."""
        self.validation_rules_executed.labels(
            rule_category=rule_category,
            rule_severity=severity
        ).inc()
        
        if not passed:
            self.validation_failures.labels(
                rule_id=rule_id,
                rule_category=rule_category,
                severity=severity
            ).inc()

    def record_phi_access(self, user_role: str, access_type: str,
                         phi_fields: List[str], success: bool = True):
        """Record PHI access for compliance monitoring."""
        status = 'success' if success else 'denied'
        
        self.phi_access_total.labels(
            user_role=user_role,
            access_type=access_type,
            status=status
        ).inc()
        
        for field in phi_fields:
            self.phi_fields_accessed.labels(
                field_type=field,
                user_role=user_role
            ).inc()

    def record_ml_prediction(self, model_name: str, prediction_time: float,
                           batch_size: int, accuracy: float = None):
        """Record ML model prediction metrics."""
        self.ml_prediction_latency.labels(
            model_name=model_name,
            batch_size_category=self._categorize_batch_size(batch_size)
        ).observe(prediction_time)
        
        self.ml_model_inference_total.labels(
            model_name=model_name,
            prediction_result='completed'
        ).inc()
        
        if accuracy is not None:
            self.ml_prediction_accuracy.labels(
                model_name=model_name,
                model_version='v1'
            ).set(accuracy * 100)  # Convert to percentage

    def _categorize_complexity(self, processing_time: float) -> str:
        """Categorize processing complexity based on time."""
        if processing_time < 1.0:
            return 'simple'
        elif processing_time < 5.0:
            return 'moderate'
        elif processing_time < 30.0:
            return 'complex'
        else:
            return 'very_complex'

    def _categorize_batch_size(self, batch_size: int) -> str:
        """Categorize batch size."""
        if batch_size < 100:
            return 'small'
        elif batch_size < 1000:
            return 'medium'
        elif batch_size < 10000:
            return 'large'
        else:
            return 'xlarge'

    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry)

    def start_metrics_server(self, port: int = 8000):
        """Start Prometheus metrics server."""
        start_http_server(port, registry=self.registry)
        logger.info("Metrics server started", port=port)


# Global metrics collector instance
metrics_collector = ComprehensiveMetricsCollector()

# Export commonly used metrics for easy access
claims_processed_total = metrics_collector.claims_processed_total
claims_failed_total = metrics_collector.claims_failed_total
processing_latency = metrics_collector.claims_processing_latency
batch_processing_time = metrics_collector.batch_processing_time
throughput_gauge = metrics_collector.batch_throughput_claims_per_second
ml_prediction_latency = metrics_collector.ml_prediction_latency
ml_prediction_accuracy = metrics_collector.ml_prediction_accuracy
ml_model_inference_total = metrics_collector.ml_model_inference_total
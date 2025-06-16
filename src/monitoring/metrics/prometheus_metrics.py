"""Prometheus metrics collection for claims processing system."""

import time
from contextlib import contextmanager
from typing import Dict, List, Optional

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
    start_http_server,
)

from src.core.config import settings


class MetricsCollector:
    """Centralized metrics collection for Prometheus."""

    def __init__(self):
        """Initialize metrics collector."""
        self._setup_metrics()

    def _setup_metrics(self) -> None:
        """Setup all Prometheus metrics."""
        
        # Business Metrics
        self.claims_processed_total = Counter(
            "claims_processed_total",
            "Total number of claims processed",
            ["facility_id", "status", "insurance_type"]
        )
        
        self.claims_failed_total = Counter(
            "claims_failed_total",
            "Total number of failed claims",
            ["facility_id", "failure_reason", "stage"]
        )
        
        self.revenue_processed_dollars = Gauge(
            "revenue_processed_dollars",
            "Total revenue processed in dollars",
            ["facility_id", "insurance_type"]
        )
        
        self.processing_latency_seconds = Histogram(
            "processing_latency_seconds",
            "Time taken to process claims",
            ["stage", "facility_id"],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
        )
        
        self.ml_prediction_accuracy = Gauge(
            "ml_prediction_accuracy",
            "ML model prediction accuracy percentage",
            ["model_type", "facility_id"]
        )
        
        self.validation_pass_rate = Gauge(
            "validation_pass_rate",
            "Validation pass rate percentage",
            ["rule_category", "facility_id"]
        )
        
        self.sla_compliance_percentage = Gauge(
            "sla_compliance_percentage",
            "SLA compliance percentage",
            ["metric_type"]
        )
        
        # Infrastructure Metrics
        self.cpu_usage_percent = Gauge(
            "cpu_usage_percent",
            "CPU usage percentage",
            ["service", "instance"]
        )
        
        self.memory_usage_mb = Gauge(
            "memory_usage_mb",
            "Memory usage in megabytes",
            ["service", "instance"]
        )
        
        self.database_connections_active = Gauge(
            "database_connections_active",
            "Active database connections",
            ["database_type", "instance"]
        )
        
        self.queue_depth = Gauge(
            "queue_depth",
            "Number of items in processing queue",
            ["queue_name", "priority"]
        )
        
        self.network_latency_seconds = Histogram(
            "network_latency_seconds",
            "Network latency in seconds",
            ["endpoint", "method"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
        )
        
        # Database Performance Metrics
        self.database_query_duration_seconds = Histogram(
            "database_query_duration_seconds",
            "Database query duration in seconds",
            ["database_type", "query_type"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
        )
        
        self.database_deadlocks_total = Counter(
            "database_deadlocks_total",
            "Total number of database deadlocks",
            ["database_type"]
        )
        
        self.slow_query_count = Counter(
            "slow_query_count",
            "Number of slow database queries",
            ["database_type", "query_type"]
        )
        
        self.connection_pool_saturation = Gauge(
            "connection_pool_saturation",
            "Connection pool saturation percentage",
            ["database_type"]
        )
        
        # Cache Metrics
        self.cache_hit_rate = Gauge(
            "cache_hit_rate",
            "Cache hit rate percentage",
            ["cache_type", "operation"]
        )
        
        self.cache_operations_total = Counter(
            "cache_operations_total",
            "Total cache operations",
            ["cache_type", "operation", "result"]
        )
        
        self.cache_size_bytes = Gauge(
            "cache_size_bytes",
            "Cache size in bytes",
            ["cache_type"]
        )
        
        # Application Health Metrics
        self.application_info = Info(
            "application_info",
            "Application information"
        )
        
        self.application_uptime_seconds = Gauge(
            "application_uptime_seconds",
            "Application uptime in seconds"
        )
        
        self.api_requests_total = Counter(
            "api_requests_total",
            "Total API requests",
            ["method", "endpoint", "status_code"]
        )
        
        self.api_request_duration_seconds = Histogram(
            "api_request_duration_seconds",
            "API request duration in seconds",
            ["method", "endpoint"],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        )
        
        # Business Logic Metrics
        self.batch_size_claims = Histogram(
            "batch_size_claims",
            "Number of claims in processing batches",
            ["facility_id"],
            buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
        )
        
        self.batch_processing_throughput = Gauge(
            "batch_processing_throughput",
            "Claims processing throughput per second",
            ["facility_id", "batch_type"]
        )
        
        self.rvu_calculation_errors_total = Counter(
            "rvu_calculation_errors_total",
            "Total RVU calculation errors",
            ["error_type", "procedure_code"]
        )
        
        self.ml_model_inference_duration_seconds = Histogram(
            "ml_model_inference_duration_seconds",
            "ML model inference duration in seconds",
            ["model_type", "batch_size_range"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
        )

    # Business Metrics Methods
    def increment_claims_processed(
        self, 
        facility_id: str, 
        status: str, 
        insurance_type: str,
        count: int = 1
    ) -> None:
        """Increment processed claims counter."""
        self.claims_processed_total.labels(
            facility_id=facility_id,
            status=status,
            insurance_type=insurance_type
        ).inc(count)

    def increment_claims_failed(
        self, 
        facility_id: str, 
        failure_reason: str, 
        stage: str,
        count: int = 1
    ) -> None:
        """Increment failed claims counter."""
        self.claims_failed_total.labels(
            facility_id=facility_id,
            failure_reason=failure_reason,
            stage=stage
        ).inc(count)

    def set_revenue_processed(
        self, 
        facility_id: str, 
        insurance_type: str, 
        amount: float
    ) -> None:
        """Set revenue processed gauge."""
        self.revenue_processed_dollars.labels(
            facility_id=facility_id,
            insurance_type=insurance_type
        ).set(amount)

    def observe_processing_latency(
        self, 
        stage: str, 
        facility_id: str, 
        duration: float
    ) -> None:
        """Observe processing latency."""
        self.processing_latency_seconds.labels(
            stage=stage,
            facility_id=facility_id
        ).observe(duration)

    def set_ml_prediction_accuracy(
        self, 
        model_type: str, 
        facility_id: str, 
        accuracy: float
    ) -> None:
        """Set ML prediction accuracy."""
        self.ml_prediction_accuracy.labels(
            model_type=model_type,
            facility_id=facility_id
        ).set(accuracy)

    def set_validation_pass_rate(
        self, 
        rule_category: str, 
        facility_id: str, 
        pass_rate: float
    ) -> None:
        """Set validation pass rate."""
        self.validation_pass_rate.labels(
            rule_category=rule_category,
            facility_id=facility_id
        ).set(pass_rate)

    def set_sla_compliance(self, metric_type: str, percentage: float) -> None:
        """Set SLA compliance percentage."""
        self.sla_compliance_percentage.labels(metric_type=metric_type).set(percentage)

    # Infrastructure Metrics Methods
    def set_cpu_usage(self, service: str, instance: str, percentage: float) -> None:
        """Set CPU usage percentage."""
        self.cpu_usage_percent.labels(service=service, instance=instance).set(percentage)

    def set_memory_usage(self, service: str, instance: str, mb: float) -> None:
        """Set memory usage in MB."""
        self.memory_usage_mb.labels(service=service, instance=instance).set(mb)

    def set_database_connections(self, database_type: str, instance: str, count: int) -> None:
        """Set active database connections."""
        self.database_connections_active.labels(
            database_type=database_type,
            instance=instance
        ).set(count)

    def set_queue_depth(self, queue_name: str, priority: str, depth: int) -> None:
        """Set queue depth."""
        self.queue_depth.labels(queue_name=queue_name, priority=priority).set(depth)

    def observe_network_latency(self, endpoint: str, method: str, latency: float) -> None:
        """Observe network latency."""
        self.network_latency_seconds.labels(endpoint=endpoint, method=method).observe(latency)

    # Database Metrics Methods
    def observe_database_query_duration(
        self, 
        database_type: str, 
        query_type: str, 
        duration: float
    ) -> None:
        """Observe database query duration."""
        self.database_query_duration_seconds.labels(
            database_type=database_type,
            query_type=query_type
        ).observe(duration)

    def increment_database_deadlocks(self, database_type: str) -> None:
        """Increment database deadlock counter."""
        self.database_deadlocks_total.labels(database_type=database_type).inc()

    def increment_slow_queries(self, database_type: str, query_type: str) -> None:
        """Increment slow query counter."""
        self.slow_query_count.labels(
            database_type=database_type,
            query_type=query_type
        ).inc()

    def set_connection_pool_saturation(self, database_type: str, percentage: float) -> None:
        """Set connection pool saturation."""
        self.connection_pool_saturation.labels(database_type=database_type).set(percentage)

    # Cache Metrics Methods
    def set_cache_hit_rate(self, cache_type: str, operation: str, hit_rate: float) -> None:
        """Set cache hit rate."""
        self.cache_hit_rate.labels(cache_type=cache_type, operation=operation).set(hit_rate)

    def increment_cache_operations(
        self, 
        cache_type: str, 
        operation: str, 
        result: str
    ) -> None:
        """Increment cache operations counter."""
        self.cache_operations_total.labels(
            cache_type=cache_type,
            operation=operation,
            result=result
        ).inc()

    def set_cache_size(self, cache_type: str, size_bytes: int) -> None:
        """Set cache size in bytes."""
        self.cache_size_bytes.labels(cache_type=cache_type).set(size_bytes)

    # API Metrics Methods
    def increment_api_requests(
        self, 
        method: str, 
        endpoint: str, 
        status_code: str
    ) -> None:
        """Increment API request counter."""
        self.api_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).inc()

    def observe_api_request_duration(
        self, 
        method: str, 
        endpoint: str, 
        duration: float
    ) -> None:
        """Observe API request duration."""
        self.api_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)

    # Business Logic Metrics Methods
    def observe_batch_size(self, facility_id: str, size: int) -> None:
        """Observe batch size."""
        self.batch_size_claims.labels(facility_id=facility_id).observe(size)

    def set_batch_throughput(
        self, 
        facility_id: str, 
        batch_type: str, 
        throughput: float
    ) -> None:
        """Set batch processing throughput."""
        self.batch_processing_throughput.labels(
            facility_id=facility_id,
            batch_type=batch_type
        ).set(throughput)

    def increment_rvu_calculation_errors(
        self, 
        error_type: str, 
        procedure_code: str
    ) -> None:
        """Increment RVU calculation errors."""
        self.rvu_calculation_errors_total.labels(
            error_type=error_type,
            procedure_code=procedure_code
        ).inc()

    def observe_ml_inference_duration(
        self, 
        model_type: str, 
        batch_size_range: str, 
        duration: float
    ) -> None:
        """Observe ML model inference duration."""
        self.ml_model_inference_duration_seconds.labels(
            model_type=model_type,
            batch_size_range=batch_size_range
        ).observe(duration)

    # Context Managers for Timing
    @contextmanager
    def time_processing_stage(self, stage: str, facility_id: str):
        """Context manager to time processing stages."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.observe_processing_latency(stage, facility_id, duration)

    @contextmanager
    def time_database_query(self, database_type: str, query_type: str):
        """Context manager to time database queries."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.observe_database_query_duration(database_type, query_type, duration)

    @contextmanager
    def time_api_request(self, method: str, endpoint: str):
        """Context manager to time API requests."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.observe_api_request_duration(method, endpoint, duration)

    @contextmanager
    def time_ml_inference(self, model_type: str, batch_size: int):
        """Context manager to time ML model inference."""
        start_time = time.time()
        
        # Determine batch size range
        if batch_size <= 10:
            batch_range = "1-10"
        elif batch_size <= 50:
            batch_range = "11-50"
        elif batch_size <= 100:
            batch_range = "51-100"
        else:
            batch_range = "100+"
            
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.observe_ml_inference_duration(model_type, batch_range, duration)

    def set_application_info(self, version: str, environment: str) -> None:
        """Set application information."""
        self.application_info.info({
            "version": version,
            "environment": environment,
            "name": settings.app_name
        })

    def update_uptime(self, start_time: float) -> None:
        """Update application uptime."""
        uptime = time.time() - start_time
        self.application_uptime_seconds.set(uptime)

    def start_metrics_server(self) -> None:
        """Start Prometheus metrics HTTP server."""
        if settings.enable_metrics:
            start_http_server(settings.prometheus_port)

    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest().decode('utf-8')


# Global metrics collector instance
metrics = MetricsCollector()
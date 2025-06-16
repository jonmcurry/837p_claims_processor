"""Predictive auto-scaling system based on queue depth and resource monitoring."""

import asyncio
import time
import logging
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import math

import numpy as np
import psutil
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import docker

from src.core.config import config
from src.monitoring.metrics.comprehensive_metrics import metrics_collector


logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling direction options."""
    UP = "up"
    DOWN = "down"
    NONE = "none"


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    WORKER_PROCESSES = "worker_processes"
    API_INSTANCES = "api_instances"
    BATCH_PROCESSORS = "batch_processors"
    ML_WORKERS = "ml_workers"
    DATABASE_CONNECTIONS = "database_connections"


@dataclass
class ResourceMetrics:
    """Current resource utilization metrics."""
    cpu_percent: float
    memory_percent: float
    disk_io_percent: float
    network_io_mbps: float
    queue_depth: int
    queue_wait_time_avg_ms: float
    active_workers: int
    completed_tasks_per_minute: float
    error_rate_percent: float
    response_time_p95_ms: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ScalingRule:
    """Auto-scaling rule configuration."""
    resource_type: ResourceType
    name: str
    condition: str  # Python expression to evaluate
    scaling_direction: ScalingDirection
    scaling_factor: float  # How much to scale (e.g., 1.5 = 50% increase)
    cooldown_minutes: int
    min_instances: int
    max_instances: int
    priority: int = 5  # 1-10, higher = more important
    enabled: bool = True


@dataclass
class ScalingAction:
    """Represents a scaling action to be executed."""
    action_id: str
    resource_type: ResourceType
    direction: ScalingDirection
    current_instances: int
    target_instances: int
    reason: str
    confidence: float
    predicted_load: float
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    status: str = "pending"  # pending, executing, completed, failed


@dataclass
class LoadPrediction:
    """Load prediction for future time periods."""
    timestamp: datetime
    predicted_queue_depth: float
    predicted_cpu_percent: float
    predicted_memory_percent: float
    predicted_throughput: float
    confidence: float
    horizon_minutes: int


class PredictiveAutoScaler:
    """Advanced auto-scaling system with ML-based load prediction."""
    
    def __init__(self,
                 docker_client: Optional[docker.DockerClient] = None,
                 prediction_horizon_minutes: int = 30,
                 scaling_check_interval_seconds: int = 60):
        
        self.docker_client = docker_client or docker.from_env()
        self.prediction_horizon_minutes = prediction_horizon_minutes
        self.scaling_check_interval_seconds = scaling_check_interval_seconds
        
        # Resource monitoring
        self._resource_metrics_history: deque = deque(maxlen=1440)  # 24 hours of minute-level data
        self._current_metrics: Optional[ResourceMetrics] = None
        self._metrics_lock = threading.RLock()
        
        # Scaling configuration
        self._scaling_rules: Dict[str, ScalingRule] = {}
        self._register_default_scaling_rules()
        
        # Scaling actions and cooldowns
        self._scaling_actions: Dict[str, ScalingAction] = {}
        self._last_scaling_time: Dict[ResourceType, datetime] = {}
        self._current_instances: Dict[ResourceType, int] = defaultdict(lambda: 1)
        
        # ML prediction models
        self._load_predictor: Optional[RandomForestRegressor] = None
        self._feature_scaler = StandardScaler()
        self._prediction_history: deque = deque(maxlen=1000)
        self._model_accuracy = 0.0
        
        # Background tasks
        self._metrics_collector_task = None
        self._scaling_monitor_task = None
        self._predictor_trainer_task = None
        self._start_background_tasks()
        
        # Performance tracking
        self._scaling_performance: Dict[str, List[float]] = defaultdict(list)
        
        # External scaling callbacks
        self._scaling_callbacks: Dict[ResourceType, List[Callable]] = defaultdict(list)
    
    def register_scaling_callback(self, 
                                resource_type: ResourceType, 
                                callback: Callable[[int, int], None]):
        """Register callback for scaling events."""
        self._scaling_callbacks[resource_type].append(callback)
        logger.info(f"Registered scaling callback for {resource_type.value}")
    
    def _register_default_scaling_rules(self):
        """Register default auto-scaling rules."""
        
        # Worker process scaling rules
        self._scaling_rules.update({
            "worker_scale_up_queue": ScalingRule(
                resource_type=ResourceType.WORKER_PROCESSES,
                name="Scale up workers based on queue depth",
                condition="queue_depth > 1000 and queue_wait_time_avg_ms > 5000",
                scaling_direction=ScalingDirection.UP,
                scaling_factor=1.5,
                cooldown_minutes=5,
                min_instances=2,
                max_instances=20,
                priority=8
            ),
            
            "worker_scale_up_cpu": ScalingRule(
                resource_type=ResourceType.WORKER_PROCESSES,
                name="Scale up workers based on CPU usage",
                condition="cpu_percent > 80 and completed_tasks_per_minute > 100",
                scaling_direction=ScalingDirection.UP,
                scaling_factor=1.3,
                cooldown_minutes=3,
                min_instances=2,
                max_instances=20,
                priority=7
            ),
            
            "worker_scale_down": ScalingRule(
                resource_type=ResourceType.WORKER_PROCESSES,
                name="Scale down workers when load is low",
                condition="queue_depth < 100 and cpu_percent < 40 and queue_wait_time_avg_ms < 1000",
                scaling_direction=ScalingDirection.DOWN,
                scaling_factor=0.8,
                cooldown_minutes=10,
                min_instances=2,
                max_instances=20,
                priority=5
            ),
        })
        
        # API instance scaling rules
        self._scaling_rules.update({
            "api_scale_up_response_time": ScalingRule(
                resource_type=ResourceType.API_INSTANCES,
                name="Scale up API instances for high response times",
                condition="response_time_p95_ms > 2000 and error_rate_percent < 5",
                scaling_direction=ScalingDirection.UP,
                scaling_factor=1.4,
                cooldown_minutes=3,
                min_instances=2,
                max_instances=10,
                priority=9
            ),
            
            "api_scale_up_error_rate": ScalingRule(
                resource_type=ResourceType.API_INSTANCES,
                name="Scale up API instances for high error rate",
                condition="error_rate_percent > 5 and response_time_p95_ms > 1000",
                scaling_direction=ScalingDirection.UP,
                scaling_factor=1.6,
                cooldown_minutes=2,
                min_instances=2,
                max_instances=10,
                priority=10
            ),
            
            "api_scale_down": ScalingRule(
                resource_type=ResourceType.API_INSTANCES,
                name="Scale down API instances when load is low",
                condition="response_time_p95_ms < 500 and error_rate_percent < 1 and cpu_percent < 30",
                scaling_direction=ScalingDirection.DOWN,
                scaling_factor=0.9,
                cooldown_minutes=15,
                min_instances=2,
                max_instances=10,
                priority=4
            ),
        })
        
        # Batch processor scaling rules
        self._scaling_rules.update({
            "batch_scale_up_throughput": ScalingRule(
                resource_type=ResourceType.BATCH_PROCESSORS,
                name="Scale up batch processors for high load",
                condition="queue_depth > 5000 and completed_tasks_per_minute < 500",
                scaling_direction=ScalingDirection.UP,
                scaling_factor=1.5,
                cooldown_minutes=5,
                min_instances=1,
                max_instances=8,
                priority=8
            ),
            
            "batch_scale_down": ScalingRule(
                resource_type=ResourceType.BATCH_PROCESSORS,
                name="Scale down batch processors when idle",
                condition="queue_depth < 500 and cpu_percent < 25",
                scaling_direction=ScalingDirection.DOWN,
                scaling_factor=0.7,
                cooldown_minutes=20,
                min_instances=1,
                max_instances=8,
                priority=3
            ),
        })
        
        # ML worker scaling rules
        self._scaling_rules.update({
            "ml_scale_up_queue": ScalingRule(
                resource_type=ResourceType.ML_WORKERS,
                name="Scale up ML workers for inference load",
                condition="queue_depth > 2000 and queue_wait_time_avg_ms > 10000",
                scaling_direction=ScalingDirection.UP,
                scaling_factor=1.4,
                cooldown_minutes=8,  # ML workers take longer to start
                min_instances=1,
                max_instances=6,
                priority=7
            ),
            
            "ml_scale_down": ScalingRule(
                resource_type=ResourceType.ML_WORKERS,
                name="Scale down ML workers when not needed",
                condition="queue_depth < 200 and cpu_percent < 20",
                scaling_direction=ScalingDirection.DOWN,
                scaling_factor=0.8,
                cooldown_minutes=30,  # Keep ML workers longer
                min_instances=1,
                max_instances=6,
                priority=2
            ),
        })
    
    async def collect_current_metrics(self) -> ResourceMetrics:
        """Collect current system and application metrics."""
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk_io = psutil.disk_io_counters()
        network_io = psutil.net_io_counters()
        
        # Calculate network throughput (simplified)
        network_mbps = (network_io.bytes_sent + network_io.bytes_recv) / (1024 * 1024) if network_io else 0
        
        # Application metrics (these would come from your monitoring system)
        # For now, we'll use mock values or actual metrics if available
        queue_depth = await self._get_queue_depth()
        queue_wait_time = await self._get_average_queue_wait_time()
        active_workers = await self._get_active_workers()
        throughput = await self._get_completed_tasks_per_minute()
        error_rate = await self._get_error_rate_percent()
        response_time_p95 = await self._get_response_time_p95()
        
        metrics = ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_io_percent=0.0,  # Would calculate from baseline
            network_io_mbps=network_mbps,
            queue_depth=queue_depth,
            queue_wait_time_avg_ms=queue_wait_time,
            active_workers=active_workers,
            completed_tasks_per_minute=throughput,
            error_rate_percent=error_rate,
            response_time_p95_ms=response_time_p95
        )
        
        # Store metrics
        with self._metrics_lock:
            self._current_metrics = metrics
            self._resource_metrics_history.append(metrics)
        
        # Record metrics for monitoring
        metrics_collector.record_system_metrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            queue_depth=queue_depth,
            active_workers=active_workers
        )
        
        return metrics
    
    async def _get_queue_depth(self) -> int:
        """Get current queue depth from processing system."""
        # This would integrate with your actual queue system
        # For now, return a mock value
        return 500
    
    async def _get_average_queue_wait_time(self) -> float:
        """Get average queue wait time in milliseconds."""
        # This would integrate with your actual queue system
        return 2000.0
    
    async def _get_active_workers(self) -> int:
        """Get number of active workers."""
        # This would check your actual worker processes
        return self._current_instances.get(ResourceType.WORKER_PROCESSES, 4)
    
    async def _get_completed_tasks_per_minute(self) -> float:
        """Get completed tasks per minute."""
        # This would come from your metrics system
        return 250.0
    
    async def _get_error_rate_percent(self) -> float:
        """Get current error rate percentage."""
        # This would come from your metrics system
        return 2.5
    
    async def _get_response_time_p95(self) -> float:
        """Get 95th percentile response time in milliseconds."""
        # This would come from your metrics system
        return 800.0
    
    async def evaluate_scaling_rules(self, metrics: ResourceMetrics) -> List[ScalingAction]:
        """Evaluate all scaling rules and return recommended actions."""
        
        recommended_actions = []
        
        # Create evaluation context
        eval_context = {
            'cpu_percent': metrics.cpu_percent,
            'memory_percent': metrics.memory_percent,
            'disk_io_percent': metrics.disk_io_percent,
            'network_io_mbps': metrics.network_io_mbps,
            'queue_depth': metrics.queue_depth,
            'queue_wait_time_avg_ms': metrics.queue_wait_time_avg_ms,
            'active_workers': metrics.active_workers,
            'completed_tasks_per_minute': metrics.completed_tasks_per_minute,
            'error_rate_percent': metrics.error_rate_percent,
            'response_time_p95_ms': metrics.response_time_p95_ms,
        }
        
        # Evaluate each rule
        for rule_id, rule in self._scaling_rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown
            if self._is_in_cooldown(rule.resource_type, rule.cooldown_minutes):
                continue
            
            try:
                # Evaluate rule condition
                condition_met = eval(rule.condition, {"__builtins__": {}}, eval_context)
                
                if condition_met:
                    # Calculate target instances
                    current_instances = self._current_instances[rule.resource_type]
                    
                    if rule.scaling_direction == ScalingDirection.UP:
                        target_instances = min(
                            math.ceil(current_instances * rule.scaling_factor),
                            rule.max_instances
                        )
                    else:  # ScalingDirection.DOWN
                        target_instances = max(
                            math.floor(current_instances * rule.scaling_factor),
                            rule.min_instances
                        )
                    
                    # Only create action if instances would actually change
                    if target_instances != current_instances:
                        action = ScalingAction(
                            action_id=f"{rule_id}_{int(time.time())}",
                            resource_type=rule.resource_type,
                            direction=rule.scaling_direction,
                            current_instances=current_instances,
                            target_instances=target_instances,
                            reason=f"Rule '{rule.name}' triggered: {rule.condition}",
                            confidence=0.8 + (rule.priority / 10) * 0.2,  # Higher priority = higher confidence
                            predicted_load=metrics.queue_depth
                        )
                        
                        recommended_actions.append(action)
                        
                        logger.info(f"Scaling rule triggered: {rule.name} - "
                                   f"{rule.resource_type.value} "
                                   f"{current_instances} -> {target_instances}")
            
            except Exception as e:
                logger.error(f"Error evaluating scaling rule {rule_id}: {e}")
        
        # Sort by priority and confidence
        recommended_actions.sort(
            key=lambda x: (self._get_rule_priority(x), x.confidence), 
            reverse=True
        )
        
        return recommended_actions
    
    def _get_rule_priority(self, action: ScalingAction) -> int:
        """Get priority for scaling action."""
        for rule in self._scaling_rules.values():
            if (rule.resource_type == action.resource_type and 
                rule.scaling_direction == action.direction):
                return rule.priority
        return 5  # Default priority
    
    def _is_in_cooldown(self, resource_type: ResourceType, cooldown_minutes: int) -> bool:
        """Check if resource is in cooldown period."""
        if resource_type not in self._last_scaling_time:
            return False
        
        time_since_last = datetime.now() - self._last_scaling_time[resource_type]
        return time_since_last.total_seconds() < (cooldown_minutes * 60)
    
    async def predict_future_load(self, horizon_minutes: int = None) -> LoadPrediction:
        """Predict future load using ML model."""
        
        if horizon_minutes is None:
            horizon_minutes = self.prediction_horizon_minutes
        
        if self._load_predictor is None or len(self._resource_metrics_history) < 10:
            # Return current load if no prediction model
            current = self._current_metrics
            if current:
                return LoadPrediction(
                    timestamp=datetime.now() + timedelta(minutes=horizon_minutes),
                    predicted_queue_depth=current.queue_depth,
                    predicted_cpu_percent=current.cpu_percent,
                    predicted_memory_percent=current.memory_percent,
                    predicted_throughput=current.completed_tasks_per_minute,
                    confidence=0.5,
                    horizon_minutes=horizon_minutes
                )
        
        try:
            # Prepare features for prediction
            features = self._extract_prediction_features()
            
            if features is not None:
                # Scale features
                features_scaled = self._feature_scaler.transform([features])
                
                # Make prediction
                prediction = self._load_predictor.predict(features_scaled)[0]
                
                # Parse prediction (assuming it's queue depth for simplicity)
                predicted_queue_depth = max(0, prediction)
                
                # Estimate other metrics based on historical correlations
                predicted_cpu = self._estimate_cpu_from_queue(predicted_queue_depth)
                predicted_memory = self._estimate_memory_from_queue(predicted_queue_depth)
                predicted_throughput = self._estimate_throughput_from_queue(predicted_queue_depth)
                
                return LoadPrediction(
                    timestamp=datetime.now() + timedelta(minutes=horizon_minutes),
                    predicted_queue_depth=predicted_queue_depth,
                    predicted_cpu_percent=predicted_cpu,
                    predicted_memory_percent=predicted_memory,
                    predicted_throughput=predicted_throughput,
                    confidence=self._model_accuracy,
                    horizon_minutes=horizon_minutes
                )
        
        except Exception as e:
            logger.warning(f"Load prediction failed: {e}")
        
        # Fallback to trend-based prediction
        return self._trend_based_prediction(horizon_minutes)
    
    def _extract_prediction_features(self) -> Optional[List[float]]:
        """Extract features for load prediction."""
        
        if len(self._resource_metrics_history) < 5:
            return None
        
        recent_metrics = list(self._resource_metrics_history)[-5:]  # Last 5 minutes
        
        # Time-based features
        now = datetime.now()
        hour_of_day = now.hour
        day_of_week = now.weekday()
        minute_of_hour = now.minute
        
        # Historical load features
        avg_queue_depth = np.mean([m.queue_depth for m in recent_metrics])
        max_queue_depth = np.max([m.queue_depth for m in recent_metrics])
        queue_trend = recent_metrics[-1].queue_depth - recent_metrics[0].queue_depth
        
        avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
        avg_memory = np.mean([m.memory_percent for m in recent_metrics])
        avg_throughput = np.mean([m.completed_tasks_per_minute for m in recent_metrics])
        
        # Volatility measures
        queue_std = np.std([m.queue_depth for m in recent_metrics])
        cpu_std = np.std([m.cpu_percent for m in recent_metrics])
        
        return [
            hour_of_day, day_of_week, minute_of_hour,
            avg_queue_depth, max_queue_depth, queue_trend,
            avg_cpu, avg_memory, avg_throughput,
            queue_std, cpu_std
        ]
    
    def _estimate_cpu_from_queue(self, queue_depth: float) -> float:
        """Estimate CPU usage from queue depth using historical correlation."""
        # Simplified correlation - in practice, this would be learned from data
        base_cpu = 20  # Base CPU usage
        queue_factor = min(queue_depth / 100, 80)  # Up to 80% additional CPU
        return min(base_cpu + queue_factor, 100)
    
    def _estimate_memory_from_queue(self, queue_depth: float) -> float:
        """Estimate memory usage from queue depth."""
        base_memory = 30  # Base memory usage
        queue_factor = min(queue_depth / 200, 50)  # Up to 50% additional memory
        return min(base_memory + queue_factor, 90)
    
    def _estimate_throughput_from_queue(self, queue_depth: float) -> float:
        """Estimate throughput based on queue depth."""
        # Higher queue depth might indicate higher throughput potential
        base_throughput = 100
        if queue_depth < 500:
            return base_throughput + queue_depth * 0.2
        else:
            # Diminishing returns for very high queues
            return base_throughput + 100 + (queue_depth - 500) * 0.05
    
    def _trend_based_prediction(self, horizon_minutes: int) -> LoadPrediction:
        """Fallback trend-based prediction."""
        
        if len(self._resource_metrics_history) < 2:
            current = self._current_metrics
            return LoadPrediction(
                timestamp=datetime.now() + timedelta(minutes=horizon_minutes),
                predicted_queue_depth=current.queue_depth if current else 500,
                predicted_cpu_percent=current.cpu_percent if current else 50,
                predicted_memory_percent=current.memory_percent if current else 60,
                predicted_throughput=current.completed_tasks_per_minute if current else 200,
                confidence=0.3,
                horizon_minutes=horizon_minutes
            )
        
        # Simple linear trend
        recent_metrics = list(self._resource_metrics_history)[-10:]  # Last 10 minutes
        
        queue_trend = (recent_metrics[-1].queue_depth - recent_metrics[0].queue_depth) / len(recent_metrics)
        cpu_trend = (recent_metrics[-1].cpu_percent - recent_metrics[0].cpu_percent) / len(recent_metrics)
        memory_trend = (recent_metrics[-1].memory_percent - recent_metrics[0].memory_percent) / len(recent_metrics)
        
        current = recent_metrics[-1]
        
        return LoadPrediction(
            timestamp=datetime.now() + timedelta(minutes=horizon_minutes),
            predicted_queue_depth=max(0, current.queue_depth + queue_trend * horizon_minutes),
            predicted_cpu_percent=max(0, min(100, current.cpu_percent + cpu_trend * horizon_minutes)),
            predicted_memory_percent=max(0, min(100, current.memory_percent + memory_trend * horizon_minutes)),
            predicted_throughput=current.completed_tasks_per_minute,
            confidence=0.6,
            horizon_minutes=horizon_minutes
        )
    
    async def execute_scaling_action(self, action: ScalingAction) -> bool:
        """Execute a scaling action."""
        
        logger.info(f"Executing scaling action: {action.resource_type.value} "
                   f"{action.current_instances} -> {action.target_instances} "
                   f"({action.reason})")
        
        action.status = "executing"
        action.executed_at = datetime.now()
        
        try:
            success = False
            
            if action.resource_type == ResourceType.WORKER_PROCESSES:
                success = await self._scale_worker_processes(action.target_instances)
            
            elif action.resource_type == ResourceType.API_INSTANCES:
                success = await self._scale_api_instances(action.target_instances)
            
            elif action.resource_type == ResourceType.BATCH_PROCESSORS:
                success = await self._scale_batch_processors(action.target_instances)
            
            elif action.resource_type == ResourceType.ML_WORKERS:
                success = await self._scale_ml_workers(action.target_instances)
            
            elif action.resource_type == ResourceType.DATABASE_CONNECTIONS:
                success = await self._scale_database_connections(action.target_instances)
            
            if success:
                # Update instance count
                self._current_instances[action.resource_type] = action.target_instances
                self._last_scaling_time[action.resource_type] = datetime.now()
                
                # Call registered callbacks
                for callback in self._scaling_callbacks[action.resource_type]:
                    try:
                        await callback(action.current_instances, action.target_instances)
                    except Exception as e:
                        logger.error(f"Scaling callback error: {e}")
                
                action.status = "completed"
                
                # Record metrics
                metrics_collector.record_scaling_action(
                    resource_type=action.resource_type.value,
                    direction=action.direction.value,
                    current_instances=action.current_instances,
                    target_instances=action.target_instances,
                    confidence=action.confidence
                )
                
                logger.info(f"Scaling action completed successfully: {action.action_id}")
                return True
            
            else:
                action.status = "failed"
                logger.error(f"Scaling action failed: {action.action_id}")
                return False
        
        except Exception as e:
            action.status = "failed"
            logger.error(f"Error executing scaling action {action.action_id}: {e}")
            return False
    
    async def _scale_worker_processes(self, target_instances: int) -> bool:
        """Scale worker processes."""
        try:
            # This would integrate with your process management system
            # For Docker, you might scale a service
            if self.docker_client:
                # Example: Scale a Docker service
                services = self.docker_client.services.list(filters={'name': 'claims-worker'})
                if services:
                    service = services[0]
                    service.update(mode={'Replicated': {'Replicas': target_instances}})
                    return True
            
            # For process-based scaling, you might use supervisor or similar
            logger.info(f"Would scale worker processes to {target_instances}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale worker processes: {e}")
            return False
    
    async def _scale_api_instances(self, target_instances: int) -> bool:
        """Scale API instances."""
        try:
            if self.docker_client:
                services = self.docker_client.services.list(filters={'name': 'claims-api'})
                if services:
                    service = services[0]
                    service.update(mode={'Replicated': {'Replicas': target_instances}})
                    return True
            
            logger.info(f"Would scale API instances to {target_instances}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale API instances: {e}")
            return False
    
    async def _scale_batch_processors(self, target_instances: int) -> bool:
        """Scale batch processors."""
        try:
            # This would scale your batch processing workers
            logger.info(f"Would scale batch processors to {target_instances}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale batch processors: {e}")
            return False
    
    async def _scale_ml_workers(self, target_instances: int) -> bool:
        """Scale ML workers."""
        try:
            # This would scale your ML inference workers
            logger.info(f"Would scale ML workers to {target_instances}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale ML workers: {e}")
            return False
    
    async def _scale_database_connections(self, target_instances: int) -> bool:
        """Scale database connection pool."""
        try:
            # This would adjust database connection pool sizes
            logger.info(f"Would scale database connections to {target_instances}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale database connections: {e}")
            return False
    
    def _start_background_tasks(self):
        """Start background monitoring and scaling tasks."""
        
        async def metrics_collector():
            """Continuously collect system metrics."""
            while True:
                try:
                    await self.collect_current_metrics()
                    await asyncio.sleep(60)  # Collect metrics every minute
                except Exception as e:
                    logger.error(f"Metrics collection error: {e}")
                    await asyncio.sleep(60)
        
        async def scaling_monitor():
            """Monitor for scaling opportunities."""
            while True:
                try:
                    await asyncio.sleep(self.scaling_check_interval_seconds)
                    
                    if self._current_metrics:
                        # Evaluate scaling rules
                        actions = await self.evaluate_scaling_rules(self._current_metrics)
                        
                        # Execute highest priority action (one at a time to avoid conflicts)
                        if actions:
                            action = actions[0]  # Highest priority
                            
                            # Get prediction to validate action
                            prediction = await self.predict_future_load()
                            
                            # Only execute if prediction supports the action
                            if self._validate_action_with_prediction(action, prediction):
                                await self.execute_scaling_action(action)
                                self._scaling_actions[action.action_id] = action
                
                except Exception as e:
                    logger.error(f"Scaling monitor error: {e}")
        
        async def predictor_trainer():
            """Train ML prediction model with recent data."""
            while True:
                try:
                    await asyncio.sleep(1800)  # Train every 30 minutes
                    await self._train_prediction_model()
                except Exception as e:
                    logger.error(f"Predictor trainer error: {e}")
        
        self._metrics_collector_task = asyncio.create_task(metrics_collector())
        self._scaling_monitor_task = asyncio.create_task(scaling_monitor())
        self._predictor_trainer_task = asyncio.create_task(predictor_trainer())
    
    def _validate_action_with_prediction(self, 
                                       action: ScalingAction, 
                                       prediction: LoadPrediction) -> bool:
        """Validate scaling action against load prediction."""
        
        # Don't scale up if load is predicted to decrease significantly
        if (action.direction == ScalingDirection.UP and 
            prediction.predicted_queue_depth < action.predicted_load * 0.7):
            logger.info(f"Skipping scale-up action - load predicted to decrease")
            return False
        
        # Don't scale down if load is predicted to increase significantly
        if (action.direction == ScalingDirection.DOWN and 
            prediction.predicted_queue_depth > action.predicted_load * 1.3):
            logger.info(f"Skipping scale-down action - load predicted to increase")
            return False
        
        # Require minimum confidence for scaling actions
        if prediction.confidence < 0.5:
            logger.info(f"Skipping action - prediction confidence too low: {prediction.confidence}")
            return False
        
        return True
    
    async def _train_prediction_model(self):
        """Train ML model for load prediction."""
        
        if len(self._resource_metrics_history) < 50:
            logger.debug("Not enough data to train prediction model")
            return
        
        try:
            # Prepare training data
            features = []
            targets = []
            
            metrics_list = list(self._resource_metrics_history)
            
            # Create feature vectors and targets
            for i in range(10, len(metrics_list) - 5):  # Leave some lookahead
                # Features: historical data
                feature_vector = []
                
                # Time features
                timestamp = metrics_list[i].timestamp
                feature_vector.extend([
                    timestamp.hour,
                    timestamp.weekday(),
                    timestamp.minute
                ])
                
                # Historical metrics (last 10 minutes)
                historical_metrics = metrics_list[i-10:i]
                
                feature_vector.extend([
                    np.mean([m.queue_depth for m in historical_metrics]),
                    np.max([m.queue_depth for m in historical_metrics]),
                    np.std([m.queue_depth for m in historical_metrics]),
                    np.mean([m.cpu_percent for m in historical_metrics]),
                    np.mean([m.memory_percent for m in historical_metrics]),
                    np.mean([m.completed_tasks_per_minute for m in historical_metrics]),
                    historical_metrics[-1].queue_depth - historical_metrics[0].queue_depth,  # Trend
                ])
                
                # Target: queue depth 5 minutes in the future
                target = metrics_list[i + 5].queue_depth
                
                features.append(feature_vector)
                targets.append(target)
            
            if len(features) > 20:
                X = np.array(features)
                y = np.array(targets)
                
                # Scale features
                X_scaled = self._feature_scaler.fit_transform(X)
                
                # Train model
                if self._load_predictor is None:
                    self._load_predictor = RandomForestRegressor(
                        n_estimators=50,
                        max_depth=10,
                        random_state=42
                    )
                
                # Split for validation
                split_idx = int(len(X_scaled) * 0.8)
                X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]
                
                # Train model
                self._load_predictor.fit(X_train, y_train)
                
                # Calculate accuracy
                y_pred = self._load_predictor.predict(X_val)
                mae = mean_absolute_error(y_val, y_pred)
                self._model_accuracy = max(0, 1 - (mae / np.mean(y_val)))
                
                logger.info(f"Updated prediction model - accuracy: {self._model_accuracy:.3f}, "
                           f"MAE: {mae:.1f}")
        
        except Exception as e:
            logger.error(f"Failed to train prediction model: {e}")
    
    async def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling system status."""
        
        # Current instances
        current_instances = dict(self._current_instances)
        
        # Recent scaling actions
        recent_actions = [
            {
                'action_id': action.action_id,
                'resource_type': action.resource_type.value,
                'direction': action.direction.value,
                'current_instances': action.current_instances,
                'target_instances': action.target_instances,
                'status': action.status,
                'created_at': action.created_at.isoformat(),
                'confidence': action.confidence
            }
            for action in sorted(self._scaling_actions.values(), 
                               key=lambda x: x.created_at, reverse=True)[:10]
        ]
        
        # Rule status
        rule_status = {
            rule_id: {
                'name': rule.name,
                'resource_type': rule.resource_type.value,
                'condition': rule.condition,
                'enabled': rule.enabled,
                'in_cooldown': self._is_in_cooldown(rule.resource_type, rule.cooldown_minutes)
            }
            for rule_id, rule in self._scaling_rules.items()
        }
        
        # Current metrics
        current_metrics = None
        if self._current_metrics:
            current_metrics = {
                'cpu_percent': self._current_metrics.cpu_percent,
                'memory_percent': self._current_metrics.memory_percent,
                'queue_depth': self._current_metrics.queue_depth,
                'queue_wait_time_avg_ms': self._current_metrics.queue_wait_time_avg_ms,
                'active_workers': self._current_metrics.active_workers,
                'completed_tasks_per_minute': self._current_metrics.completed_tasks_per_minute,
                'error_rate_percent': self._current_metrics.error_rate_percent,
                'response_time_p95_ms': self._current_metrics.response_time_p95_ms,
                'timestamp': self._current_metrics.timestamp.isoformat()
            }
        
        # Load prediction
        try:
            prediction = await self.predict_future_load()
            load_prediction = {
                'predicted_queue_depth': prediction.predicted_queue_depth,
                'predicted_cpu_percent': prediction.predicted_cpu_percent,
                'predicted_memory_percent': prediction.predicted_memory_percent,
                'predicted_throughput': prediction.predicted_throughput,
                'confidence': prediction.confidence,
                'horizon_minutes': prediction.horizon_minutes,
                'timestamp': prediction.timestamp.isoformat()
            }
        except Exception:
            load_prediction = None
        
        return {
            'current_instances': current_instances,
            'current_metrics': current_metrics,
            'load_prediction': load_prediction,
            'recent_scaling_actions': recent_actions,
            'scaling_rules': rule_status,
            'model_accuracy': self._model_accuracy,
            'prediction_horizon_minutes': self.prediction_horizon_minutes,
            'auto_scaling_enabled': True
        }
    
    async def shutdown(self):
        """Gracefully shutdown the auto-scaler."""
        
        # Cancel background tasks
        if self._metrics_collector_task:
            self._metrics_collector_task.cancel()
        
        if self._scaling_monitor_task:
            self._scaling_monitor_task.cancel()
        
        if self._predictor_trainer_task:
            self._predictor_trainer_task.cancel()
        
        logger.info("Predictive auto-scaler shut down")


# Global auto-scaler instance
predictive_auto_scaler: Optional[PredictiveAutoScaler] = None

def initialize_auto_scaler(docker_client: Optional[docker.DockerClient] = None) -> PredictiveAutoScaler:
    """Initialize the global auto-scaler instance."""
    global predictive_auto_scaler
    predictive_auto_scaler = PredictiveAutoScaler(
        docker_client=docker_client,
        prediction_horizon_minutes=config.get_processing_settings().get('prediction_horizon_minutes', 30),
        scaling_check_interval_seconds=config.get_processing_settings().get('scaling_check_interval_seconds', 60)
    )
    return predictive_auto_scaler
"""Advanced database query optimization and materialized view management."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import asyncpg
import psutil
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import config
from src.monitoring.metrics.comprehensive_metrics import metrics_collector


logger = logging.getLogger(__name__)


class RefreshPriority(Enum):
    """Priority levels for materialized view refresh."""
    CRITICAL = "critical"    # Real-time views (5 minutes)
    HIGH = "high"           # Hourly views
    MEDIUM = "medium"       # Daily views
    LOW = "low"             # Weekly views


@dataclass
class MaterializedView:
    """Materialized view configuration."""
    name: str
    refresh_priority: RefreshPriority
    refresh_interval_minutes: int
    depends_on: List[str]  # Dependencies (other views or tables)
    estimated_rows: int
    size_mb: float = 0.0
    last_refresh: Optional[datetime] = None
    refresh_duration_seconds: float = 0.0
    query_frequency: int = 0
    is_populated: bool = False


@dataclass
class QueryPerformanceMetrics:
    """Query performance tracking."""
    query_hash: str
    query_text: str
    execution_count: int
    total_time_ms: float
    avg_time_ms: float
    rows_examined: int
    cache_hit_ratio: float
    last_executed: datetime


class QueryOptimizer:
    """Advanced query optimization and materialized view management system."""
    
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        
        # Materialized view registry
        self.materialized_views: Dict[str, MaterializedView] = {}
        self._register_materialized_views()
        
        # Query performance tracking
        self.query_metrics: Dict[str, QueryPerformanceMetrics] = {}
        
        # Optimization settings
        self.auto_refresh_enabled = True
        self.adaptive_refresh_enabled = True
        self.query_caching_enabled = True
        
        # Background tasks
        self._refresh_scheduler_task = None
        self._performance_monitor_task = None
        self._start_background_tasks()
    
    def _register_materialized_views(self):
        """Register all materialized views with their configurations."""
        
        # Real-time views (5-minute refresh)
        self.materialized_views.update({
            'mv_realtime_metrics': MaterializedView(
                name='mv_realtime_metrics',
                refresh_priority=RefreshPriority.CRITICAL,
                refresh_interval_minutes=5,
                depends_on=['claims'],
                estimated_rows=500,  # 24 hours * 12 intervals * 2 facilities
            ),
            'mv_hourly_claims_throughput': MaterializedView(
                name='mv_hourly_claims_throughput',
                refresh_priority=RefreshPriority.CRITICAL,
                refresh_interval_minutes=5,
                depends_on=['claims'],
                estimated_rows=168,  # 7 days * 24 hours
            ),
        })
        
        # High-frequency views (hourly refresh)
        self.materialized_views.update({
            'mv_facility_dashboard': MaterializedView(
                name='mv_facility_dashboard',
                refresh_priority=RefreshPriority.HIGH,
                refresh_interval_minutes=60,
                depends_on=['claims', 'failed_claims', 'facilities'],
                estimated_rows=300,  # 30 days * 10 facilities
            ),
            'mv_failed_claims_analysis': MaterializedView(
                name='mv_failed_claims_analysis',
                refresh_priority=RefreshPriority.HIGH,
                refresh_interval_minutes=60,
                depends_on=['failed_claims'],
                estimated_rows=2700,  # 90 days * 10 facilities * 3 categories
            ),
        })
        
        # Medium-frequency views (daily refresh)
        self.materialized_views.update({
            'mv_daily_claims_summary': MaterializedView(
                name='mv_daily_claims_summary',
                refresh_priority=RefreshPriority.MEDIUM,
                refresh_interval_minutes=60,  # Hourly for now, can be daily
                depends_on=['claims'],
                estimated_rows=900,  # 90 days * 10 facilities
            ),
            'mv_provider_performance': MaterializedView(
                name='mv_provider_performance',
                refresh_priority=RefreshPriority.MEDIUM,
                refresh_interval_minutes=240,  # 4 hours
                depends_on=['claims'],
                estimated_rows=12000,  # 12 months * 100 providers * 10 facilities
            ),
            'mv_diagnosis_analytics': MaterializedView(
                name='mv_diagnosis_analytics',
                refresh_priority=RefreshPriority.MEDIUM,
                refresh_interval_minutes=240,
                depends_on=['claims', 'claim_line_items'],
                estimated_rows=6000,  # 12 months * 500 diagnosis codes
            ),
            'mv_procedure_performance': MaterializedView(
                name='mv_procedure_performance',
                refresh_priority=RefreshPriority.MEDIUM,
                refresh_interval_minutes=240,
                depends_on=['claim_line_items', 'claims'],
                estimated_rows=24000,  # 12 months * 200 procedures * 10 payers
            ),
        })
        
        # Low-frequency views (weekly refresh)
        self.materialized_views.update({
            'mv_batch_performance': MaterializedView(
                name='mv_batch_performance',
                refresh_priority=RefreshPriority.LOW,
                refresh_interval_minutes=1440,  # Daily
                depends_on=['batch_metadata'],
                estimated_rows=168,  # 7 days * 24 hours
            ),
        })
    
    async def refresh_materialized_view(self, 
                                      view_name: str, 
                                      concurrent: bool = True,
                                      force: bool = False) -> Tuple[bool, float]:
        """Refresh a specific materialized view."""
        
        if view_name not in self.materialized_views:
            logger.error(f"Unknown materialized view: {view_name}")
            return False, 0.0
        
        view_config = self.materialized_views[view_name]
        
        # Check if refresh is needed (unless forced)
        if not force and view_config.last_refresh:
            time_since_refresh = datetime.now() - view_config.last_refresh
            if time_since_refresh.total_seconds() < (view_config.refresh_interval_minutes * 60):
                logger.debug(f"Skipping refresh for {view_name} - not due yet")
                return True, 0.0
        
        start_time = datetime.now()
        
        try:
            # Check dependencies first
            if not await self._check_view_dependencies(view_config):
                logger.warning(f"Dependencies not ready for {view_name}")
                return False, 0.0
            
            # Refresh the view
            refresh_sql = f"REFRESH MATERIALIZED VIEW {'CONCURRENTLY' if concurrent else ''} {view_name}"
            
            logger.info(f"Refreshing materialized view: {view_name}")
            await self.db_session.execute(text(refresh_sql))
            await self.db_session.commit()
            
            # Update metadata
            refresh_duration = (datetime.now() - start_time).total_seconds()
            view_config.last_refresh = datetime.now()
            view_config.refresh_duration_seconds = refresh_duration
            
            # Update view statistics
            await self._update_view_statistics(view_name)
            
            # Record metrics
            metrics_collector.record_database_operation(
                operation_type="materialized_view_refresh",
                table_name=view_name,
                duration_ms=refresh_duration * 1000,
                rows_affected=view_config.estimated_rows
            )
            
            logger.info(f"Successfully refreshed {view_name} in {refresh_duration:.2f}s")
            return True, refresh_duration
            
        except Exception as e:
            logger.error(f"Failed to refresh materialized view {view_name}: {e}")
            
            # Record error metrics
            metrics_collector.record_database_error(
                operation_type="materialized_view_refresh",
                table_name=view_name,
                error_message=str(e)
            )
            
            return False, 0.0
    
    async def refresh_views_by_priority(self, priority: RefreshPriority) -> Dict[str, bool]:
        """Refresh all views of a specific priority level."""
        views_to_refresh = [
            view_name for view_name, view_config in self.materialized_views.items()
            if view_config.refresh_priority == priority
        ]
        
        logger.info(f"Refreshing {len(views_to_refresh)} views with priority {priority.value}")
        
        # Refresh views in dependency order
        results = {}
        
        for view_name in self._get_refresh_order(views_to_refresh):
            success, duration = await self.refresh_materialized_view(view_name)
            results[view_name] = success
            
            if not success:
                logger.error(f"Failed to refresh {view_name}, stopping priority batch")
                break
        
        return results
    
    async def _check_view_dependencies(self, view_config: MaterializedView) -> bool:
        """Check if all dependencies are ready for view refresh."""
        
        for dependency in view_config.depends_on:
            # Check if dependency table has recent data
            try:
                if dependency.startswith('mv_'):
                    # Dependency is another materialized view
                    dep_view = self.materialized_views.get(dependency)
                    if dep_view and dep_view.last_refresh:
                        time_since_refresh = datetime.now() - dep_view.last_refresh
                        if time_since_refresh.total_seconds() > (dep_view.refresh_interval_minutes * 60 * 2):
                            logger.warning(f"Dependency {dependency} is stale")
                            return False
                else:
                    # Dependency is a regular table - check for recent data
                    check_sql = f"""
                    SELECT COUNT(*) 
                    FROM {dependency} 
                    WHERE created_at >= NOW() - INTERVAL '1 hour'
                    """
                    result = await self.db_session.execute(text(check_sql))
                    count = result.scalar()
                    
                    if count == 0:
                        logger.warning(f"No recent data in dependency table {dependency}")
                        # Don't fail for this - just warn
            
            except Exception as e:
                logger.warning(f"Could not check dependency {dependency}: {e}")
        
        return True
    
    def _get_refresh_order(self, view_names: List[str]) -> List[str]:
        """Get views in dependency order for refresh."""
        ordered_views = []
        remaining_views = set(view_names)
        
        while remaining_views:
            # Find views with no unresolved dependencies
            ready_views = []
            
            for view_name in remaining_views:
                view_config = self.materialized_views[view_name]
                
                # Check if all dependencies are already refreshed or not in our list
                deps_ready = all(
                    dep not in remaining_views or not dep.startswith('mv_')
                    for dep in view_config.depends_on
                )
                
                if deps_ready:
                    ready_views.append(view_name)
            
            if not ready_views:
                # Circular dependency or other issue - just add remaining
                ready_views = list(remaining_views)
                logger.warning(f"Potential circular dependency in views: {remaining_views}")
            
            ordered_views.extend(ready_views)
            remaining_views -= set(ready_views)
        
        return ordered_views
    
    async def _update_view_statistics(self, view_name: str):
        """Update materialized view statistics."""
        
        try:
            # Get view size and row count
            stats_sql = """
            SELECT 
                pg_total_relation_size($1) as size_bytes,
                (SELECT reltuples::bigint FROM pg_class WHERE relname = $2) as estimated_rows,
                (SELECT ispopulated FROM pg_matviews WHERE matviewname = $3) as is_populated
            """
            
            result = await self.db_session.execute(
                text(stats_sql), 
                (view_name, view_name, view_name)
            )
            row = result.fetchone()
            
            if row:
                view_config = self.materialized_views[view_name]
                view_config.size_mb = row[0] / (1024 * 1024) if row[0] else 0.0
                view_config.estimated_rows = row[1] or 0
                view_config.is_populated = row[2] or False
                
                logger.debug(f"Updated stats for {view_name}: "
                           f"{view_config.size_mb:.1f}MB, "
                           f"{view_config.estimated_rows:,} rows")
        
        except Exception as e:
            logger.warning(f"Could not update statistics for {view_name}: {e}")
    
    async def optimize_query(self, query: str, parameters: Optional[Dict] = None) -> str:
        """Optimize a query using available materialized views and indexes."""
        
        # Analyze query to identify optimization opportunities
        optimizations = []
        original_query = query.lower()
        
        # Check if query can use materialized views
        if "select" in original_query and "from claims" in original_query:
            
            # Daily aggregations
            if ("group by date(" in original_query or 
                "date_trunc('day'" in original_query):
                optimizations.append({
                    'type': 'materialized_view',
                    'suggestion': 'mv_daily_claims_summary',
                    'reason': 'Query aggregates claims by day'
                })
            
            # Hourly aggregations
            elif ("date_trunc('hour'" in original_query or
                  "group by extract(hour" in original_query):
                optimizations.append({
                    'type': 'materialized_view',
                    'suggestion': 'mv_hourly_claims_throughput',
                    'reason': 'Query aggregates claims by hour'
                })
            
            # Provider analysis
            elif "billing_provider_npi" in original_query and "group by" in original_query:
                optimizations.append({
                    'type': 'materialized_view',
                    'suggestion': 'mv_provider_performance',
                    'reason': 'Query analyzes provider performance'
                })
            
            # Diagnosis code analysis
            elif "primary_diagnosis_code" in original_query and "group by" in original_query:
                optimizations.append({
                    'type': 'materialized_view',
                    'suggestion': 'mv_diagnosis_analytics',
                    'reason': 'Query analyzes diagnosis codes'
                })
        
        # Check for failed claims analysis
        if "from failed_claims" in original_query and "group by" in original_query:
            optimizations.append({
                'type': 'materialized_view',
                'suggestion': 'mv_failed_claims_analysis',
                'reason': 'Query analyzes failed claims'
            })
        
        # Index suggestions
        if "where created_at" in original_query and "facility_id" in original_query:
            optimizations.append({
                'type': 'index',
                'suggestion': 'idx_claims_created_at_facility',
                'reason': 'Query filters by date and facility'
            })
        
        # Log optimization suggestions
        if optimizations:
            logger.info(f"Query optimization suggestions: {optimizations}")
            
            # Record metrics
            metrics_collector.record_database_optimization(
                query_hash=str(hash(query)),
                optimizations_found=len(optimizations),
                materialized_views_suggested=sum(1 for opt in optimizations if opt['type'] == 'materialized_view')
            )
        
        return query  # Return original query for now
    
    async def analyze_query_performance(self, 
                                      query: str, 
                                      execution_time_ms: float,
                                      rows_examined: int = 0) -> Dict[str, Any]:
        """Analyze query performance and provide optimization recommendations."""
        
        query_hash = str(hash(query))
        
        # Update performance metrics
        if query_hash in self.query_metrics:
            metrics = self.query_metrics[query_hash]
            metrics.execution_count += 1
            metrics.total_time_ms += execution_time_ms
            metrics.avg_time_ms = metrics.total_time_ms / metrics.execution_count
            metrics.rows_examined += rows_examined
            metrics.last_executed = datetime.now()
        else:
            self.query_metrics[query_hash] = QueryPerformanceMetrics(
                query_hash=query_hash,
                query_text=query[:500],  # Truncate for storage
                execution_count=1,
                total_time_ms=execution_time_ms,
                avg_time_ms=execution_time_ms,
                rows_examined=rows_examined,
                cache_hit_ratio=0.0,
                last_executed=datetime.now()
            )
        
        metrics = self.query_metrics[query_hash]
        
        # Analyze performance
        performance_analysis = {
            'query_hash': query_hash,
            'avg_execution_time_ms': metrics.avg_time_ms,
            'total_executions': metrics.execution_count,
            'performance_rating': self._rate_query_performance(metrics),
            'optimization_opportunities': []
        }
        
        # Identify optimization opportunities
        if metrics.avg_time_ms > 1000:  # Slow query (>1 second)
            performance_analysis['optimization_opportunities'].append({
                'type': 'slow_query',
                'severity': 'high' if metrics.avg_time_ms > 5000 else 'medium',
                'suggestion': 'Consider query optimization or indexing'
            })
        
        if metrics.execution_count > 100 and metrics.avg_time_ms > 100:
            performance_analysis['optimization_opportunities'].append({
                'type': 'frequent_query',
                'severity': 'medium',
                'suggestion': 'Consider creating materialized view for this query pattern'
            })
        
        return performance_analysis
    
    def _rate_query_performance(self, metrics: QueryPerformanceMetrics) -> str:
        """Rate query performance based on execution time and frequency."""
        
        if metrics.avg_time_ms < 50:
            return "excellent"
        elif metrics.avg_time_ms < 200:
            return "good"
        elif metrics.avg_time_ms < 1000:
            return "fair"
        elif metrics.avg_time_ms < 5000:
            return "poor"
        else:
            return "critical"
    
    def _start_background_tasks(self):
        """Start background optimization tasks."""
        
        async def refresh_scheduler():
            """Background task to refresh materialized views on schedule."""
            while True:
                try:
                    current_time = datetime.now()
                    
                    # Check each view's refresh schedule
                    for view_name, view_config in self.materialized_views.items():
                        if not view_config.last_refresh:
                            # Never refreshed - do it now
                            await self.refresh_materialized_view(view_name)
                            continue
                        
                        time_since_refresh = current_time - view_config.last_refresh
                        refresh_due = time_since_refresh.total_seconds() >= (view_config.refresh_interval_minutes * 60)
                        
                        if refresh_due:
                            success, duration = await self.refresh_materialized_view(view_name)
                            if success:
                                logger.info(f"Scheduled refresh of {view_name} completed in {duration:.2f}s")
                    
                    # Sleep for 1 minute before next check
                    await asyncio.sleep(60)
                    
                except Exception as e:
                    logger.error(f"Refresh scheduler error: {e}")
                    await asyncio.sleep(60)
        
        async def performance_monitor():
            """Monitor query performance and suggest optimizations."""
            while True:
                try:
                    await asyncio.sleep(300)  # Check every 5 minutes
                    
                    # Analyze slow queries
                    slow_queries = [
                        metrics for metrics in self.query_metrics.values()
                        if metrics.avg_time_ms > 1000 and metrics.execution_count > 5
                    ]
                    
                    if slow_queries:
                        logger.warning(f"Found {len(slow_queries)} slow query patterns")
                        
                        for metrics in slow_queries[:5]:  # Top 5 slow queries
                            logger.warning(f"Slow query: {metrics.avg_time_ms:.1f}ms avg, "
                                         f"{metrics.execution_count} executions - "
                                         f"{metrics.query_text[:100]}...")
                    
                    # Check materialized view health
                    stale_views = [
                        view_name for view_name, view_config in self.materialized_views.items()
                        if view_config.last_refresh and 
                           (datetime.now() - view_config.last_refresh).total_seconds() > 
                           (view_config.refresh_interval_minutes * 60 * 2)
                    ]
                    
                    if stale_views:
                        logger.warning(f"Stale materialized views: {stale_views}")
                
                except Exception as e:
                    logger.error(f"Performance monitor error: {e}")
        
        if self.auto_refresh_enabled:
            self._refresh_scheduler_task = asyncio.create_task(refresh_scheduler())
        
        self._performance_monitor_task = asyncio.create_task(performance_monitor())
    
    async def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status and statistics."""
        
        # Materialized view status
        view_status = {}
        total_size_mb = 0
        
        for view_name, view_config in self.materialized_views.items():
            view_status[view_name] = {
                'last_refresh': view_config.last_refresh.isoformat() if view_config.last_refresh else None,
                'refresh_interval_minutes': view_config.refresh_interval_minutes,
                'size_mb': view_config.size_mb,
                'estimated_rows': view_config.estimated_rows,
                'is_populated': view_config.is_populated,
                'refresh_duration_seconds': view_config.refresh_duration_seconds,
                'priority': view_config.refresh_priority.value
            }
            total_size_mb += view_config.size_mb
        
        # Query performance summary
        total_queries = len(self.query_metrics)
        slow_queries = sum(1 for m in self.query_metrics.values() if m.avg_time_ms > 1000)
        
        if self.query_metrics:
            avg_execution_time = sum(m.avg_time_ms for m in self.query_metrics.values()) / total_queries
        else:
            avg_execution_time = 0
        
        return {
            'materialized_views': {
                'total_views': len(self.materialized_views),
                'total_size_mb': total_size_mb,
                'view_status': view_status
            },
            'query_performance': {
                'total_tracked_queries': total_queries,
                'slow_queries': slow_queries,
                'avg_execution_time_ms': avg_execution_time
            },
            'optimization_settings': {
                'auto_refresh_enabled': self.auto_refresh_enabled,
                'adaptive_refresh_enabled': self.adaptive_refresh_enabled,
                'query_caching_enabled': self.query_caching_enabled
            }
        }
    
    async def shutdown(self):
        """Shutdown the query optimizer."""
        if self._refresh_scheduler_task:
            self._refresh_scheduler_task.cancel()
        
        if self._performance_monitor_task:
            self._performance_monitor_task.cancel()
        
        logger.info("Query optimizer shut down")


# Global query optimizer instance (will be initialized with db session)
query_optimizer: Optional[QueryOptimizer] = None

def initialize_query_optimizer(db_session: AsyncSession):
    """Initialize the global query optimizer instance."""
    global query_optimizer
    query_optimizer = QueryOptimizer(db_session)
    return query_optimizer
"""Advanced analytics engine for claims data analysis."""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import structlog
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database.base import get_postgres_session, get_sqlserver_session
from src.core.database.models import (
    Claim,
    ClaimLineItem,
    FailedClaim,
    PerformanceMetrics,
)

logger = structlog.get_logger(__name__)


class ClaimsAnalytics:
    """Advanced analytics engine for claims processing insights."""

    def __init__(self):
        """Initialize analytics engine."""
        self.cache_ttl = 3600  # 1 hour cache

    async def get_processing_dashboard_data(
        self, 
        facility_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get comprehensive dashboard data for claims processing."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        try:
            async with get_postgres_session() as session:
                # Get basic processing statistics
                processing_stats = await self._get_processing_statistics(
                    session, facility_id, start_date, end_date
                )
                
                # Get failure analysis
                failure_analysis = await self._get_failure_analysis(
                    session, facility_id, start_date, end_date
                )
                
                # Get throughput trends
                throughput_trends = await self._get_throughput_trends(
                    session, facility_id, start_date, end_date
                )
                
                # Get revenue analysis
                revenue_analysis = await self._get_revenue_analysis(
                    session, facility_id, start_date, end_date
                )
                
                # Get RVU analysis
                rvu_analysis = await self._get_rvu_analysis(
                    session, facility_id, start_date, end_date
                )
                
                return {
                    "period": {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "days": days
                    },
                    "facility_id": facility_id,
                    "processing_stats": processing_stats,
                    "failure_analysis": failure_analysis,
                    "throughput_trends": throughput_trends,
                    "revenue_analysis": revenue_analysis,
                    "rvu_analysis": rvu_analysis,
                    "generated_at": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.exception("Failed to generate dashboard data", error=str(e))
            raise

    async def _get_processing_statistics(
        self,
        session: AsyncSession,
        facility_id: Optional[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get basic processing statistics."""
        base_query = select(
            func.count(Claim.id).label("total_claims"),
            func.sum(Claim.total_charges).label("total_charges"),
            func.sum(Claim.expected_reimbursement).label("total_expected_reimbursement"),
            func.count(
                func.case([(Claim.processing_status == "completed", 1)])
            ).label("completed_claims"),
            func.count(
                func.case([(Claim.processing_status == "failed", 1)])
            ).label("failed_claims"),
            func.avg(Claim.total_charges).label("avg_claim_amount")
        ).where(
            Claim.created_at.between(start_date, end_date)
        )
        
        if facility_id:
            base_query = base_query.where(Claim.facility_id == facility_id)
        
        result = await session.execute(base_query)
        row = result.first()
        
        if not row:
            return self._empty_processing_stats()
        
        total_claims = row.total_claims or 0
        completed_claims = row.completed_claims or 0
        failed_claims = row.failed_claims or 0
        
        return {
            "total_claims": total_claims,
            "completed_claims": completed_claims,
            "failed_claims": failed_claims,
            "pending_claims": total_claims - completed_claims - failed_claims,
            "success_rate": (completed_claims / total_claims * 100) if total_claims > 0 else 0,
            "failure_rate": (failed_claims / total_claims * 100) if total_claims > 0 else 0,
            "total_charges": float(row.total_charges or 0),
            "total_expected_reimbursement": float(row.total_expected_reimbursement or 0),
            "avg_claim_amount": float(row.avg_claim_amount or 0)
        }

    async def _get_failure_analysis(
        self,
        session: AsyncSession,
        facility_id: Optional[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get detailed failure analysis."""
        # Get failure counts by category
        failure_query = select(
            FailedClaim.failure_category,
            func.count(FailedClaim.id).label("count"),
            func.sum(FailedClaim.charge_amount).label("total_amount")
        ).where(
            FailedClaim.failed_at.between(start_date, end_date)
        ).group_by(FailedClaim.failure_category)
        
        if facility_id:
            failure_query = failure_query.where(FailedClaim.facility_id == facility_id)
        
        failure_result = await session.execute(failure_query)
        
        failure_categories = []
        total_failed_amount = 0
        
        for row in failure_result:
            amount = float(row.total_amount or 0)
            total_failed_amount += amount
            
            failure_categories.append({
                "category": row.failure_category,
                "count": row.count,
                "amount": amount,
                "percentage": 0  # Will be calculated below
            })
        
        # Calculate percentages
        total_failed_count = sum(cat["count"] for cat in failure_categories)
        for category in failure_categories:
            category["percentage"] = (
                category["count"] / total_failed_count * 100 
                if total_failed_count > 0 else 0
            )
        
        # Sort by count descending
        failure_categories.sort(key=lambda x: x["count"], reverse=True)
        
        return {
            "failure_categories": failure_categories,
            "total_failed_claims": total_failed_count,
            "total_failed_amount": total_failed_amount,
            "top_failure_category": failure_categories[0]["category"] if failure_categories else None
        }

    async def _get_throughput_trends(
        self,
        session: AsyncSession,
        facility_id: Optional[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get throughput trends over time."""
        # Get daily processing counts
        daily_query = select(
            func.date(Claim.created_at).label("date"),
            func.count(Claim.id).label("claims_count"),
            func.sum(Claim.total_charges).label("total_charges")
        ).where(
            Claim.created_at.between(start_date, end_date)
        ).group_by(
            func.date(Claim.created_at)
        ).order_by("date")
        
        if facility_id:
            daily_query = daily_query.where(Claim.facility_id == facility_id)
        
        daily_result = await session.execute(daily_query)
        
        daily_trends = []
        for row in daily_result:
            daily_trends.append({
                "date": row.date.isoformat(),
                "claims_count": row.claims_count,
                "total_charges": float(row.total_charges or 0)
            })
        
        # Calculate moving averages
        if len(daily_trends) >= 7:
            for i in range(6, len(daily_trends)):
                week_data = daily_trends[i-6:i+1]
                avg_claims = sum(d["claims_count"] for d in week_data) / 7
                daily_trends[i]["moving_avg_claims"] = avg_claims
        
        return {
            "daily_trends": daily_trends,
            "peak_day": max(daily_trends, key=lambda x: x["claims_count"]) if daily_trends else None,
            "avg_daily_claims": sum(d["claims_count"] for d in daily_trends) / len(daily_trends) if daily_trends else 0
        }

    async def _get_revenue_analysis(
        self,
        session: AsyncSession,
        facility_id: Optional[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get revenue analysis by payer and other dimensions."""
        # Revenue by insurance type
        payer_query = select(
            Claim.insurance_type,
            func.count(Claim.id).label("claim_count"),
            func.sum(Claim.total_charges).label("total_charges"),
            func.sum(Claim.expected_reimbursement).label("expected_reimbursement"),
            func.avg(Claim.total_charges).label("avg_charges")
        ).where(
            Claim.created_at.between(start_date, end_date),
            Claim.processing_status == "completed"
        ).group_by(Claim.insurance_type)
        
        if facility_id:
            payer_query = payer_query.where(Claim.facility_id == facility_id)
        
        payer_result = await session.execute(payer_query)
        
        payer_analysis = []
        total_revenue = 0
        
        for row in payer_result:
            expected_reimbursement = float(row.expected_reimbursement or 0)
            total_revenue += expected_reimbursement
            
            payer_analysis.append({
                "insurance_type": row.insurance_type,
                "claim_count": row.claim_count,
                "total_charges": float(row.total_charges or 0),
                "expected_reimbursement": expected_reimbursement,
                "avg_charges": float(row.avg_charges or 0),
                "reimbursement_rate": (
                    expected_reimbursement / float(row.total_charges) * 100
                    if row.total_charges and row.total_charges > 0 else 0
                )
            })
        
        # Calculate percentages of total revenue
        for payer in payer_analysis:
            payer["revenue_percentage"] = (
                payer["expected_reimbursement"] / total_revenue * 100
                if total_revenue > 0 else 0
            )
        
        # Sort by revenue descending
        payer_analysis.sort(key=lambda x: x["expected_reimbursement"], reverse=True)
        
        return {
            "payer_analysis": payer_analysis,
            "total_revenue": total_revenue,
            "top_payer": payer_analysis[0]["insurance_type"] if payer_analysis else None
        }

    async def _get_rvu_analysis(
        self,
        session: AsyncSession,
        facility_id: Optional[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get RVU analysis by procedure codes and categories."""
        # Top procedures by RVU
        procedure_query = select(
            ClaimLineItem.procedure_code,
            func.count(ClaimLineItem.id).label("count"),
            func.sum(ClaimLineItem.rvu_total).label("total_rvu"),
            func.sum(ClaimLineItem.expected_reimbursement).label("total_reimbursement"),
            func.avg(ClaimLineItem.rvu_total).label("avg_rvu")
        ).join(
            Claim, ClaimLineItem.claim_id == Claim.id
        ).where(
            Claim.created_at.between(start_date, end_date),
            Claim.processing_status == "completed",
            ClaimLineItem.rvu_total.isnot(None)
        ).group_by(
            ClaimLineItem.procedure_code
        ).order_by(
            func.sum(ClaimLineItem.rvu_total).desc()
        ).limit(20)
        
        if facility_id:
            procedure_query = procedure_query.where(Claim.facility_id == facility_id)
        
        procedure_result = await session.execute(procedure_query)
        
        top_procedures = []
        for row in procedure_result:
            top_procedures.append({
                "procedure_code": row.procedure_code,
                "count": row.count,
                "total_rvu": float(row.total_rvu or 0),
                "total_reimbursement": float(row.total_reimbursement or 0),
                "avg_rvu": float(row.avg_rvu or 0)
            })
        
        # RVU summary statistics
        rvu_summary_query = select(
            func.sum(ClaimLineItem.rvu_total).label("total_rvu"),
            func.avg(ClaimLineItem.rvu_total).label("avg_rvu"),
            func.count(ClaimLineItem.id).label("total_line_items")
        ).join(
            Claim, ClaimLineItem.claim_id == Claim.id
        ).where(
            Claim.created_at.between(start_date, end_date),
            Claim.processing_status == "completed",
            ClaimLineItem.rvu_total.isnot(None)
        )
        
        if facility_id:
            rvu_summary_query = rvu_summary_query.where(Claim.facility_id == facility_id)
        
        rvu_result = await session.execute(rvu_summary_query)
        rvu_row = rvu_result.first()
        
        return {
            "top_procedures": top_procedures,
            "rvu_summary": {
                "total_rvu": float(rvu_row.total_rvu or 0) if rvu_row else 0,
                "avg_rvu": float(rvu_row.avg_rvu or 0) if rvu_row else 0,
                "total_line_items": rvu_row.total_line_items if rvu_row else 0
            }
        }

    def _empty_processing_stats(self) -> Dict[str, Any]:
        """Return empty processing statistics."""
        return {
            "total_claims": 0,
            "completed_claims": 0,
            "failed_claims": 0,
            "pending_claims": 0,
            "success_rate": 0,
            "failure_rate": 0,
            "total_charges": 0,
            "total_expected_reimbursement": 0,
            "avg_claim_amount": 0
        }

    async def get_diagnosis_analysis(
        self,
        facility_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get diagnosis code analysis."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        try:
            async with get_postgres_session() as session:
                # Top diagnosis codes
                dx_query = select(
                    Claim.primary_diagnosis_code,
                    func.count(Claim.id).label("count"),
                    func.sum(Claim.total_charges).label("total_charges"),
                    func.avg(Claim.total_charges).label("avg_charges")
                ).where(
                    Claim.created_at.between(start_date, end_date),
                    Claim.processing_status == "completed",
                    Claim.primary_diagnosis_code.isnot(None)
                ).group_by(
                    Claim.primary_diagnosis_code
                ).order_by(
                    func.count(Claim.id).desc()
                ).limit(20)
                
                if facility_id:
                    dx_query = dx_query.where(Claim.facility_id == facility_id)
                
                dx_result = await session.execute(dx_query)
                
                top_diagnoses = []
                for row in dx_result:
                    top_diagnoses.append({
                        "diagnosis_code": row.primary_diagnosis_code,
                        "count": row.count,
                        "total_charges": float(row.total_charges or 0),
                        "avg_charges": float(row.avg_charges or 0)
                    })
                
                return {
                    "period": {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "days": days
                    },
                    "facility_id": facility_id,
                    "top_diagnoses": top_diagnoses,
                    "generated_at": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.exception("Failed to generate diagnosis analysis", error=str(e))
            raise

    async def get_performance_metrics(
        self,
        facility_id: Optional[str] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get system performance metrics."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=hours)
        
        try:
            async with get_postgres_session() as session:
                # Get performance metrics
                metrics_query = select(
                    PerformanceMetrics.metric_type,
                    PerformanceMetrics.metric_name,
                    func.avg(PerformanceMetrics.metric_value).label("avg_value"),
                    func.max(PerformanceMetrics.metric_value).label("max_value"),
                    func.min(PerformanceMetrics.metric_value).label("min_value"),
                    func.count(PerformanceMetrics.id).label("count")
                ).where(
                    PerformanceMetrics.recorded_at.between(start_date, end_date)
                ).group_by(
                    PerformanceMetrics.metric_type,
                    PerformanceMetrics.metric_name
                )
                
                if facility_id:
                    metrics_query = metrics_query.where(
                        PerformanceMetrics.facility_id == facility_id
                    )
                
                metrics_result = await session.execute(metrics_query)
                
                performance_data = {}
                for row in metrics_result:
                    metric_type = row.metric_type
                    if metric_type not in performance_data:
                        performance_data[metric_type] = {}
                    
                    performance_data[metric_type][row.metric_name] = {
                        "avg_value": float(row.avg_value),
                        "max_value": float(row.max_value),
                        "min_value": float(row.min_value),
                        "count": row.count
                    }
                
                return {
                    "period": {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "hours": hours
                    },
                    "facility_id": facility_id,
                    "performance_metrics": performance_data,
                    "generated_at": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.exception("Failed to generate performance metrics", error=str(e))
            raise


# Global analytics instance
claims_analytics = ClaimsAnalytics()
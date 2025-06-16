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
    # SQL Server Analytics Models
    ClaimAnalytics,
    ClaimLineItemAnalytics,
    FailedClaimAnalytics,
    PerformanceMetricsAnalytics,
    DailyProcessingSummary,
    Facility,
    FacilityFinancialClass,
    RVUData,
    CoreStandardPayer,
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
            # Use PostgreSQL for staging data and SQL Server for analytics
            async with get_postgres_session() as pg_session:
                # Get basic processing statistics from staging
                processing_stats = await self._get_processing_statistics(
                    pg_session, facility_id, start_date, end_date
                )
                
                # Get failure analysis from staging
                failure_analysis = await self._get_failure_analysis(
                    pg_session, facility_id, start_date, end_date
                )
                
            # Get analytics data from SQL Server
            async with get_sqlserver_session() as sql_session:
                # Get throughput trends from analytics
                throughput_trends = await self._get_throughput_trends_analytics(
                    sql_session, facility_id, start_date, end_date
                )
                
                # Get revenue analysis from analytics
                revenue_analysis = await self._get_revenue_analysis_analytics(
                    sql_session, facility_id, start_date, end_date
                )
                
                # Get RVU analysis from analytics
                rvu_analysis = await self._get_rvu_analysis_analytics(
                    sql_session, facility_id, start_date, end_date
                )
                
                # Get facility hierarchy data
                facility_data = await self._get_facility_hierarchy(
                    sql_session, facility_id
                )
                
                return {
                    "period": {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "days": days
                    },
                    "facility_id": facility_id,
                    "facility_data": facility_data,
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
        """Get RVU analysis by procedure codes and categories (PostgreSQL staging)."""
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

    async def _get_rvu_analysis_analytics(
        self,
        session: AsyncSession,
        facility_id: Optional[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get RVU analysis from SQL Server analytics database with enhanced RVU data."""
        # Top procedures by RVU with enhanced RVU data
        procedure_query = select(
            ClaimLineItemAnalytics.procedure_code,
            RVUData.description,
            RVUData.category,
            RVUData.subcategory,
            func.count(ClaimLineItemAnalytics.line_number).label("count"),
            func.sum(ClaimLineItemAnalytics.rvu_value).label("total_rvu"),
            func.sum(ClaimLineItemAnalytics.reimbursement_amount).label("total_reimbursement"),
            func.avg(ClaimLineItemAnalytics.rvu_value).label("avg_rvu"),
            func.sum(ClaimLineItemAnalytics.charge_amount).label("total_charges"),
            func.sum(ClaimLineItemAnalytics.units).label("total_units")
        ).join(
            RVUData, ClaimLineItemAnalytics.procedure_code == RVUData.procedure_code
        ).where(
            ClaimLineItemAnalytics.created_at.between(start_date, end_date),
            ClaimLineItemAnalytics.rvu_value.isnot(None)
        ).group_by(
            ClaimLineItemAnalytics.procedure_code,
            RVUData.description,
            RVUData.category,
            RVUData.subcategory
        ).order_by(
            func.sum(ClaimLineItemAnalytics.rvu_value).desc()
        ).limit(20)
        
        if facility_id:
            procedure_query = procedure_query.where(ClaimLineItemAnalytics.facility_id == facility_id)
        
        procedure_result = await session.execute(procedure_query)
        
        top_procedures = []
        for row in procedure_result:
            top_procedures.append({
                "procedure_code": row.procedure_code,
                "description": row.description,
                "category": row.category,
                "subcategory": row.subcategory,
                "count": row.count,
                "total_rvu": float(row.total_rvu or 0),
                "total_reimbursement": float(row.total_reimbursement or 0),
                "avg_rvu": float(row.avg_rvu or 0),
                "total_charges": float(row.total_charges or 0),
                "total_units": row.total_units,
                "reimbursement_rate": (
                    float(row.total_reimbursement) / float(row.total_charges) * 100
                    if row.total_charges and row.total_charges > 0 else 0
                )
            })
        
        # RVU summary by category
        category_query = select(
            RVUData.category,
            func.count(ClaimLineItemAnalytics.line_number).label("count"),
            func.sum(ClaimLineItemAnalytics.rvu_value).label("total_rvu"),
            func.sum(ClaimLineItemAnalytics.reimbursement_amount).label("total_reimbursement")
        ).join(
            RVUData, ClaimLineItemAnalytics.procedure_code == RVUData.procedure_code
        ).where(
            ClaimLineItemAnalytics.created_at.between(start_date, end_date),
            ClaimLineItemAnalytics.rvu_value.isnot(None),
            RVUData.category.isnot(None)
        ).group_by(
            RVUData.category
        ).order_by(
            func.sum(ClaimLineItemAnalytics.rvu_value).desc()
        )
        
        if facility_id:
            category_query = category_query.where(ClaimLineItemAnalytics.facility_id == facility_id)
        
        category_result = await session.execute(category_query)
        
        rvu_by_category = []
        total_category_rvu = 0
        
        for row in category_result:
            rvu = float(row.total_rvu or 0)
            total_category_rvu += rvu
            rvu_by_category.append({
                "category": row.category,
                "count": row.count,
                "total_rvu": rvu,
                "total_reimbursement": float(row.total_reimbursement or 0)
            })
        
        # Calculate percentages
        for category in rvu_by_category:
            category["rvu_percentage"] = (
                category["total_rvu"] / total_category_rvu * 100
                if total_category_rvu > 0 else 0
            )
        
        # Overall RVU summary
        rvu_summary_query = select(
            func.sum(ClaimLineItemAnalytics.rvu_value).label("total_rvu"),
            func.avg(ClaimLineItemAnalytics.rvu_value).label("avg_rvu"),
            func.count(ClaimLineItemAnalytics.line_number).label("total_line_items"),
            func.sum(ClaimLineItemAnalytics.reimbursement_amount).label("total_reimbursement"),
            func.count(func.distinct(ClaimLineItemAnalytics.procedure_code)).label("unique_procedures")
        ).where(
            ClaimLineItemAnalytics.created_at.between(start_date, end_date),
            ClaimLineItemAnalytics.rvu_value.isnot(None)
        )
        
        if facility_id:
            rvu_summary_query = rvu_summary_query.where(ClaimLineItemAnalytics.facility_id == facility_id)
        
        rvu_result = await session.execute(rvu_summary_query)
        rvu_row = rvu_result.first()
        
        return {
            "top_procedures": top_procedures,
            "rvu_by_category": rvu_by_category,
            "rvu_summary": {
                "total_rvu": float(rvu_row.total_rvu or 0) if rvu_row else 0,
                "avg_rvu": float(rvu_row.avg_rvu or 0) if rvu_row else 0,
                "total_line_items": rvu_row.total_line_items if rvu_row else 0,
                "total_reimbursement": float(rvu_row.total_reimbursement or 0) if rvu_row else 0,
                "unique_procedures": rvu_row.unique_procedures if rvu_row else 0
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


    async def _get_throughput_trends_analytics(
        self,
        session: AsyncSession,
        facility_id: Optional[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get throughput trends from SQL Server analytics database."""
        # Use daily processing summary if available
        daily_query = select(
            DailyProcessingSummary.summary_date,
            DailyProcessingSummary.total_claims_processed,
            DailyProcessingSummary.total_claims_failed,
            DailyProcessingSummary.total_charge_amount,
            DailyProcessingSummary.throughput_claims_per_hour,
            DailyProcessingSummary.error_rate_percentage
        ).where(
            DailyProcessingSummary.summary_date.between(start_date, end_date)
        ).order_by(DailyProcessingSummary.summary_date)
        
        if facility_id:
            daily_query = daily_query.where(DailyProcessingSummary.facility_id == facility_id)
        
        daily_result = await session.execute(daily_query)
        
        daily_trends = []
        for row in daily_result:
            daily_trends.append({
                "date": row.summary_date.isoformat(),
                "claims_count": row.total_claims_processed or 0,
                "failed_claims": row.total_claims_failed or 0,
                "total_charges": float(row.total_charge_amount or 0),
                "throughput_per_hour": float(row.throughput_claims_per_hour or 0),
                "error_rate": float(row.error_rate_percentage or 0)
            })
        
        # Calculate moving averages and trends
        if len(daily_trends) >= 7:
            for i in range(6, len(daily_trends)):
                week_data = daily_trends[i-6:i+1]
                avg_claims = sum(d["claims_count"] for d in week_data) / 7
                avg_throughput = sum(d["throughput_per_hour"] for d in week_data) / 7
                daily_trends[i]["moving_avg_claims"] = avg_claims
                daily_trends[i]["moving_avg_throughput"] = avg_throughput
        
        return {
            "daily_trends": daily_trends,
            "peak_day": max(daily_trends, key=lambda x: x["claims_count"]) if daily_trends else None,
            "avg_daily_claims": sum(d["claims_count"] for d in daily_trends) / len(daily_trends) if daily_trends else 0,
            "avg_throughput": sum(d["throughput_per_hour"] for d in daily_trends) / len(daily_trends) if daily_trends else 0
        }

    async def _get_revenue_analysis_analytics(
        self,
        session: AsyncSession,
        facility_id: Optional[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get revenue analysis from SQL Server analytics database with enhanced payer data."""
        # Revenue by financial class and payer
        payer_query = select(
            FacilityFinancialClass.financial_class_name,
            CoreStandardPayer.payer_name,
            CoreStandardPayer.payer_type,
            FacilityFinancialClass.reimbursement_rate,
            func.count(ClaimAnalytics.patient_account_number).label("claim_count"),
            func.sum(ClaimLineItemAnalytics.charge_amount).label("total_charges"),
            func.sum(ClaimLineItemAnalytics.reimbursement_amount).label("total_reimbursement"),
            func.avg(ClaimLineItemAnalytics.charge_amount).label("avg_charges")
        ).join(
            ClaimAnalytics, 
            ClaimAnalytics.financial_class_id == FacilityFinancialClass.financial_class_id
        ).join(
            ClaimLineItemAnalytics,
            (ClaimLineItemAnalytics.facility_id == ClaimAnalytics.facility_id) &
            (ClaimLineItemAnalytics.patient_account_number == ClaimAnalytics.patient_account_number)
        ).join(
            CoreStandardPayer, 
            FacilityFinancialClass.payer_id == CoreStandardPayer.payer_id
        ).where(
            ClaimAnalytics.created_at.between(start_date, end_date)
        ).group_by(
            FacilityFinancialClass.financial_class_name,
            CoreStandardPayer.payer_name,
            CoreStandardPayer.payer_type,
            FacilityFinancialClass.reimbursement_rate
        )
        
        if facility_id:
            payer_query = payer_query.where(
                ClaimAnalytics.facility_id == facility_id,
                FacilityFinancialClass.facility_id == facility_id
            )
        
        payer_result = await session.execute(payer_query)
        
        payer_analysis = []
        total_revenue = 0
        
        for row in payer_result:
            total_reimbursement = float(row.total_reimbursement or 0)
            total_charges = float(row.total_charges or 0)
            total_revenue += total_reimbursement
            
            payer_analysis.append({
                "financial_class": row.financial_class_name,
                "payer_name": row.payer_name,
                "payer_type": row.payer_type,
                "contracted_rate": float(row.reimbursement_rate or 0),
                "claim_count": row.claim_count,
                "total_charges": total_charges,
                "total_reimbursement": total_reimbursement,
                "avg_charges": float(row.avg_charges or 0),
                "actual_reimbursement_rate": (
                    total_reimbursement / total_charges * 100
                    if total_charges > 0 else 0
                )
            })
        
        # Calculate percentages of total revenue
        for payer in payer_analysis:
            payer["revenue_percentage"] = (
                payer["total_reimbursement"] / total_revenue * 100
                if total_revenue > 0 else 0
            )
        
        # Sort by revenue descending
        payer_analysis.sort(key=lambda x: x["total_reimbursement"], reverse=True)
        
        return {
            "payer_analysis": payer_analysis,
            "total_revenue": total_revenue,
            "top_payer": payer_analysis[0]["payer_name"] if payer_analysis else None,
            "payer_count": len(payer_analysis)
        }

    async def _get_facility_hierarchy(
        self,
        session: AsyncSession,
        facility_id: Optional[str]
    ) -> Dict[str, Any]:
        """Get facility hierarchy information from SQL Server analytics."""
        if not facility_id:
            return {}
        
        # Get facility with organization and region data
        facility_query = select(
            Facility.facility_id,
            Facility.facility_name,
            Facility.city,
            Facility.state,
            Facility.beds,
            Facility.fiscal_month,
            Facility.region_id,
            Facility.org_id
        ).where(
            Facility.facility_id == facility_id,
            Facility.active == True
        )
        
        facility_result = await session.execute(facility_query)
        facility_row = facility_result.first()
        
        if not facility_row:
            return {"error": "Facility not found"}
        
        facility_data = {
            "facility_id": facility_row.facility_id,
            "facility_name": facility_row.facility_name,
            "city": facility_row.city,
            "state": facility_row.state,
            "beds": facility_row.beds,
            "fiscal_month": facility_row.fiscal_month,
            "organization": None,
            "region": None
        }
        
        # Get organization data
        if facility_row.org_id:
            org_query = select(
                text("org_name")
            ).select_from(
                text("facility_organization")
            ).where(
                text("org_id = :org_id")
            )
            
            org_result = await session.execute(org_query, {"org_id": facility_row.org_id})
            org_row = org_result.first()
            if org_row:
                facility_data["organization"] = org_row[0]
        
        # Get region data
        if facility_row.region_id:
            region_query = select(
                text("region_name")
            ).select_from(
                text("facility_region")
            ).where(
                text("region_id = :region_id")
            )
            
            region_result = await session.execute(region_query, {"region_id": facility_row.region_id})
            region_row = region_result.first()
            if region_row:
                facility_data["region"] = region_row[0]
        
        return facility_data

    async def get_facility_performance_comparison(
        self,
        org_id: Optional[int] = None,
        region_id: Optional[int] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Compare performance across facilities in an organization or region."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        try:
            async with get_sqlserver_session() as session:
                # Get facilities in organization/region
                facility_query = select(
                    Facility.facility_id,
                    Facility.facility_name,
                    Facility.city,
                    Facility.state,
                    Facility.beds
                ).where(
                    Facility.active == True
                )
                
                if org_id:
                    facility_query = facility_query.where(Facility.org_id == org_id)
                if region_id:
                    facility_query = facility_query.where(Facility.region_id == region_id)
                
                facility_result = await session.execute(facility_query)
                
                facility_performance = []
                
                for facility_row in facility_result:
                    # Get performance data for each facility
                    perf_query = select(
                        func.sum(DailyProcessingSummary.total_claims_processed).label("total_claims"),
                        func.sum(DailyProcessingSummary.total_claims_failed).label("total_failed"),
                        func.sum(DailyProcessingSummary.total_charge_amount).label("total_charges"),
                        func.sum(DailyProcessingSummary.total_reimbursement_amount).label("total_reimbursement"),
                        func.avg(DailyProcessingSummary.throughput_claims_per_hour).label("avg_throughput"),
                        func.avg(DailyProcessingSummary.error_rate_percentage).label("avg_error_rate")
                    ).where(
                        DailyProcessingSummary.facility_id == facility_row.facility_id,
                        DailyProcessingSummary.summary_date.between(start_date, end_date)
                    )
                    
                    perf_result = await session.execute(perf_query)
                    perf_row = perf_result.first()
                    
                    if perf_row and perf_row.total_claims:
                        total_claims = perf_row.total_claims or 0
                        total_failed = perf_row.total_failed or 0
                        
                        facility_performance.append({
                            "facility_id": facility_row.facility_id,
                            "facility_name": facility_row.facility_name,
                            "city": facility_row.city,
                            "state": facility_row.state,
                            "beds": facility_row.beds,
                            "total_claims": total_claims,
                            "total_failed": total_failed,
                            "success_rate": (total_claims - total_failed) / total_claims * 100 if total_claims > 0 else 0,
                            "total_charges": float(perf_row.total_charges or 0),
                            "total_reimbursement": float(perf_row.total_reimbursement or 0),
                            "avg_throughput": float(perf_row.avg_throughput or 0),
                            "avg_error_rate": float(perf_row.avg_error_rate or 0),
                            "claims_per_bed": total_claims / facility_row.beds if facility_row.beds else 0
                        })
                
                # Sort by total claims descending
                facility_performance.sort(key=lambda x: x["total_claims"], reverse=True)
                
                return {
                    "period": {
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "days": days
                    },
                    "org_id": org_id,
                    "region_id": region_id,
                    "facility_performance": facility_performance,
                    "facility_count": len(facility_performance),
                    "generated_at": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.exception("Failed to generate facility performance comparison", error=str(e))
            raise


# Global analytics instance
claims_analytics = ClaimsAnalytics()
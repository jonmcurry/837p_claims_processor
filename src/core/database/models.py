"""Database models for claims processing system."""

import enum
from datetime import datetime
from decimal import Decimal
from typing import List, Optional

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class ProcessingStatus(str, enum.Enum):
    """Claim processing status enumeration."""

    PENDING = "pending"
    PROCESSING = "processing"
    VALIDATED = "validated"
    FAILED = "failed"
    COMPLETED = "completed"
    REPROCESSING = "reprocessing"


class FailureCategory(str, enum.Enum):
    """Claim failure category enumeration."""

    VALIDATION_ERROR = "validation_error"
    MISSING_DATA = "missing_data"
    DUPLICATE_CLAIM = "duplicate_claim"
    INVALID_FACILITY = "invalid_facility"
    INVALID_PROVIDER = "invalid_provider"
    INVALID_PROCEDURE = "invalid_procedure"
    INVALID_DIAGNOSIS = "invalid_diagnosis"
    DATE_RANGE_ERROR = "date_range_error"
    FINANCIAL_ERROR = "financial_error"
    ML_REJECTION = "ml_rejection"
    SYSTEM_ERROR = "system_error"


class ClaimPriority(str, enum.Enum):
    """Claim processing priority."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Claim(Base):
    """Main claim model for staging database."""

    __tablename__ = "claims"

    # Primary key
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    claim_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    
    # Facility and account information
    facility_id: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    patient_account_number: Mapped[str] = mapped_column(String(50), nullable=False)
    medical_record_number: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Patient demographics (encrypted in production)
    patient_first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    patient_last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    patient_middle_name: Mapped[Optional[str]] = mapped_column(String(100))
    patient_date_of_birth: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    patient_ssn: Mapped[Optional[str]] = mapped_column(String(20))  # Encrypted
    
    # Service information
    admission_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    discharge_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    service_from_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    service_to_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    
    # Financial information
    financial_class: Mapped[str] = mapped_column(String(50), nullable=False)
    total_charges: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    expected_reimbursement: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2))
    
    # Insurance information
    insurance_type: Mapped[str] = mapped_column(String(50), nullable=False)
    insurance_plan_id: Mapped[Optional[str]] = mapped_column(String(50))
    subscriber_id: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Provider information
    billing_provider_npi: Mapped[str] = mapped_column(String(10), nullable=False)
    billing_provider_name: Mapped[str] = mapped_column(String(200), nullable=False)
    attending_provider_npi: Mapped[Optional[str]] = mapped_column(String(10))
    attending_provider_name: Mapped[Optional[str]] = mapped_column(String(200))
    
    # Diagnosis codes
    primary_diagnosis_code: Mapped[str] = mapped_column(String(10), nullable=False)
    diagnosis_codes: Mapped[List[str]] = mapped_column(JSON, default=list)
    
    # Processing metadata
    processing_status: Mapped[ProcessingStatus] = mapped_column(
        Enum(ProcessingStatus), default=ProcessingStatus.PENDING, nullable=False, index=True
    )
    priority: Mapped[ClaimPriority] = mapped_column(
        Enum(ClaimPriority), default=ClaimPriority.MEDIUM, nullable=False
    )
    correlation_id: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    batch_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("batch_metadata.id"))
    
    # ML prediction results
    ml_prediction_score: Mapped[Optional[float]] = mapped_column(Numeric(5, 4))
    ml_prediction_result: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Validation results
    validation_errors: Mapped[Optional[List[dict]]] = mapped_column(JSON)
    validation_warnings: Mapped[Optional[List[dict]]] = mapped_column(JSON)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Retry information
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_retry_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Relationships
    line_items: Mapped[List["ClaimLineItem"]] = relationship(
        "ClaimLineItem", back_populates="claim", cascade="all, delete-orphan"
    )
    batch: Mapped[Optional["BatchMetadata"]] = relationship("BatchMetadata", back_populates="claims")
    
    # Indexes
    __table_args__ = (
        Index("idx_claims_facility_status", "facility_id", "processing_status"),
        Index("idx_claims_created_at_btree", "created_at"),
        Index(
            "idx_claims_active_processing",
            "facility_id",
            "created_at",
            postgresql_where=(processing_status.in_(["pending", "processing"])),
        ),
        UniqueConstraint("claim_id", "facility_id", name="uq_claim_facility"),
    )


class ClaimLineItem(Base):
    """Claim line item model for service details."""

    __tablename__ = "claim_line_items"

    # Primary key
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    claim_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("claims.id"), nullable=False)
    line_number: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Service information
    service_date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    procedure_code: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    procedure_description: Mapped[Optional[str]] = mapped_column(String(500))
    modifier_codes: Mapped[Optional[List[str]]] = mapped_column(JSON)
    
    # Quantity and charges
    units: Mapped[int] = mapped_column(Integer, nullable=False)
    charge_amount: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    
    # Provider information
    rendering_provider_npi: Mapped[Optional[str]] = mapped_column(String(10))
    rendering_provider_name: Mapped[Optional[str]] = mapped_column(String(200))
    
    # RVU information
    rvu_work: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    rvu_practice_expense: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    rvu_malpractice: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    rvu_total: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4))
    expected_reimbursement: Mapped[Optional[Decimal]] = mapped_column(Numeric(10, 2))
    
    # Diagnosis pointers
    diagnosis_pointers: Mapped[Optional[List[int]]] = mapped_column(JSON)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )
    
    # Relationships
    claim: Mapped["Claim"] = relationship("Claim", back_populates="line_items")
    
    # Indexes
    __table_args__ = (
        Index("idx_line_items_claim_procedure", "claim_id", "procedure_code"),
        UniqueConstraint("claim_id", "line_number", name="uq_claim_line_number"),
    )


class BatchMetadata(Base):
    """Batch processing metadata."""

    __tablename__ = "batch_metadata"

    # Primary key
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    batch_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    
    # Batch information
    facility_id: Mapped[Optional[str]] = mapped_column(String(20))
    source_system: Mapped[str] = mapped_column(String(50), nullable=False)
    file_name: Mapped[Optional[str]] = mapped_column(String(500))
    
    # Processing metadata
    status: Mapped[ProcessingStatus] = mapped_column(
        Enum(ProcessingStatus), default=ProcessingStatus.PENDING, nullable=False, index=True
    )
    priority: Mapped[ClaimPriority] = mapped_column(
        Enum(ClaimPriority), default=ClaimPriority.MEDIUM, nullable=False
    )
    
    # Statistics
    total_claims: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    processed_claims: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failed_claims: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_amount: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 2))
    
    # User information
    submitted_by: Mapped[str] = mapped_column(String(100), nullable=False)
    approved_by: Mapped[Optional[str]] = mapped_column(String(100))
    
    # Timestamps
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Processing metrics
    processing_time_seconds: Mapped[Optional[float]] = mapped_column(Numeric(10, 3))
    throughput_per_second: Mapped[Optional[float]] = mapped_column(Numeric(10, 3))
    
    # Error information
    error_summary: Mapped[Optional[dict]] = mapped_column(JSON)
    
    # Relationships
    claims: Mapped[List["Claim"]] = relationship("Claim", back_populates="batch")
    
    # Indexes
    __table_args__ = (
        Index("idx_batch_metadata_status_priority", "status", "priority", "submitted_at"),
    )


class FailedClaim(Base):
    """Failed claims tracking for investigation and reprocessing."""

    __tablename__ = "failed_claims"

    # Primary key
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    original_claim_id: Mapped[Optional[int]] = mapped_column(BigInteger)
    claim_reference: Mapped[str] = mapped_column(String(50), nullable=False)
    facility_id: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    
    # Failure information
    failure_category: Mapped[FailureCategory] = mapped_column(
        Enum(FailureCategory), nullable=False, index=True
    )
    failure_reason: Mapped[str] = mapped_column(Text, nullable=False)
    failure_details: Mapped[dict] = mapped_column(JSON, nullable=False)
    
    # Original claim data
    claim_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    
    # Resolution tracking
    resolution_status: Mapped[str] = mapped_column(
        String(50), default="pending", nullable=False, index=True
    )
    assigned_to: Mapped[Optional[str]] = mapped_column(String(100))
    resolution_notes: Mapped[Optional[str]] = mapped_column(Text)
    resolved_by: Mapped[Optional[str]] = mapped_column(String(100))
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Reprocessing information
    can_reprocess: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    reprocess_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_reprocess_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Financial impact
    charge_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    expected_reimbursement: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 2))
    
    # Timestamps
    failed_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_failed_claims_resolution", "resolution_status", "assigned_to"),
        Index("idx_failed_claims_category_facility", "failure_category", "facility_id"),
        Index("idx_failed_claims_failed_at", "failed_at"),
    )


class AuditLog(Base):
    """Audit logging for HIPAA compliance."""

    __tablename__ = "audit_logs"

    # Primary key
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    
    # Action information
    action_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    action_description: Mapped[str] = mapped_column(Text, nullable=False)
    
    # User information
    user_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    user_name: Mapped[str] = mapped_column(String(200), nullable=False)
    user_role: Mapped[str] = mapped_column(String(50), nullable=False)
    ip_address: Mapped[str] = mapped_column(String(45), nullable=False)
    user_agent: Mapped[Optional[str]] = mapped_column(String(500))
    
    # Resource information
    resource_type: Mapped[str] = mapped_column(String(50), nullable=False)
    resource_id: Mapped[Optional[str]] = mapped_column(String(100))
    
    # PHI access tracking
    accessed_phi: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    phi_fields_accessed: Mapped[Optional[List[str]]] = mapped_column(JSON)
    business_justification: Mapped[Optional[str]] = mapped_column(Text)
    
    # Request information
    request_id: Mapped[str] = mapped_column(String(100), nullable=False)
    session_id: Mapped[Optional[str]] = mapped_column(String(100))
    
    # Additional context
    additional_context: Mapped[Optional[dict]] = mapped_column(JSON)
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False, index=True
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_audit_logs_user_time", "user_id", "created_at"),
        Index("idx_audit_logs_phi_access", "accessed_phi", "created_at"),
    )


class PerformanceMetrics(Base):
    """System performance metrics tracking."""

    __tablename__ = "performance_metrics"

    # Primary key
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    
    # Metric information
    metric_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    metric_value: Mapped[float] = mapped_column(Numeric(20, 6), nullable=False)
    unit: Mapped[str] = mapped_column(String(20), nullable=False)
    
    # Context
    facility_id: Mapped[Optional[str]] = mapped_column(String(20))
    batch_id: Mapped[Optional[str]] = mapped_column(String(100))
    service_name: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Tags for filtering
    tags: Mapped[Optional[dict]] = mapped_column(JSON)
    
    # Timestamp
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False, index=True
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_performance_metrics_type_time", "metric_type", "recorded_at"),
        Index("idx_performance_metrics_service", "service_name", "recorded_at"),
    )
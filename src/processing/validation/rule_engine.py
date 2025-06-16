"""Advanced rule engine for claims validation using Python rule-engine library."""

import re
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import rule_engine
import structlog
from dateutil import parser

from src.cache.redis_cache import cache_manager
from src.core.database.models import Claim, ClaimLineItem

logger = structlog.get_logger(__name__)


class ValidationError:
    """Represents a validation error."""

    def __init__(self, field: str, message: str, severity: str = "error", code: str = None):
        self.field = field
        self.message = message
        self.severity = severity  # error, warning, info
        self.code = code or f"VAL_{field.upper()}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "field": self.field,
            "message": self.message,
            "severity": self.severity,
            "code": self.code,
        }


class ClaimContext:
    """Context object for rule evaluation."""

    def __init__(self, claim: Claim, line_items: List[ClaimLineItem] = None):
        """Initialize claim context for rule evaluation."""
        self.claim = claim
        self.line_items = line_items or []
        
        # Pre-calculate derived fields for rule evaluation
        self._setup_derived_fields()

    def _setup_derived_fields(self) -> None:
        """Setup derived fields for rule evaluation."""
        # Patient age calculation
        today = datetime.utcnow().date()
        dob = self.claim.patient_date_of_birth.date()
        self.patient_age = (today - dob).days // 365

        # Service period calculations
        self.service_days = (self.claim.service_to_date - self.claim.service_from_date).days + 1
        self.admission_days = (self.claim.discharge_date - self.claim.admission_date).days + 1
        
        # Financial calculations
        self.total_line_items = len(self.line_items)
        self.line_item_total = sum(item.charge_amount for item in self.line_items)
        
        # Date validations
        self.is_future_service = self.claim.service_from_date.date() > today
        self.is_discharge_before_admission = self.claim.discharge_date < self.claim.admission_date

    def get_field_value(self, field_path: str) -> Any:
        """Get field value using dot notation."""
        obj = self
        for field in field_path.split('.'):
            if hasattr(obj, field):
                obj = getattr(obj, field)
            else:
                return None
        return obj


class RuleDefinition:
    """Defines a validation rule."""

    def __init__(
        self,
        name: str,
        description: str,
        rule_expression: str,
        error_message: str,
        severity: str = "error",
        category: str = "general",
        enabled: bool = True,
    ):
        self.name = name
        self.description = description
        self.rule_expression = rule_expression
        self.error_message = error_message
        self.severity = severity
        self.category = category
        self.enabled = enabled
        
        # Compile the rule expression
        try:
            self.compiled_rule = rule_engine.Rule(rule_expression)
        except Exception as e:
            logger.error("Failed to compile rule", rule_name=name, error=str(e))
            self.enabled = False


class ClaimValidator:
    """Advanced claims validator using rule engine."""

    def __init__(self):
        """Initialize the validator with predefined rules."""
        self.rules: List[RuleDefinition] = []
        self.reference_data_cache = {}
        self._setup_validation_rules()

    def _setup_validation_rules(self) -> None:
        """Setup comprehensive validation rules."""
        
        # Basic data validation rules
        basic_rules = [
            RuleDefinition(
                name="required_claim_id",
                description="Claim ID must be present and non-empty",
                rule_expression="claim.claim_id is not None and len(claim.claim_id) > 0",
                error_message="Claim ID is required",
                category="required_fields",
            ),
            RuleDefinition(
                name="required_facility_id",
                description="Facility ID must be present and valid format",
                rule_expression="claim.facility_id is not None and len(claim.facility_id) >= 3",
                error_message="Valid facility ID is required",
                category="required_fields",
            ),
            RuleDefinition(
                name="required_patient_name",
                description="Patient first and last name are required",
                rule_expression="claim.patient_first_name is not None and claim.patient_last_name is not None and len(claim.patient_first_name) > 0 and len(claim.patient_last_name) > 0",
                error_message="Patient first and last name are required",
                category="required_fields",
            ),
            RuleDefinition(
                name="required_patient_dob",
                description="Patient date of birth is required",
                rule_expression="claim.patient_date_of_birth is not None",
                error_message="Patient date of birth is required",
                category="required_fields",
            ),
        ]

        # Date validation rules
        date_rules = [
            RuleDefinition(
                name="future_service_date",
                description="Service dates cannot be in the future",
                rule_expression="not is_future_service",
                error_message="Service date cannot be in the future",
                category="date_validation",
            ),
            RuleDefinition(
                name="discharge_after_admission",
                description="Discharge date must be after or equal to admission date",
                rule_expression="not is_discharge_before_admission",
                error_message="Discharge date must be after admission date",
                category="date_validation",
            ),
            RuleDefinition(
                name="service_within_admission",
                description="Service dates must be within admission period",
                rule_expression="claim.service_from_date >= claim.admission_date and claim.service_to_date <= claim.discharge_date",
                error_message="Service dates must be within admission period",
                category="date_validation",
            ),
            RuleDefinition(
                name="reasonable_service_period",
                description="Service period should not exceed 365 days",
                rule_expression="service_days <= 365",
                error_message="Service period exceeds maximum allowed duration",
                category="date_validation",
                severity="warning",
            ),
        ]

        # Financial validation rules
        financial_rules = [
            RuleDefinition(
                name="positive_charges",
                description="Total charges must be positive",
                rule_expression="claim.total_charges > 0",
                error_message="Total charges must be greater than zero",
                category="financial",
            ),
            RuleDefinition(
                name="reasonable_charge_amount",
                description="Charge amount should be reasonable (< $1M)",
                rule_expression="claim.total_charges < 1000000",
                error_message="Charge amount appears unreasonably high",
                category="financial",
                severity="warning",
            ),
            RuleDefinition(
                name="line_items_match_total",
                description="Line item charges should approximately match total",
                rule_expression="abs(line_item_total - claim.total_charges) < 1.00",
                error_message="Line item charges do not match claim total",
                category="financial",
                severity="warning",
            ),
        ]

        # Provider validation rules
        provider_rules = [
            RuleDefinition(
                name="valid_billing_npi",
                description="Billing provider NPI must be valid format",
                rule_expression="claim.billing_provider_npi is not None and len(claim.billing_provider_npi) == 10 and claim.billing_provider_npi.isdigit()",
                error_message="Invalid billing provider NPI format",
                category="provider",
            ),
            RuleDefinition(
                name="provider_name_present",
                description="Billing provider name must be present",
                rule_expression="claim.billing_provider_name is not None and len(claim.billing_provider_name) > 0",
                error_message="Billing provider name is required",
                category="provider",
            ),
        ]

        # Patient demographic rules
        demographic_rules = [
            RuleDefinition(
                name="reasonable_patient_age",
                description="Patient age should be reasonable (0-150 years)",
                rule_expression="patient_age >= 0 and patient_age <= 150",
                error_message="Patient age appears unreasonable",
                category="demographics",
            ),
            RuleDefinition(
                name="valid_ssn_format",
                description="SSN format validation if provided",
                rule_expression="claim.patient_ssn is None or len(claim.patient_ssn) in [9, 11]",
                error_message="Invalid SSN format",
                category="demographics",
                severity="warning",
            ),
        ]

        # Diagnosis validation rules
        diagnosis_rules = [
            RuleDefinition(
                name="primary_diagnosis_present",
                description="Primary diagnosis code is required",
                rule_expression="claim.primary_diagnosis_code is not None and len(claim.primary_diagnosis_code) >= 3",
                error_message="Primary diagnosis code is required",
                category="diagnosis",
            ),
            RuleDefinition(
                name="valid_icd10_format",
                description="Primary diagnosis should follow ICD-10 format",
                rule_expression="self._validate_icd10_format(claim.primary_diagnosis_code)",
                error_message="Invalid ICD-10 diagnosis code format",
                category="diagnosis",
            ),
        ]

        # Insurance validation rules
        insurance_rules = [
            RuleDefinition(
                name="insurance_type_present",
                description="Insurance type is required",
                rule_expression="claim.insurance_type is not None and len(claim.insurance_type) > 0",
                error_message="Insurance type is required",
                category="insurance",
            ),
            RuleDefinition(
                name="financial_class_present",
                description="Financial class is required",
                rule_expression="claim.financial_class is not None and len(claim.financial_class) > 0",
                error_message="Financial class is required",
                category="insurance",
            ),
        ]

        # Line item validation rules
        line_item_rules = [
            RuleDefinition(
                name="line_items_present",
                description="At least one line item is required",
                rule_expression="total_line_items > 0",
                error_message="At least one line item is required",
                category="line_items",
            ),
            RuleDefinition(
                name="reasonable_line_item_count",
                description="Line item count should be reasonable",
                rule_expression="total_line_items <= 50",
                error_message="Excessive number of line items",
                category="line_items",
                severity="warning",
            ),
        ]

        # Combine all rules
        all_rules = (
            basic_rules + date_rules + financial_rules + provider_rules +
            demographic_rules + diagnosis_rules + insurance_rules + line_item_rules
        )

        # Add custom facility-specific rules
        facility_rules = self._get_facility_specific_rules()
        all_rules.extend(facility_rules)

        self.rules = all_rules
        logger.info("Initialized validator with rules", rule_count=len(self.rules))

    def _get_facility_specific_rules(self) -> List[RuleDefinition]:
        """Get facility-specific validation rules from cache."""
        # This would typically load from database/cache
        return [
            RuleDefinition(
                name="facility_procedure_codes",
                description="Procedure codes must be valid for facility",
                rule_expression="self._validate_facility_procedures(claim.facility_id, line_items)",
                error_message="Invalid procedure code for this facility",
                category="facility_specific",
                severity="warning",
            ),
        ]

    async def validate_claim(self, claim: Claim, line_items: Optional[List[ClaimLineItem]] = None) -> Tuple[bool, List[ValidationError]]:
        """Validate a claim against all active rules."""
        context = ClaimContext(claim, line_items or [])
        errors = []
        warnings = []

        for rule in self.rules:
            if not rule.enabled:
                continue

            try:
                # Evaluate the rule
                result = await self._evaluate_rule(rule, context)
                
                if not result:
                    error = ValidationError(
                        field=rule.category,
                        message=rule.error_message,
                        severity=rule.severity,
                        code=rule.name.upper(),
                    )
                    
                    if rule.severity == "error":
                        errors.append(error)
                    else:
                        warnings.append(error)

            except Exception as e:
                logger.exception("Rule evaluation failed", rule_name=rule.name, error=str(e))
                # Don't fail validation due to rule evaluation errors
                continue

        # Additional complex validations
        complex_errors = await self._run_complex_validations(context)
        errors.extend(complex_errors)

        is_valid = len(errors) == 0
        all_issues = errors + warnings

        logger.debug("Claim validation completed",
                    claim_id=claim.claim_id,
                    is_valid=is_valid,
                    error_count=len(errors),
                    warning_count=len(warnings))

        return is_valid, all_issues

    async def _evaluate_rule(self, rule: RuleDefinition, context: ClaimContext) -> bool:
        """Evaluate a single rule against claim context."""
        try:
            # Create evaluation context with claim data and helper methods
            eval_context = {
                'claim': context.claim,
                'line_items': context.line_items,
                'patient_age': context.patient_age,
                'service_days': context.service_days,
                'admission_days': context.admission_days,
                'total_line_items': context.total_line_items,
                'line_item_total': context.line_item_total,
                'is_future_service': context.is_future_service,
                'is_discharge_before_admission': context.is_discharge_before_admission,
                'self': self,  # Allow rules to call validator methods
            }

            return rule.compiled_rule.matches(eval_context)

        except Exception as e:
            logger.error("Rule evaluation error", rule_name=rule.name, error=str(e))
            return True  # Default to passing on evaluation errors

    async def _run_complex_validations(self, context: ClaimContext) -> List[ValidationError]:
        """Run complex validations that require external data or multiple checks."""
        errors = []

        # Validate facility exists and is active
        facility_error = await self._validate_facility(context.claim.facility_id)
        if facility_error:
            errors.append(facility_error)

        # Validate provider NPIs
        npi_errors = await self._validate_npis(context)
        errors.extend(npi_errors)

        # Validate procedure codes
        procedure_errors = await self._validate_procedures(context)
        errors.extend(procedure_errors)

        # Validate diagnosis codes
        diagnosis_errors = await self._validate_diagnoses(context)
        errors.extend(diagnosis_errors)

        return errors

    async def _validate_facility(self, facility_id: str) -> Optional[ValidationError]:
        """Validate facility exists and is active."""
        try:
            facility_info = await cache_manager.get_facility_info(facility_id)
            if not facility_info:
                return ValidationError(
                    field="facility_id",
                    message=f"Facility {facility_id} not found",
                    code="FACILITY_NOT_FOUND"
                )
            
            if not facility_info.get("is_active", False):
                return ValidationError(
                    field="facility_id",
                    message=f"Facility {facility_id} is inactive",
                    code="FACILITY_INACTIVE"
                )

        except Exception as e:
            logger.warning("Facility validation failed", facility_id=facility_id, error=str(e))
            # Don't fail validation if external service is down
            
        return None

    async def _validate_npis(self, context: ClaimContext) -> List[ValidationError]:
        """Validate NPI numbers against registry."""
        errors = []
        
        npis_to_check = [context.claim.billing_provider_npi]
        if context.claim.attending_provider_npi:
            npis_to_check.append(context.claim.attending_provider_npi)

        for line_item in context.line_items:
            if line_item.rendering_provider_npi:
                npis_to_check.append(line_item.rendering_provider_npi)

        for npi in set(npis_to_check):  # Remove duplicates
            if npi and not await self._is_valid_npi(npi):
                errors.append(ValidationError(
                    field="provider_npi",
                    message=f"Invalid or inactive NPI: {npi}",
                    code="INVALID_NPI"
                ))

        return errors

    async def _validate_procedures(self, context: ClaimContext) -> List[ValidationError]:
        """Validate procedure codes."""
        errors = []
        
        for line_item in context.line_items:
            if not await self._is_valid_cpt_code(line_item.procedure_code):
                errors.append(ValidationError(
                    field="procedure_code",
                    message=f"Invalid CPT code: {line_item.procedure_code}",
                    code="INVALID_CPT"
                ))

        return errors

    async def _validate_diagnoses(self, context: ClaimContext) -> List[ValidationError]:
        """Validate diagnosis codes."""
        errors = []
        
        # Validate primary diagnosis
        if not await self._is_valid_icd10_code(context.claim.primary_diagnosis_code):
            errors.append(ValidationError(
                field="primary_diagnosis_code",
                message=f"Invalid ICD-10 code: {context.claim.primary_diagnosis_code}",
                code="INVALID_ICD10"
            ))

        # Validate additional diagnosis codes
        for i, dx_code in enumerate(context.claim.diagnosis_codes or []):
            if dx_code and not await self._is_valid_icd10_code(dx_code):
                errors.append(ValidationError(
                    field=f"diagnosis_codes[{i}]",
                    message=f"Invalid ICD-10 code: {dx_code}",
                    code="INVALID_ICD10"
                ))

        return errors

    def _validate_icd10_format(self, code: str) -> bool:
        """Validate ICD-10 code format."""
        if not code:
            return False
        
        # Basic ICD-10 format: A00.0 to Z99.9
        pattern = r'^[A-TV-Z][0-9][0-9AB]\.?[0-9A-TV-Z]{0,4}$'
        return bool(re.match(pattern, code.upper()))

    async def _is_valid_npi(self, npi: str) -> bool:
        """Check if NPI is valid and active."""
        if not npi or len(npi) != 10 or not npi.isdigit():
            return False
        
        # Check Luhn algorithm for NPI validation
        if not self._validate_npi_checksum(npi):
            return False
        
        # Check against cached NPI registry data
        try:
            npi_info = await cache_manager.get_npi_info(npi)
            return npi_info and npi_info.get("status") == "active"
        except Exception:
            # If external validation fails, accept format validation
            return True

    def _validate_npi_checksum(self, npi: str) -> bool:
        """Validate NPI using Luhn algorithm."""
        # NPI validation using Luhn algorithm
        digits = [int(d) for d in npi]
        checksum = sum(digits[i] if i % 2 == 0 else sum(divmod(digits[i] * 2, 10)) for i in range(10))
        return checksum % 10 == 0

    async def _is_valid_cpt_code(self, code: str) -> bool:
        """Check if CPT code is valid."""
        if not code:
            return False
        
        # Basic CPT format validation
        if not (code.isdigit() and len(code) == 5):
            return False
        
        try:
            cpt_info = await cache_manager.get_cpt_info(code)
            return cpt_info and cpt_info.get("status") == "active"
        except Exception:
            # If external validation fails, accept format validation
            return True

    async def _is_valid_icd10_code(self, code: str) -> bool:
        """Check if ICD-10 code is valid."""
        if not self._validate_icd10_format(code):
            return False
        
        try:
            icd_info = await cache_manager.get_icd10_info(code)
            return icd_info and icd_info.get("status") == "active"
        except Exception:
            # If external validation fails, accept format validation
            return True

    def _validate_facility_procedures(self, facility_id: str, line_items: List[ClaimLineItem]) -> bool:
        """Validate procedures are allowed for facility."""
        # This would check against facility-specific procedure allowlists
        return True  # Placeholder implementation

    async def get_validation_summary(self, facility_id: Optional[str] = None) -> Dict[str, Any]:
        """Get validation rule summary and statistics."""
        active_rules = [r for r in self.rules if r.enabled]
        
        summary = {
            "total_rules": len(self.rules),
            "active_rules": len(active_rules),
            "categories": {},
            "severity_breakdown": {"error": 0, "warning": 0, "info": 0},
        }

        for rule in active_rules:
            # Count by category
            if rule.category not in summary["categories"]:
                summary["categories"][rule.category] = 0
            summary["categories"][rule.category] += 1
            
            # Count by severity
            summary["severity_breakdown"][rule.severity] += 1

        return summary
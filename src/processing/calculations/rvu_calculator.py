"""RVU-based reimbursement calculation engine."""

from decimal import Decimal
from typing import Dict, List, Optional, Tuple

import structlog

from src.cache.redis_cache import cache_manager
from src.core.config import settings
from src.core.database.models import Claim, ClaimLineItem

logger = structlog.get_logger(__name__)


class RVUData:
    """RVU data structure for a procedure code."""

    def __init__(
        self,
        procedure_code: str,
        work_rvu: Decimal,
        practice_expense_rvu: Decimal,
        malpractice_rvu: Decimal,
        total_rvu: Optional[Decimal] = None,
        modifier_adjustments: Optional[Dict[str, Decimal]] = None,
    ):
        self.procedure_code = procedure_code
        self.work_rvu = Decimal(str(work_rvu))
        self.practice_expense_rvu = Decimal(str(practice_expense_rvu))
        self.malpractice_rvu = Decimal(str(malpractice_rvu))
        self.total_rvu = total_rvu or (self.work_rvu + self.practice_expense_rvu + self.malpractice_rvu)
        self.modifier_adjustments = modifier_adjustments or {}


class ConversionFactors:
    """Medicare conversion factors and geographic adjustments."""

    # 2024 Medicare Physician Fee Schedule conversion factor
    MEDICARE_CF_2024 = Decimal("36.04")
    
    # Geographic Practice Cost Index (GPCI) - sample values
    DEFAULT_GPCI = {
        "work": Decimal("1.000"),
        "practice_expense": Decimal("1.000"),
        "malpractice": Decimal("1.000"),
    }

    @classmethod
    async def get_conversion_factor(cls, insurance_type: str, year: int = 2024) -> Decimal:
        """Get conversion factor based on insurance type and year."""
        # This would typically come from a database or cache
        if insurance_type.upper() in ["MEDICARE", "MEDICARE_ADVANTAGE"]:
            return cls.MEDICARE_CF_2024
        elif insurance_type.upper() in ["MEDICAID"]:
            return cls.MEDICARE_CF_2024 * Decimal("0.85")  # Medicaid typically pays less
        else:
            return cls.MEDICARE_CF_2024 * Decimal("1.10")  # Commercial typically pays more

    @classmethod
    async def get_gpci_values(cls, facility_id: str) -> Dict[str, Decimal]:
        """Get GPCI values for a facility location."""
        try:
            facility_info = await cache_manager.get_facility_info(facility_id)
            if facility_info and "gpci" in facility_info:
                return facility_info["gpci"]
        except Exception as e:
            logger.warning("Failed to get GPCI values", facility_id=facility_id, error=str(e))
        
        return cls.DEFAULT_GPCI


class ModifierAdjustments:
    """Handle CPT modifier adjustments to RVU values."""

    # Common modifier adjustments (percentage of base RVU)
    MODIFIER_ADJUSTMENTS = {
        "26": Decimal("0.26"),  # Professional component only
        "TC": Decimal("0.74"),  # Technical component only
        "50": Decimal("1.50"),  # Bilateral procedure
        "51": Decimal("0.50"),  # Multiple procedures (50% for additional)
        "52": Decimal("0.50"),  # Reduced services
        "53": Decimal("0.00"),  # Discontinued procedure
        "22": Decimal("1.25"),  # Increased procedural services
        "78": Decimal("0.70"),  # Unplanned return to OR
        "79": Decimal("1.00"),  # Unrelated procedure during postop
        "80": Decimal("0.16"),  # Assistant surgeon
        "81": Decimal("0.16"),  # Minimum assistant surgeon
        "82": Decimal("0.16"),  # Assistant surgeon when qualified surgeon not available
        "AS": Decimal("0.16"),  # Assistant at surgery
    }

    @classmethod
    def apply_modifiers(cls, base_rvu: Decimal, modifiers: List[str]) -> Decimal:
        """Apply modifier adjustments to base RVU value."""
        if not modifiers:
            return base_rvu

        adjusted_rvu = base_rvu
        
        for modifier in modifiers:
            if modifier in cls.MODIFIER_ADJUSTMENTS:
                adjustment = cls.MODIFIER_ADJUSTMENTS[modifier]
                adjusted_rvu = adjusted_rvu * adjustment
                logger.debug("Applied modifier adjustment", 
                           modifier=modifier, 
                           adjustment=adjustment,
                           original_rvu=base_rvu,
                           adjusted_rvu=adjusted_rvu)

        return adjusted_rvu


class RVUCalculator:
    """Calculate RVU values and expected reimbursements for claims."""

    def __init__(self):
        """Initialize RVU calculator."""
        self.rvu_cache = {}
        self.conversion_factors_cache = {}

    async def calculate_claim_rvus(self, claim: Claim) -> None:
        """Calculate RVU values for all line items in a claim."""
        try:
            total_work_rvu = Decimal("0")
            total_practice_expense_rvu = Decimal("0")
            total_malpractice_rvu = Decimal("0")
            total_expected_reimbursement = Decimal("0")

            # Get conversion factor for this claim
            conversion_factor = await ConversionFactors.get_conversion_factor(claim.insurance_type)
            gpci_values = await ConversionFactors.get_gpci_values(claim.facility_id)

            for line_item in claim.line_items:
                await self._calculate_line_item_rvu(line_item, conversion_factor, gpci_values)
                
                # Accumulate totals
                total_work_rvu += line_item.rvu_work or Decimal("0")
                total_practice_expense_rvu += line_item.rvu_practice_expense or Decimal("0")
                total_malpractice_rvu += line_item.rvu_malpractice or Decimal("0")
                total_expected_reimbursement += line_item.expected_reimbursement or Decimal("0")

            # Update claim totals
            claim.expected_reimbursement = total_expected_reimbursement

            logger.info("Calculated claim RVUs",
                       claim_id=claim.claim_id,
                       total_work_rvu=total_work_rvu,
                       total_expected_reimbursement=total_expected_reimbursement)

        except Exception as e:
            logger.exception("RVU calculation failed", claim_id=claim.claim_id, error=str(e))
            raise

    async def _calculate_line_item_rvu(
        self, 
        line_item: ClaimLineItem, 
        conversion_factor: Decimal,
        gpci_values: Dict[str, Decimal]
    ) -> None:
        """Calculate RVU values for a single line item."""
        try:
            # Get RVU data for the procedure code
            rvu_data = await self._get_rvu_data(line_item.procedure_code)
            
            if not rvu_data:
                logger.warning("No RVU data found", procedure_code=line_item.procedure_code)
                return

            # Apply modifier adjustments
            modifiers = line_item.modifier_codes or []
            adjusted_work_rvu = ModifierAdjustments.apply_modifiers(rvu_data.work_rvu, modifiers)
            adjusted_pe_rvu = ModifierAdjustments.apply_modifiers(rvu_data.practice_expense_rvu, modifiers)
            adjusted_mp_rvu = ModifierAdjustments.apply_modifiers(rvu_data.malpractice_rvu, modifiers)

            # Apply geographic adjustments (GPCI)
            geographic_work_rvu = adjusted_work_rvu * gpci_values["work"]
            geographic_pe_rvu = adjusted_pe_rvu * gpci_values["practice_expense"]
            geographic_mp_rvu = adjusted_mp_rvu * gpci_values["malpractice"]

            # Calculate total adjusted RVU
            total_adjusted_rvu = geographic_work_rvu + geographic_pe_rvu + geographic_mp_rvu

            # Apply units
            final_work_rvu = geographic_work_rvu * Decimal(str(line_item.units))
            final_pe_rvu = geographic_pe_rvu * Decimal(str(line_item.units))
            final_mp_rvu = geographic_mp_rvu * Decimal(str(line_item.units))
            final_total_rvu = total_adjusted_rvu * Decimal(str(line_item.units))

            # Calculate expected reimbursement
            expected_reimbursement = final_total_rvu * conversion_factor

            # Update line item with calculated values
            line_item.rvu_work = final_work_rvu
            line_item.rvu_practice_expense = final_pe_rvu
            line_item.rvu_malpractice = final_mp_rvu
            line_item.rvu_total = final_total_rvu
            line_item.expected_reimbursement = expected_reimbursement

            logger.debug("Calculated line item RVU",
                        procedure_code=line_item.procedure_code,
                        units=line_item.units,
                        total_rvu=final_total_rvu,
                        expected_reimbursement=expected_reimbursement)

        except Exception as e:
            logger.exception("Line item RVU calculation failed", 
                           procedure_code=line_item.procedure_code, 
                           error=str(e))
            # Set default values to prevent null constraint violations
            line_item.rvu_work = Decimal("0")
            line_item.rvu_practice_expense = Decimal("0")
            line_item.rvu_malpractice = Decimal("0")
            line_item.rvu_total = Decimal("0")
            line_item.expected_reimbursement = Decimal("0")

    async def _get_rvu_data(self, procedure_code: str) -> Optional[RVUData]:
        """Get RVU data for a procedure code."""
        # Check cache first
        if procedure_code in self.rvu_cache:
            return self.rvu_cache[procedure_code]

        try:
            # Try to get from Redis cache
            rvu_data = await cache_manager.get_rvu_data(procedure_code)
            
            if rvu_data:
                rvu_obj = RVUData(
                    procedure_code=procedure_code,
                    work_rvu=rvu_data["work_rvu"],
                    practice_expense_rvu=rvu_data["practice_expense_rvu"],
                    malpractice_rvu=rvu_data["malpractice_rvu"],
                )
                
                # Cache locally
                self.rvu_cache[procedure_code] = rvu_obj
                return rvu_obj

        except Exception as e:
            logger.warning("Failed to get RVU data from cache", 
                         procedure_code=procedure_code, 
                         error=str(e))

        # Fallback to default RVU values for common codes
        return self._get_default_rvu_data(procedure_code)

    def _get_default_rvu_data(self, procedure_code: str) -> Optional[RVUData]:
        """Get default RVU data for common procedure codes."""
        # Default RVU values for common procedures
        default_rvus = {
            "99213": RVUData("99213", Decimal("0.97"), Decimal("0.85"), Decimal("0.04")),  # Office visit
            "99214": RVUData("99214", Decimal("1.50"), Decimal("1.30"), Decimal("0.06")),  # Office visit
            "99215": RVUData("99215", Decimal("2.11"), Decimal("1.83"), Decimal("0.09")),  # Office visit
            "99223": RVUData("99223", Decimal("3.05"), Decimal("1.20"), Decimal("0.12")),  # Hospital visit
            "99232": RVUData("99232", Decimal("1.28"), Decimal("0.80"), Decimal("0.05")),  # Hospital subsequent
            "99233": RVUData("99233", Decimal("1.93"), Decimal("1.10"), Decimal("0.08")),  # Hospital subsequent
            "99283": RVUData("99283", Decimal("1.42"), Decimal("1.95"), Decimal("0.07")),  # Emergency dept
            "99284": RVUData("99284", Decimal("2.60"), Decimal("3.52"), Decimal("0.12")),  # Emergency dept
            "99285": RVUData("99285", Decimal("4.16"), Decimal("5.71"), Decimal("0.20")),  # Emergency dept
        }

        if procedure_code in default_rvus:
            rvu_data = default_rvus[procedure_code]
            self.rvu_cache[procedure_code] = rvu_data
            return rvu_data

        # If no default available, create minimal RVU
        logger.warning("No RVU data available, using minimal values", 
                      procedure_code=procedure_code)
        minimal_rvu = RVUData(procedure_code, Decimal("0.1"), Decimal("0.1"), Decimal("0.01"))
        self.rvu_cache[procedure_code] = minimal_rvu
        return minimal_rvu

    async def get_reimbursement_estimate(
        self, 
        procedure_code: str, 
        units: int, 
        insurance_type: str,
        facility_id: str,
        modifiers: Optional[List[str]] = None
    ) -> Dict[str, Decimal]:
        """Get reimbursement estimate for a specific procedure."""
        try:
            # Get RVU data
            rvu_data = await self._get_rvu_data(procedure_code)
            if not rvu_data:
                return {"error": "No RVU data available"}

            # Get conversion factor and GPCI
            conversion_factor = await ConversionFactors.get_conversion_factor(insurance_type)
            gpci_values = await ConversionFactors.get_gpci_values(facility_id)

            # Apply modifiers
            modifiers = modifiers or []
            adjusted_work_rvu = ModifierAdjustments.apply_modifiers(rvu_data.work_rvu, modifiers)
            adjusted_pe_rvu = ModifierAdjustments.apply_modifiers(rvu_data.practice_expense_rvu, modifiers)
            adjusted_mp_rvu = ModifierAdjustments.apply_modifiers(rvu_data.malpractice_rvu, modifiers)

            # Apply geographic adjustments
            geographic_work_rvu = adjusted_work_rvu * gpci_values["work"]
            geographic_pe_rvu = adjusted_pe_rvu * gpci_values["practice_expense"]
            geographic_mp_rvu = adjusted_mp_rvu * gpci_values["malpractice"]

            # Calculate total RVU
            total_rvu = (geographic_work_rvu + geographic_pe_rvu + geographic_mp_rvu) * Decimal(str(units))

            # Calculate reimbursement
            expected_reimbursement = total_rvu * conversion_factor

            return {
                "procedure_code": procedure_code,
                "units": Decimal(str(units)),
                "work_rvu": geographic_work_rvu * Decimal(str(units)),
                "practice_expense_rvu": geographic_pe_rvu * Decimal(str(units)),
                "malpractice_rvu": geographic_mp_rvu * Decimal(str(units)),
                "total_rvu": total_rvu,
                "conversion_factor": conversion_factor,
                "expected_reimbursement": expected_reimbursement,
                "modifiers_applied": modifiers,
            }

        except Exception as e:
            logger.exception("Reimbursement estimate failed", 
                           procedure_code=procedure_code, 
                           error=str(e))
            return {"error": str(e)}

    async def validate_rvu_calculations(self, claim: Claim) -> List[Dict[str, str]]:
        """Validate RVU calculations for accuracy."""
        issues = []

        for line_item in claim.line_items:
            # Check for missing RVU data
            if not line_item.rvu_total or line_item.rvu_total == 0:
                issues.append({
                    "type": "missing_rvu",
                    "line_item": str(line_item.line_number),
                    "procedure_code": line_item.procedure_code,
                    "message": "Missing or zero RVU calculation"
                })

            # Check for unreasonably high RVU values
            if line_item.rvu_total and line_item.rvu_total > 50:
                issues.append({
                    "type": "high_rvu",
                    "line_item": str(line_item.line_number),
                    "procedure_code": line_item.procedure_code,
                    "rvu_total": str(line_item.rvu_total),
                    "message": "Unusually high RVU value"
                })

            # Check reimbursement vs charge amount
            if (line_item.expected_reimbursement and 
                line_item.charge_amount and 
                line_item.expected_reimbursement > line_item.charge_amount * 2):
                issues.append({
                    "type": "high_reimbursement",
                    "line_item": str(line_item.line_number),
                    "procedure_code": line_item.procedure_code,
                    "expected_reimbursement": str(line_item.expected_reimbursement),
                    "charge_amount": str(line_item.charge_amount),
                    "message": "Expected reimbursement exceeds charge amount significantly"
                })

        return issues


# Global calculator instance
rvu_calculator = RVUCalculator()
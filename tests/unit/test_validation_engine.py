"""Unit tests for the comprehensive validation engine."""

import pytest
from datetime import datetime, date
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch

from src.processing.validation.comprehensive_rules import (
    ComprehensiveValidationEngine,
    ClaimValidationContext,
    ValidationResult,
    ValidationSeverity,
    RuleCategory
)
from src.core.database.models import Claim, ClaimLineItem


class TestClaimValidationContext:
    """Test ClaimValidationContext functionality."""
    
    def test_context_initialization(self, sample_claim_data):
        """Test context initialization with claim data."""
        claim = Claim(**sample_claim_data)
        line_items = [
            ClaimLineItem(
                line_number=1,
                procedure_code="99213",
                units=1,
                charge_amount=Decimal("150.00"),
                service_date=datetime.now().date()
            )
        ]
        
        context = ClaimValidationContext(
            claim=claim,
            line_items=line_items,
            patient_age=0,
            total_line_item_charges=Decimal('0'),
            line_item_count=0,
            service_days=0,
            unique_procedure_codes=set(),
            unique_diagnosis_codes=set(),
            is_emergency=False,
            is_outpatient=False,
            is_inpatient=False
        )
        
        assert context.claim == claim
        assert context.line_items == line_items
        assert isinstance(context.patient_age, int)
        assert context.patient_age >= 0
        
    def test_derived_fields_calculation(self, sample_claim_data):
        """Test calculation of derived fields."""
        # Set a specific date of birth for age calculation
        sample_claim_data["patient_date_of_birth"] = date(1980, 1, 1)
        claim = Claim(**sample_claim_data)
        
        line_items = [
            ClaimLineItem(
                line_number=1,
                procedure_code="99213",
                units=1,
                charge_amount=Decimal("150.00"),
                service_date=datetime.now().date()
            ),
            ClaimLineItem(
                line_number=2,
                procedure_code="99283",  # Emergency code
                units=1,
                charge_amount=Decimal("300.00"),
                service_date=datetime.now().date()
            )
        ]
        
        context = ClaimValidationContext(
            claim=claim,
            line_items=line_items,
            patient_age=0,
            total_line_item_charges=Decimal('0'),
            line_item_count=0,
            service_days=0,
            unique_procedure_codes=set(),
            unique_diagnosis_codes=set(),
            is_emergency=False,
            is_outpatient=False,
            is_inpatient=False
        )
        
        # Verify age calculation
        current_year = date.today().year
        expected_age = current_year - 1980
        assert context.patient_age in [expected_age - 1, expected_age]  # Account for birthday
        
        # Verify aggregations
        assert context.total_line_item_charges == Decimal("450.00")
        assert context.line_item_count == 2
        
        # Verify procedure analysis
        assert len(context.unique_procedure_codes) == 2
        assert "99213" in context.unique_procedure_codes
        assert "99283" in context.unique_procedure_codes
        
        # Verify emergency detection
        assert context.is_emergency is True


class TestValidationRules:
    """Test individual validation rules."""
    
    @pytest.fixture
    def validation_engine(self):
        """Create validation engine for testing."""
        return ComprehensiveValidationEngine()
    
    @pytest.fixture
    def valid_context(self, sample_claim_data):
        """Create valid claim context for testing."""
        claim = Claim(**sample_claim_data)
        line_items = [
            ClaimLineItem(
                line_number=1,
                procedure_code="99213",
                units=1,
                charge_amount=Decimal("150.00"),
                service_date=datetime.now().date()
            )
        ]
        
        return ClaimValidationContext(
            claim=claim,
            line_items=line_items,
            patient_age=44,  # Based on 1980 birth year
            total_line_item_charges=Decimal('150.00'),
            line_item_count=1,
            service_days=1,
            unique_procedure_codes={"99213"},
            unique_diagnosis_codes={"Z00.00"},
            is_emergency=False,
            is_outpatient=True,
            is_inpatient=False
        )
    
    def test_patient_name_validation_success(self, validation_engine, valid_context):
        """Test successful patient name validation."""
        # Get the specific validation rule
        patient_demo_rules = validation_engine.rules[RuleCategory.PATIENT_DEMOGRAPHICS]
        name_rule = next(rule for rule in patient_demo_rules if rule['id'] == 'PDR001')
        
        result = name_rule['func'](valid_context)
        
        assert isinstance(result, ValidationResult)
        assert result.passed is True
        assert result.rule_id == "PDR001"
        assert result.severity == ValidationSeverity.ERROR
        assert result.category == RuleCategory.PATIENT_DEMOGRAPHICS
    
    def test_patient_name_validation_failure(self, validation_engine, valid_context):
        """Test failed patient name validation."""
        # Modify context to have invalid name
        valid_context.claim.patient_first_name = ""
        
        patient_demo_rules = validation_engine.rules[RuleCategory.PATIENT_DEMOGRAPHICS]
        name_rule = next(rule for rule in patient_demo_rules if rule['id'] == 'PDR001')
        
        result = name_rule['func'](valid_context)
        
        assert result.passed is False
        assert "required" in result.message.lower()
    
    def test_patient_age_validation_success(self, validation_engine, valid_context):
        """Test successful patient age validation."""
        patient_demo_rules = validation_engine.rules[RuleCategory.PATIENT_DEMOGRAPHICS]
        age_rule = next(rule for rule in patient_demo_rules if rule['id'] == 'PDR002')
        
        result = age_rule['func'](valid_context)
        
        assert result.passed is True
        assert result.rule_id == "PDR002"
    
    def test_patient_age_validation_failure(self, validation_engine, valid_context):
        """Test failed patient age validation."""
        # Set invalid age
        valid_context.patient_age = 200
        
        patient_demo_rules = validation_engine.rules[RuleCategory.PATIENT_DEMOGRAPHICS]
        age_rule = next(rule for rule in patient_demo_rules if rule['id'] == 'PDR002')
        
        result = age_rule['func'](valid_context)
        
        assert result.passed is False
        assert "invalid" in result.message.lower()
        assert result.value == 200
    
    def test_future_date_validation(self, validation_engine, valid_context):
        """Test future date validation."""
        # Set future date of birth
        from datetime import timedelta
        future_date = date.today() + timedelta(days=1)
        valid_context.claim.patient_date_of_birth = future_date
        
        patient_demo_rules = validation_engine.rules[RuleCategory.PATIENT_DEMOGRAPHICS]
        date_rule = next(rule for rule in patient_demo_rules if rule['id'] == 'PDR003')
        
        result = date_rule['func'](valid_context)
        
        assert result.passed is False
        assert "future" in result.message.lower()
    
    def test_ssn_format_validation(self, validation_engine, valid_context):
        """Test SSN format validation."""
        # Test valid SSN format
        valid_context.claim.patient_ssn = "123-45-6789"
        
        patient_demo_rules = validation_engine.rules[RuleCategory.PATIENT_DEMOGRAPHICS]
        ssn_rule = next(rule for rule in patient_demo_rules if rule['id'] == 'PDR004')
        
        result = ssn_rule['func'](valid_context)
        assert result.passed is True
        
        # Test invalid SSN format
        valid_context.claim.patient_ssn = "invalid-ssn"
        result = ssn_rule['func'](valid_context)
        assert result.passed is False
        assert result.severity == ValidationSeverity.WARNING  # SSN is warning, not error
    
    def test_financial_validation_rules(self, validation_engine, valid_context):
        """Test financial validation rules."""
        financial_rules = validation_engine.rules[RuleCategory.FINANCIAL_VALIDATION]
        
        # Test total charges validation
        charges_rule = next(rule for rule in financial_rules if rule['id'] == 'FVR001')
        
        # Valid charges
        result = charges_rule['func'](valid_context)
        assert result.passed is True
        
        # Invalid charges (zero or negative)
        valid_context.claim.total_charges = Decimal('0')
        result = charges_rule['func'](valid_context)
        assert result.passed is False
        
        # Very high charges (warning)
        valid_context.claim.total_charges = Decimal('1500000')  # $1.5M
        result = charges_rule['func'](valid_context)
        assert result.passed is False
        assert result.severity == ValidationSeverity.WARNING
    
    def test_line_item_sum_validation(self, validation_engine, valid_context):
        """Test line item sum validation."""
        financial_rules = validation_engine.rules[RuleCategory.FINANCIAL_VALIDATION]
        sum_rule = next(rule for rule in financial_rules if rule['id'] == 'FVR002')
        
        # Matching totals should pass
        valid_context.claim.total_charges = Decimal('150.00')
        valid_context.total_line_item_charges = Decimal('150.00')
        
        result = sum_rule['func'](valid_context)
        assert result.passed is True
        
        # Mismatched totals should fail
        valid_context.claim.total_charges = Decimal('200.00')
        valid_context.total_line_item_charges = Decimal('150.00')
        
        result = sum_rule['func'](valid_context)
        assert result.passed is False
        assert result.expected == Decimal('150.00')
        assert result.value == Decimal('200.00')


class TestValidationEngine:
    """Test the complete validation engine."""
    
    @pytest.fixture
    def validation_engine(self):
        """Create validation engine for testing."""
        return ComprehensiveValidationEngine()
    
    @pytest.mark.asyncio
    async def test_validate_claim_success(self, validation_engine, sample_claim_data):
        """Test successful claim validation."""
        claim = Claim(**sample_claim_data)
        line_items = [
            ClaimLineItem(
                line_number=1,
                procedure_code="99213",
                units=1,
                charge_amount=Decimal("150.00"),
                service_date=datetime.now().date()
            )
        ]
        
        # Mock the lookup data loading
        with patch.object(validation_engine, '_load_lookup_data', new_callable=AsyncMock):
            passed, results = await validation_engine.validate_claim(claim, line_items)
        
        assert isinstance(passed, bool)
        assert isinstance(results, list)
        assert all(isinstance(result, ValidationResult) for result in results)
        
        # Should have results from all implemented rule categories
        rule_ids = {result.rule_id for result in results}
        expected_rules = {'PDR001', 'PDR002', 'PDR003', 'PDR004', 'PDR005', 'FVR001', 'FVR002', 'FVR003'}
        assert expected_rules.issubset(rule_ids)
    
    @pytest.mark.asyncio
    async def test_validate_claim_with_errors(self, validation_engine, sample_claim_data):
        """Test claim validation with errors."""
        # Create claim with validation errors
        sample_claim_data['patient_first_name'] = ""  # Missing name
        sample_claim_data['total_charges'] = 0  # Invalid charges
        
        claim = Claim(**sample_claim_data)
        line_items = []
        
        with patch.object(validation_engine, '_load_lookup_data', new_callable=AsyncMock):
            passed, results = await validation_engine.validate_claim(claim, line_items)
        
        assert passed is False  # Should fail due to errors
        
        # Check for specific errors
        error_results = [r for r in results if not r.passed and r.severity == ValidationSeverity.ERROR]
        assert len(error_results) > 0
        
        # Should have patient name error
        name_errors = [r for r in error_results if r.rule_id == 'PDR001']
        assert len(name_errors) > 0
        
        # Should have charges error
        charges_errors = [r for r in error_results if r.rule_id == 'FVR001']
        assert len(charges_errors) > 0
    
    @pytest.mark.asyncio
    async def test_validate_claim_exception_handling(self, validation_engine, sample_claim_data):
        """Test validation engine exception handling."""
        claim = Claim(**sample_claim_data)
        
        # Mock a rule to throw an exception
        original_rules = validation_engine.rules[RuleCategory.PATIENT_DEMOGRAPHICS]
        
        def failing_rule(context):
            raise ValueError("Test exception")
        
        # Add a rule that will fail
        validation_engine.rules[RuleCategory.PATIENT_DEMOGRAPHICS] = [
            {
                'id': 'FAIL001',
                'name': 'Failing Rule',
                'category': RuleCategory.PATIENT_DEMOGRAPHICS,
                'severity': ValidationSeverity.ERROR,
                'func': failing_rule,
                'field': 'test_field'
            }
        ]
        
        with patch.object(validation_engine, '_load_lookup_data', new_callable=AsyncMock):
            passed, results = await validation_engine.validate_claim(claim, [])
        
        # Should fail due to rule execution error
        assert passed is False
        
        # Should have a critical error result
        critical_errors = [r for r in results if r.severity == ValidationSeverity.CRITICAL]
        assert len(critical_errors) > 0
        
        critical_error = critical_errors[0]
        assert critical_error.rule_id == 'FAIL001'
        assert "Rule execution error" in critical_error.message
        
        # Restore original rules
        validation_engine.rules[RuleCategory.PATIENT_DEMOGRAPHICS] = original_rules
    
    def test_rule_count(self, validation_engine):
        """Test that validation engine has expected number of rules."""
        total_rules = sum(len(rules) for rules in validation_engine.rules.values())
        
        # Should have rules from all implemented categories
        assert total_rules >= 10  # At least 10 rules implemented
        assert validation_engine.rule_count == total_rules
        
        # Check that we have rules in expected categories
        assert RuleCategory.PATIENT_DEMOGRAPHICS in validation_engine.rules
        assert RuleCategory.FINANCIAL_VALIDATION in validation_engine.rules
        assert RuleCategory.CLINICAL_CODING in validation_engine.rules
        assert len(validation_engine.rules[RuleCategory.PATIENT_DEMOGRAPHICS]) >= 5
        assert len(validation_engine.rules[RuleCategory.FINANCIAL_VALIDATION]) >= 3


class TestPerformanceValidation:
    """Test validation engine performance."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_validation_performance(self, validation_engine, performance_test_data):
        """Test validation performance with large dataset."""
        import time
        
        # Create test claims
        claims_data = performance_test_data(1000)
        claims = [Claim(**claim_data) for claim_data in claims_data]
        
        # Mock lookup data loading
        with patch.object(validation_engine, '_load_lookup_data', new_callable=AsyncMock):
            start_time = time.perf_counter()
            
            # Validate all claims
            results = []
            for claim in claims:
                passed, claim_results = await validation_engine.validate_claim(claim, [])
                results.append((passed, claim_results))
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
        
        # Performance assertions
        claims_per_second = len(claims) / execution_time
        
        # Should validate at least 100 claims per second
        assert claims_per_second >= 100, f"Validation too slow: {claims_per_second:.2f} claims/sec"
        
        # All claims should have validation results
        assert len(results) == len(claims)
        
        print(f"Validated {len(claims)} claims in {execution_time:.3f}s ({claims_per_second:.2f} claims/sec)")
    
    @pytest.mark.performance
    def test_memory_usage(self, validation_engine, performance_test_data):
        """Test validation engine memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large dataset
        claims_data = performance_test_data(5000)
        claims = [Claim(**claim_data) for claim_data in claims_data]
        
        # Create contexts (this is memory intensive)
        contexts = []
        for claim in claims:
            context = ClaimValidationContext(
                claim=claim,
                line_items=[],
                patient_age=44,
                total_line_item_charges=Decimal('0'),
                line_item_count=0,
                service_days=1,
                unique_procedure_codes=set(),
                unique_diagnosis_codes=set(),
                is_emergency=False,
                is_outpatient=True,
                is_inpatient=False
            )
            contexts.append(context)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for 5000 claims)
        assert memory_increase < 100, f"Memory usage too high: {memory_increase:.2f}MB"
        
        print(f"Memory increase: {memory_increase:.2f}MB for {len(claims)} claims")
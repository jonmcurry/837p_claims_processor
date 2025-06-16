"""Integration tests for batch processing pipeline."""

import pytest
import asyncio
from datetime import datetime, date
from decimal import Decimal
from unittest.mock import AsyncMock, patch
import json

from src.processing.batch_processor.ultra_pipeline import (
    UltraHighPerformancePipeline,
    BatchMetadata,
    ProcessingResult
)
from src.core.database.models import Claim, ClaimLineItem, FailedClaim
from src.processing.validation.comprehensive_rules import ComprehensiveValidationEngine
from src.processing.ml_pipeline.advanced_predictor import AdvancedMLPredictor
from src.core.security.hipaa_security import security_manager
from src.monitoring.metrics.comprehensive_metrics import metrics_collector


@pytest.mark.integration
class TestBatchProcessingIntegration:
    """Integration tests for the complete batch processing pipeline."""
    
    @pytest.fixture
    async def pipeline(self, db_session, redis_client, mock_settings):
        """Create pipeline instance for testing."""
        pipeline = UltraHighPerformancePipeline()
        
        # Mock external dependencies
        with patch.object(pipeline, 'validation_engine') as mock_validation:
            mock_validation.validate_claim = AsyncMock(return_value=(True, []))
            
            with patch.object(pipeline, 'ml_predictor') as mock_ml:
                mock_ml.predict_batch_optimized = AsyncMock(return_value=[
                    {"should_process": True, "confidence": 0.95} for _ in range(100)
                ])
                
                yield pipeline
    
    @pytest.fixture
    def sample_batch_claims(self):
        """Generate sample claims for batch processing."""
        claims = []
        for i in range(100):  # 100 claims for integration testing
            claim_data = {
                "claim_id": f"INTEG_CLM_{i:06d}",
                "facility_id": f"FAC{(i % 5) + 1:03d}",
                "patient_account_number": f"PAT_{i:06d}",
                "patient_first_name": f"Patient{i}",
                "patient_last_name": "Integration",
                "patient_date_of_birth": date(1980, 1, 1),
                "admission_date": datetime.now().date(),
                "discharge_date": datetime.now().date(),
                "service_from_date": datetime.now().date(),
                "service_to_date": datetime.now().date(),
                "financial_class": "INPATIENT",
                "total_charges": Decimal(str(1000 + (i * 10))),
                "insurance_type": "MEDICARE",
                "billing_provider_npi": "1234567890",
                "billing_provider_name": "Dr. Integration Test",
                "primary_diagnosis_code": "Z00.00",
            }
            
            line_items = [
                {
                    "line_number": 1,
                    "procedure_code": "99213",
                    "units": 1,
                    "charge_amount": Decimal("150.00"),
                    "service_date": datetime.now().date()
                }
            ]
            
            claims.append((claim_data, line_items))
        
        return claims
    
    @pytest.mark.asyncio
    async def test_complete_batch_processing_workflow(
        self, 
        pipeline, 
        db_session, 
        redis_client, 
        sample_batch_claims,
        integration_test_setup
    ):
        """Test complete batch processing workflow from submission to completion."""
        # Create batch metadata
        batch_metadata = BatchMetadata(
            batch_id="INTEG_BATCH_001",
            facility_id="FAC001",
            source_system="INTEGRATION_TEST",
            total_claims=len(sample_batch_claims),
            submitted_by="integration_test_user"
        )
        
        # Submit batch for processing
        batch_result = await pipeline.process_batch_optimized(
            batch_metadata=batch_metadata,
            claims_data=sample_batch_claims,
            priority="high"
        )
        
        # Verify batch processing results
        assert batch_result is not None
        assert batch_result.batch_id == "INTEG_BATCH_001"
        assert batch_result.total_claims == len(sample_batch_claims)
        assert batch_result.processed_claims > 0
        
        # Verify claims were processed (should be in database)
        processed_claims = await db_session.execute(
            "SELECT COUNT(*) FROM claims WHERE claim_id LIKE 'INTEG_CLM_%'"
        )
        claim_count = processed_claims.scalar()
        assert claim_count > 0
        
        # Verify processing metrics were recorded
        # This would be checked through the metrics system in a real scenario
        assert batch_result.processing_time > 0
        assert batch_result.throughput > 0
    
    @pytest.mark.asyncio
    async def test_validation_integration(
        self, 
        pipeline, 
        db_session, 
        sample_batch_claims
    ):
        """Test integration between batch processor and validation engine."""
        validation_engine = ComprehensiveValidationEngine()
        
        # Process a smaller batch to test validation integration
        test_claims = sample_batch_claims[:10]
        
        # Create one claim with validation errors
        invalid_claim_data, line_items = test_claims[0]
        invalid_claim_data["patient_first_name"] = ""  # This should fail validation
        invalid_claim_data["total_charges"] = Decimal("0")  # This should also fail
        
        batch_metadata = BatchMetadata(
            batch_id="VALIDATION_BATCH_001",
            facility_id="FAC001",
            source_system="VALIDATION_TEST",
            total_claims=len(test_claims),
            submitted_by="validation_test_user"
        )
        
        # Mock the validation engine to actually validate
        with patch.object(pipeline, 'validation_engine', validation_engine):
            batch_result = await pipeline.process_batch_optimized(
                batch_metadata=batch_metadata,
                claims_data=test_claims,
                priority="high"
            )
        
        # Should have some failed claims due to validation errors
        assert batch_result.failed_claims > 0
        
        # Check that failed claims were recorded in the database
        failed_claims = await db_session.execute(
            "SELECT COUNT(*) FROM failed_claims WHERE batch_id = 'VALIDATION_BATCH_001'"
        )
        failed_count = failed_claims.scalar()
        assert failed_count > 0
    
    @pytest.mark.asyncio
    async def test_ml_integration(
        self, 
        pipeline, 
        db_session, 
        sample_batch_claims,
        temp_model_dir
    ):
        """Test integration with ML prediction pipeline."""
        ml_predictor = AdvancedMLPredictor()
        
        # Mock model loading since we don't have actual models in test
        with patch.object(ml_predictor, '_load_models') as mock_load:
            mock_load.return_value = None
            
            with patch.object(ml_predictor, 'predict_batch_optimized') as mock_predict:
                # Simulate ML rejecting some claims
                predictions = []
                for i, _ in enumerate(sample_batch_claims[:20]):
                    predictions.append({
                        "should_process": i % 10 != 0,  # Reject every 10th claim
                        "confidence": 0.85 if i % 10 != 0 else 0.30,
                        "reason": "Validation passed" if i % 10 != 0 else "High risk claim"
                    })
                mock_predict.return_value = predictions
                
                batch_metadata = BatchMetadata(
                    batch_id="ML_BATCH_001",
                    facility_id="FAC001",
                    source_system="ML_TEST",
                    total_claims=20,
                    submitted_by="ml_test_user"
                )
                
                with patch.object(pipeline, 'ml_predictor', ml_predictor):
                    batch_result = await pipeline.process_batch_optimized(
                        batch_metadata=batch_metadata,
                        claims_data=sample_batch_claims[:20],
                        priority="high"
                    )
                
                # Should have some ML-rejected claims
                assert batch_result.processed_claims < batch_result.total_claims
                
                # Verify ML predictions were called
                mock_predict.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_security_integration(
        self, 
        pipeline, 
        db_session, 
        sample_batch_claims,
        mock_security_manager
    ):
        """Test integration with security and audit logging."""
        
        batch_metadata = BatchMetadata(
            batch_id="SECURITY_BATCH_001",
            facility_id="FAC001",
            source_system="SECURITY_TEST",
            total_claims=5,
            submitted_by="security_test_user"
        )
        
        # Mock security manager
        with patch('src.core.security.hipaa_security.security_manager', mock_security_manager):
            batch_result = await pipeline.process_batch_optimized(
                batch_metadata=batch_metadata,
                claims_data=sample_batch_claims[:5],
                priority="high"
            )
        
        # Verify audit logging was called
        assert mock_security_manager.audit_logger.log_system_event.called
        
        # Verify PHI access was logged
        call_args = mock_security_manager.audit_logger.log_system_event.call_args_list
        assert any("batch_processing" in str(call) for call in call_args)
    
    @pytest.mark.asyncio 
    async def test_metrics_integration(
        self, 
        pipeline, 
        db_session, 
        sample_batch_claims,
        mock_metrics_collector
    ):
        """Test integration with metrics collection."""
        
        batch_metadata = BatchMetadata(
            batch_id="METRICS_BATCH_001",
            facility_id="FAC001",
            source_system="METRICS_TEST",
            total_claims=10,
            submitted_by="metrics_test_user"
        )
        
        with patch('src.monitoring.metrics.comprehensive_metrics.metrics_collector', mock_metrics_collector):
            batch_result = await pipeline.process_batch_optimized(
                batch_metadata=batch_metadata,
                claims_data=sample_batch_claims[:10],
                priority="high"
            )
        
        # Verify metrics were recorded
        assert mock_metrics_collector.record_batch_processing.called
        assert mock_metrics_collector.record_claim_processed.called
        
        # Check batch processing metrics
        batch_calls = mock_metrics_collector.record_batch_processing.call_args_list
        assert len(batch_calls) > 0
        
        # Verify the metrics contain expected data
        batch_call_args = batch_calls[0][1]  # Get keyword arguments
        assert batch_call_args['batch_id'] == "METRICS_BATCH_001"
        assert batch_call_args['total_claims'] == 10
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(
        self, 
        pipeline, 
        db_session, 
        sample_batch_claims
    ):
        """Test error handling and recovery mechanisms."""
        
        # Create a batch with some problematic claims
        problematic_claims = sample_batch_claims[:5]
        
        # Make one claim cause a processing error
        bad_claim_data, line_items = problematic_claims[2]
        bad_claim_data["total_charges"] = "invalid_amount"  # This should cause an error
        
        batch_metadata = BatchMetadata(
            batch_id="ERROR_BATCH_001",
            facility_id="FAC001",
            source_system="ERROR_TEST",
            total_claims=len(problematic_claims),
            submitted_by="error_test_user"
        )
        
        # Process batch - should handle errors gracefully
        batch_result = await pipeline.process_batch_optimized(
            batch_metadata=batch_metadata,
            claims_data=problematic_claims,
            priority="high"
        )
        
        # Batch should complete even with some errors
        assert batch_result is not None
        assert batch_result.batch_id == "ERROR_BATCH_001"
        
        # Should have some failed claims due to processing errors
        assert batch_result.failed_claims > 0
        
        # Verify error was properly logged and claim was marked as failed
        failed_claims = await db_session.execute(
            "SELECT * FROM failed_claims WHERE batch_id = 'ERROR_BATCH_001'"
        )
        failed_records = failed_claims.fetchall()
        assert len(failed_records) > 0
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_concurrent_batch_processing(
        self, 
        pipeline, 
        db_session, 
        sample_batch_claims
    ):
        """Test concurrent processing of multiple batches."""
        
        # Create multiple small batches
        batch_size = 20
        num_batches = 3
        
        async def process_single_batch(batch_num):
            batch_claims = sample_batch_claims[batch_num * batch_size:(batch_num + 1) * batch_size]
            
            batch_metadata = BatchMetadata(
                batch_id=f"CONCURRENT_BATCH_{batch_num:03d}",
                facility_id="FAC001",
                source_system="CONCURRENT_TEST",
                total_claims=len(batch_claims),
                submitted_by=f"concurrent_test_user_{batch_num}"
            )
            
            return await pipeline.process_batch_optimized(
                batch_metadata=batch_metadata,
                claims_data=batch_claims,
                priority="medium"
            )
        
        # Process batches concurrently
        batch_tasks = [process_single_batch(i) for i in range(num_batches)]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # All batches should complete successfully
        for i, result in enumerate(batch_results):
            assert not isinstance(result, Exception), f"Batch {i} failed: {result}"
            assert result.batch_id == f"CONCURRENT_BATCH_{i:03d}"
            assert result.processed_claims > 0
        
        # Total processed claims should equal sum of all batches
        total_processed = sum(result.processed_claims for result in batch_results)
        assert total_processed > 0
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_performance_under_load(
        self, 
        pipeline, 
        db_session, 
        performance_test_data,
        test_helpers
    ):
        """Test performance under realistic load conditions."""
        import time
        
        # Generate larger dataset for performance testing
        claims_data = performance_test_data(1000)  # 1000 claims
        
        batch_metadata = BatchMetadata(
            batch_id="PERFORMANCE_BATCH_001",
            facility_id="FAC001",
            source_system="PERFORMANCE_TEST",
            total_claims=len(claims_data),
            submitted_by="performance_test_user"
        )
        
        # Convert to the format expected by the pipeline
        formatted_claims = []
        for claim_data in claims_data:
            line_items = [{
                "line_number": 1,
                "procedure_code": "99213",
                "units": 1,
                "charge_amount": Decimal("150.00"),
                "service_date": datetime.now().date()
            }]
            formatted_claims.append((claim_data, line_items))
        
        # Measure processing time
        start_time = time.perf_counter()
        
        batch_result = await pipeline.process_batch_optimized(
            batch_metadata=batch_metadata,
            claims_data=formatted_claims,
            priority="high"
        )
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # Performance assertions
        test_helpers.assert_performance_target(
            execution_time=execution_time,
            target_claims=1000,
            target_time=15.0  # Should process 1000 claims in under 15 seconds
        )
        
        # Verify processing results
        assert batch_result.processed_claims > 0
        assert batch_result.throughput > 0
        
        print(f"Processed {batch_result.processed_claims} claims in {execution_time:.3f}s "
              f"({batch_result.throughput:.2f} claims/sec)")
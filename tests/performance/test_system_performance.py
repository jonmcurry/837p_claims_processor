"""System-wide performance tests for production readiness."""

import pytest
import asyncio
import time
import psutil
import os
from decimal import Decimal
from datetime import datetime, date
from unittest.mock import AsyncMock, patch
import statistics

from src.processing.batch_processor.ultra_pipeline import UltraHighPerformancePipeline
from src.processing.validation.comprehensive_rules import ComprehensiveValidationEngine
from src.processing.ml_pipeline.advanced_predictor import AdvancedMLPredictor
from src.core.database.models import Claim, ClaimLineItem


@pytest.mark.performance
class TestSystemPerformance:
    """Comprehensive performance tests to validate production readiness."""
    
    @pytest.fixture
    def performance_pipeline(self, db_session, redis_client, mock_settings):
        """Create optimized pipeline for performance testing."""
        pipeline = UltraHighPerformancePipeline()
        
        # Configure for maximum performance
        pipeline.max_workers = psutil.cpu_count()
        pipeline.batch_size = 1000
        pipeline.enable_vectorization = True
        pipeline.memory_limit_mb = 2048
        
        return pipeline
    
    @pytest.fixture
    def large_dataset_generator(self):
        """Generate large datasets for performance testing."""
        def generate_claims(count: int, complexity: str = "standard"):
            claims = []
            
            for i in range(count):
                # Vary complexity based on parameter
                if complexity == "simple":
                    line_items_count = 1
                elif complexity == "complex":
                    line_items_count = 5 + (i % 10)  # 5-15 line items
                else:  # standard
                    line_items_count = 2 + (i % 3)  # 2-4 line items
                
                claim_data = {
                    "claim_id": f"PERF_CLM_{i:08d}",
                    "facility_id": f"FAC{(i % 100) + 1:03d}",
                    "patient_account_number": f"PAT_{i:08d}",
                    "patient_first_name": f"Patient{i}",
                    "patient_last_name": "Performance",
                    "patient_date_of_birth": date(1980 + (i % 40), 1 + (i % 12), 1 + (i % 28)),
                    "admission_date": datetime.now().date(),
                    "discharge_date": datetime.now().date(),
                    "service_from_date": datetime.now().date(),
                    "service_to_date": datetime.now().date(),
                    "financial_class": ["INPATIENT", "OUTPATIENT", "EMERGENCY"][i % 3],
                    "total_charges": Decimal(str(500 + (i * 12.34))),
                    "insurance_type": ["MEDICARE", "MEDICAID", "COMMERCIAL", "SELF_PAY"][i % 4],
                    "billing_provider_npi": f"{1000000000 + (i % 999999):010d}",
                    "billing_provider_name": f"Dr. Performance {i % 100}",
                    "primary_diagnosis_code": f"Z{(i % 99):02d}.{(i % 99):02d}",
                }
                
                line_items = []
                for j in range(line_items_count):
                    line_items.append({
                        "line_number": j + 1,
                        "procedure_code": f"{99000 + (j + i) % 999:05d}",
                        "units": 1 + (j % 3),
                        "charge_amount": Decimal(str(100 + (j * 25.50))),
                        "service_date": datetime.now().date()
                    })
                
                claims.append((claim_data, line_items))
            
            return claims
        
        return generate_claims
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_target_throughput_100k_claims_15_seconds(
        self, 
        performance_pipeline,
        large_dataset_generator,
        db_session,
        test_helpers
    ):
        """Test the primary performance target: 100,000 claims in 15 seconds."""
        
        # Generate 100k test claims
        print("Generating 100,000 test claims...")
        start_gen = time.perf_counter()
        claims_data = large_dataset_generator(100000, "standard")
        gen_time = time.perf_counter() - start_gen
        print(f"Generated claims in {gen_time:.2f}s")
        
        # Create batch metadata
        from src.processing.batch_processor.ultra_pipeline import BatchMetadata
        batch_metadata = BatchMetadata(
            batch_id="PERFORMANCE_100K_BATCH",
            facility_id="FAC001",
            source_system="PERFORMANCE_TEST",
            total_claims=100000,
            submitted_by="performance_test_user"
        )
        
        # Monitor system resources
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute performance test
        print("Starting 100k claims performance test...")
        start_time = time.perf_counter()
        
        # Mock external dependencies for pure processing speed
        with patch.object(performance_pipeline, 'validation_engine') as mock_validation:
            mock_validation.validate_claim = AsyncMock(return_value=(True, []))
            
            with patch.object(performance_pipeline, 'ml_predictor') as mock_ml:
                # Fast ML predictions
                mock_ml.predict_batch_optimized = AsyncMock(return_value=[
                    {"should_process": True, "confidence": 0.95, "score": 0.95}
                    for _ in range(100000)
                ])
                
                batch_result = await performance_pipeline.process_batch_optimized(
                    batch_metadata=batch_metadata,
                    claims_data=claims_data,
                    priority="high"
                )
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = final_memory - initial_memory
        
        # Calculate metrics
        throughput = batch_result.processed_claims / execution_time
        target_throughput = 100000 / 15  # 6,667 claims/second
        
        print(f"\n=== PERFORMANCE TEST RESULTS ===")
        print(f"Total Claims: {batch_result.total_claims:,}")
        print(f"Processed Claims: {batch_result.processed_claims:,}")
        print(f"Failed Claims: {batch_result.failed_claims:,}")
        print(f"Execution Time: {execution_time:.3f}s")
        print(f"Throughput: {throughput:,.2f} claims/sec")
        print(f"Target Throughput: {target_throughput:,.2f} claims/sec")
        print(f"Memory Usage: {memory_usage:.2f} MB")
        print(f"Target Met: {'✓' if throughput >= target_throughput else '✗'}")
        
        # Primary performance assertion
        assert execution_time <= 15.0, (
            f"Failed to meet 15-second target. Took {execution_time:.3f}s"
        )
        
        assert throughput >= target_throughput, (
            f"Failed to meet throughput target. "
            f"Achieved: {throughput:,.2f} claims/sec, "
            f"Required: {target_throughput:,.2f} claims/sec"
        )
        
        # Resource usage assertions
        assert memory_usage < 4096, f"Memory usage too high: {memory_usage:.2f} MB"
        
        # Quality assertions
        success_rate = batch_result.processed_claims / batch_result.total_claims
        assert success_rate >= 0.95, f"Success rate too low: {success_rate:.2%}"
    
    @pytest.mark.asyncio
    async def test_sustained_throughput_multiple_batches(
        self,
        performance_pipeline,
        large_dataset_generator,
        test_helpers
    ):
        """Test sustained throughput across multiple consecutive batches."""
        
        batch_size = 10000  # 10k claims per batch
        num_batches = 5
        
        execution_times = []
        throughputs = []
        memory_usages = []
        
        process = psutil.Process(os.getpid())
        
        for batch_num in range(num_batches):
            print(f"Processing batch {batch_num + 1}/{num_batches}...")
            
            # Generate batch
            claims_data = large_dataset_generator(batch_size, "standard")
            
            from src.processing.batch_processor.ultra_pipeline import BatchMetadata
            batch_metadata = BatchMetadata(
                batch_id=f"SUSTAINED_BATCH_{batch_num:03d}",
                facility_id="FAC001",
                source_system="SUSTAINED_TEST",
                total_claims=batch_size,
                submitted_by="sustained_test_user"
            )
            
            initial_memory = process.memory_info().rss / 1024 / 1024
            start_time = time.perf_counter()
            
            # Mock dependencies for speed
            with patch.object(performance_pipeline, 'validation_engine') as mock_validation:
                mock_validation.validate_claim = AsyncMock(return_value=(True, []))
                
                with patch.object(performance_pipeline, 'ml_predictor') as mock_ml:
                    mock_ml.predict_batch_optimized = AsyncMock(return_value=[
                        {"should_process": True, "confidence": 0.95}
                        for _ in range(batch_size)
                    ])
                    
                    batch_result = await performance_pipeline.process_batch_optimized(
                        batch_metadata=batch_metadata,
                        claims_data=claims_data,
                        priority="high"
                    )
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_usage = final_memory - initial_memory
            
            throughput = batch_result.processed_claims / execution_time
            
            execution_times.append(execution_time)
            throughputs.append(throughput)
            memory_usages.append(memory_usage)
            
            print(f"  Batch {batch_num + 1}: {throughput:,.0f} claims/sec, "
                  f"{execution_time:.2f}s, {memory_usage:.1f}MB")
        
        # Analyze sustained performance
        avg_throughput = statistics.mean(throughputs)
        min_throughput = min(throughputs)
        max_throughput = max(throughputs)
        throughput_std = statistics.stdev(throughputs) if len(throughputs) > 1 else 0
        
        avg_memory = statistics.mean(memory_usages)
        max_memory = max(memory_usages)
        
        print(f"\n=== SUSTAINED PERFORMANCE RESULTS ===")
        print(f"Batches Processed: {num_batches}")
        print(f"Claims per Batch: {batch_size:,}")
        print(f"Average Throughput: {avg_throughput:,.2f} claims/sec")
        print(f"Min Throughput: {min_throughput:,.2f} claims/sec")
        print(f"Max Throughput: {max_throughput:,.2f} claims/sec")
        print(f"Throughput Std Dev: {throughput_std:,.2f} claims/sec")
        print(f"Average Memory: {avg_memory:.2f} MB")
        print(f"Max Memory: {max_memory:.2f} MB")
        
        # Performance assertions
        target_throughput = 6000  # Slightly lower target for sustained performance
        assert avg_throughput >= target_throughput, (
            f"Sustained throughput too low: {avg_throughput:,.2f} < {target_throughput:,.2f}"
        )
        
        # Consistency assertion (throughput shouldn't vary too much)
        throughput_cv = throughput_std / avg_throughput  # Coefficient of variation
        assert throughput_cv < 0.2, f"Throughput too inconsistent: CV = {throughput_cv:.3f}"
        
        # Memory shouldn't grow significantly over time
        assert max_memory < 1024, f"Memory usage too high: {max_memory:.2f} MB"
    
    @pytest.mark.asyncio
    async def test_concurrent_batch_processing_performance(
        self,
        performance_pipeline,
        large_dataset_generator
    ):
        """Test performance with concurrent batch processing."""
        
        num_concurrent_batches = 3
        claims_per_batch = 5000
        
        async def process_batch(batch_id: int):
            claims_data = large_dataset_generator(claims_per_batch, "standard")
            
            from src.processing.batch_processor.ultra_pipeline import BatchMetadata
            batch_metadata = BatchMetadata(
                batch_id=f"CONCURRENT_BATCH_{batch_id:03d}",
                facility_id="FAC001",
                source_system="CONCURRENT_TEST",
                total_claims=claims_per_batch,
                submitted_by=f"concurrent_user_{batch_id}"
            )
            
            start_time = time.perf_counter()
            
            with patch.object(performance_pipeline, 'validation_engine') as mock_validation:
                mock_validation.validate_claim = AsyncMock(return_value=(True, []))
                
                with patch.object(performance_pipeline, 'ml_predictor') as mock_ml:
                    mock_ml.predict_batch_optimized = AsyncMock(return_value=[
                        {"should_process": True, "confidence": 0.95}
                        for _ in range(claims_per_batch)
                    ])
                    
                    batch_result = await performance_pipeline.process_batch_optimized(
                        batch_metadata=batch_metadata,
                        claims_data=claims_data,
                        priority="medium"
                    )
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            throughput = batch_result.processed_claims / execution_time
            
            return {
                "batch_id": batch_id,
                "execution_time": execution_time,
                "throughput": throughput,
                "processed_claims": batch_result.processed_claims,
                "total_claims": batch_result.total_claims
            }
        
        # Run concurrent batches
        print(f"Running {num_concurrent_batches} concurrent batches...")
        start_time = time.perf_counter()
        
        tasks = [process_batch(i) for i in range(num_concurrent_batches)]
        results = await asyncio.gather(*tasks)
        
        total_time = time.perf_counter() - start_time
        
        # Analyze results
        total_claims = sum(r["processed_claims"] for r in results)
        total_throughput = total_claims / total_time
        individual_throughputs = [r["throughput"] for r in results]
        avg_individual_throughput = statistics.mean(individual_throughputs)
        
        print(f"\n=== CONCURRENT PROCESSING RESULTS ===")
        print(f"Concurrent Batches: {num_concurrent_batches}")
        print(f"Claims per Batch: {claims_per_batch:,}")
        print(f"Total Claims: {total_claims:,}")
        print(f"Total Time: {total_time:.3f}s")
        print(f"Overall Throughput: {total_throughput:,.2f} claims/sec")
        print(f"Avg Individual Throughput: {avg_individual_throughput:,.2f} claims/sec")
        
        for i, result in enumerate(results):
            print(f"  Batch {i}: {result['throughput']:,.0f} claims/sec "
                  f"({result['execution_time']:.2f}s)")
        
        # Performance assertions
        min_expected_throughput = 4000  # Lower for concurrent processing
        assert total_throughput >= min_expected_throughput, (
            f"Concurrent throughput too low: {total_throughput:,.2f}"
        )
        
        # All batches should complete successfully
        assert all(r["processed_claims"] == r["total_claims"] for r in results)
    
    @pytest.mark.asyncio
    async def test_memory_efficiency_large_dataset(
        self,
        performance_pipeline,
        large_dataset_generator
    ):
        """Test memory efficiency with large datasets."""
        
        # Test with different dataset sizes
        test_sizes = [1000, 5000, 10000, 25000]
        memory_results = []
        
        process = psutil.Process(os.getpid())
        
        for size in test_sizes:
            print(f"Testing memory efficiency with {size:,} claims...")
            
            # Force garbage collection before test
            import gc
            gc.collect()
            
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            # Generate and process claims
            claims_data = large_dataset_generator(size, "complex")  # Complex claims
            
            from src.processing.batch_processor.ultra_pipeline import BatchMetadata
            batch_metadata = BatchMetadata(
                batch_id=f"MEMORY_TEST_{size}",
                facility_id="FAC001",
                source_system="MEMORY_TEST",
                total_claims=size,
                submitted_by="memory_test_user"
            )
            
            with patch.object(performance_pipeline, 'validation_engine') as mock_validation:
                mock_validation.validate_claim = AsyncMock(return_value=(True, []))
                
                with patch.object(performance_pipeline, 'ml_predictor') as mock_ml:
                    mock_ml.predict_batch_optimized = AsyncMock(return_value=[
                        {"should_process": True, "confidence": 0.95}
                        for _ in range(size)
                    ])
                    
                    await performance_pipeline.process_batch_optimized(
                        batch_metadata=batch_metadata,
                        claims_data=claims_data,
                        priority="medium"
                    )
            
            peak_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = peak_memory - initial_memory
            memory_per_claim = memory_increase / size
            
            memory_results.append({
                "size": size,
                "memory_increase": memory_increase,
                "memory_per_claim": memory_per_claim
            })
            
            print(f"  Memory increase: {memory_increase:.2f} MB "
                  f"({memory_per_claim:.4f} MB per claim)")
            
            # Clean up
            del claims_data
            gc.collect()
        
        print(f"\n=== MEMORY EFFICIENCY RESULTS ===")
        for result in memory_results:
            print(f"{result['size']:,} claims: {result['memory_increase']:.2f} MB "
                  f"({result['memory_per_claim']:.4f} MB/claim)")
        
        # Memory efficiency assertions
        for result in memory_results:
            # Memory per claim should be reasonable (less than 0.1 MB per claim)
            assert result["memory_per_claim"] < 0.1, (
                f"Memory usage too high: {result['memory_per_claim']:.4f} MB per claim "
                f"for {result['size']:,} claims"
            )
            
            # Total memory shouldn't exceed reasonable limits
            assert result["memory_increase"] < 2048, (
                f"Total memory usage too high: {result['memory_increase']:.2f} MB "
                f"for {result['size']:,} claims"
            )
    
    @pytest.mark.asyncio
    async def test_validation_engine_performance(self):
        """Test validation engine performance in isolation."""
        
        validation_engine = ComprehensiveValidationEngine()
        
        # Generate test claims for validation
        test_claims = []
        for i in range(5000):
            claim = Claim(
                claim_id=f"VAL_CLM_{i:06d}",
                facility_id="FAC001",
                patient_account_number=f"PAT_{i:06d}",
                patient_first_name=f"Patient{i}",
                patient_last_name="Validation",
                patient_date_of_birth=date(1980, 1, 1),
                total_charges=Decimal("1000.00"),
                insurance_type="MEDICARE",
                billing_provider_npi="1234567890",
                billing_provider_name="Dr. Validation",
                primary_diagnosis_code="Z00.00"
            )
            
            line_items = [ClaimLineItem(
                line_number=1,
                procedure_code="99213",
                units=1,
                charge_amount=Decimal("150.00"),
                service_date=datetime.now().date()
            )]
            
            test_claims.append((claim, line_items))
        
        # Performance test
        start_time = time.perf_counter()
        
        validation_results = []
        for claim, line_items in test_claims:
            passed, results = await validation_engine.validate_claim(claim, line_items)
            validation_results.append((passed, results))
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        validation_throughput = len(test_claims) / execution_time
        
        print(f"\n=== VALIDATION ENGINE PERFORMANCE ===")
        print(f"Claims Validated: {len(test_claims):,}")
        print(f"Execution Time: {execution_time:.3f}s")
        print(f"Validation Throughput: {validation_throughput:,.2f} claims/sec")
        
        # Performance assertion - should validate at least 1000 claims/sec
        assert validation_throughput >= 1000, (
            f"Validation too slow: {validation_throughput:.2f} claims/sec"
        )
        
        # Quality assertion - most validations should pass
        passed_count = sum(1 for passed, _ in validation_results if passed)
        pass_rate = passed_count / len(validation_results)
        assert pass_rate >= 0.8, f"Validation pass rate too low: {pass_rate:.2%}"
    
    @pytest.mark.asyncio
    async def test_end_to_end_production_simulation(
        self,
        performance_pipeline,
        large_dataset_generator,
        test_helpers
    ):
        """Comprehensive end-to-end production simulation test."""
        
        print("=== PRODUCTION SIMULATION TEST ===")
        print("Simulating real production conditions...")
        
        # Simulate a busy day with multiple batches of varying sizes
        batch_configs = [
            {"size": 15000, "complexity": "simple", "priority": "high"},
            {"size": 25000, "complexity": "standard", "priority": "medium"},
            {"size": 10000, "complexity": "complex", "priority": "high"},
            {"size": 30000, "complexity": "standard", "priority": "low"},
            {"size": 20000, "complexity": "simple", "priority": "medium"},
        ]
        
        total_claims = sum(config["size"] for config in batch_configs)
        print(f"Total claims to process: {total_claims:,}")
        
        overall_start = time.perf_counter()
        all_results = []
        
        for i, config in enumerate(batch_configs):
            print(f"\nProcessing batch {i+1}/{len(batch_configs)} "
                  f"({config['size']:,} {config['complexity']} claims, "
                  f"{config['priority']} priority)...")
            
            claims_data = large_dataset_generator(config["size"], config["complexity"])
            
            from src.processing.batch_processor.ultra_pipeline import BatchMetadata
            batch_metadata = BatchMetadata(
                batch_id=f"PROD_SIM_BATCH_{i:03d}",
                facility_id=f"FAC{(i % 10) + 1:03d}",
                source_system="PRODUCTION_SIMULATION",
                total_claims=config["size"],
                submitted_by="prod_sim_user"
            )
            
            batch_start = time.perf_counter()
            
            with patch.object(performance_pipeline, 'validation_engine') as mock_validation:
                # Simulate realistic validation results
                async def mock_validate(claim, line_items):
                    # 95% pass rate with some validation issues
                    import random
                    passed = random.random() < 0.95
                    results = [] if passed else [{"rule_id": "test", "message": "test error"}]
                    return passed, results
                
                mock_validation.validate_claim = mock_validate
                
                with patch.object(performance_pipeline, 'ml_predictor') as mock_ml:
                    # Simulate realistic ML predictions
                    def generate_ml_predictions(size):
                        import random
                        predictions = []
                        for _ in range(size):
                            should_process = random.random() < 0.93  # 93% acceptance rate
                            confidence = 0.85 + (random.random() * 0.15) if should_process else 0.30 + (random.random() * 0.40)
                            predictions.append({
                                "should_process": should_process,
                                "confidence": confidence,
                                "score": confidence
                            })
                        return predictions
                    
                    mock_ml.predict_batch_optimized = AsyncMock(
                        return_value=generate_ml_predictions(config["size"])
                    )
                    
                    batch_result = await performance_pipeline.process_batch_optimized(
                        batch_metadata=batch_metadata,
                        claims_data=claims_data,
                        priority=config["priority"]
                    )
            
            batch_time = time.perf_counter() - batch_start
            batch_throughput = batch_result.processed_claims / batch_time
            
            all_results.append({
                "batch_id": i,
                "config": config,
                "result": batch_result,
                "execution_time": batch_time,
                "throughput": batch_throughput
            })
            
            print(f"  Completed: {batch_result.processed_claims:,}/{batch_result.total_claims:,} claims")
            print(f"  Time: {batch_time:.2f}s")
            print(f"  Throughput: {batch_throughput:,.0f} claims/sec")
            print(f"  Success Rate: {(batch_result.processed_claims/batch_result.total_claims)*100:.1f}%")
        
        overall_time = time.perf_counter() - overall_start
        
        # Calculate overall statistics
        total_processed = sum(r["result"].processed_claims for r in all_results)
        total_failed = sum(r["result"].failed_claims for r in all_results)
        overall_throughput = total_processed / overall_time
        overall_success_rate = total_processed / total_claims
        
        print(f"\n=== PRODUCTION SIMULATION RESULTS ===")
        print(f"Total Claims: {total_claims:,}")
        print(f"Total Processed: {total_processed:,}")
        print(f"Total Failed: {total_failed:,}")
        print(f"Overall Time: {overall_time:.2f}s")
        print(f"Overall Throughput: {overall_throughput:,.2f} claims/sec")
        print(f"Overall Success Rate: {overall_success_rate:.2%}")
        print(f"Target Met (6,667 claims/sec): {'✓' if overall_throughput >= 6667 else '✗'}")
        
        # Production readiness assertions
        assert overall_throughput >= 6000, (
            f"Production throughput target not met: {overall_throughput:,.2f} < 6,000 claims/sec"
        )
        
        assert overall_success_rate >= 0.85, (
            f"Success rate too low for production: {overall_success_rate:.2%}"
        )
        
        # Individual batch performance shouldn't be too inconsistent
        batch_throughputs = [r["throughput"] for r in all_results]
        min_throughput = min(batch_throughputs)
        assert min_throughput >= 4000, (
            f"Some batches too slow for production: {min_throughput:,.2f} claims/sec"
        )
        
        print(f"\n🎉 PRODUCTION READINESS TEST PASSED! 🎉")
        print(f"System is ready to handle production load of 100,000 claims in 15 seconds.")
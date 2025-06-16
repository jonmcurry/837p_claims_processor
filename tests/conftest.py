"""Pytest configuration and fixtures for comprehensive testing."""

import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import Mock, AsyncMock
import tempfile
import shutil
from pathlib import Path

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import StaticPool
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

from src.core.config import settings
from src.core.database.base import Base, get_postgres_session
from src.core.database.models import *
from src.core.security.hipaa_security import security_manager
from src.processing.ml_pipeline.advanced_predictor import advanced_predictor
from src.monitoring.metrics.comprehensive_metrics import metrics_collector


# Test Configuration
class TestConfig:
    """Test-specific configuration."""
    DATABASE_URL = "postgresql+asyncpg://test:test@localhost:5432/test_claims"
    REDIS_URL = "redis://localhost:6379/1"
    TESTING = True
    DEBUG = True
    ENCRYPTION_KEY = "test-encryption-key-32-characters-long"
    JWT_SECRET_KEY = "test-jwt-secret-key"
    ML_MODEL_PATH = "/tmp/test_models"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def postgres_container() -> AsyncGenerator[PostgresContainer, None]:
    """Start PostgreSQL test container."""
    with PostgresContainer("postgres:15") as postgres:
        postgres.start()
        yield postgres


@pytest.fixture(scope="session")
async def redis_container() -> AsyncGenerator[RedisContainer, None]:
    """Start Redis test container."""
    with RedisContainer("redis:7-alpine") as redis_c:
        redis_c.start()
        yield redis_c


@pytest.fixture(scope="session")
async def test_db_engine(postgres_container):
    """Create test database engine."""
    database_url = postgres_container.get_connection_url().replace("psycopg2", "asyncpg")
    
    engine = create_async_engine(
        database_url,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest.fixture
async def db_session(test_db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    async with AsyncSession(test_db_engine, expire_on_commit=False) as session:
        yield session
        await session.rollback()


@pytest.fixture
async def redis_client(redis_container) -> AsyncGenerator[redis.Redis, None]:
    """Create test Redis client."""
    redis_url = redis_container.get_connection_url()
    client = redis.from_url(redis_url)
    
    yield client
    
    await client.flushdb()
    await client.close()


@pytest.fixture
def temp_model_dir() -> Generator[Path, None, None]:
    """Create temporary directory for ML models."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_settings(monkeypatch, temp_model_dir):
    """Mock settings for testing."""
    test_config = TestConfig()
    
    for attr_name in dir(test_config):
        if not attr_name.startswith('_'):
            monkeypatch.setattr(settings, attr_name, getattr(test_config, attr_name))
    
    monkeypatch.setattr(settings, "ML_MODEL_PATH", str(temp_model_dir))


# Test Data Fixtures
@pytest.fixture
def sample_claim_data():
    """Sample claim data for testing."""
    return {
        "claim_id": "CLM001",
        "facility_id": "FAC001",
        "patient_account_number": "PAT001",
        "patient_first_name": "John",
        "patient_last_name": "Doe",
        "patient_date_of_birth": "1980-01-01",
        "admission_date": "2024-01-01",
        "discharge_date": "2024-01-03",
        "service_from_date": "2024-01-01",
        "service_to_date": "2024-01-03",
        "financial_class": "INPATIENT",
        "total_charges": 5000.00,
        "insurance_type": "MEDICARE",
        "billing_provider_npi": "1234567890",
        "billing_provider_name": "Dr. Smith",
        "primary_diagnosis_code": "Z00.00",
    }


@pytest.fixture
def sample_line_items():
    """Sample claim line items for testing."""
    return [
        {
            "line_number": 1,
            "service_date": "2024-01-01",
            "procedure_code": "99213",
            "procedure_description": "Office visit",
            "units": 1,
            "charge_amount": 150.00,
        },
        {
            "line_number": 2,
            "service_date": "2024-01-02",
            "procedure_code": "80053",
            "procedure_description": "Comprehensive metabolic panel",
            "units": 1,
            "charge_amount": 75.00,
        },
    ]


@pytest.fixture
def sample_batch_data(sample_claim_data):
    """Sample batch data for testing."""
    return {
        "batch_id": "BATCH001",
        "facility_id": "FAC001",
        "source_system": "TEST_SYSTEM",
        "total_claims": 1,
        "submitted_by": "test_user",
        "claims": [sample_claim_data],
    }


@pytest.fixture
async def test_claim(db_session, sample_claim_data) -> Claim:
    """Create test claim in database."""
    claim = Claim(**sample_claim_data)
    db_session.add(claim)
    await db_session.commit()
    await db_session.refresh(claim)
    return claim


@pytest.fixture
async def test_failed_claim(db_session, test_claim) -> FailedClaim:
    """Create test failed claim in database."""
    failed_claim = FailedClaim(
        original_claim_id=test_claim.id,
        claim_reference=test_claim.claim_id,
        facility_id=test_claim.facility_id,
        failure_category=FailureCategory.VALIDATION_ERROR,
        failure_reason="Test validation error",
        failure_details={"error": "test error"},
        claim_data={"claim_id": test_claim.claim_id},
        charge_amount=test_claim.total_charges,
    )
    db_session.add(failed_claim)
    await db_session.commit()
    await db_session.refresh(failed_claim)
    return failed_claim


@pytest.fixture
async def test_batch(db_session, sample_batch_data) -> BatchMetadata:
    """Create test batch in database."""
    batch = BatchMetadata(
        batch_id=sample_batch_data["batch_id"],
        facility_id=sample_batch_data["facility_id"],
        source_system=sample_batch_data["source_system"],
        total_claims=sample_batch_data["total_claims"],
        submitted_by=sample_batch_data["submitted_by"],
    )
    db_session.add(batch)
    await db_session.commit()
    await db_session.refresh(batch)
    return batch


# Mock Fixtures
@pytest.fixture
def mock_security_manager():
    """Mock security manager for testing."""
    mock = Mock()
    mock.authenticate_user = AsyncMock(return_value={
        "access_token": "test_token",
        "user_data": {
            "user_id": "test_user",
            "username": "testuser",
            "role": "claims_processor"
        }
    })
    mock.get_access_context = AsyncMock()
    mock.audit_logger = Mock()
    mock.audit_logger.log_phi_access = AsyncMock()
    mock.audit_logger.log_system_event = AsyncMock()
    return mock


@pytest.fixture
def mock_ml_predictor():
    """Mock ML predictor for testing."""
    mock = Mock()
    mock.predict_single = AsyncMock(return_value={
        "should_process": True,
        "confidence": 0.95,
        "score": 0.95,
        "reason": "Passed validation"
    })
    mock.predict_batch_optimized = AsyncMock()
    return mock


@pytest.fixture
def mock_metrics_collector():
    """Mock metrics collector for testing."""
    mock = Mock()
    mock.record_claim_processed = Mock()
    mock.record_claim_failed = Mock()
    mock.record_batch_processing = Mock()
    mock.record_validation_result = Mock()
    mock.record_phi_access = Mock()
    mock.record_ml_prediction = Mock()
    return mock


@pytest.fixture
def mock_cache_manager():
    """Mock cache manager for testing."""
    mock = Mock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock()
    mock.delete = AsyncMock()
    mock.ping = AsyncMock()
    return mock


# Performance Testing Fixtures
@pytest.fixture
def performance_test_data():
    """Generate data for performance testing."""
    def generate_claims(count: int):
        claims = []
        for i in range(count):
            claims.append({
                "claim_id": f"CLM{i:06d}",
                "facility_id": f"FAC{(i % 10) + 1:03d}",
                "patient_account_number": f"PAT{i:06d}",
                "patient_first_name": f"Patient{i}",
                "patient_last_name": "Test",
                "patient_date_of_birth": "1980-01-01",
                "admission_date": "2024-01-01",
                "discharge_date": "2024-01-03",
                "service_from_date": "2024-01-01",
                "service_to_date": "2024-01-03",
                "financial_class": "INPATIENT",
                "total_charges": 1000.00 + (i * 10),
                "insurance_type": "MEDICARE",
                "billing_provider_npi": "1234567890",
                "billing_provider_name": "Dr. Test",
                "primary_diagnosis_code": "Z00.00",
            })
        return claims
    
    return generate_claims


# Integration Test Fixtures
@pytest.fixture
async def integration_test_setup(db_session, redis_client, mock_settings):
    """Setup for integration tests."""
    # Create test facilities
    facilities = [
        {"facility_id": "FAC001", "facility_name": "Test Hospital 1"},
        {"facility_id": "FAC002", "facility_name": "Test Hospital 2"},
    ]
    
    for facility_data in facilities:
        facility = Facility(**facility_data)
        db_session.add(facility)
    
    # Create test providers
    providers = [
        {"npi": "1234567890", "provider_name": "Dr. Test Smith"},
        {"npi": "0987654321", "provider_name": "Dr. Test Jones"},
    ]
    
    for provider_data in providers:
        provider = Provider(**provider_data)
        db_session.add(provider)
    
    await db_session.commit()
    
    return {
        "facilities": facilities,
        "providers": providers,
    }


# Custom Pytest Markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as a security test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Test Utilities
class TestHelpers:
    """Helper utilities for testing."""
    
    @staticmethod
    async def create_test_claims(session: AsyncSession, count: int, facility_id: str = "FAC001"):
        """Create multiple test claims."""
        claims = []
        for i in range(count):
            claim = Claim(
                claim_id=f"CLM{i:06d}",
                facility_id=facility_id,
                patient_account_number=f"PAT{i:06d}",
                patient_first_name=f"Patient{i}",
                patient_last_name="Test",
                patient_date_of_birth="1980-01-01",
                admission_date="2024-01-01",
                discharge_date="2024-01-03",
                service_from_date="2024-01-01",
                service_to_date="2024-01-03",
                financial_class="INPATIENT",
                total_charges=1000.00 + (i * 10),
                insurance_type="MEDICARE",
                billing_provider_npi="1234567890",
                billing_provider_name="Dr. Test",
                primary_diagnosis_code="Z00.00",
            )
            session.add(claim)
            claims.append(claim)
        
        await session.commit()
        return claims
    
    @staticmethod
    def assert_performance_target(execution_time: float, target_claims: int, target_time: float):
        """Assert that performance targets are met."""
        throughput = target_claims / execution_time
        target_throughput = target_claims / target_time
        
        assert throughput >= target_throughput, (
            f"Performance target not met. "
            f"Achieved: {throughput:.2f} claims/sec, "
            f"Target: {target_throughput:.2f} claims/sec"
        )
    
    @staticmethod
    def assert_security_compliance(audit_logs: list):
        """Assert security compliance requirements."""
        assert len(audit_logs) > 0, "No audit logs generated"
        
        for log in audit_logs:
            assert log.get("user_id"), "Missing user_id in audit log"
            assert log.get("action_type"), "Missing action_type in audit log"
            assert log.get("timestamp"), "Missing timestamp in audit log"


@pytest.fixture
def test_helpers():
    """Provide test helper utilities."""
    return TestHelpers
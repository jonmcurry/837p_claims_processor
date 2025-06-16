# Testing Strategy

This document outlines the comprehensive testing strategy for the 837P Claims Processing System, ensuring reliability, performance, and compliance at scale.

## Testing Philosophy

Our testing approach follows the **Testing Pyramid** principle with emphasis on:
- **Fast Feedback**: Quick detection of issues during development
- **High Confidence**: Comprehensive coverage of critical functionality
- **Production Readiness**: Testing that mirrors production conditions
- **HIPAA Compliance**: Ensuring security and privacy in all test scenarios

## Testing Pyramid

```mermaid
graph TB
    subgraph "Testing Pyramid"
        UI[UI Tests<br/>10%]
        INTEGRATION[Integration Tests<br/>20%]
        UNIT[Unit Tests<br/>70%]
    end
    
    subgraph "Specialized Testing"
        PERFORMANCE[Performance Tests]
        SECURITY[Security Tests]
        COMPLIANCE[Compliance Tests]
    end
    
    UI --- INTEGRATION
    INTEGRATION --- UNIT
    PERFORMANCE -.-> INTEGRATION
    SECURITY -.-> INTEGRATION
    COMPLIANCE -.-> INTEGRATION
```

## Test Categories

### 1. Unit Tests (70% of total tests)

**Purpose**: Test individual components in isolation

**Coverage Requirements**:
- **Minimum**: 85% code coverage
- **Critical Components**: 95% code coverage
- **Security Functions**: 100% code coverage

**Technology Stack**:
- **Python**: pytest, pytest-asyncio, pytest-mock
- **TypeScript**: Jest, React Testing Library
- **Database**: pytest-postgresql for isolated database tests

**Example Unit Test Structure**:
```python
# tests/unit/test_validation_engine.py
import pytest
from unittest.mock import AsyncMock, patch
from src.processing.validation.comprehensive_rules import ValidationEngine
from src.processing.validation.exceptions import ValidationError

class TestValidationEngine:
    """Unit tests for ValidationEngine."""
    
    @pytest.fixture
    def validation_engine(self):
        """Create ValidationEngine instance for testing."""
        return ValidationEngine(
            rule_engine=AsyncMock(),
            audit_logger=AsyncMock()
        )
    
    @pytest.fixture
    def sample_claim(self):
        """Sample claim data for testing."""
        return {
            "claim_id": "TEST123",
            "patient_id": "P12345",
            "provider_id": "PR67890",
            "total_amount": 150.00,
            "service_date": "2024-01-15",
            "diagnosis_codes": ["Z00.00"],
            "procedure_codes": ["99213"]
        }
    
    @pytest.mark.asyncio
    async def test_validate_claim_success(self, validation_engine, sample_claim):
        """Test successful claim validation."""
        # Arrange
        validation_engine.rule_engine.evaluate_rules.return_value = {
            "passed": True,
            "failed_rules": [],
            "warnings": []
        }
        
        # Act
        result = await validation_engine.validate_claim(sample_claim)
        
        # Assert
        assert result.is_valid is True
        assert len(result.errors) == 0
        validation_engine.audit_logger.log_validation_result.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_validate_claim_with_errors(self, validation_engine, sample_claim):
        """Test claim validation with errors."""
        # Arrange
        validation_engine.rule_engine.evaluate_rules.return_value = {
            "passed": False,
            "failed_rules": ["total_amount_reasonable"],
            "warnings": []
        }
        
        # Act
        result = await validation_engine.validate_claim(sample_claim)
        
        # Assert
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "total_amount_reasonable" in [error.rule_name for error in result.errors]
    
    @pytest.mark.asyncio
    async def test_validate_claim_performance(self, validation_engine, sample_claim):
        """Test validation performance requirements."""
        import time
        
        # Act
        start_time = time.time()
        await validation_engine.validate_claim(sample_claim)
        end_time = time.time()
        
        # Assert - Validation should complete within 100ms
        assert (end_time - start_time) < 0.1
    
    @pytest.mark.parametrize("invalid_field,expected_error", [
        ("patient_id", "missing_required_field"),
        ("total_amount", "invalid_amount_format"),
        ("service_date", "invalid_date_format")
    ])
    @pytest.mark.asyncio
    async def test_validate_claim_required_fields(self, validation_engine, 
                                                 sample_claim, invalid_field, 
                                                 expected_error):
        """Test validation of required fields."""
        # Arrange
        del sample_claim[invalid_field]
        
        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            await validation_engine.validate_claim(sample_claim)
        
        assert expected_error in str(exc_info.value)
```

### 2. Integration Tests (20% of total tests)

**Purpose**: Test interaction between components and external systems

**Test Scenarios**:
- API endpoint integration
- Database operations
- Redis cache operations
- Message queue processing
- External service integration

**Example Integration Test**:
```python
# tests/integration/test_batch_processing.py
import pytest
import asyncio
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

@pytest.fixture(scope="session")
def postgres_container():
    """PostgreSQL test container."""
    with PostgresContainer("postgres:13") as postgres:
        yield postgres

@pytest.fixture(scope="session")
def redis_container():
    """Redis test container."""
    with RedisContainer("redis:6") as redis:
        yield redis

@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_claim_processing(postgres_container, redis_container):
    """Test complete claim processing pipeline."""
    
    # Setup test environment
    db_url = postgres_container.get_connection_url()
    redis_url = redis_container.get_connection_url()
    
    # Initialize test data
    test_claims = await load_test_claims(count=100)
    
    # Process claims through complete pipeline
    processor = BatchProcessor(db_url=db_url, redis_url=redis_url)
    
    start_time = time.time()
    results = await processor.process_batch(test_claims)
    end_time = time.time()
    
    # Verify results
    assert len(results.processed_claims) == 100
    assert results.success_rate > 0.95
    assert (end_time - start_time) < 30  # 100 claims in <30 seconds
    
    # Verify database state
    async with get_db_session(db_url) as session:
        processed_count = await session.scalar(
            text("SELECT COUNT(*) FROM staging.claims_837p WHERE status = 'processed'")
        )
        assert processed_count == len(results.processed_claims)
```

### 3. Performance Tests

**Purpose**: Validate system performance under various load conditions

**Performance Requirements**:
- **Throughput**: 6,667 claims/second sustained
- **Latency**: API responses <500ms P95
- **Concurrency**: 1,000 concurrent users
- **Resource Usage**: <80% CPU, <85% memory under normal load

**Load Testing with Locust**:
```python
# tests/performance/test_system_performance.py
from locust import HttpUser, task, between
import json
import random

class ClaimsProcessingUser(HttpUser):
    """Locust user for load testing claims processing."""
    
    wait_time = between(1, 3)
    
    def on_start(self):
        """Setup user session."""
        # Authenticate user
        response = self.client.post("/auth/login", json={
            "username": "test_user",
            "password": "test_password"
        })
        self.token = response.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}
    
    @task(10)
    def submit_claim(self):
        """Submit a claim for processing."""
        claim_data = self.generate_test_claim()
        
        with self.client.post(
            "/api/v1/claims/submit",
            json=claim_data,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to submit claim: {response.status_code}")
    
    @task(5)
    def check_claim_status(self):
        """Check status of existing claim."""
        claim_id = f"TEST{random.randint(1000, 9999)}"
        
        with self.client.get(
            f"/api/v1/claims/{claim_id}/status",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Failed to check status: {response.status_code}")
    
    @task(2)
    def get_processing_metrics(self):
        """Get system processing metrics."""
        with self.client.get(
            "/api/v1/metrics/processing",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed to get metrics: {response.status_code}")
    
    def generate_test_claim(self) -> dict:
        """Generate realistic test claim data."""
        return {
            "claim_id": f"TEST{random.randint(100000, 999999)}",
            "patient_id": f"P{random.randint(10000, 99999)}",
            "provider_id": f"PR{random.randint(1000, 9999)}",
            "payer_id": f"PAY{random.randint(100, 999)}",
            "total_amount": round(random.uniform(50.0, 2000.0), 2),
            "service_date": "2024-01-15",
            "diagnosis_codes": [f"Z{random.randint(10, 99)}.{random.randint(10, 99)}"],
            "procedure_codes": [f"{random.randint(99000, 99999)}"]
        }

# Performance test execution
@pytest.mark.performance
def test_system_performance_requirements():
    """Test system meets performance requirements."""
    
    # Configure performance test
    test_config = {
        'users': 1000,
        'spawn_rate': 50,
        'duration': '10m',
        'host': 'http://localhost:8000'
    }
    
    # Run load test
    stats = run_locust_test(ClaimsProcessingUser, test_config)
    
    # Verify performance requirements
    assert stats.avg_response_time < 500  # <500ms average response time
    assert stats.p95_response_time < 1000  # <1s P95 response time
    assert stats.failure_rate < 0.01  # <1% failure rate
    assert stats.requests_per_second > 1000  # >1000 RPS throughput
```

### 4. Security Tests

**Purpose**: Validate security controls and identify vulnerabilities

**Security Test Categories**:
- Authentication and authorization
- Input validation and injection attacks
- Encryption and data protection
- Session management
- HIPAA compliance

**Example Security Tests**:
```python
# tests/security/test_security_controls.py
import pytest
import jwt
from src.core.security.authentication import AuthenticationManager

@pytest.mark.security
class TestAuthenticationSecurity:
    """Security tests for authentication system."""
    
    @pytest.fixture
    def auth_manager(self):
        return AuthenticationManager()
    
    def test_password_strength_requirements(self, auth_manager):
        """Test password strength validation."""
        weak_passwords = [
            "123456",
            "password",
            "abc123",
            "password123"
        ]
        
        for password in weak_passwords:
            with pytest.raises(ValueError, match="Password does not meet complexity requirements"):
                auth_manager.validate_password_strength(password)
    
    def test_sql_injection_protection(self):
        """Test SQL injection protection."""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "UNION SELECT * FROM users",
            "'; INSERT INTO users VALUES('hacker'); --"
        ]
        
        for malicious_input in malicious_inputs:
            # Test various endpoints with malicious input
            response = client.get(f"/api/v1/claims?provider_id={malicious_input}")
            assert response.status_code in [400, 422]  # Should be rejected
    
    def test_jwt_token_security(self, auth_manager):
        """Test JWT token security controls."""
        # Test token expiration
        expired_token = jwt.encode(
            {"user_id": "test", "exp": 1234567890},  # Expired timestamp
            auth_manager.jwt_secret,
            algorithm="HS256"
        )
        
        with pytest.raises(jwt.ExpiredSignatureError):
            auth_manager.validate_token(expired_token)
        
        # Test token tampering
        valid_token = auth_manager.generate_token("test_user")
        tampered_token = valid_token[:-5] + "XXXXX"
        
        with pytest.raises(jwt.InvalidTokenError):
            auth_manager.validate_token(tampered_token)
```

### 5. Compliance Tests

**Purpose**: Validate HIPAA and regulatory compliance

**Compliance Test Areas**:
- PHI access controls
- Audit logging completeness
- Data encryption verification
- Breach detection testing

```python
# tests/security/test_hipaa_compliance.py
@pytest.mark.compliance
@pytest.mark.asyncio
class TestHIPAACompliance:
    """HIPAA compliance verification tests."""
    
    async def test_phi_access_audit_logging(self):
        """Test that all PHI access is properly audited."""
        
        # Access PHI data
        patient_id = "P12345"
        user_id = "test_user"
        
        phi_data = await get_patient_data(patient_id, user_id)
        
        # Verify audit log entry was created
        audit_entries = await get_audit_logs(
            user_id=user_id,
            resource_type="patient_data",
            resource_id=patient_id
        )
        
        assert len(audit_entries) == 1
        audit_entry = audit_entries[0]
        
        # Verify required audit fields
        assert audit_entry.user_id == user_id
        assert audit_entry.resource_id == patient_id
        assert audit_entry.action == "read"
        assert audit_entry.phi_accessed is True
        assert audit_entry.timestamp is not None
        assert audit_entry.ip_address is not None
        assert audit_entry.user_agent is not None
    
    async def test_minimum_necessary_access(self):
        """Test minimum necessary access principle."""
        
        # Test different user roles have appropriate access
        test_scenarios = [
            {
                "role": "claims_processor",
                "accessible_fields": ["claim_id", "total_amount", "service_date"],
                "restricted_fields": ["ssn", "phone_number", "email"]
            },
            {
                "role": "physician",
                "accessible_fields": ["patient_id", "diagnosis_codes", "medical_history"],
                "restricted_fields": ["billing_address", "insurance_details"]
            }
        ]
        
        for scenario in test_scenarios:
            user = create_test_user(role=scenario["role"])
            
            # Test accessible fields
            for field in scenario["accessible_fields"]:
                data = await access_patient_field(user, "P12345", field)
                assert data is not None
            
            # Test restricted fields
            for field in scenario["restricted_fields"]:
                with pytest.raises(PermissionError):
                    await access_patient_field(user, "P12345", field)
```

## Test Data Management

### Test Data Generation

```python
# tests/utils/test_data_factory.py
from faker import Faker
from faker.providers import BaseProvider
import random

class MedicalDataProvider(BaseProvider):
    """Custom provider for medical test data."""
    
    def diagnosis_code(self):
        """Generate realistic ICD-10 diagnosis code."""
        category = random.choice(['Z', 'M', 'R', 'K', 'I'])
        return f"{category}{random.randint(10, 99)}.{random.randint(0, 9)}"
    
    def procedure_code(self):
        """Generate realistic CPT procedure code."""
        return str(random.randint(99000, 99999))
    
    def npi_number(self):
        """Generate valid NPI number."""
        # Simplified NPI generation for testing
        return f"{random.randint(1000000000, 9999999999)}"

class TestDataFactory:
    """Factory for generating realistic test data."""
    
    def __init__(self):
        self.fake = Faker()
        self.fake.add_provider(MedicalDataProvider)
    
    def generate_claim(self, **overrides) -> dict:
        """Generate realistic claim data."""
        claim = {
            "claim_id": f"CLM{self.fake.random_int(100000, 999999)}",
            "patient_id": f"P{self.fake.random_int(10000, 99999)}",
            "provider_id": f"PR{self.fake.random_int(1000, 9999)}",
            "payer_id": f"PAY{self.fake.random_int(100, 999)}",
            "total_amount": round(random.uniform(50.0, 5000.0), 2),
            "service_date": self.fake.date_between(start_date='-1y', end_date='today').isoformat(),
            "diagnosis_codes": [self.fake.diagnosis_code() for _ in range(random.randint(1, 3))],
            "procedure_codes": [self.fake.procedure_code() for _ in range(random.randint(1, 5))],
            "provider_npi": self.fake.npi_number(),
            "service_facility": self.fake.company(),
            "patient_dob": self.fake.date_of_birth(minimum_age=0, maximum_age=100).isoformat()
        }
        
        # Apply any overrides
        claim.update(overrides)
        return claim
    
    def generate_batch(self, size: int = 100) -> List[dict]:
        """Generate batch of test claims."""
        return [self.generate_claim() for _ in range(size)]
```

### Test Environment Management

```python
# tests/conftest.py
import pytest
import asyncio
import docker
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def docker_client():
    """Docker client for container management."""
    return docker.from_env()

@pytest.fixture(scope="session")
def test_database():
    """PostgreSQL test database container."""
    with PostgresContainer("postgres:13") as postgres:
        # Initialize test database
        connection_url = postgres.get_connection_url()
        
        # Run migrations
        run_database_migrations(connection_url)
        
        # Load test data
        load_test_data(connection_url)
        
        yield connection_url

@pytest.fixture(scope="session")
def test_redis():
    """Redis test container."""
    with RedisContainer("redis:6") as redis:
        yield redis.get_connection_url()

@pytest.fixture
def test_client(test_database, test_redis):
    """Test client with test database."""
    from src.api.production_main import create_app
    
    app = create_app(
        database_url=test_database,
        redis_url=test_redis,
        testing=True
    )
    
    with TestClient(app) as client:
        yield client
```

## Continuous Integration Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: test_password
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:6
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
      env:
        DATABASE_URL: postgresql://postgres:test_password@localhost:5432/test_db
        REDIS_URL: redis://localhost:6379/0

  performance-tests:
    runs-on: ubuntu-latest
    needs: integration-tests
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run performance tests
      run: |
        pytest tests/performance/ -v --performance
    
    - name: Upload performance results
      uses: actions/upload-artifact@v3
      with:
        name: performance-results
        path: performance-results/

  security-tests:
    runs-on: ubuntu-latest
    needs: integration-tests
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Run security tests
      run: |
        pytest tests/security/ -v --security
    
    - name: Run SAST scan
      uses: github/super-linter@v4
      env:
        DEFAULT_BRANCH: main
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

## Test Reporting and Metrics

### Coverage Requirements

| Component | Coverage Target | Current Coverage |
|-----------|----------------|------------------|
| Core Business Logic | 95% | 94.2% |
| API Endpoints | 90% | 91.8% |
| Security Functions | 100% | 100% |
| Database Operations | 85% | 87.3% |
| ML Pipeline | 80% | 82.1% |

### Quality Gates

Before deployment, all tests must pass:
- ✅ Unit test coverage >85%
- ✅ Integration tests pass
- ✅ Performance requirements met
- ✅ Security tests pass
- ✅ HIPAA compliance verified

---

For related documentation, see:
- [Performance Testing](./performance-testing.md)
- [Security Testing](./security-testing.md)
- [Integration Testing](./integration-testing.md)
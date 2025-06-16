"""Integration tests for API endpoints."""

import pytest
import json
from decimal import Decimal
from datetime import datetime, date
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient
from fastapi.testclient import TestClient

from src.api.production_main import app
from src.core.database.models import (
    Claim, FailedClaim, BatchMetadata, ClaimLineItem,
    FailureCategory, ResolutionStatus
)
from src.core.security.hipaa_security import security_manager


@pytest.mark.integration
class TestAPIEndpointsIntegration:
    """Integration tests for all API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    async def async_client(self):
        """Create async test client."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    @pytest.fixture
    def auth_headers(self, mock_security_manager):
        """Create authentication headers for testing."""
        # Mock JWT token
        test_token = "test_jwt_token_for_integration_testing"
        return {"Authorization": f"Bearer {test_token}"}
    
    @pytest.fixture
    async def test_claims_in_db(self, db_session, test_helpers):
        """Create test claims in database."""
        claims = await test_helpers.create_test_claims(db_session, 10, "FAC001")
        
        # Create some failed claims
        failed_claims = []
        for i, claim in enumerate(claims[:3]):
            failed_claim = FailedClaim(
                original_claim_id=claim.id,
                claim_reference=claim.claim_id,
                facility_id=claim.facility_id,
                failure_category=FailureCategory.VALIDATION_ERROR,
                failure_reason=f"Test validation error {i}",
                failure_details={"error": f"test error {i}"},
                claim_data={"claim_id": claim.claim_id},
                charge_amount=claim.total_charges,
                resolution_status=ResolutionStatus.PENDING
            )
            db_session.add(failed_claim)
            failed_claims.append(failed_claim)
        
        await db_session.commit()
        return claims, failed_claims
    
    @pytest.mark.asyncio
    async def test_health_endpoints(self, async_client):
        """Test health check endpoints."""
        # Basic health check
        response = await async_client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        
        # Detailed health check
        response = await async_client.get("/health/detailed")
        assert response.status_code == 200
        
        data = response.json()
        assert "database" in data
        assert "redis" in data
        assert "services" in data
    
    @pytest.mark.asyncio
    async def test_authentication_endpoints(self, async_client, mock_security_manager):
        """Test authentication endpoints."""
        # Mock the security manager
        with patch('src.api.production_main.security_manager', mock_security_manager):
            # Test login
            login_data = {
                "username": "testuser",
                "password": "testpass"
            }
            
            response = await async_client.post("/auth/login", json=login_data)
            assert response.status_code == 200
            
            data = response.json()
            assert "access_token" in data
            assert "user_data" in data
            assert data["user_data"]["username"] == "testuser"
    
    @pytest.mark.asyncio
    async def test_claims_endpoints(
        self, 
        async_client, 
        auth_headers, 
        test_claims_in_db,
        mock_security_manager
    ):
        """Test claims-related endpoints."""
        claims, _ = test_claims_in_db
        test_claim = claims[0]
        
        with patch('src.api.production_main.security_manager', mock_security_manager):
            # Test get claim by ID
            response = await async_client.get(
                f"/claims/{test_claim.claim_id}",
                headers=auth_headers,
                params={"business_justification": "Integration testing"}
            )
            assert response.status_code == 200
            
            data = response.json()
            assert data["claim_id"] == test_claim.claim_id
            assert data["facility_id"] == test_claim.facility_id
    
    @pytest.mark.asyncio
    async def test_batch_submission_endpoint(
        self, 
        async_client, 
        auth_headers,
        mock_security_manager,
        sample_claim_data
    ):
        """Test batch submission endpoint."""
        batch_data = {
            "facility_id": "FAC001",
            "claims": [sample_claim_data] * 5,  # 5 test claims
            "priority": "high",
            "submitted_by": "integration_test_user"
        }
        
        with patch('src.api.production_main.security_manager', mock_security_manager):
            with patch('src.api.production_main.batch_processor') as mock_processor:
                # Mock the batch processor response
                mock_processor.submit_batch.return_value = {
                    "batch_id": "TEST_BATCH_001",
                    "status": "submitted",
                    "claims_count": 5,
                    "estimated_completion": datetime.now().isoformat()
                }
                
                response = await async_client.post(
                    "/claims/batch",
                    json=batch_data,
                    headers=auth_headers
                )
                assert response.status_code == 200
                
                data = response.json()
                assert data["batch_id"] == "TEST_BATCH_001"
                assert data["claims_count"] == 5
                assert data["status"] == "submitted"
    
    @pytest.mark.asyncio
    async def test_failed_claims_endpoints(
        self, 
        async_client, 
        auth_headers,
        test_claims_in_db,
        mock_security_manager
    ):
        """Test failed claims management endpoints."""
        _, failed_claims = test_claims_in_db
        
        with patch('src.api.production_main.security_manager', mock_security_manager):
            # Test get failed claims with filters
            response = await async_client.get(
                "/failed-claims",
                headers=auth_headers,
                params={
                    "facility_id": "FAC001",
                    "failure_category": "validation_error",
                    "limit": 10,
                    "offset": 0
                }
            )
            assert response.status_code == 200
            
            data = response.json()
            assert "failed_claims" in data
            assert "total_count" in data
            assert len(data["failed_claims"]) > 0
            
            # Test claim resolution
            failed_claim = failed_claims[0]
            resolution_data = {
                "resolution_type": "manual_fix",
                "resolution_notes": "Fixed during integration testing",
                "corrected_data": {"patient_first_name": "Corrected Name"},
                "business_justification": "Integration test resolution"
            }
            
            response = await async_client.post(
                f"/failed-claims/{failed_claim.id}/resolve",
                json=resolution_data,
                headers=auth_headers
            )
            assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_bulk_operations_endpoints(
        self, 
        async_client, 
        auth_headers,
        test_claims_in_db,
        mock_security_manager
    ):
        """Test bulk operations on failed claims."""
        _, failed_claims = test_claims_in_db
        
        with patch('src.api.production_main.security_manager', mock_security_manager):
            # Test bulk assignment
            claim_ids = [str(claim.id) for claim in failed_claims[:2]]
            assignment_data = {
                "claim_ids": claim_ids,
                "assigned_to": "integration_test_assignee"
            }
            
            response = await async_client.post(
                "/failed-claims/assign",
                json=assignment_data,
                headers=auth_headers
            )
            assert response.status_code == 200
            
            # Test bulk export
            export_data = {
                "claim_ids": claim_ids
            }
            
            response = await async_client.post(
                "/failed-claims/export",
                json=export_data,
                headers=auth_headers
            )
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/octet-stream"
    
    @pytest.mark.asyncio
    async def test_dashboard_endpoints(
        self, 
        async_client, 
        auth_headers,
        test_claims_in_db,
        mock_security_manager,
        mock_metrics_collector
    ):
        """Test dashboard and metrics endpoints."""
        with patch('src.api.production_main.security_manager', mock_security_manager):
            with patch('src.api.production_main.metrics_collector', mock_metrics_collector):
                # Mock metrics data
                mock_metrics_collector.get_current_metrics.return_value = {
                    "total_claims_today": 1500,
                    "successful_claims_today": 1450,
                    "failed_claims_today": 50,
                    "avg_processing_time": 2.3,
                    "current_throughput": 6500,
                    "target_throughput": 6667,
                    "system_health": "healthy"
                }
                
                response = await async_client.get(
                    "/dashboard/metrics",
                    headers=auth_headers
                )
                assert response.status_code == 200
                
                data = response.json()
                assert "total_claims_today" in data
                assert "current_throughput" in data
                assert "system_health" in data
                assert data["system_health"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_analytics_endpoints(
        self, 
        async_client, 
        auth_headers,
        mock_security_manager
    ):
        """Test analytics endpoints."""
        with patch('src.api.production_main.security_manager', mock_security_manager):
            # Test analytics data endpoint
            response = await async_client.get(
                "/analytics",
                headers=auth_headers,
                params={
                    "start_date": "2024-01-01T00:00:00Z",
                    "end_date": "2024-01-31T23:59:59Z"
                }
            )
            assert response.status_code == 200
            
            data = response.json()
            assert "processing_trends" in data
            assert "facility_performance" in data
            assert "error_analysis" in data
    
    @pytest.mark.asyncio
    async def test_rate_limiting(
        self, 
        async_client, 
        auth_headers,
        mock_security_manager
    ):
        """Test rate limiting middleware."""
        with patch('src.api.production_main.security_manager', mock_security_manager):
            # Make multiple rapid requests to test rate limiting
            responses = []
            for i in range(20):  # Exceed rate limit
                response = await async_client.get(
                    "/health",
                    headers=auth_headers
                )
                responses.append(response)
            
            # Some requests should be rate limited (429 status)
            status_codes = [r.status_code for r in responses]
            assert 429 in status_codes  # At least some should be rate limited
    
    @pytest.mark.asyncio
    async def test_error_handling(
        self, 
        async_client, 
        auth_headers,
        mock_security_manager
    ):
        """Test API error handling."""
        with patch('src.api.production_main.security_manager', mock_security_manager):
            # Test 404 for non-existent claim
            response = await async_client.get(
                "/claims/NON_EXISTENT_CLAIM",
                headers=auth_headers
            )
            assert response.status_code == 404
            
            data = response.json()
            assert "detail" in data
            assert "not found" in data["detail"].lower()
            
            # Test validation error for invalid batch data
            invalid_batch_data = {
                "facility_id": "",  # Required field is empty
                "claims": [],  # No claims provided
                "submitted_by": ""  # Required field is empty
            }
            
            response = await async_client.post(
                "/claims/batch",
                json=invalid_batch_data,
                headers=auth_headers
            )
            assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_authentication_required(self, async_client):
        """Test that protected endpoints require authentication."""
        # Test without authentication headers
        response = await async_client.get("/claims/CLM123456")
        assert response.status_code == 401
        
        response = await async_client.get("/failed-claims")
        assert response.status_code == 401
        
        response = await async_client.get("/dashboard/metrics")
        assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_audit_logging_integration(
        self, 
        async_client, 
        auth_headers,
        test_claims_in_db,
        mock_security_manager
    ):
        """Test that API calls are properly audit logged."""
        claims, _ = test_claims_in_db
        test_claim = claims[0]
        
        with patch('src.api.production_main.security_manager', mock_security_manager):
            # Make API call that accesses PHI
            response = await async_client.get(
                f"/claims/{test_claim.claim_id}",
                headers=auth_headers,
                params={"business_justification": "Integration testing"}
            )
            assert response.status_code == 200
            
            # Verify audit logging was called
            assert mock_security_manager.audit_logger.log_phi_access.called
            
            # Check the audit log call arguments
            call_args = mock_security_manager.audit_logger.log_phi_access.call_args
            assert call_args[1]["resource_type"] == "claim"
            assert call_args[1]["business_justification"] == "Integration testing"
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_api_performance(
        self, 
        async_client, 
        auth_headers,
        test_claims_in_db,
        mock_security_manager
    ):
        """Test API performance under load."""
        import time
        import asyncio
        
        claims, failed_claims = test_claims_in_db
        
        async def make_request():
            """Make a single API request."""
            response = await async_client.get(
                "/failed-claims",
                headers=auth_headers,
                params={"limit": 10}
            )
            return response.status_code == 200
        
        with patch('src.api.production_main.security_manager', mock_security_manager):
            # Make 50 concurrent requests
            start_time = time.perf_counter()
            
            tasks = [make_request() for _ in range(50)]
            results = await asyncio.gather(*tasks)
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
        
        # All requests should succeed
        assert all(results)
        
        # Should handle 50 requests in reasonable time (under 5 seconds)
        assert execution_time < 5.0
        
        requests_per_second = len(results) / execution_time
        print(f"API handled {requests_per_second:.2f} requests/second")
    
    @pytest.mark.asyncio
    async def test_websocket_integration(
        self, 
        async_client,
        auth_headers,
        mock_security_manager
    ):
        """Test WebSocket integration with API."""
        with patch('src.api.production_main.security_manager', mock_security_manager):
            # Test WebSocket connection (basic connectivity test)
            # In a real scenario, this would test real-time notifications
            
            # This is a placeholder for WebSocket testing
            # Full WebSocket testing would require additional setup
            async with async_client.websocket_connect("/ws") as websocket:
                # Send authentication
                await websocket.send_json({
                    "type": "auth",
                    "token": "test_token"
                })
                
                # Should receive acknowledgment
                data = await websocket.receive_json()
                assert data.get("type") == "auth_success"
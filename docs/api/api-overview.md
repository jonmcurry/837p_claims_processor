# API Overview

This document provides a comprehensive overview of the 837P Claims Processing System REST API, including endpoints, authentication, request/response formats, and usage examples.

## API Architecture

### Design Principles

- **RESTful Design**: Following REST architectural principles
- **HIPAA Compliant**: All endpoints implement HIPAA security requirements
- **High Performance**: Optimized for 6,667 claims/second throughput
- **Comprehensive Monitoring**: Full request/response logging and metrics
- **Developer Friendly**: Clear documentation with examples

### Technology Stack

- **Framework**: FastAPI with Pydantic validation
- **Authentication**: JWT with refresh tokens
- **Rate Limiting**: Redis-based adaptive rate limiting
- **Documentation**: Auto-generated OpenAPI/Swagger docs
- **Monitoring**: Prometheus metrics integration

## Base URL and Versioning

```
Production: https://api.claims-processor.com/api/v1
Staging: https://staging-api.claims-processor.com/api/v1
Development: http://localhost:8000/api/v1
```

### API Versioning Strategy
- **URL Path Versioning**: `/api/v1/`, `/api/v2/`
- **Backward Compatibility**: Maintained for 2 major versions
- **Deprecation Timeline**: 12 months notice for breaking changes

## Authentication

### JWT Token Authentication

All API endpoints require authentication via JWT tokens in the Authorization header:

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Authentication Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Auth Service
    participant Database
    
    Client->>API: POST /auth/login (credentials)
    API->>Auth Service: Validate credentials
    Auth Service->>Database: Check user & password
    Database-->>Auth Service: User data
    Auth Service-->>API: User validated
    API-->>Client: Access token + Refresh token
    
    Note over Client,API: Subsequent requests
    Client->>API: Request with Bearer token
    API->>Auth Service: Validate token
    Auth Service-->>API: Token valid + user info
    API-->>Client: Response
```

### Token Management

```python
# Example token usage
import requests

# Login to get tokens
response = requests.post('http://localhost:8000/auth/login', json={
    'username': 'your_username',
    'password': 'your_password'
})

tokens = response.json()
access_token = tokens['access_token']
refresh_token = tokens['refresh_token']

# Use access token for API calls
headers = {'Authorization': f'Bearer {access_token}'}
response = requests.get('http://localhost:8000/api/v1/claims', headers=headers)

# Refresh token when access token expires
response = requests.post('http://localhost:8000/auth/refresh', json={
    'refresh_token': refresh_token
})
new_tokens = response.json()
```

## Core API Endpoints

### 1. Claims Processing

#### Submit Claim
```http
POST /api/v1/claims/submit
Content-Type: application/json
Authorization: Bearer {token}

{
  "claim_id": "CLM123456",
  "patient_id": "P12345",
  "provider_id": "PR67890",
  "payer_id": "PAY001",
  "total_amount": 150.50,
  "service_date": "2024-01-15",
  "diagnosis_codes": ["Z00.00", "M79.3"],
  "procedure_codes": ["99213", "29125"],
  "line_items": [
    {
      "line_number": 1,
      "procedure_code": "99213",
      "diagnosis_code": "Z00.00",
      "service_units": 1,
      "charge_amount": 125.00,
      "place_of_service": "11"
    }
  ]
}
```

**Response:**
```json
{
  "claim_id": "CLM123456",
  "submission_id": "SUB789012",
  "status": "submitted",
  "submitted_at": "2024-01-15T10:30:00Z",
  "estimated_processing_time_seconds": 45,
  "tracking_url": "/api/v1/claims/CLM123456/status"
}
```

#### Get Claim Status
```http
GET /api/v1/claims/{claim_id}/status
Authorization: Bearer {token}
```

**Response:**
```json
{
  "claim_id": "CLM123456",
  "status": "processing",
  "current_stage": "validation",
  "progress_percentage": 75,
  "processing_history": [
    {
      "stage": "received",
      "timestamp": "2024-01-15T10:30:00Z",
      "duration_ms": 5
    },
    {
      "stage": "validation",
      "timestamp": "2024-01-15T10:30:05Z",
      "duration_ms": 1250,
      "details": {
        "rules_passed": 45,
        "rules_failed": 0,
        "warnings": 2
      }
    }
  ],
  "estimated_completion": "2024-01-15T10:31:30Z"
}
```

#### List Claims
```http
GET /api/v1/claims?page=1&limit=50&status=processing&provider_id=PR67890
Authorization: Bearer {token}
```

**Query Parameters:**
- `page`: Page number (default: 1)
- `limit`: Items per page (max: 100, default: 50)
- `status`: Filter by status (submitted, processing, approved, denied, failed)
- `provider_id`: Filter by provider ID
- `payer_id`: Filter by payer ID
- `date_from`: Filter claims from date (ISO format)
- `date_to`: Filter claims to date (ISO format)
- `sort_by`: Sort field (submitted_at, total_amount, status)
- `sort_order`: Sort order (asc, desc)

### 2. Failed Claims Management

#### Get Failed Claims
```http
GET /api/v1/claims/failed?page=1&limit=20&error_type=validation
Authorization: Bearer {token}
```

**Response:**
```json
{
  "failed_claims": [
    {
      "claim_id": "CLM123457",
      "failure_reason": "validation_error",
      "error_details": [
        {
          "rule": "total_amount_reasonable",
          "message": "Total amount exceeds reasonable limits for procedure",
          "severity": "error",
          "field": "total_amount"
        }
      ],
      "failed_at": "2024-01-15T10:35:00Z",
      "retry_count": 2,
      "can_retry": true,
      "original_claim": { /* original claim data */ }
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total_items": 156,
    "total_pages": 8,
    "has_next": true,
    "has_previous": false
  }
}
```

#### Retry Failed Claim
```http
POST /api/v1/claims/{claim_id}/retry
Authorization: Bearer {token}

{
  "corrected_data": {
    "total_amount": 145.00
  },
  "retry_reason": "Corrected amount based on provider feedback"
}
```

### 3. System Monitoring

#### System Health
```http
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:40:00Z",
  "version": "1.2.0",
  "environment": "production",
  "services": {
    "database": {
      "status": "healthy",
      "response_time_ms": 12,
      "connections_active": 25,
      "connections_max": 100
    },
    "redis": {
      "status": "healthy",
      "response_time_ms": 3,
      "memory_usage_mb": 2048,
      "memory_max_mb": 4096
    },
    "ml_pipeline": {
      "status": "healthy",
      "models_loaded": 3,
      "gpu_utilization_percent": 78
    }
  },
  "performance_metrics": {
    "claims_per_second": 1247,
    "avg_processing_time_ms": 67,
    "success_rate_percent": 99.2
  }
}
```

#### Processing Metrics
```http
GET /api/v1/metrics/processing?time_range=1h
Authorization: Bearer {token}
```

**Response:**
```json
{
  "time_range": "1h",
  "metrics": {
    "claims_processed": 4500000,
    "claims_per_second_avg": 1250,
    "claims_per_second_peak": 1890,
    "success_rate_percent": 99.3,
    "avg_processing_time_ms": 68,
    "p95_processing_time_ms": 145,
    "p99_processing_time_ms": 298
  },
  "status_breakdown": {
    "approved": 3465000,
    "denied": 225000,
    "pending": 45000,
    "failed": 315000
  },
  "error_categories": {
    "validation_errors": 189000,
    "system_errors": 126000
  }
}
```

### 4. Analytics and Reporting

#### Claims Analytics
```http
GET /api/v1/analytics/claims?date_from=2024-01-01&date_to=2024-01-31&group_by=provider
Authorization: Bearer {token}
```

**Response:**
```json
{
  "analytics_period": {
    "start_date": "2024-01-01",
    "end_date": "2024-01-31",
    "group_by": "provider"
  },
  "summary": {
    "total_claims": 150000,
    "total_amount": 45000000.00,
    "average_claim_amount": 300.00,
    "approval_rate_percent": 92.5
  },
  "grouped_data": [
    {
      "provider_id": "PR67890",
      "provider_name": "General Medical Practice",
      "claims_count": 12500,
      "total_amount": 3750000.00,
      "approval_rate_percent": 94.2,
      "avg_processing_time_ms": 65
    }
  ],
  "trending": {
    "daily_volumes": [
      {
        "date": "2024-01-01",
        "claims_count": 4850,
        "total_amount": 1455000.00
      }
    ]
  }
}
```

#### Revenue Analytics
```http
GET /api/v1/analytics/revenue?period=monthly&include_rvus=true
Authorization: Bearer {token}
```

**Response:**
```json
{
  "revenue_summary": {
    "period": "2024-01",
    "total_billed": 45000000.00,
    "total_approved": 41625000.00,
    "total_denied": 3375000.00,
    "approval_rate_percent": 92.5,
    "total_rvus": 234567.89
  },
  "rvu_breakdown": {
    "work_rvus": 123456.78,
    "practice_expense_rvus": 89012.34,
    "malpractice_rvus": 22098.77
  },
  "top_procedures": [
    {
      "procedure_code": "99213",
      "description": "Office visit, established patient",
      "claim_count": 12500,
      "total_amount": 1562500.00,
      "total_rvus": 15625.00
    }
  ],
  "payer_performance": [
    {
      "payer_id": "PAY001",
      "payer_name": "Major Insurance Co",
      "total_claims": 45000,
      "approval_rate_percent": 95.2,
      "avg_payment_time_days": 14
    }
  ]
}
```

## Request/Response Formats

### Standard Request Headers

```http
Content-Type: application/json
Authorization: Bearer {jwt_token}
Accept: application/json
User-Agent: YourApp/1.0
X-Request-ID: uuid4-request-id (optional)
```

### Standard Response Format

```json
{
  "data": { /* response data */ },
  "meta": {
    "request_id": "req_123456789",
    "timestamp": "2024-01-15T10:40:00Z",
    "processing_time_ms": 45,
    "api_version": "v1"
  },
  "pagination": { /* if applicable */ },
  "links": { /* if applicable */ }
}
```

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "One or more validation errors occurred",
    "details": [
      {
        "field": "total_amount",
        "message": "Must be a positive number",
        "invalid_value": -150.00
      }
    ],
    "request_id": "req_123456789",
    "timestamp": "2024-01-15T10:40:00Z",
    "documentation_url": "https://docs.claims-processor.com/errors/validation"
  }
}
```

## Rate Limiting

### Rate Limit Headers

All responses include rate limiting information:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642262400
X-RateLimit-Window: 3600
```

### Rate Limits by Endpoint

| Endpoint Category | Requests per Hour | Burst Limit |
|------------------|-------------------|-------------|
| Authentication | 100 | 10 |
| Claims Submission | 10,000 | 100 |
| Claims Status | 50,000 | 500 |
| Analytics | 1,000 | 50 |
| System Health | Unlimited | 100 |

### Rate Limit Exceeded Response

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Please try again later.",
    "retry_after_seconds": 300,
    "limit": 1000,
    "window_seconds": 3600
  }
}
```

## Pagination

### Request Parameters

```http
GET /api/v1/claims?page=2&limit=50&sort_by=submitted_at&sort_order=desc
```

### Response Format

```json
{
  "data": [ /* items array */ ],
  "pagination": {
    "page": 2,
    "limit": 50,
    "total_items": 1250,
    "total_pages": 25,
    "has_next": true,
    "has_previous": true
  },
  "links": {
    "first": "/api/v1/claims?page=1&limit=50",
    "previous": "/api/v1/claims?page=1&limit=50",
    "next": "/api/v1/claims?page=3&limit=50",
    "last": "/api/v1/claims?page=25&limit=50"
  }
}
```

## WebSocket Real-time Updates

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/claims-updates');
ws.onopen = function(event) {
    // Send authentication
    ws.send(JSON.stringify({
        type: 'auth',
        token: 'your-jwt-token'
    }));
};
```

### Message Types

```javascript
// Subscribe to claim updates
ws.send(JSON.stringify({
    type: 'subscribe',
    channels: ['claim_status', 'system_metrics']
}));

// Receive real-time updates
ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    
    switch(message.type) {
        case 'claim_status_update':
            console.log('Claim updated:', message.data);
            break;
        case 'system_metrics':
            console.log('Metrics:', message.data);
            break;
    }
};
```

## SDK and Client Libraries

### Python SDK
```python
# Install: pip install claims-processor-sdk
from claims_processor_sdk import ClaimsClient

client = ClaimsClient(
    base_url='https://api.claims-processor.com/api/v1',
    username='your_username',
    password='your_password'
)

# Submit claim
result = await client.submit_claim({
    'claim_id': 'CLM123456',
    'patient_id': 'P12345',
    # ... other fields
})

# Get status
status = await client.get_claim_status('CLM123456')
```

### JavaScript SDK
```javascript
// Install: npm install @claims-processor/sdk
import { ClaimsClient } from '@claims-processor/sdk';

const client = new ClaimsClient({
    baseUrl: 'https://api.claims-processor.com/api/v1',
    username: 'your_username',
    password: 'your_password'
});

// Submit claim
const result = await client.submitClaim({
    claimId: 'CLM123456',
    patientId: 'P12345',
    // ... other fields
});
```

## Error Handling

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `AUTHENTICATION_REQUIRED` | 401 | Missing or invalid authentication |
| `INSUFFICIENT_PERMISSIONS` | 403 | User lacks required permissions |
| `RESOURCE_NOT_FOUND` | 404 | Requested resource not found |
| `VALIDATION_ERROR` | 422 | Request validation failed |
| `RATE_LIMIT_EXCEEDED` | 429 | Rate limit exceeded |
| `INTERNAL_ERROR` | 500 | Internal server error |

### Error Handling Best Practices

```python
import requests
from requests.exceptions import RequestException

def submit_claim_with_retry(claim_data, max_retries=3):
    """Submit claim with exponential backoff retry."""
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                'http://localhost:8000/api/v1/claims/submit',
                json=claim_data,
                headers={'Authorization': f'Bearer {token}'},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limited
                retry_after = int(response.headers.get('Retry-After', 60))
                time.sleep(retry_after)
                continue
            elif response.status_code == 422:  # Validation error
                error_data = response.json()
                raise ValueError(f"Validation error: {error_data['error']['message']}")
            else:
                response.raise_for_status()
                
        except RequestException as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

---

For detailed endpoint documentation, see:
- [Authentication](./authentication.md)
- [Claims Processing](./claims-processing.md)
- [Failed Claims Management](./failed-claims.md)
#!/bin/bash

# Claims Processing System Deployment Script
# Supports deployment to development, staging, and production environments

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_ROOT/config"

# Default values
ENVIRONMENT="${1:-development}"
DOCKER_COMPOSE_FILE=""
ENV_FILE=""
BACKUP_BEFORE_DEPLOY="${BACKUP_BEFORE_DEPLOY:-false}"
RUN_MIGRATIONS="${RUN_MIGRATIONS:-true}"
RUN_TESTS="${RUN_TESTS:-true}"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
    fi
    
    # Check if docker-compose is installed
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
    fi
    
    # Check if kubectl is installed (for production)
    if [[ "$ENVIRONMENT" == "production" ]] && ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed (required for production deployment)"
    fi
    
    log_success "Prerequisites check passed"
}

validate_environment() {
    log_info "Validating environment: $ENVIRONMENT"
    
    case "$ENVIRONMENT" in
        development|dev)
            ENVIRONMENT="development"
            DOCKER_COMPOSE_FILE="docker-compose.dev.yml"
            ENV_FILE="$CONFIG_DIR/.env.development"
            ;;
        staging)
            DOCKER_COMPOSE_FILE="docker-compose.staging.yml"
            ENV_FILE="$CONFIG_DIR/.env.staging"
            ;;
        production|prod)
            ENVIRONMENT="production"
            DOCKER_COMPOSE_FILE="docker-compose.prod.yml"
            ENV_FILE="$CONFIG_DIR/.env.production"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT. Valid options: development, staging, production"
            ;;
    esac
    
    # Check if docker-compose file exists
    if [[ ! -f "$PROJECT_ROOT/$DOCKER_COMPOSE_FILE" ]]; then
        log_error "Docker Compose file not found: $DOCKER_COMPOSE_FILE"
    fi
    
    # Check if environment file exists
    if [[ ! -f "$ENV_FILE" ]]; then
        log_warning "Environment file not found: $ENV_FILE"
        log_info "Please create $ENV_FILE based on $CONFIG_DIR/.env.example"
        
        if [[ "$ENVIRONMENT" != "development" ]]; then
            log_error "Environment file is required for $ENVIRONMENT deployment"
        fi
    fi
    
    log_success "Environment validation passed"
}

run_tests() {
    if [[ "$RUN_TESTS" == "true" ]]; then
        log_info "Running tests..."
        
        # Export environment for tests
        export ENVIRONMENT="$ENVIRONMENT"
        
        # Run unit tests
        docker-compose -f "$PROJECT_ROOT/docker-compose.test.yml" run --rm test-runner pytest tests/unit -v
        
        # Run integration tests for staging/production
        if [[ "$ENVIRONMENT" != "development" ]]; then
            docker-compose -f "$PROJECT_ROOT/docker-compose.test.yml" run --rm test-runner pytest tests/integration -v
        fi
        
        log_success "Tests passed"
    else
        log_warning "Skipping tests (RUN_TESTS=false)"
    fi
}

backup_database() {
    if [[ "$BACKUP_BEFORE_DEPLOY" == "true" ]] && [[ "$ENVIRONMENT" != "development" ]]; then
        log_info "Creating database backup..."
        
        BACKUP_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        BACKUP_FILE="backup_${ENVIRONMENT}_${BACKUP_TIMESTAMP}.sql"
        
        # Create backup using docker
        docker-compose -f "$PROJECT_ROOT/$DOCKER_COMPOSE_FILE" --env-file "$ENV_FILE" exec -T postgres \
            pg_dump -U claims_user claims_processor > "$PROJECT_ROOT/backups/$BACKUP_FILE"
        
        log_success "Database backup created: $BACKUP_FILE"
    else
        log_info "Skipping database backup"
    fi
}

build_images() {
    log_info "Building Docker images..."
    
    # Build images with environment-specific tags
    docker-compose -f "$PROJECT_ROOT/$DOCKER_COMPOSE_FILE" --env-file "$ENV_FILE" build
    
    log_success "Docker images built successfully"
}

run_migrations() {
    if [[ "$RUN_MIGRATIONS" == "true" ]]; then
        log_info "Running database migrations..."
        
        # Wait for database to be ready
        docker-compose -f "$PROJECT_ROOT/$DOCKER_COMPOSE_FILE" --env-file "$ENV_FILE" run --rm api \
            python -c "
import asyncio
from src.core.database.base import check_database_connection
asyncio.run(check_database_connection())
"
        
        # Run migrations
        docker-compose -f "$PROJECT_ROOT/$DOCKER_COMPOSE_FILE" --env-file "$ENV_FILE" run --rm api \
            alembic upgrade head
        
        log_success "Database migrations completed"
    else
        log_warning "Skipping database migrations (RUN_MIGRATIONS=false)"
    fi
}

deploy_services() {
    log_info "Deploying services..."
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        # Production deployment using Kubernetes
        log_info "Deploying to Kubernetes..."
        
        # Apply Kubernetes manifests
        kubectl apply -f "$PROJECT_ROOT/k8s/namespace.yml"
        kubectl apply -f "$PROJECT_ROOT/k8s/configmap.yml"
        kubectl apply -f "$PROJECT_ROOT/k8s/secrets.yml"
        kubectl apply -f "$PROJECT_ROOT/k8s/postgres.yml"
        kubectl apply -f "$PROJECT_ROOT/k8s/redis.yml"
        kubectl apply -f "$PROJECT_ROOT/k8s/api.yml"
        kubectl apply -f "$PROJECT_ROOT/k8s/worker.yml"
        kubectl apply -f "$PROJECT_ROOT/k8s/frontend.yml"
        kubectl apply -f "$PROJECT_ROOT/k8s/monitoring.yml"
        
        # Wait for deployment to be ready
        kubectl rollout status deployment/claims-api -n claims-processor
        kubectl rollout status deployment/claims-worker -n claims-processor
        kubectl rollout status deployment/claims-frontend -n claims-processor
        
    else
        # Development/Staging deployment using Docker Compose
        log_info "Deploying with Docker Compose..."
        
        # Stop existing services
        docker-compose -f "$PROJECT_ROOT/$DOCKER_COMPOSE_FILE" --env-file "$ENV_FILE" down
        
        # Start services
        docker-compose -f "$PROJECT_ROOT/$DOCKER_COMPOSE_FILE" --env-file "$ENV_FILE" up -d
        
        # Wait for services to be healthy
        log_info "Waiting for services to be healthy..."
        sleep 30
        
        # Check service health
        docker-compose -f "$PROJECT_ROOT/$DOCKER_COMPOSE_FILE" --env-file "$ENV_FILE" ps
    fi
    
    log_success "Services deployed successfully"
}

run_health_checks() {
    log_info "Running health checks..."
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        # Kubernetes health checks
        API_POD=$(kubectl get pods -n claims-processor -l app=claims-api -o jsonpath='{.items[0].metadata.name}')
        kubectl exec -n claims-processor "$API_POD" -- curl -f http://localhost:8000/health
        
    else
        # Docker Compose health checks
        MAX_RETRIES=30
        RETRY_COUNT=0
        
        while [[ $RETRY_COUNT -lt $MAX_RETRIES ]]; do
            if curl -f http://localhost:8000/health &> /dev/null; then
                log_success "API health check passed"
                break
            fi
            
            RETRY_COUNT=$((RETRY_COUNT + 1))
            log_info "Health check attempt $RETRY_COUNT/$MAX_RETRIES..."
            sleep 10
        done
        
        if [[ $RETRY_COUNT -eq $MAX_RETRIES ]]; then
            log_error "Health checks failed after $MAX_RETRIES attempts"
        fi
    fi
    
    log_success "Health checks passed"
}

run_smoke_tests() {
    log_info "Running smoke tests..."
    
    # Basic API endpoints test
    if [[ "$ENVIRONMENT" == "production" ]]; then
        API_URL="https://api.claims-processor.company.com"
    else
        API_URL="http://localhost:8000"
    fi
    
    # Test health endpoint
    curl -f "$API_URL/health" || log_error "Health endpoint failed"
    
    # Test API version endpoint
    curl -f "$API_URL/api/v1/health" || log_error "API v1 health endpoint failed"
    
    log_success "Smoke tests passed"
}

cleanup() {
    log_info "Cleaning up temporary files..."
    
    # Remove temporary files
    rm -f /tmp/claims_deploy_*
    
    # Clean up old Docker images (keep last 3 versions)
    docker image prune -f
    
    log_success "Cleanup completed"
}

show_deployment_info() {
    log_success "Deployment completed successfully!"
    
    echo ""
    echo "=== Deployment Information ==="
    echo "Environment: $ENVIRONMENT"
    echo "Timestamp: $(date)"
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        echo "API URL: https://api.claims-processor.company.com"
        echo "Frontend URL: https://claims-processor.company.com"
        echo "Monitoring: https://monitoring.claims-processor.company.com"
    else
        echo "API URL: http://localhost:8000"
        echo "Frontend URL: http://localhost:3000"
        echo "Monitoring: http://localhost:3001"
    fi
    
    echo ""
    echo "=== Next Steps ==="
    echo "1. Monitor logs: docker-compose logs -f (dev/staging) or kubectl logs -f deployment/claims-api (prod)"
    echo "2. Check metrics: Navigate to monitoring dashboard"
    echo "3. Run integration tests: pytest tests/integration"
    echo ""
}

# Main execution
main() {
    log_info "Starting deployment to $ENVIRONMENT environment..."
    
    check_prerequisites
    validate_environment
    
    if [[ "$ENVIRONMENT" != "development" ]]; then
        backup_database
    fi
    
    run_tests
    build_images
    
    if [[ "$ENVIRONMENT" != "development" ]]; then
        run_migrations
    fi
    
    deploy_services
    run_health_checks
    run_smoke_tests
    cleanup
    show_deployment_info
}

# Script usage
show_usage() {
    echo "Usage: $0 <environment> [options]"
    echo ""
    echo "Environments:"
    echo "  development, dev    - Deploy to development environment"
    echo "  staging            - Deploy to staging environment"
    echo "  production, prod   - Deploy to production environment"
    echo ""
    echo "Environment Variables:"
    echo "  BACKUP_BEFORE_DEPLOY  - Create database backup before deploy (default: false)"
    echo "  RUN_MIGRATIONS        - Run database migrations (default: true)"
    echo "  RUN_TESTS            - Run tests before deployment (default: true)"
    echo ""
    echo "Examples:"
    echo "  $0 development"
    echo "  BACKUP_BEFORE_DEPLOY=true $0 staging"
    echo "  RUN_TESTS=false $0 production"
}

# Handle script arguments
if [[ $# -eq 0 ]]; then
    show_usage
    exit 1
fi

if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    show_usage
    exit 0
fi

# Run main function
main "$@"
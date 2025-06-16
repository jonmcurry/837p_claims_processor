# Installation Guide

This guide provides comprehensive instructions for installing and configuring the 837P Claims Processing System in different environments.

## Prerequisites

### System Requirements

#### Minimum Requirements (Development)
- **CPU**: 4 cores, 2.5GHz
- **RAM**: 16GB
- **Storage**: 100GB SSD
- **Network**: 1Gbps connection

#### Recommended Requirements (Production)
- **CPU**: 16+ cores, 3.0GHz
- **RAM**: 64GB+
- **Storage**: 1TB NVMe SSD
- **Network**: 10Gbps connection
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for ML workloads)

### Software Dependencies

#### Core Dependencies
- **Python**: 3.9 or higher
- **PostgreSQL**: 13.0 or higher
- **Redis**: 6.0 or higher
- **Docker**: 20.10 or higher
- **Docker Compose**: 2.0 or higher

#### Optional Dependencies
- **Kubernetes**: 1.20+ (for production orchestration)
- **NGINX**: 1.20+ (for load balancing)
- **Prometheus**: 2.30+ (for monitoring)
- **Grafana**: 8.0+ (for dashboards)

## Installation Methods

### Method 1: Docker Compose (Recommended for Development)

#### 1. Clone Repository
```bash
git clone https://github.com/jonmcurry/837p_claims_processor.git
cd 837p_claims_processor
```

#### 2. Environment Configuration
```bash
# Copy environment template
cp config/.env.example config/.env.development

# Edit configuration (see Configuration section below)
nano config/.env.development
```

#### 3. Start Services
```bash
# Start all services
docker-compose up -d

# Verify services are running
docker-compose ps

# Check logs
docker-compose logs -f claims_processor
```

#### 4. Initialize Database
```bash
# Run database migrations
docker-compose exec claims_processor python -m alembic upgrade head

# Load initial data
docker-compose exec claims_processor python -m scripts.load_initial_data
```

#### 5. Verify Installation
```bash
# Test API endpoint
curl http://localhost:8000/health

# Check processing pipeline
curl http://localhost:8000/api/v1/status
```

### Method 2: Manual Installation

#### 1. Install System Dependencies

**Ubuntu/Debian:**
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install PostgreSQL
sudo apt install postgresql postgresql-contrib -y
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Install Redis
sudo apt install redis-server -y
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Install Python 3.9+
sudo apt install python3.9 python3.9-venv python3.9-dev -y

# Install additional dependencies
sudo apt install build-essential libpq-dev -y
```

**CentOS/RHEL:**
```bash
# Install PostgreSQL
sudo dnf install postgresql postgresql-server postgresql-contrib -y
sudo postgresql-setup --initdb
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Install Redis
sudo dnf install redis -y
sudo systemctl start redis
sudo systemctl enable redis

# Install Python 3.9+
sudo dnf install python39 python39-devel -y
```

#### 2. Create Database and User
```sql
-- Connect as postgres user
sudo -u postgres psql

-- Create database and user
CREATE DATABASE claims_processor;
CREATE USER claims_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE claims_processor TO claims_user;

-- Enable extensions
\c claims_processor
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
```

#### 3. Python Environment Setup
```bash
# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

#### 4. Environment Configuration
```bash
# Copy environment template
cp config/.env.example config/.env.development

# Edit configuration file
nano config/.env.development
```

#### 5. Database Migration
```bash
# Run database migrations
python -m alembic upgrade head

# Load initial data
python -m scripts.load_initial_data
```

#### 6. Start Application
```bash
# Start API server
python -m src.api.production_main

# Start background workers (in separate terminals)
python -m src.processing.batch_processor.ultra_pipeline
python -m src.processing.ml_pipeline.advanced_predictor
```

### Method 3: Kubernetes Deployment

#### 1. Prerequisites
```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

#### 2. Create Namespace
```bash
kubectl create namespace claims-processor
kubectl config set-context --current --namespace=claims-processor
```

#### 3. Deploy PostgreSQL
```bash
# Add Bitnami repository
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Deploy PostgreSQL
helm install postgresql bitnami/postgresql \
  --set auth.postgresPassword=secure_password \
  --set auth.database=claims_processor \
  --set primary.persistence.size=100Gi \
  --set metrics.enabled=true
```

#### 4. Deploy Redis
```bash
# Deploy Redis
helm install redis bitnami/redis \
  --set auth.password=secure_password \
  --set master.persistence.size=10Gi \
  --set metrics.enabled=true
```

#### 5. Deploy Application
```bash
# Create ConfigMap
kubectl create configmap claims-config --from-env-file=config/.env.production

# Deploy application
kubectl apply -f k8s/
```

## Configuration

### Environment Variables

Create and configure your environment file:

```bash
# config/.env.development
# Database Configuration
DATABASE_URL=postgresql://claims_user:secure_password@localhost:5432/claims_processor
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=
REDIS_MAX_CONNECTIONS=50

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_DEBUG=false

# Processing Configuration
BATCH_SIZE=1000
MAX_WORKERS=8
ASYNC_WORKERS=4

# ML Configuration
ML_MODEL_PATH=/models/
ML_ENABLE_GPU=true
ML_BATCH_SIZE=32

# Security Configuration
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-here
ENCRYPTION_KEY=your-32-byte-encryption-key

# Monitoring Configuration
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
GRAFANA_ENABLED=true
GRAFANA_PORT=3000

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/var/log/claims_processor.log
```

### Database Configuration

#### PostgreSQL Optimization Settings
```sql
-- postgresql.conf optimizations
shared_buffers = 8GB                    # 25% of RAM
effective_cache_size = 24GB             # 75% of RAM
work_mem = 256MB                        # For complex queries
maintenance_work_mem = 2GB              # For VACUUM, CREATE INDEX
checkpoint_completion_target = 0.9      # Smooth checkpoints
wal_buffers = 16MB                      # WAL buffer size
default_statistics_target = 100        # Query planning statistics
random_page_cost = 1.1                 # SSD optimization
effective_io_concurrency = 200         # Concurrent I/O operations

# Enable query statistics
shared_preload_libraries = 'pg_stat_statements'
pg_stat_statements.track = all
pg_stat_statements.save = on
```

#### Redis Configuration
```conf
# redis.conf optimizations
maxmemory 4gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
rdbcompression yes
rdbchecksum yes
```

### Security Configuration

#### SSL/TLS Setup
```bash
# Generate SSL certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/ssl/private/claims_processor.key \
  -out /etc/ssl/certs/claims_processor.crt

# Set proper permissions
chmod 600 /etc/ssl/private/claims_processor.key
chmod 644 /etc/ssl/certs/claims_processor.crt
```

#### Firewall Configuration
```bash
# Ubuntu UFW
sudo ufw allow 8000/tcp  # API
sudo ufw allow 9090/tcp  # Prometheus
sudo ufw allow 3000/tcp  # Grafana
sudo ufw enable

# CentOS/RHEL firewalld
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --permanent --add-port=9090/tcp
sudo firewall-cmd --permanent --add-port=3000/tcp
sudo firewall-cmd --reload
```

## Post-Installation Setup

### 1. Load Initial Data

```bash
# Load medical codes and validation rules
python -m scripts.load_medical_codes --source=cms_codes.csv
python -m scripts.load_validation_rules --source=business_rules.json

# Load provider and payer data
python -m scripts.load_providers --source=providers.csv
python -m scripts.load_payers --source=payers.csv
```

### 2. Create Admin User

```bash
# Create first administrator user
python -m scripts.create_admin_user \
  --username=admin \
  --email=admin@example.com \
  --password=secure_admin_password
```

### 3. Configure Monitoring

```bash
# Start monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Import Grafana dashboards
python -m scripts.import_grafana_dashboards --dashboard-dir=monitoring/grafana/dashboards/
```

### 4. Performance Tuning

```bash
# Run performance optimization
python -m scripts.optimize_database_performance
python -m scripts.warm_cache_startup
python -m scripts.tune_worker_processes
```

## Verification and Testing

### Health Check
```bash
# Basic health check
curl -f http://localhost:8000/health || exit 1

# Detailed system status
curl http://localhost:8000/api/v1/system/status
```

### Database Connectivity
```bash
# Test database connection
python -c "
from src.core.database import get_db_session
async def test():
    async with get_db_session() as session:
        result = await session.execute('SELECT 1')
        print('Database connection successful')
import asyncio
asyncio.run(test())
"
```

### Processing Pipeline Test
```bash
# Submit test claim
curl -X POST http://localhost:8000/api/v1/claims/submit \
  -H "Content-Type: application/json" \
  -d @test_data/sample_claim.json

# Check processing status
curl http://localhost:8000/api/v1/claims/status/test-claim-id
```

### Performance Test
```bash
# Run performance benchmarks
python -m tests.performance.test_system_performance

# Load test with sample data
python -m scripts.load_test --claims=1000 --duration=300
```

## Troubleshooting

### Common Installation Issues

#### Database Connection Issues
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test connection
psql -h localhost -U claims_user -d claims_processor -c "SELECT 1;"

# Check logs
sudo tail -f /var/log/postgresql/postgresql-*.log
```

#### Redis Connection Issues
```bash
# Check Redis status
sudo systemctl status redis-server

# Test connection
redis-cli ping

# Check configuration
redis-cli config get "*"
```

#### Permission Issues
```bash
# Fix file permissions
sudo chown -R claims_user:claims_user /opt/claims_processor/
sudo chmod -R 755 /opt/claims_processor/

# Fix log permissions
sudo mkdir -p /var/log/claims_processor/
sudo chown claims_user:claims_user /var/log/claims_processor/
```

#### Memory Issues
```bash
# Check system memory
free -h

# Monitor application memory usage
ps aux | grep python

# Adjust worker processes
export MAX_WORKERS=4  # Reduce if memory constrained
```

### Log Analysis
```bash
# Application logs
tail -f /var/log/claims_processor/app.log

# Error logs
tail -f /var/log/claims_processor/error.log

# Performance logs
tail -f /var/log/claims_processor/performance.log

# Access logs
tail -f /var/log/claims_processor/access.log
```

### Service Management
```bash
# Docker Compose
docker-compose restart claims_processor
docker-compose logs -f claims_processor

# Systemd (manual installation)
sudo systemctl restart claims_processor
sudo systemctl status claims_processor
sudo journalctl -u claims_processor -f
```

## Next Steps

After successful installation:

1. **Review Configuration**: See [Configuration Management](./configuration.md)
2. **Security Setup**: Review [Security Documentation](../security/)
3. **Performance Tuning**: See [Performance Testing](../testing/performance-testing.md)
4. **Monitoring Setup**: Configure [Monitoring and Alerting](../architecture/system-overview.md#monitoring--observability)
5. **Production Deployment**: Follow [Production Deployment Guide](./production-deployment.md)

---

For deployment-specific configurations, see:
- [Docker Deployment](./docker-deployment.md)
- [Production Deployment](./production-deployment.md)
- [Configuration Management](./configuration.md)
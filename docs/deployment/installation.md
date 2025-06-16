# Installation Guide

This guide provides comprehensive instructions for installing and configuring the 837P Claims Processing System on Windows Server.

## Prerequisites

### System Requirements

#### Minimum Requirements (Development)
- **OS**: Windows Server 2019 or higher / Windows 10 Pro
- **CPU**: 4 cores, 2.5GHz
- **RAM**: 16GB
- **Storage**: 100GB SSD
- **Network**: 1Gbps connection

#### Recommended Requirements (Production)
- **OS**: Windows Server 2022 
- **CPU**: 16+ cores, 3.0GHz
- **RAM**: 64GB+
- **Storage**: 1TB NVMe SSD
- **Network**: 10Gbps connection
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for ML workloads)

### Software Dependencies

#### Core Dependencies
- **Python**: 3.9 or higher
- **PostgreSQL**: 13.0 or higher
- **Redis**: 6.0 or higher (Redis for Windows or Memurai)
- **Git**: Latest version
- **Visual Studio Build Tools**: For compiling Python packages

#### Optional Dependencies
- **IIS**: For reverse proxy (alternative to NGINX)
- **Prometheus**: 2.30+ (for monitoring)
- **Grafana**: 8.0+ (for dashboards)

## Installation Steps

### Step 1: Install Python

Download and install Python from the official website:

```powershell
# Download Python 3.9+ from python.org
# Or use Chocolatey package manager
choco install python --version=3.9.13

# Verify installation
python --version
pip --version
```

### Step 2: Install PostgreSQL

```powershell
# Download PostgreSQL installer from postgresql.org
# Or use Chocolatey
choco install postgresql --params '/Password:YourSecurePassword'

# Verify installation
psql --version

# Start PostgreSQL service
Start-Service postgresql-x64-13
Set-Service postgresql-x64-13 -StartupType Automatic
```

### Step 3: Install Redis

```powershell
# Option 1: Install Redis for Windows (community version)
# Download from https://github.com/microsoftarchive/redis/releases

# Option 2: Install Memurai (Redis-compatible, Windows-native)
choco install memurai-developer

# Start Redis service
Start-Service Redis
Set-Service Redis -StartupType Automatic

# Verify Redis is running
redis-cli ping
```

### Step 4: Install Git and Visual Studio Build Tools

```powershell
# Install Git
choco install git

# Install Visual Studio Build Tools (required for Python package compilation)
choco install visualstudio2022buildtools --package-parameters "--add Microsoft.VisualStudio.Workload.VCTools"

# Install SQL Server ODBC driver for Python connectivity
choco install sql-server-odbc-driver -y
```

### Step 5: Clone Repository and Setup Environment

```powershell
# Clone repository
git clone https://github.com/jonmcurry/837p_claims_processor.git
cd 837p_claims_processor

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Install additional Windows-specific dependencies
pip install pywin32 wmi psutil

# Install SQL Server connectivity packages
pip install pyodbc sqlalchemy[mssql]
```

### Step 6: Install SQL Server

```powershell
# Download and install SQL Server 2022 Developer/Standard Edition
# For production, use Standard or Enterprise edition
choco install sql-server-2022 -y

# Install SQL Server Management Studio (SSMS)
choco install sql-server-management-studio -y

# Start SQL Server services
Start-Service MSSQLSERVER
Set-Service MSSQLSERVER -StartupType Automatic

Start-Service SQLSERVERAGENT
Set-Service SQLSERVERAGENT -StartupType Automatic

# Verify SQL Server installation
sqlcmd -S localhost -E -Q "SELECT @@VERSION"
```

### Step 7: Database Setup

#### PostgreSQL Setup (Staging Database)
```powershell
# Create PostgreSQL database and user
psql -U postgres

# In PostgreSQL prompt:
CREATE DATABASE claims_processor_staging;
CREATE USER claims_staging_user WITH PASSWORD 'YourSecurePassword';
GRANT ALL PRIVILEGES ON DATABASE claims_processor_staging TO claims_staging_user;

# Enable required extensions
\c claims_processor_staging
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
\q
```

#### SQL Server Setup (Analytics Database)
```powershell
# Create SQL Server database and user
sqlcmd -S localhost -E -Q "CREATE DATABASE ClaimsProcessingProduction"

# Create login and user for claims processing
sqlcmd -S localhost -E -Q "CREATE LOGIN claims_analytics_user WITH PASSWORD = 'YourSecureSQLPassword'"
sqlcmd -S localhost -E -d ClaimsProcessingProduction -Q "CREATE USER claims_analytics_user FOR LOGIN claims_analytics_user"
sqlcmd -S localhost -E -d ClaimsProcessingProduction -Q "ALTER ROLE db_owner ADD MEMBER claims_analytics_user"

# Create data directories
New-Item -ItemType Directory -Force -Path "C:\Data"
New-Item -ItemType Directory -Force -Path "C:\Logs"
New-Item -ItemType Directory -Force -Path "C:\TempDB"

# Apply SQL Server schema
sqlcmd -S localhost -E -d ClaimsProcessingProduction -i "database\sqlserver_schema.sql"

# Create materialized views
sqlcmd -S localhost -E -d ClaimsProcessingProduction -i "database\materialized_views.sql"

# Verify database setup
sqlcmd -S localhost -E -d ClaimsProcessingProduction -Q "SELECT COUNT(*) FROM sys.tables"
```

### Step 8: Environment Configuration

```powershell
# Copy environment template
copy config\.env.example config\.env.production

# Edit configuration file
notepad config\.env.production
```

Configure the following key settings:

```ini
# config/.env.production

# Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Database Configuration - Dual Database Setup
# PostgreSQL (Staging/Processing Database)
DATABASE_URL=postgresql://claims_staging_user:YourSecurePassword@localhost:5432/claims_processor_staging
DATABASE_POOL_SIZE=50
DATABASE_MAX_OVERFLOW=100

# SQL Server (Analytics Database)
ANALYTICS_DATABASE_URL=mssql+pyodbc://claims_analytics_user:YourSecureSQLPassword@localhost/ClaimsProcessingProduction?driver=ODBC+Driver+17+for+SQL+Server
ANALYTICS_DATABASE_POOL_SIZE=30
ANALYTICS_DATABASE_MAX_OVERFLOW=60

# Redis Configuration  
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=
REDIS_MAX_CONNECTIONS=100

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=8

# Security Configuration
SECRET_KEY=your-secret-key-32-characters-minimum
JWT_SECRET_KEY=your-jwt-secret-32-characters-minimum
ENCRYPTION_KEY=your-32-character-encryption-key

# File Paths (Windows-style)
UPLOAD_DIR=C:\Claims_Processor\uploads\
LOG_FILE=C:\Claims_Processor\logs\claims_processor.log
TEMP_DIR=C:\Claims_Processor\temp\

# Windows Service Configuration
WINDOWS_SERVICE_NAME=ClaimsProcessor
WINDOWS_SERVICE_DISPLAY_NAME=837P Claims Processor
WINDOWS_SERVICE_DESCRIPTION=High-performance HIPAA-compliant claims processing service
```

### Step 9: Database Migration

```powershell
# Run PostgreSQL migrations
python -m alembic upgrade head

# Run SQL Server schema setup (already done in Step 7)
# Verify SQL Server tables are created
sqlcmd -S localhost -E -d ClaimsProcessingProduction -Q "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"

# Load initial data
python -m scripts.load_initial_data --env=production
```

### Step 10: Create Required Directories

```powershell
# Create application directories
New-Item -ItemType Directory -Force -Path "C:\Claims_Processor\logs"
New-Item -ItemType Directory -Force -Path "C:\Claims_Processor\uploads"
New-Item -ItemType Directory -Force -Path "C:\Claims_Processor\temp"
New-Item -ItemType Directory -Force -Path "C:\Claims_Processor\backups"

# Set appropriate permissions
icacls "C:\Claims_Processor" /grant "IIS_IUSRS:(OI)(CI)F" /T
```

### Step 11: Install as Windows Service

Create a Windows service installer script:

```python
# scripts/install_windows_service.py
import os
import sys
import win32serviceutil
import win32service
import win32event
import servicemanager
import socket
import time
import logging

class ClaimsProcessorService(win32serviceutil.ServiceFramework):
    _svc_name_ = "ClaimsProcessor"
    _svc_display_name_ = "837P Claims Processor"
    _svc_description_ = "High-performance HIPAA-compliant claims processing service"
    
    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        socket.setdefaulttimeout(60)
        
    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)
        
    def SvcDoRun(self):
        servicemanager.LogMsg(servicemanager.EVENTLOG_INFORMATION_TYPE,
                            servicemanager.PYS_SERVICE_STARTED,
                            (self._svc_name_, ''))
        self.main()
        
    def main(self):
        # Import and start the application
        import uvicorn
        from src.api.production_main import app
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            workers=1,  # Use 1 worker for Windows service
            loop="asyncio"
        )

if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(ClaimsProcessorService)
```

Install the service:

```powershell
# Install the Windows service
python scripts\install_windows_service.py install

# Start the service
python scripts\install_windows_service.py start

# Set service to start automatically
sc config ClaimsProcessor start= auto
```

### Step 12: Configure IIS Reverse Proxy (Optional)

If using IIS as a reverse proxy:

```powershell
# Install IIS and URL Rewrite module
Enable-WindowsOptionalFeature -Online -FeatureName IIS-WebServerRole
Enable-WindowsOptionalFeature -Online -FeatureName IIS-WebServer
Enable-WindowsOptionalFeature -Online -FeatureName IIS-HttpRedirect

# Download and install URL Rewrite module from Microsoft
# Create web.config for reverse proxy
```

web.config example:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <system.webServer>
        <rewrite>
            <rules>
                <rule name="Claims Processor API" stopProcessing="true">
                    <match url="(.*)" />
                    <action type="Rewrite" url="http://localhost:8000/{R:1}" />
                </rule>
            </rules>
        </rewrite>
    </system.webServer>
</configuration>
```

## Post-Installation Setup

### 1. Verify Installation

```powershell
# Test API endpoint
Invoke-RestMethod -Uri "http://localhost:8000/health"

# Check Windows service status
Get-Service -Name "ClaimsProcessor"

# Check logs
Get-Content "C:\Claims_Processor\logs\claims_processor.log" -Tail 20
```

### 2. Load Test Data

```powershell
# Load medical codes and validation rules
python -m scripts.load_medical_codes --source="data\cms_codes.csv"
python -m scripts.load_validation_rules --source="data\business_rules.json"

# Load provider and payer data
python -m scripts.load_providers --source="data\providers.csv"
python -m scripts.load_payers --source="data\payers.csv"
```

### 3. Create Admin User

```powershell
# Create first administrator user
python -m scripts.create_admin_user --username=admin --email=admin@company.com --password=SecurePassword123!
```

### 4. Configure Windows Firewall

```powershell
# Allow API port through Windows Firewall
New-NetFirewallRule -DisplayName "Claims Processor API" -Direction Inbound -Port 8000 -Protocol TCP -Action Allow

# Allow Prometheus port (if using monitoring)
New-NetFirewallRule -DisplayName "Prometheus" -Direction Inbound -Port 9090 -Protocol TCP -Action Allow

# Allow Grafana port (if using monitoring)
New-NetFirewallRule -DisplayName "Grafana" -Direction Inbound -Port 3000 -Protocol TCP -Action Allow
```

### 5. Setup Monitoring (Optional)

```powershell
# Install Prometheus for Windows
# Download from prometheus.io/download
# Extract to C:\Prometheus\

# Create Prometheus configuration
# Copy monitoring\prometheus\prometheus.yml to C:\Prometheus\

# Install as Windows service
# Use NSSM (Non-Sucking Service Manager) or similar tool
choco install nssm

# Install Prometheus service
nssm install prometheus "C:\Prometheus\prometheus.exe"
nssm set prometheus Parameters "--config.file=C:\Prometheus\prometheus.yml"
nssm set prometheus Start SERVICE_AUTO_START

# Start Prometheus
nssm start prometheus
```

## Troubleshooting

### Common Installation Issues

#### Python Package Installation Failures
```powershell
# Ensure Visual Studio Build Tools are installed
# Install specific packages that might fail
pip install --only-binary=all psycopg2-binary
pip install --upgrade setuptools wheel
```

#### PostgreSQL Connection Issues
```powershell
# Check PostgreSQL service status
Get-Service -Name "postgresql*"

# Test connection
psql -h localhost -U claims_user -d claims_processor

# Check pg_hba.conf configuration
# Usually located in: C:\Program Files\PostgreSQL\13\data\pg_hba.conf
```

#### Redis Connection Issues
```powershell
# Check Redis/Memurai service
Get-Service -Name "Redis*" -Or -Name "Memurai*"

# Test Redis connection
redis-cli ping

# Check Redis configuration
# Default config location varies by installation method
```

#### Windows Service Issues
```powershell
# Check service status and logs
Get-Service -Name "ClaimsProcessor"
Get-EventLog -LogName Application -Source "ClaimsProcessor" -Newest 10

# Restart service
Restart-Service -Name "ClaimsProcessor"

# Uninstall and reinstall service if needed
python scripts\install_windows_service.py remove
python scripts\install_windows_service.py install
```

#### Performance Issues
```powershell
# Check system resources
Get-Counter "\Processor(_Total)\% Processor Time"
Get-Counter "\Memory\Available MBytes"

# Monitor application performance
# Use Windows Performance Monitor (perfmon.exe)
# Monitor custom application metrics
```

## Security Considerations

### Windows-Specific Security

1. **User Account Control**: Run services under dedicated service account
2. **File Permissions**: Restrict access to application directories
3. **Network Security**: Configure Windows Firewall appropriately
4. **Update Management**: Keep Windows Server and dependencies updated

```powershell
# Create dedicated service account
New-LocalUser -Name "claims_service" -Password (ConvertTo-SecureString "SecurePassword123!" -AsPlainText -Force) -AccountNeverExpires
Add-LocalGroupMember -Group "Users" -Member "claims_service"

# Set service to run under dedicated account
sc config ClaimsProcessor obj= ".\claims_service" password= "SecurePassword123!"
```

## Next Steps

After successful installation:

1. **Review Configuration**: See [Configuration Management](./configuration.md)
2. **Security Setup**: Review [Security Documentation](../security/)
3. **Performance Tuning**: See [Performance Testing](../testing/performance-testing.md)
4. **Monitoring Setup**: Configure [System Monitoring](../architecture/system-overview.md#monitoring--observability)
5. **Production Deployment**: Follow [Production Deployment Guide](./production-deployment.md)

---

For Windows-specific deployment configurations, see:
- [Windows Server Deployment](./windows-server-deployment.md)
- [Production Deployment](./production-deployment.md)
- [Configuration Management](./configuration.md)
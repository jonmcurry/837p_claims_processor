# Windows Server Deployment

This guide provides detailed instructions for deploying the 837P Claims Processing System on Windows Server in production environments.

## Prerequisites

### Windows Server Requirements

#### Recommended Production Configuration
- **OS**: Windows Server 2022 (latest patches)
- **CPU**: 32+ cores, 3.0GHz+ (Intel Xeon or AMD EPYC)
- **RAM**: 128GB+ for high-volume processing
- **Storage**: 
  - 500GB NVMe SSD for OS and application
  - 2TB+ SSD for database storage
  - 1TB+ for logs and backups
- **Network**: 10Gbps network interface
- **GPU**: NVIDIA A100 or RTX 4090 (for ML workloads)

#### High-Availability Configuration
- **Load Balancer**: Windows NLB or external load balancer
- **Application Servers**: 3+ Windows Server nodes
- **Database**: SQL Server Always On Availability Groups
- **Shared Storage**: SAN or Azure File Share

## Installation Process

### Step 1: Prepare Windows Server

```powershell
# Install required Windows features
Enable-WindowsOptionalFeature -Online -FeatureName IIS-WebServerRole
Enable-WindowsOptionalFeature -Online -FeatureName IIS-WebServer
Enable-WindowsOptionalFeature -Online -FeatureName IIS-HttpRedirect
Enable-WindowsOptionalFeature -Online -FeatureName IIS-HttpErrors
Enable-WindowsOptionalFeature -Online -FeatureName IIS-HttpLogging
Enable-WindowsOptionalFeature -Online -FeatureName IIS-Security
Enable-WindowsOptionalFeature -Online -FeatureName IIS-RequestFiltering

# Install Chocolatey package manager
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))

# Configure Windows for high performance
powercfg -setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c  # High Performance plan

# Configure memory settings for large workloads
bcdedit /set increaseuserva 3072
```

### Step 2: Install Dependencies

```powershell
# Install Python 3.9+
choco install python --version=3.9.13 -y

# Install PostgreSQL
choco install postgresql --params '/Password:SecurePassword123!' -y

# Install SQL Server (Enterprise/Standard for production)
choco install sql-server-2022 -y

# Install Redis (Memurai - Redis-compatible for Windows)
choco install memurai-developer -y

# Install Git
choco install git -y

# Install Visual Studio Build Tools
choco install visualstudio2022buildtools --package-parameters "--add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Workload.MSBuildTools" -y

# Install SQL Server ODBC Driver
choco install sql-server-odbc-driver -y

# Install NVIDIA drivers and CUDA (if using GPU)
choco install cuda -y

# Install monitoring tools
choco install nssm -y  # Non-Sucking Service Manager
```

### Step 3: Configure Services

```powershell
# Configure PostgreSQL
Start-Service postgresql-x64-13
Set-Service postgresql-x64-13 -StartupType Automatic

# Configure SQL Server
Start-Service MSSQLSERVER
Set-Service MSSQLSERVER -StartupType Automatic
Start-Service SQLSERVERAGENT
Set-Service SQLSERVERAGENT -StartupType Automatic

# Configure Memurai/Redis
Start-Service Memurai
Set-Service Memurai -StartupType Automatic

# Verify services are running
Get-Service postgresql-x64-13, MSSQLSERVER, SQLSERVERAGENT, Memurai

# Test connectivity
psql -U postgres -c "SELECT version();"
sqlcmd -S localhost -E -Q "SELECT @@VERSION"
redis-cli ping
```

### Step 4: Security Configuration

```powershell
# Create dedicated service account
$securePassword = ConvertTo-SecureString "SecureServicePassword123!" -AsPlainText -Force
New-LocalUser -Name "claims_service" -Password $securePassword -PasswordNeverExpires -AccountNeverExpires
Add-LocalGroupMember -Group "Log on as a service" -Member "claims_service"

# Configure Windows Firewall
New-NetFirewallRule -DisplayName "Claims Processor API" -Direction Inbound -Port 8000 -Protocol TCP -Action Allow
New-NetFirewallRule -DisplayName "PostgreSQL" -Direction Inbound -Port 5432 -Protocol TCP -Action Allow -RemoteAddress "192.168.1.0/24"
New-NetFirewallRule -DisplayName "Redis/Memurai" -Direction Inbound -Port 6379 -Protocol TCP -Action Allow -RemoteAddress "192.168.1.0/24"

# Disable unnecessary services for security
Stop-Service -Name "Themes", "Windows Search" -Force
Set-Service -Name "Themes", "Windows Search" -StartupType Disabled
```

### Step 5: Application Deployment

```powershell
# Create application directory structure
$appRoot = "C:\Claims_Processor"
New-Item -ItemType Directory -Force -Path $appRoot
New-Item -ItemType Directory -Force -Path "$appRoot\app"
New-Item -ItemType Directory -Force -Path "$appRoot\logs"
New-Item -ItemType Directory -Force -Path "$appRoot\uploads"
New-Item -ItemType Directory -Force -Path "$appRoot\temp"
New-Item -ItemType Directory -Force -Path "$appRoot\backups"
New-Item -ItemType Directory -Force -Path "$appRoot\scripts"

# Clone application repository
Set-Location $appRoot
git clone https://github.com/jonmcurry/837p_claims_processor.git app
Set-Location "$appRoot\app"

# Create Python virtual environment
python -m venv "$appRoot\venv"
& "$appRoot\venv\Scripts\Activate.ps1"

# Install Python dependencies
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Install Windows-specific packages
pip install pywin32 wmi psutil

# Install SQL Server connectivity packages
pip install pyodbc sqlalchemy[mssql]

# Set file permissions
icacls $appRoot /grant "claims_service:(OI)(CI)F" /T
icacls $appRoot /grant "IIS_IUSRS:(OI)(CI)RX" /T
```

### Step 6: Database Setup

#### PostgreSQL Setup (Staging Database)
```powershell
# Create PostgreSQL production database
psql -U postgres << EOF
CREATE DATABASE claims_processor_staging;
CREATE USER claims_staging_user WITH PASSWORD 'SecureDatabasePassword123!';
GRANT ALL PRIVILEGES ON DATABASE claims_processor_staging TO claims_staging_user;

-- Enable extensions
\c claims_processor_staging
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
EOF

# Configure PostgreSQL for production
$pgConfigPath = "C:\Program Files\PostgreSQL\13\data\postgresql.conf"
$pgHbaPath = "C:\Program Files\PostgreSQL\13\data\pg_hba.conf"

# Update PostgreSQL configuration
Add-Content -Path $pgConfigPath -Value @"
# Claims Processor Production Settings
max_connections = 200
shared_buffers = 8GB
effective_cache_size = 24GB
work_mem = 256MB
maintenance_work_mem = 2GB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
max_worker_processes = 16
max_parallel_workers_per_gather = 4
max_parallel_workers = 16
"@

# Restart PostgreSQL to apply settings
Restart-Service postgresql-x64-13
```

#### SQL Server Setup (Analytics Database)
```powershell
# Install SQL Server 2022 Enterprise/Standard for production
# Download from Microsoft Volume Licensing or use evaluation
# choco install sql-server-2022 -y  # Use for Standard edition

# For Enterprise edition, run installer manually with production license

# Install SQL Server Management Studio
choco install sql-server-management-studio -y

# Configure SQL Server services
Start-Service MSSQLSERVER
Set-Service MSSQLSERVER -StartupType Automatic
Start-Service SQLSERVERAGENT  
Set-Service SQLSERVERAGENT -StartupType Automatic

# Create production analytics database
sqlcmd -S localhost -E -Q "CREATE DATABASE ClaimsProcessingProduction"

# Create dedicated SQL login for analytics
sqlcmd -S localhost -E -Q "CREATE LOGIN claims_analytics_user WITH PASSWORD = 'SecureAnalyticsPassword123!', CHECK_POLICY = OFF"
sqlcmd -S localhost -E -d ClaimsProcessingProduction -Q "CREATE USER claims_analytics_user FOR LOGIN claims_analytics_user"
sqlcmd -S localhost -E -d ClaimsProcessingProduction -Q "ALTER ROLE db_owner ADD MEMBER claims_analytics_user"

# Create data directories for SQL Server files
New-Item -ItemType Directory -Force -Path "C:\SQLData"
New-Item -ItemType Directory -Force -Path "C:\SQLLogs" 
New-Item -ItemType Directory -Force -Path "C:\SQLTempDB"

# Configure SQL Server for high performance
sqlcmd -S localhost -E -Q @"
EXEC sp_configure 'max server memory (MB)', 32768;  -- 32GB for SQL Server
EXEC sp_configure 'max degree of parallelism', 8;
EXEC sp_configure 'cost threshold for parallelism', 25;
EXEC sp_configure 'optimize for ad hoc workloads', 1;
RECONFIGURE WITH OVERRIDE;
"@

# Apply SQL Server schema for analytics database
Set-Location "$appRoot\app"
sqlcmd -S localhost -E -d ClaimsProcessingProduction -i "database\sqlserver_schema.sql"

# Create materialized views for analytics
sqlcmd -S localhost -E -d ClaimsProcessingProduction -i "database\materialized_views.sql"

# Verify SQL Server setup
sqlcmd -S localhost -E -d ClaimsProcessingProduction -Q "SELECT COUNT(*) as TableCount FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"
```

### Step 7: Application Configuration

```powershell
# Create production configuration
Copy-Item "$appRoot\app\config\.env.example" "$appRoot\app\config\.env.production"

# Update configuration file (edit manually or use script)
$configPath = "$appRoot\app\config\.env.production"
$configContent = @"
# Production Configuration
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING

# Database Configuration - Dual Database Setup
# PostgreSQL (Staging/Processing Database)
DATABASE_URL=postgresql://claims_staging_user:SecureDatabasePassword123!@localhost:5432/claims_processor_staging
DATABASE_POOL_SIZE=50
DATABASE_MAX_OVERFLOW=100

# SQL Server (Analytics Database)
ANALYTICS_DATABASE_URL=mssql+pyodbc://claims_analytics_user:SecureAnalyticsPassword123!@localhost/ClaimsProcessingProduction?driver=ODBC+Driver+17+for+SQL+Server
ANALYTICS_DATABASE_POOL_SIZE=30
ANALYTICS_DATABASE_MAX_OVERFLOW=60

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=100

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=16

# Security Configuration
SECRET_KEY=your-production-secret-key-32-characters-minimum
JWT_SECRET_KEY=your-production-jwt-secret-32-characters-minimum
ENCRYPTION_KEY=your-32-character-encryption-key

# File Paths
UPLOAD_DIR=C:\Claims_Processor\uploads\
LOG_FILE=C:\Claims_Processor\logs\claims_processor.log
TEMP_DIR=C:\Claims_Processor\temp\

# Windows Service Configuration
WINDOWS_SERVICE_NAME=ClaimsProcessor
WINDOWS_SERVICE_DISPLAY_NAME=837P Claims Processor
WINDOWS_SERVICE_DESCRIPTION=Production HIPAA-compliant claims processing service

# Performance Settings
BATCH_SIZE=2000
MAX_WORKERS=20
ASYNC_WORKERS=8

# Monitoring
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
GRAFANA_ENABLED=true
GRAFANA_PORT=3000
"@

Set-Content -Path $configPath -Value $configContent
```

### Step 8: Database Migration and Initial Data

```powershell
# Set environment variables
$env:DATABASE_URL = "postgresql://claims_prod_user:SecureDatabasePassword123!@localhost:5432/claims_processor_prod"

# Run database migrations
Set-Location "$appRoot\app"
& "$appRoot\venv\Scripts\Activate.ps1"
python -m alembic upgrade head

# Load initial data
python -m scripts.load_initial_data --env=production
python -m scripts.load_medical_codes --source="data\cms_codes.csv"
python -m scripts.load_validation_rules --source="data\business_rules.json"

# Create admin user
python -m scripts.create_admin_user --username=admin --email=admin@company.com --password=AdminPassword123!
```

### Step 9: Install as Windows Service

Create the Windows service script:

```python
# scripts/windows_service_production.py
import os
import sys
import win32serviceutil
import win32service
import win32event
import servicemanager
import socket
import time
import logging
from pathlib import Path

# Add application path to Python path
app_root = Path("C:/Claims_Processor/app")
sys.path.insert(0, str(app_root))

class ClaimsProcessorProductionService(win32serviceutil.ServiceFramework):
    _svc_name_ = "ClaimsProcessor"
    _svc_display_name_ = "837P Claims Processor"
    _svc_description_ = "Production HIPAA-compliant claims processing service"
    
    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        socket.setdefaulttimeout(60)
        
        # Setup logging
        logging.basicConfig(
            filename='C:/Claims_Processor/logs/service.log',
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        self.logger.info("Service stop requested")
        win32event.SetEvent(self.hWaitStop)
        
    def SvcDoRun(self):
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, '')
        )
        self.logger.info("Service starting")
        self.main()
        
    def main(self):
        try:
            # Set environment variables
            os.environ['ENVIRONMENT'] = 'production'
            os.environ['PYTHONPATH'] = str(app_root)
            
            # Change to application directory
            os.chdir(app_root)
            
            # Import and start the application
            import uvicorn
            from src.api.production_main import app
            
            self.logger.info("Starting Claims Processor application")
            
            uvicorn.run(
                app,
                host="0.0.0.0",
                port=8000,
                workers=1,  # Single worker for Windows service
                loop="asyncio",
                access_log=True,
                log_config={
                    "version": 1,
                    "disable_existing_loggers": False,
                    "formatters": {
                        "default": {
                            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                        }
                    },
                    "handlers": {
                        "file": {
                            "class": "logging.handlers.RotatingFileHandler",
                            "filename": "C:/Claims_Processor/logs/app.log",
                            "maxBytes": 100*1024*1024,  # 100MB
                            "backupCount": 10,
                            "formatter": "default"
                        }
                    },
                    "root": {
                        "level": "INFO",
                        "handlers": ["file"]
                    }
                }
            )
            
        except Exception as e:
            self.logger.error(f"Service error: {e}")
            servicemanager.LogMsg(
                servicemanager.EVENTLOG_ERROR_TYPE,
                servicemanager.PYS_SERVICE_STOPPED,
                (self._svc_name_, str(e))
            )

if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(ClaimsProcessorProductionService)
```

Install and start the service:

```powershell
# Install the Windows service
Copy-Item "$appRoot\app\scripts\windows_service_production.py" "$appRoot\scripts\"
Set-Location "$appRoot\scripts"
& "$appRoot\venv\Scripts\python.exe" windows_service_production.py install

# Configure service to run under dedicated account
sc config ClaimsProcessor obj= ".\claims_service" password= "SecureServicePassword123!"

# Set service recovery options
sc failure ClaimsProcessor reset= 86400 actions= restart/60000/restart/60000/restart/60000

# Start the service
sc start ClaimsProcessor

# Set to start automatically
sc config ClaimsProcessor start= auto

# Verify service is running
Get-Service -Name "ClaimsProcessor"
```

## Production Monitoring Setup

### Step 10: Install Prometheus and Grafana

```powershell
# Download and install Prometheus
$prometheusVersion = "2.40.7"
$prometheusUrl = "https://github.com/prometheus/prometheus/releases/download/v$prometheusVersion/prometheus-$prometheusVersion.windows-amd64.zip"
Invoke-WebRequest -Uri $prometheusUrl -OutFile "C:\temp\prometheus.zip"
Expand-Archive -Path "C:\temp\prometheus.zip" -DestinationPath "C:\Prometheus"

# Create Prometheus configuration
$prometheusConfig = @"
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'claims-processor'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
    
  - job_name: 'windows-exporter'
    static_configs:
      - targets: ['localhost:9182']
    scrape_interval: 30s
"@

Set-Content -Path "C:\Prometheus\prometheus.yml" -Value $prometheusConfig

# Install Prometheus as Windows service
nssm install prometheus "C:\Prometheus\prometheus.exe"
nssm set prometheus Parameters "--config.file=C:\Prometheus\prometheus.yml --storage.tsdb.path=C:\Prometheus\data"
nssm set prometheus Start SERVICE_AUTO_START
nssm start prometheus

# Install Windows Exporter for system metrics
choco install prometheus-windows-exporter.install -y

# Install Grafana
choco install grafana -y
Start-Service Grafana
Set-Service Grafana -StartupType Automatic
```

### Step 11: Configure IIS Load Balancer (Optional)

```powershell
# Install URL Rewrite module for IIS
$urlRewriteUrl = "https://download.microsoft.com/download/1/2/8/128E2E22-C1B9-44A4-BE2A-5859ED1D4592/rewrite_amd64_en-US.msi"
Invoke-WebRequest -Uri $urlRewriteUrl -OutFile "C:\temp\urlrewrite.msi"
Start-Process msiexec.exe -Wait -ArgumentList "/i C:\temp\urlrewrite.msi /quiet"

# Create IIS site for load balancing
Import-Module WebAdministration
New-WebSite -Name "ClaimsProcessorLB" -Port 80 -PhysicalPath "C:\inetpub\wwwroot\claims" -Force
```

Create web.config for load balancing:

```xml
<!-- C:\inetpub\wwwroot\claims\web.config -->
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <system.webServer>
        <rewrite>
            <rules>
                <rule name="Claims Processor Load Balancer" stopProcessing="true">
                    <match url="(.*)" />
                    <action type="Rewrite" url="http://localhost:8000/{R:1}" />
                    <serverVariables>
                        <set name="HTTP_X_FORWARDED_PROTO" value="http" />
                        <set name="HTTP_X_FORWARDED_FOR" value="{REMOTE_ADDR}" />
                    </serverVariables>
                </rule>
            </rules>
        </rewrite>
        <httpRedirect enabled="false" />
    </system.webServer>
</configuration>
```

## High Availability Configuration

### Database High Availability

```powershell
# Configure PostgreSQL streaming replication
# On primary server
$pgConfigPath = "C:\Program Files\PostgreSQL\13\data\postgresql.conf"
Add-Content -Path $pgConfigPath -Value @"
# Replication settings
wal_level = replica
max_wal_senders = 3
max_replication_slots = 3
archive_mode = on
archive_command = 'copy "%p" "C:\PostgreSQL\archive\%f"'
"@

# Create replication user
psql -U postgres -c "CREATE USER replicator REPLICATION LOGIN ENCRYPTED PASSWORD 'ReplicationPassword123!';"
```

### Application Clustering

```powershell
# Configure Windows NLB (Network Load Balancing)
Install-WindowsFeature NLB -IncludeManagementTools

# Create NLB cluster (run on each node)
New-NlbCluster -InterfaceName "Ethernet" -OperationMode Multicast -ClusterName "ClaimsProcessorCluster" -ClusterPrimaryIP "192.168.1.100"

# Add port rule for Claims Processor
New-NlbClusterPortRule -Port 8000 -Protocol Tcp -Mode Multiple -Affinity None
```

## Security Hardening

### Step 12: Security Configuration

```powershell
# Configure Windows Defender
Set-MpPreference -DisableRealtimeMonitoring $false
Set-MpPreference -DisableBehaviorMonitoring $false
Set-MpPreference -DisableScriptScanning $false

# Configure audit policies
auditpol /set /category:"Logon/Logoff" /success:enable /failure:enable
auditpol /set /category:"Account Management" /success:enable /failure:enable
auditpol /set /category:"Privilege Use" /success:enable /failure:enable

# Configure SSL/TLS
# Generate self-signed certificate for development/testing
$cert = New-SelfSignedCertificate -DnsName "claims-processor.local" -CertStoreLocation "cert:\LocalMachine\My"
$thumbprint = $cert.Thumbprint

# For production, use proper SSL certificate from CA
# Import-PfxCertificate -FilePath "C:\Certificates\claims-processor.pfx" -CertStoreLocation "cert:\LocalMachine\My" -Password (ConvertTo-SecureString "CertPassword" -AsPlainText -Force)
```

## Performance Optimization

### Step 13: Performance Tuning

```powershell
# Configure Windows for high performance
powercfg -setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c  # High Performance

# Optimize TCP settings
netsh int tcp set global autotuninglevel=normal
netsh int tcp set global chimney=enabled
netsh int tcp set global rss=enabled
netsh int tcp set global netdma=enabled

# Configure memory settings
# Increase virtual memory for large datasets
$computerSystem = Get-WmiObject -Class Win32_ComputerSystem
$totalRam = [math]::Round($computerSystem.TotalPhysicalMemory / 1GB)
$pagingFileSize = $totalRam * 1.5
$pagingFile = Get-WmiObject -Class Win32_PageFileSetting
$pagingFile.InitialSize = $pagingFileSize * 1024
$pagingFile.MaximumSize = $pagingFileSize * 1024
$pagingFile.Put()

# Configure process priorities
Get-Process -Name "python" | ForEach-Object { $_.PriorityClass = "High" }
```

## Backup and Recovery

### Step 14: Configure Backups

```powershell
# Create backup script
$backupScript = @"
# Claims Processor Backup Script
param(
    [string]`$BackupPath = "C:\Claims_Processor\backups"
)

`$date = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
`$backupDir = "`$BackupPath\backup_`$date"
New-Item -ItemType Directory -Force -Path `$backupDir

# Database backup
pg_dump -U claims_prod_user -h localhost claims_processor_prod > "`$backupDir\database_backup.sql"

# Application data backup
Copy-Item -Path "C:\Claims_Processor\uploads" -Destination "`$backupDir\uploads" -Recurse
Copy-Item -Path "C:\Claims_Processor\logs" -Destination "`$backupDir\logs" -Recurse
Copy-Item -Path "C:\Claims_Processor\app\config" -Destination "`$backupDir\config" -Recurse

# Compress backup
Compress-Archive -Path `$backupDir -DestinationPath "`$backupDir.zip"
Remove-Item -Path `$backupDir -Recurse -Force

# Cleanup old backups (keep 30 days)
Get-ChildItem -Path `$BackupPath -Name "backup_*.zip" | Where-Object { `$_.CreationTime -lt (Get-Date).AddDays(-30) } | Remove-Item -Force

Write-Host "Backup completed: `$backupDir.zip"
"@

Set-Content -Path "C:\Claims_Processor\scripts\backup.ps1" -Value $backupScript

# Schedule daily backups
$action = New-ScheduledTaskAction -Execute "PowerShell.exe" -Argument "-File C:\Claims_Processor\scripts\backup.ps1"
$trigger = New-ScheduledTaskTrigger -Daily -At 2AM
Register-ScheduledTask -Action $action -Trigger $trigger -TaskName "ClaimsProcessorBackup" -Description "Daily backup of Claims Processor"
```

## Verification and Testing

### Step 15: Production Verification

```powershell
# Verify service status
Get-Service -Name "ClaimsProcessor", "postgresql-x64-13", "Memurai", "prometheus"

# Test API endpoints
Invoke-RestMethod -Uri "http://localhost:8000/health"
Invoke-RestMethod -Uri "http://localhost:8000/api/v1/system/status"

# Check logs
Get-Content "C:\Claims_Processor\logs\claims_processor.log" -Tail 20
Get-EventLog -LogName Application -Source "ClaimsProcessor" -Newest 10

# Monitor performance
Get-Counter "\Process(python)\% Processor Time", "\Process(python)\Working Set", "\Memory\Available MBytes"

# Test database connectivity
psql -U claims_prod_user -d claims_processor_prod -c "SELECT COUNT(*) FROM staging.claims_837p;"

# Test Redis connectivity
redis-cli ping
redis-cli info memory
```

## Maintenance Procedures

### Regular Maintenance Tasks

```powershell
# Weekly maintenance script
$maintenanceScript = @"
# Claims Processor Weekly Maintenance

# 1. Update statistics
psql -U claims_prod_user -d claims_processor_prod -c "ANALYZE;"

# 2. Vacuum database
psql -U claims_prod_user -d claims_processor_prod -c "VACUUM;"

# 3. Check disk space
Get-WmiObject -Class Win32_LogicalDisk | Select-Object DeviceID, @{Name="Size(GB)";Expression={[math]::Round(`$_.Size/1GB,2)}}, @{Name="FreeSpace(GB)";Expression={[math]::Round(`$_.FreeSpace/1GB,2)}}, @{Name="PercentFree";Expression={[math]::Round((`$_.FreeSpace/`$_.Size)*100,2)}}

# 4. Restart Redis to clear memory fragmentation
Restart-Service Memurai

# 5. Check Windows updates
Get-WindowsUpdate -Install -AcceptAll -AutoReboot

# 6. Generate maintenance report
`$report = @{
    "Date" = Get-Date
    "Services" = Get-Service -Name "ClaimsProcessor", "postgresql-x64-13", "Memurai"
    "DiskSpace" = Get-WmiObject -Class Win32_LogicalDisk
    "Memory" = Get-Counter "\Memory\Available MBytes"
}
`$report | ConvertTo-Json | Out-File "C:\Claims_Processor\logs\maintenance_`$(Get-Date -Format 'yyyy-MM-dd').json"
"@

Set-Content -Path "C:\Claims_Processor\scripts\weekly_maintenance.ps1" -Value $maintenanceScript

# Schedule weekly maintenance
$action = New-ScheduledTaskAction -Execute "PowerShell.exe" -Argument "-File C:\Claims_Processor\scripts\weekly_maintenance.ps1"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 3AM
Register-ScheduledTask -Action $action -Trigger $trigger -TaskName "ClaimsProcessorMaintenance" -Description "Weekly maintenance tasks"
```

## Troubleshooting

### Common Issues and Solutions

#### Service Won't Start
```powershell
# Check service logs
Get-EventLog -LogName Application -Source "ClaimsProcessor" -Newest 10

# Check permissions
icacls "C:\Claims_Processor" /T

# Verify Python environment
& "C:\Claims_Processor\venv\Scripts\python.exe" --version
& "C:\Claims_Processor\venv\Scripts\pip.exe" list
```

#### Database Connection Issues
```powershell
# Test database connection
psql -U claims_prod_user -h localhost -d claims_processor_prod

# Check PostgreSQL service
Get-Service postgresql-x64-13
Get-EventLog -LogName Application -Source "postgresql-x64-13" -Newest 10

# Check network connectivity
Test-NetConnection -ComputerName localhost -Port 5432
```

#### Performance Issues
```powershell
# Monitor system resources
Get-Counter "\Processor(_Total)\% Processor Time", "\Memory\Available MBytes", "\PhysicalDisk(_Total)\% Disk Time"

# Check application metrics
Invoke-RestMethod -Uri "http://localhost:8000/metrics"

# Review application logs
Get-Content "C:\Claims_Processor\logs\app.log" -Tail 100 | Select-String "ERROR"
```

---

This Windows Server deployment guide provides a comprehensive production deployment strategy optimized for Windows environments. For additional configuration details, see:
- [Installation Guide](./installation.md)
- [Configuration Management](./configuration.md)
- [Security Documentation](../security/)
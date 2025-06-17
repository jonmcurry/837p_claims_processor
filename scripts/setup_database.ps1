# PowerShell Script to Setup Smart Pro Claims Databases
# 
# This script sets up PostgreSQL and/or SQL Server databases with schema and sample data
#
# Usage:
#   .\scripts\setup_database.ps1 -PostgresHost "localhost" -PostgresUser "postgres" -PostgresPassword "mypassword"
#   .\scripts\setup_database.ps1 -SqlServerHost "localhost" -IntegratedAuth
#   .\scripts\setup_database.ps1 -PostgresHost "localhost" -PostgresUser "postgres" -PostgresPassword "pg_pass" -SqlServerHost "localhost" -SqlServerUser "sa" -SqlServerPassword "sql_pass"

param(
    # PostgreSQL parameters
    [Parameter(Mandatory=$false)]
    [string]$PostgresHost,
    
    [Parameter(Mandatory=$false)]
    [int]$PostgresPort = 5432,
    
    [Parameter(Mandatory=$false)]
    [string]$PostgresUser,
    
    [Parameter(Mandatory=$false)]
    [string]$PostgresPassword,
    
    [Parameter(Mandatory=$false)]
    [string]$PostgresDatabase = "smart_pro_claims",
    
    # SQL Server parameters
    [Parameter(Mandatory=$false)]
    [string]$SqlServerHost,
    
    [Parameter(Mandatory=$false)]
    [string]$SqlServerUser,
    
    [Parameter(Mandatory=$false)]
    [string]$SqlServerPassword,
    
    [Parameter(Mandatory=$false)]
    [string]$SqlServerDatabase = "smart_pro_claims",
    
    [Parameter(Mandatory=$false)]
    [switch]$IntegratedAuth,
    
    # General parameters
    [Parameter(Mandatory=$false)]
    [switch]$SkipSampleData,
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipClaimsData
)

# Colors for output
$Color = @{
    Info = "Cyan"
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
    Header = "Magenta"
}

function Write-ColoredOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Show-Banner {
    Write-ColoredOutput "================================================================" $Color.Header
    Write-ColoredOutput "  SMART PRO CLAIMS - DATABASE SETUP" $Color.Header
    Write-ColoredOutput "  Automated database creation, schema loading, and sample data" $Color.Header
    Write-ColoredOutput "================================================================" $Color.Header
    Write-ColoredOutput ""
}

function Test-PythonEnvironment {
    Write-ColoredOutput "🐍 Checking Python environment..." $Color.Info
    
    # Check if Python is available
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-ColoredOutput "✓ Python found: $pythonVersion" $Color.Success
        } else {
            throw "Python not found"
        }
    } catch {
        Write-ColoredOutput "❌ Python is not installed or not in PATH" $Color.Error
        Write-ColoredOutput "Please install Python 3.11+ and add it to your PATH" $Color.Warning
        return $false
    }
    
    return $true
}

function Install-Requirements {
    Write-ColoredOutput "📦 Installing Python requirements..." $Color.Info
    
    $requirementsFile = "requirements.txt"
    
    if (Test-Path $requirementsFile) {
        try {
            pip install -r $requirementsFile
            if ($LASTEXITCODE -eq 0) {
                Write-ColoredOutput "✓ Requirements installed successfully" $Color.Success
            } else {
                throw "Failed to install requirements"
            }
        } catch {
            Write-ColoredOutput "❌ Failed to install Python requirements" $Color.Error
            Write-ColoredOutput "Please install manually: pip install -r requirements.txt" $Color.Warning
            return $false
        }
    } else {
        Write-ColoredOutput "❌ Requirements file not found: $requirementsFile" $Color.Error
        Write-ColoredOutput "Please run from project root directory" $Color.Warning
        return $false
    }
    
    return $true
}

function Build-Arguments {
    $arguments = @("scripts\setup_database.py")
    
    # PostgreSQL arguments
    if ($PostgresHost -and $PostgresUser -and $PostgresPassword) {
        $arguments += "--postgres-host", $PostgresHost
        $arguments += "--postgres-port", $PostgresPort.ToString()
        $arguments += "--postgres-user", $PostgresUser
        $arguments += "--postgres-password", $PostgresPassword
        $arguments += "--postgres-database", $PostgresDatabase
        Write-ColoredOutput "✓ PostgreSQL configuration added" $Color.Success
    }
    
    # SQL Server arguments
    if ($SqlServerHost) {
        $arguments += "--sqlserver-host", $SqlServerHost
        $arguments += "--sqlserver-database", $SqlServerDatabase
        
        if ($IntegratedAuth) {
            $arguments += "--integrated-auth"
            Write-ColoredOutput "✓ SQL Server configuration added (Integrated Auth)" $Color.Success
        } elseif ($SqlServerUser -and $SqlServerPassword) {
            $arguments += "--sqlserver-user", $SqlServerUser
            $arguments += "--sqlserver-password", $SqlServerPassword
            Write-ColoredOutput "✓ SQL Server configuration added (SQL Auth)" $Color.Success
        } else {
            Write-ColoredOutput "❌ SQL Server credentials missing" $Color.Error
            return $null
        }
    }
    
    # General arguments
    if ($SkipSampleData) {
        $arguments += "--skip-sample-data"
    }
    
    if ($SkipClaimsData) {
        $arguments += "--skip-claims-data"
    }
    
    return $arguments
}

function Invoke-DatabaseSetup {
    param([array]$Arguments)
    
    Write-ColoredOutput "🚀 Starting database setup..." $Color.Header
    
    try {
        Write-ColoredOutput "Executing: python $($Arguments -join ' ')" $Color.Info
        & python @Arguments
        
        if ($LASTEXITCODE -eq 0) {
            Write-ColoredOutput "`n🎉 Database setup completed successfully!" $Color.Success
        } else {
            throw "Python script failed with exit code $LASTEXITCODE"
        }
    } catch {
        Write-ColoredOutput "❌ Failed to run database setup" $Color.Error
        Write-ColoredOutput "Error: $_" $Color.Error
        return $false
    }
    
    return $true
}

function Show-Usage {
    Write-ColoredOutput "`nUsage Examples:" $Color.Info
    Write-ColoredOutput "  # Setup PostgreSQL only" $Color.Info
    Write-ColoredOutput "  .\scripts\setup_database.ps1 -PostgresHost localhost -PostgresUser postgres -PostgresPassword mypassword" $Color.Info
    Write-ColoredOutput ""
    Write-ColoredOutput "  # Setup SQL Server with integrated auth" $Color.Info
    Write-ColoredOutput "  .\scripts\setup_database.ps1 -SqlServerHost localhost -IntegratedAuth" $Color.Info
    Write-ColoredOutput ""
    Write-ColoredOutput "  # Setup both databases" $Color.Info
    Write-ColoredOutput "  .\scripts\setup_database.ps1 -PostgresHost localhost -PostgresUser postgres -PostgresPassword pg_pass -SqlServerHost localhost -SqlServerUser sa -SqlServerPassword sql_pass" $Color.Info
    Write-ColoredOutput ""
}

# Main execution
try {
    Show-Banner
    
    # Validate arguments
    $setupPostgres = $PostgresHost -and $PostgresUser -and $PostgresPassword
    $setupSqlServer = $SqlServerHost -and ($IntegratedAuth -or ($SqlServerUser -and $SqlServerPassword))
    
    if (-not $setupPostgres -and -not $setupSqlServer) {
        Write-ColoredOutput "❌ No valid database configuration provided" $Color.Error
        Show-Usage
        exit 1
    }
    
    # Show configuration summary
    Write-ColoredOutput "Configuration:" $Color.Info
    if ($setupPostgres) {
        Write-ColoredOutput "  PostgreSQL: $PostgresHost:$PostgresPort (Database: $PostgresDatabase)" $Color.Info
    }
    if ($setupSqlServer) {
        $authType = if ($IntegratedAuth) { "Integrated" } else { "SQL Server" }
        Write-ColoredOutput "  SQL Server: $SqlServerHost (Database: $SqlServerDatabase, Auth: $authType)" $Color.Info
    }
    if ($SkipSampleData) {
        Write-ColoredOutput "  Sample Data: Skipped" $Color.Warning
    } elseif ($SkipClaimsData) {
        Write-ColoredOutput "  Sample Data: Configuration only (no claims)" $Color.Info
    } else {
        Write-ColoredOutput "  Sample Data: Full dataset (100,000 claims)" $Color.Info
    }
    Write-ColoredOutput ""
    
    # Step 1: Check Python environment
    if (-not (Test-PythonEnvironment)) {
        exit 1
    }
    
    # Step 2: Install requirements
    if (-not (Install-Requirements)) {
        exit 1
    }
    
    # Step 3: Build arguments
    $arguments = Build-Arguments
    if (-not $arguments) {
        exit 1
    }
    
    # Step 4: Run database setup
    if (-not (Invoke-DatabaseSetup -Arguments $arguments)) {
        exit 1
    }
    
    Write-ColoredOutput "`n🏥 Databases are now ready for use!" $Color.Success
    Write-ColoredOutput "You can now test analytics, reporting, and claims processing functionality." $Color.Info
    
} catch {
    Write-ColoredOutput "`n❌ Script execution failed: $_" $Color.Error
    exit 1
}
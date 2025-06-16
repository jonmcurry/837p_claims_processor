# PowerShell Script to Load Sample Data for Smart Pro Claims Database
# 
# This script sets up the environment and runs the Python sample data loader
#
# Usage:
#   .\scripts\load_sample_data.ps1 -ServerName "localhost" -DatabaseName "smart_pro_claims" -Username "claims_analytics_user" -Password "YourPassword"
#
# Or for integrated authentication:
#   .\scripts\load_sample_data.ps1 -ServerName "localhost" -DatabaseName "smart_pro_claims" -IntegratedAuth

param(
    [Parameter(Mandatory=$true)]
    [string]$ServerName,
    
    [Parameter(Mandatory=$false)]
    [string]$DatabaseName = "smart_pro_claims",
    
    [Parameter(Mandatory=$false)]
    [string]$Username,
    
    [Parameter(Mandatory=$false)]
    [string]$Password,
    
    [Parameter(Mandatory=$false)]
    [switch]$IntegratedAuth,
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipClaims,
    
    [Parameter(Mandatory=$false)]
    [string]$Port = "1433"
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
        Write-ColoredOutput "Please install Python 3.9+ and add it to your PATH" $Color.Warning
        exit 1
    }
    
    # Check if pip is available
    try {
        $pipVersion = pip --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-ColoredOutput "✓ pip found: $pipVersion" $Color.Success
        } else {
            throw "pip not found"
        }
    } catch {
        Write-ColoredOutput "❌ pip is not available" $Color.Error
        exit 1
    }
}

function Install-Requirements {
    Write-ColoredOutput "📦 Installing Python requirements..." $Color.Info
    
    $requirementsFile = "scripts\requirements_sample_data.txt"
    
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
            Write-ColoredOutput "Please install manually: pip install sqlalchemy pyodbc faker" $Color.Warning
            exit 1
        }
    } else {
        Write-ColoredOutput "❌ Requirements file not found: $requirementsFile" $Color.Error
        Write-ColoredOutput "Installing packages individually..." $Color.Warning
        pip install sqlalchemy pyodbc faker
    }
}

function Build-ConnectionString {
    param(
        [string]$Server,
        [string]$Database,
        [string]$User,
        [string]$Pass,
        [string]$Port,
        [bool]$IntegratedAuth
    )
    
    $driver = "ODBC+Driver+17+for+SQL+Server"
    
    if ($IntegratedAuth) {
        return "mssql+pyodbc://$Server`:$Port/$Database`?driver=$driver&trusted_connection=yes"
    } else {
        if ([string]::IsNullOrEmpty($User) -or [string]::IsNullOrEmpty($Pass)) {
            Write-ColoredOutput "❌ Username and Password required when not using Integrated Authentication" $Color.Error
            exit 1
        }
        return "mssql+pyodbc://$User`:$Pass@$Server`:$Port/$Database`?driver=$driver"
    }
}

function Test-DatabaseConnection {
    param([string]$ConnectionString)
    
    Write-ColoredOutput "🔌 Testing database connection..." $Color.Info
    
    # Extract server and database from connection string for sqlcmd test
    $serverPart = if ($IntegratedAuth) { "-E" } else { "-U $Username" }
    
    try {
        if ($IntegratedAuth) {
            $testResult = sqlcmd -S $ServerName -E -d $DatabaseName -Q "SELECT 1 as test" -h -1 2>&1
        } else {
            # For security, we'll just test the Python connection directly
            Write-ColoredOutput "✓ Connection string built successfully" $Color.Success
            return
        }
        
        if ($LASTEXITCODE -eq 0) {
            Write-ColoredOutput "✓ Database connection successful" $Color.Success
        } else {
            throw "Connection failed"
        }
    } catch {
        Write-ColoredOutput "❌ Failed to connect to database" $Color.Error
        Write-ColoredOutput "Please verify server name, credentials, and database name" $Color.Warning
        Write-ColoredOutput "Connection details: Server=$ServerName, Database=$DatabaseName" $Color.Info
        exit 1
    }
}

function Invoke-SampleDataLoader {
    param(
        [string]$ConnectionString,
        [bool]$SkipClaims
    )
    
    Write-ColoredOutput "🚀 Starting sample data loading..." $Color.Header
    
    $scriptPath = "scripts\load_sample_data.py"
    
    if (!(Test-Path $scriptPath)) {
        Write-ColoredOutput "❌ Sample data script not found: $scriptPath" $Color.Error
        exit 1
    }
    
    $arguments = @("$scriptPath", "--connection-string", "`"$ConnectionString`"")
    
    if ($SkipClaims) {
        $arguments += "--skip-claims"
        Write-ColoredOutput "⏭️  Claims data loading will be skipped" $Color.Warning
    }
    
    try {
        Write-ColoredOutput "Executing: python $($arguments -join ' ')" $Color.Info
        & python @arguments
        
        if ($LASTEXITCODE -eq 0) {
            Write-ColoredOutput "`n🎉 Sample data loading completed successfully!" $Color.Success
        } else {
            throw "Python script failed with exit code $LASTEXITCODE"
        }
    } catch {
        Write-ColoredOutput "❌ Failed to run sample data loader" $Color.Error
        Write-ColoredOutput "Error: $_" $Color.Error
        exit 1
    }
}

function Show-Summary {
    Write-ColoredOutput "`n" + "="*60 $Color.Header
    Write-ColoredOutput "  SMART PRO CLAIMS - SAMPLE DATA LOADER" $Color.Header
    Write-ColoredOutput "="*60 $Color.Header
    Write-ColoredOutput ""
    Write-ColoredOutput "Server: $ServerName" $Color.Info
    Write-ColoredOutput "Database: $DatabaseName" $Color.Info
    Write-ColoredOutput "Authentication: $(if ($IntegratedAuth) { 'Integrated' } else { 'SQL Server' })" $Color.Info
    Write-ColoredOutput "Skip Claims: $(if ($SkipClaims) { 'Yes' } else { 'No' })" $Color.Info
    Write-ColoredOutput ""
}

# Main execution
try {
    Show-Summary
    
    # Step 1: Check Python environment
    Test-PythonEnvironment
    
    # Step 2: Install requirements
    Install-Requirements
    
    # Step 3: Build connection string
    $connectionString = Build-ConnectionString -Server $ServerName -Database $DatabaseName -User $Username -Pass $Password -Port $Port -IntegratedAuth $IntegratedAuth
    
    # Step 4: Test connection
    Test-DatabaseConnection -ConnectionString $connectionString
    
    # Step 5: Run sample data loader
    Invoke-SampleDataLoader -ConnectionString $connectionString -SkipClaims $SkipClaims
    
    Write-ColoredOutput "`n🏥 Database is now ready with sample data!" $Color.Success
    Write-ColoredOutput "You can now test analytics, reporting, and claims processing functionality." $Color.Info
    
} catch {
    Write-ColoredOutput "`n❌ Script execution failed: $_" $Color.Error
    exit 1
}
#!/usr/bin/env python3
"""
Database Setup Script for Smart Pro Claims

This script:
1. Reads database configuration from config/.env.example
2. Checks if PostgreSQL and SQL Server databases exist
3. Creates databases if they don't exist
4. Loads the appropriate schema
5. Loads sample data

Usage:
    python scripts/setup_database.py                    # Uses config/.env.example
    python scripts/setup_database.py --env config/.env  # Uses specific env file
    python scripts/setup_database.py --postgres-only    # Setup only PostgreSQL
    python scripts/setup_database.py --sqlserver-only   # Setup only SQL Server
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
import time

try:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import pyodbc
    SQLSERVER_AVAILABLE = True
except ImportError:
    SQLSERVER_AVAILABLE = False

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table

console = Console()

def load_env_file(env_path='config/.env'):
    """Load environment variables from .env file."""
    env_vars = {}
    
    # First try the provided path
    if not os.path.exists(env_path):
        # Try .env.example if .env doesn't exist
        if os.path.exists('config/.env.example'):
            env_path = 'config/.env.example'
            console.print(f"[yellow]WARNING: Using .env.example - copy to .env for production use[/yellow]")
        else:
            console.print(f"[red]ERROR: Environment file '{env_path}' not found![/red]")
            return None
    
    try:
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    # Split on first = only
                    if '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
        return env_vars
    except Exception as e:
        console.print(f"[red]ERROR: Error loading env file: {e}[/red]")
        return None

def build_config_from_env(env_vars, postgres_only=False, sqlserver_only=False):
    """Build configuration from environment variables."""
    config = {}
    
    # PostgreSQL configuration
    if not sqlserver_only:
        config['setup_postgres'] = True
        config['postgres_host'] = env_vars.get('POSTGRES_HOST', 'localhost')
        config['postgres_port'] = int(env_vars.get('POSTGRES_PORT', '5432'))
        config['postgres_user'] = env_vars.get('POSTGRES_USER', 'postgres')
        config['postgres_password'] = env_vars.get('POSTGRES_PASSWORD', '')
        
        # Use claims_staging for processing workflow
        pg_db = env_vars.get('POSTGRES_DB', 'claims_processor_dev')
        if pg_db == 'claims_processor_dev':
            pg_db = 'claims_staging'
            console.print(f"[blue]NOTE: Using 'claims_staging' database for PostgreSQL[/blue]")
        config['postgres_database'] = pg_db
    else:
        config['setup_postgres'] = False
    
    # SQL Server configuration
    if not postgres_only:
        config['setup_sqlserver'] = True
        config['sqlserver_host'] = env_vars.get('SQLSERVER_HOST', 'localhost')
        config['sqlserver_user'] = env_vars.get('SQLSERVER_USER', 'sa')
        config['sqlserver_password'] = env_vars.get('SQLSERVER_PASSWORD', '')
        config['sqlserver_integrated_auth'] = False  # Use SQL auth from env
        
        # Use smart_pro_claims for analytics
        ss_db = env_vars.get('SQLSERVER_DB', 'claims_analytics_dev')
        if ss_db == 'claims_analytics_dev':
            ss_db = 'smart_pro_claims'
            console.print(f"[blue]NOTE: Using 'smart_pro_claims' database for SQL Server[/blue]")
        config['sqlserver_database'] = ss_db
    else:
        config['setup_sqlserver'] = False
    
    # General settings
    config['load_sample_data'] = True
    config['skip_claims_data'] = False
    
    return config

class DatabaseSetup:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.project_root = Path(__file__).parent.parent
        self.postgres_schema_path = self.project_root / "database" / "postgresql_claims_processing_schema.sql"
        self.sqlserver_schema_path = self.project_root / "database" / "sqlserver_schema.sql"
        self.sample_data_script = self.project_root / "scripts" / "load_sample_data.py"

    def show_banner(self):
        """Display startup banner"""
        console.print(Panel.fit(
            "[bold cyan]Smart Pro Claims Database Setup[/bold cyan]\n"
            "[dim]Automated database creation, schema loading, and sample data generation[/dim]\n"
            "[dim]Reading configuration from config/.env.example[/dim]",
            border_style="cyan"
        ))

    def check_dependencies(self):
        """Check if required dependencies are available"""
        console.print("\n[yellow]Checking dependencies...[/yellow]")
        
        missing_deps = []
        
        if self.config.get('setup_postgres') and not POSTGRES_AVAILABLE:
            missing_deps.append("psycopg2-binary (for PostgreSQL)")
        
        if self.config.get('setup_sqlserver') and not SQLSERVER_AVAILABLE:
            missing_deps.append("pyodbc (for SQL Server)")
        
        if missing_deps:
            console.print(f"[red]Missing dependencies: {', '.join(missing_deps)}[/red]")
            console.print("[yellow]Install with: pip install psycopg2-binary pyodbc[/yellow]")
            return False
        
        console.print("[green]✓ All dependencies available[/green]")
        return True

    def postgres_database_exists(self, db_name: str) -> bool:
        """Check if PostgreSQL database exists"""
        try:
            # Connect to postgres database to check if target database exists
            conn = psycopg2.connect(
                host=self.config['postgres_host'],
                port=self.config['postgres_port'],
                user=self.config['postgres_user'],
                password=self.config['postgres_password'],
                database='postgres'
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
            exists = cursor.fetchone() is not None
            
            cursor.close()
            conn.close()
            return exists
            
        except Exception as e:
            console.print(f"[red]Error checking PostgreSQL database: {e}[/red]")
            return False

    def create_postgres_database(self, db_name: str) -> bool:
        """Create PostgreSQL database"""
        try:
            conn = psycopg2.connect(
                host=self.config['postgres_host'],
                port=self.config['postgres_port'],
                user=self.config['postgres_user'],
                password=self.config['postgres_password'],
                database='postgres'
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            
            cursor = conn.cursor()
            cursor.execute(f'CREATE DATABASE "{db_name}"')
            
            cursor.close()
            conn.close()
            console.print(f"[green]✓ Created PostgreSQL database: {db_name}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Error creating PostgreSQL database: {e}[/red]")
            return False

    def load_postgres_schema(self, db_name: str) -> bool:
        """Load PostgreSQL schema"""
        try:
            if not self.postgres_schema_path.exists():
                console.print(f"[red]PostgreSQL schema file not found: {self.postgres_schema_path}[/red]")
                return False
            
            # Use psql command for better SQL script execution
            cmd = [
                'psql',
                f'-h{self.config["postgres_host"]}',
                f'-p{self.config["postgres_port"]}',
                f'-U{self.config["postgres_user"]}',
                f'-d{db_name}',
                f'-f{self.postgres_schema_path}',
                '-v', 'ON_ERROR_STOP=1'
            ]
            
            env = os.environ.copy()
            env['PGPASSWORD'] = self.config['postgres_password']
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode == 0:
                console.print(f"[green]✓ PostgreSQL schema loaded successfully[/green]")
                return True
            else:
                console.print(f"[red]Error loading PostgreSQL schema: {result.stderr}[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]Error loading PostgreSQL schema: {e}[/red]")
            return False

    def sqlserver_database_exists(self, db_name: str) -> bool:
        """Check if SQL Server database exists"""
        try:
            if self.config.get('sqlserver_integrated_auth'):
                conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={self.config['sqlserver_host']};DATABASE=master;Trusted_Connection=yes;"
            else:
                conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={self.config['sqlserver_host']};DATABASE=master;UID={self.config['sqlserver_user']};PWD={self.config['sqlserver_password']};"
            
            conn = pyodbc.connect(conn_str)
            cursor = conn.cursor()
            
            cursor.execute("SELECT 1 FROM sys.databases WHERE name = ?", (db_name,))
            exists = cursor.fetchone() is not None
            
            cursor.close()
            conn.close()
            return exists
            
        except Exception as e:
            console.print(f"[red]Error checking SQL Server database: {e}[/red]")
            return False

    def create_sqlserver_database(self, db_name: str) -> bool:
        """Create SQL Server database"""
        try:
            if self.config.get('sqlserver_integrated_auth'):
                conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={self.config['sqlserver_host']};DATABASE=master;Trusted_Connection=yes;"
            else:
                conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={self.config['sqlserver_host']};DATABASE=master;UID={self.config['sqlserver_user']};PWD={self.config['sqlserver_password']};"
            
            conn = pyodbc.connect(conn_str)
            conn.autocommit = True
            cursor = conn.cursor()
            
            cursor.execute(f"CREATE DATABASE [{db_name}]")
            
            cursor.close()
            conn.close()
            console.print(f"[green]✓ Created SQL Server database: {db_name}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Error creating SQL Server database: {e}[/red]")
            return False

    def load_sqlserver_schema(self, db_name: str) -> bool:
        """Load SQL Server schema"""
        try:
            if not self.sqlserver_schema_path.exists():
                console.print(f"[red]SQL Server schema file not found: {self.sqlserver_schema_path}[/red]")
                return False
            
            # Read the schema file and modify it for dynamic execution
            schema_content = self.sqlserver_schema_path.read_text(encoding='utf-8')
            
            # Skip the database creation section since we already created it
            # Remove the database creation and USE statements
            lines = schema_content.split('\n')
            filtered_lines = []
            skip_until_go = False
            
            for line in lines:
                # Skip database creation block (including IF EXISTS check)
                if 'IF NOT EXISTS' in line and 'smart_pro_claims' in line:
                    skip_until_go = True
                    continue
                if 'CREATE DATABASE smart_pro_claims' in line:
                    skip_until_go = True
                    continue
                if skip_until_go and line.strip() == 'GO':
                    skip_until_go = False
                    continue
                if skip_until_go:
                    continue
                    
                # Skip USE database statement and its GO
                if line.strip().startswith('USE smart_pro_claims'):
                    skip_until_go = True
                    continue
                    
                # Skip filegroup creation that requires specific paths
                if ('ADD FILEGROUP' in line or 'ADD FILE' in line or 'FILENAME =' in line or 
                    'ALTER DATABASE smart_pro_claims' in line):
                    skip_until_go = True
                    continue
                    
                # Skip partition scheme that depends on filegroups
                if 'CREATE PARTITION FUNCTION' in line or 'CREATE PARTITION SCHEME' in line:
                    skip_until_go = True
                    continue
                
                # Skip PRINT statements that may cause issues
                if line.strip().startswith('PRINT '):
                    continue
                    
                # Skip ON ClaimsDatePartitionScheme clauses
                if 'ON ClaimsDatePartitionScheme' in line:
                    # Replace with default placement
                    line = line.replace('ON ClaimsDatePartitionScheme(created_at)', '')
                    line = line.replace('ON ClaimsDatePartitionScheme(failed_at)', '')
                    line = line.replace('ON ClaimsDatePartitionScheme(operation_timestamp)', '')
                    line = line.replace('ON ClaimsDatePartitionScheme(access_timestamp)', '')
                    line = line.replace('ON ClaimsDatePartitionScheme(metric_date)', '')
                
                # Ensure GO statements before stored procedures/functions
                if ('CREATE PROCEDURE' in line or 'CREATE FUNCTION' in line) and len(filtered_lines) > 0:
                    last_line = filtered_lines[-1].strip() if filtered_lines else ''
                    if last_line != 'GO' and last_line != '':
                        filtered_lines.append('GO')
                
                filtered_lines.append(line)
            
            # Add schema fixes for column size issues
            filtered_lines.append('GO')
            filtered_lines.append('')
            filtered_lines.append('-- Fix column sizes for data loading')
            filtered_lines.append('ALTER TABLE dbo.facility_place_of_service ALTER COLUMN place_of_service_name VARCHAR(60) NOT NULL;')
            filtered_lines.append('GO')
            
            # Create a temporary schema file
            temp_schema_path = self.project_root / "temp_sqlserver_schema.sql"
            temp_schema_path.write_text('\n'.join(filtered_lines), encoding='utf-8')
            
            try:
                # Use sqlcmd for better SQL script execution
                if self.config.get('sqlserver_integrated_auth'):
                    cmd = [
                        'sqlcmd',
                        '-S', self.config['sqlserver_host'],
                        '-E',  # Use integrated authentication
                        '-d', db_name,
                        '-i', str(temp_schema_path),
                        '-b'  # Exit with error code on failure
                    ]
                else:
                    cmd = [
                        'sqlcmd',
                        '-S', self.config['sqlserver_host'],
                        '-U', self.config['sqlserver_user'],
                        '-P', self.config['sqlserver_password'],
                        '-d', db_name,
                        '-i', str(temp_schema_path),
                        '-b'
                    ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    console.print(f"[green]✓ SQL Server schema loaded successfully[/green]")
                    return True
                else:
                    console.print(f"[red]Error loading SQL Server schema: {result.stderr}[/red]")
                    console.print(f"[yellow]Command output: {result.stdout}[/yellow]")
                    return False
                    
            finally:
                # Clean up temporary file
                if temp_schema_path.exists():
                    temp_schema_path.unlink()
                
        except Exception as e:
            console.print(f"[red]Error loading SQL Server schema: {e}[/red]")
            return False

    def setup_postgres(self) -> bool:
        """Setup PostgreSQL database"""
        if not self.config.get('setup_postgres'):
            return True
            
        console.print("\n[cyan]Setting up PostgreSQL database...[/cyan]")
        
        db_name = self.config.get('postgres_database', 'claims_staging')
        
        # Check if database exists
        if self.postgres_database_exists(db_name):
            console.print(f"[yellow]PostgreSQL database '{db_name}' already exists[/yellow]")
        else:
            console.print(f"[blue]Creating PostgreSQL database: {db_name}[/blue]")
            if not self.create_postgres_database(db_name):
                return False
        
        # Load schema
        console.print("[blue]Loading PostgreSQL schema...[/blue]")
        return self.load_postgres_schema(db_name)

    def setup_sqlserver(self) -> bool:
        """Setup SQL Server database"""
        if not self.config.get('setup_sqlserver'):
            return True
            
        console.print("\n[cyan]Setting up SQL Server database...[/cyan]")
        
        db_name = self.config.get('sqlserver_database', 'smart_pro_claims')
        
        # Check if database exists
        if self.sqlserver_database_exists(db_name):
            console.print(f"[yellow]SQL Server database '{db_name}' already exists[/yellow]")
        else:
            console.print(f"[blue]Creating SQL Server database: {db_name}[/blue]")
            if not self.create_sqlserver_database(db_name):
                return False
        
        # Load schema
        console.print("[blue]Loading SQL Server schema...[/blue]")
        return self.load_sqlserver_schema(db_name)

    def load_sample_data(self) -> bool:
        """Load sample data using the modified load_sample_data.py script"""
        if not self.config.get('load_sample_data', True):
            console.print("[yellow]Skipping sample data loading[/yellow]")
            return True
            
        console.print("\n[cyan]Loading sample data...[/cyan]")
        
        if not self.sample_data_script.exists():
            console.print(f"[red]Sample data script not found: {self.sample_data_script}[/red]")
            return False
        
        # PRIORITY: Use PostgreSQL for claims processing workflow when available
        if self.config.get('setup_postgres'):
            postgres_db = self.config.get('postgres_database', 'claims_staging')
            console.print(f"[green]>>> LOADING CLAIMS INTO POSTGRESQL: {postgres_db}[/green]")
            console.print("[blue]Claims will be loaded into public.claims table for processing workflow[/blue]")
        elif self.config.get('setup_sqlserver'):
            console.print(f"[yellow]>>> SQL Server configured but sample claims loader focuses on PostgreSQL[/yellow]")
            console.print("[dim]Note: For claims processing workflow, setup PostgreSQL as well[/dim]")
            return True  # Skip sample data for SQL Server only setup
        else:
            console.print("[red]No database configured for sample data loading[/red]")
            return False
        
        # Run sample data loader (it will read from config/.env.example automatically)
        cmd = [
            sys.executable,
            str(self.sample_data_script)
        ]
        
        if self.config.get('skip_claims_data'):
            cmd.append('--skip-claims')
        
        try:
            console.print(f"[blue]Running: {' '.join(cmd)} (reads config/.env.example automatically)[/blue]")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            console.print("[green]✓ Sample data loaded successfully[/green]")
            return True
            
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Error loading sample data: {e}[/red]")
            if e.stdout:
                console.print(f"[yellow]stdout: {e.stdout}[/yellow]")
            if e.stderr:
                console.print(f"[yellow]stderr: {e.stderr}[/yellow]")
            return False

    def show_summary(self):
        """Show setup summary"""
        table = Table(title="Database Setup Summary")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="dim")
        
        if self.config.get('setup_postgres'):
            table.add_row(
                "PostgreSQL",
                "✓ Configured",
                f"Host: {self.config['postgres_host']}:{self.config['postgres_port']}"
            )
        
        if self.config.get('setup_sqlserver'):
            auth_type = "Integrated" if self.config.get('sqlserver_integrated_auth') else "SQL Server"
            table.add_row(
                "SQL Server", 
                "✓ Configured",
                f"Host: {self.config['sqlserver_host']}, Auth: {auth_type}"
            )
        
        if self.config.get('load_sample_data', True):
            claims_count = "Configuration only" if self.config.get('skip_claims_data') else "100,000 claims"
            table.add_row("Sample Data", "✓ Loaded", claims_count)
        
        console.print(table)

    def run(self) -> bool:
        """Run the complete database setup"""
        self.show_banner()
        
        if not self.check_dependencies():
            return False
        
        success = True
        
        # Setup PostgreSQL if requested
        if not self.setup_postgres():
            success = False
        
        # Setup SQL Server if requested
        if not self.setup_sqlserver():
            success = False
        
        # Load sample data if requested and at least one database was set up successfully
        if success and (self.config.get('setup_postgres') or self.config.get('setup_sqlserver')):
            if not self.load_sample_data():
                success = False
        
        if success:
            console.print("\n[bold green]SUCCESS: Database setup completed successfully![/bold green]")
            self.show_summary()
        else:
            console.print("\n[bold red]ERROR: Database setup failed[/bold red]")
        
        return success


def main():
    parser = argparse.ArgumentParser(
        description="Setup Smart Pro Claims databases using config/.env.example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup both databases using config/.env.example (RECOMMENDED)
  python scripts/setup_database.py
  
  # Setup only PostgreSQL for claims processing
  python scripts/setup_database.py --postgres-only
  
  # Setup only SQL Server for analytics/reporting  
  python scripts/setup_database.py --sqlserver-only
  
  # Use custom environment file
  python scripts/setup_database.py --env config/.env
  
  # Skip sample data loading
  python scripts/setup_database.py --skip-sample-data
        """
    )
    
    # Environment configuration
    parser.add_argument(
        '--env', 
        default='config/.env',
        help='Environment file path (default: config/.env, fallback: config/.env.example)'
    )
    
    # Database selection options
    parser.add_argument(
        '--postgres-only', 
        action='store_true', 
        help='Setup only PostgreSQL database (for claims processing)'
    )
    parser.add_argument(
        '--sqlserver-only', 
        action='store_true', 
        help='Setup only SQL Server database (for analytics/reporting)'
    )
    
    # General options
    parser.add_argument(
        '--skip-sample-data', 
        action='store_true', 
        help='Skip loading sample data'
    )
    parser.add_argument(
        '--skip-claims-data', 
        action='store_true', 
        help='Skip loading claims data (configuration only)'
    )
    
    args = parser.parse_args()
    
    # Validate conflicting options
    if args.postgres_only and args.sqlserver_only:
        parser.error("Cannot specify both --postgres-only and --sqlserver-only")
    
    # Load environment configuration
    env_vars = load_env_file(args.env)
    if not env_vars:
        console.print("[red]ERROR: Failed to load environment configuration. Exiting.[/red]")
        console.print("[yellow]   Make sure config/.env exists (copy from config/.env.example)[/yellow]")
        sys.exit(1)
    
    # Build configuration from environment
    config = build_config_from_env(env_vars, args.postgres_only, args.sqlserver_only)
    
    # Apply command line overrides
    if args.skip_sample_data:
        config['load_sample_data'] = False
    if args.skip_claims_data:
        config['skip_claims_data'] = True
    
    # Show configuration summary
    console.print("\n[cyan]Database Setup Configuration:[/cyan]")
    console.print(f"[blue]Environment file: {args.env}[/blue]")
    if config.get('setup_postgres'):
        console.print(f"[green]PostgreSQL: {config['postgres_host']}:{config['postgres_port']} -> {config['postgres_database']}[/green]")
    if config.get('setup_sqlserver'):
        console.print(f"[yellow]SQL Server: {config['sqlserver_host']} -> {config['sqlserver_database']}[/yellow]")
    if config.get('load_sample_data'):
        claims_note = " (config only)" if config.get('skip_claims_data') else " (with 100k claims)"
        console.print(f"[blue]Sample Data: Enabled{claims_note}[/blue]")
    
    setup = DatabaseSetup(config)
    success = setup.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
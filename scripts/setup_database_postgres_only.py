#!/usr/bin/env python3
"""
PostgreSQL-Only Database Setup Script for Smart Pro Claims

This script creates both databases in PostgreSQL:
1. smart_claims_staging - For claims processing workflow
2. smart_pro_claims - For production data (migrated from SQL Server)

Usage:
    python scripts/setup_database_postgres_only.py                    # Uses config/.env.example
    python scripts/setup_database_postgres_only.py --env config/.env  # Uses specific env file
    python scripts/setup_database_postgres_only.py --staging-only     # Setup only staging DB
    python scripts/setup_database_postgres_only.py --production-only  # Setup only production DB
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

def build_config_from_env(env_vars, staging_only=False, production_only=False):
    """Build configuration from environment variables."""
    config = {}
    
    # PostgreSQL configuration
    config['postgres_host'] = env_vars.get('POSTGRES_HOST', 'localhost')
    config['postgres_port'] = int(env_vars.get('POSTGRES_PORT', '5432'))
    config['postgres_user'] = env_vars.get('POSTGRES_USER', 'postgres')
    config['postgres_password'] = env_vars.get('POSTGRES_PASSWORD', '')
    
    # Database configuration
    if not production_only:
        config['setup_staging'] = True
        config['staging_database'] = 'smart_claims_staging'
    else:
        config['setup_staging'] = False
    
    if not staging_only:
        config['setup_production'] = True
        config['production_database'] = 'smart_pro_claims'
    else:
        config['setup_production'] = False
    
    # General settings
    config['load_sample_data'] = True
    config['skip_claims_data'] = False
    
    return config

class PostgreSQLSetup:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.project_root = Path(__file__).parent.parent
        self.staging_schema_path = self.project_root / "database" / "postgresql_claims_processing_schema.sql"
        self.production_schema_path = self.project_root / "database" / "postgresql_smart_pro_claims_schema.sql"
        self.sample_data_script = self.project_root / "scripts" / "load_sample_data_postgres.py"

    def show_banner(self):
        """Display startup banner"""
        console.print(Panel.fit(
            "[bold cyan]Smart Pro Claims PostgreSQL-Only Setup[/bold cyan]\n"
            "[dim]Automated database creation for both staging and production databases[/dim]\n"
            "[dim]- smart_claims_staging: Claims processing workflow[/dim]\n"
            "[dim]- smart_pro_claims: Production data (migrated from SQL Server)[/dim]",
            border_style="cyan"
        ))

    def check_dependencies(self):
        """Check if required dependencies are available"""
        console.print("\n[yellow]Checking dependencies...[/yellow]")
        
        if not POSTGRES_AVAILABLE:
            console.print(f"[red]Missing dependency: psycopg2-binary (for PostgreSQL)[/red]")
            console.print("[yellow]Install with: pip install psycopg2-binary[/yellow]")
            return False
        
        console.print("[green]All dependencies available[/green]")
        return True

    def database_exists(self, db_name: str) -> bool:
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

    def create_database(self, db_name: str) -> bool:
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
            console.print(f"[green]Created PostgreSQL database: {db_name}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Error creating PostgreSQL database: {e}[/red]")
            return False

    def load_schema(self, db_name: str, schema_path: Path) -> bool:
        """Load PostgreSQL schema"""
        try:
            if not schema_path.exists():
                console.print(f"[red]Schema file not found: {schema_path}[/red]")
                return False
            
            # Use psql command for better SQL script execution
            cmd = [
                'psql',
                f'-h{self.config["postgres_host"]}',
                f'-p{self.config["postgres_port"]}',
                f'-U{self.config["postgres_user"]}',
                f'-d{db_name}',
                f'-f{schema_path}',
                '-v', 'ON_ERROR_STOP=1'
            ]
            
            env = os.environ.copy()
            env['PGPASSWORD'] = self.config['postgres_password']
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode == 0:
                console.print(f"[green]Schema loaded successfully for {db_name}[/green]")
                return True
            else:
                console.print(f"[red]Error loading schema for {db_name}: {result.stderr}[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]Error loading schema for {db_name}: {e}[/red]")
            return False

    def setup_staging_database(self) -> bool:
        """Setup smart_claims_staging database"""
        if not self.config.get('setup_staging'):
            return True
            
        console.print("\n[cyan]Setting up claims_staging database...[/cyan]")
        
        db_name = self.config.get('staging_database', 'claims_staging')
        
        # Check if database exists
        if self.database_exists(db_name):
            console.print(f"[yellow]Database '{db_name}' already exists[/yellow]")
        else:
            console.print(f"[blue]Creating database: {db_name}[/blue]")
            if not self.create_database(db_name):
                return False
        
        # Load schema
        console.print(f"[blue]Loading schema for {db_name}...[/blue]")
        return self.load_schema(db_name, self.staging_schema_path)

    def setup_production_database(self) -> bool:
        """Setup smart_pro_claims database"""
        if not self.config.get('setup_production'):
            return True
            
        console.print("\n[cyan]Setting up smart_pro_claims database...[/cyan]")
        
        db_name = self.config.get('production_database', 'smart_pro_claims')
        
        # Check if database exists
        if self.database_exists(db_name):
            console.print(f"[yellow]Database '{db_name}' already exists[/yellow]")
        else:
            console.print(f"[blue]Creating database: {db_name}[/blue]")
            if not self.create_database(db_name):
                return False
        
        # Load schema
        console.print(f"[blue]Loading schema for {db_name}...[/blue]")
        return self.load_schema(db_name, self.production_schema_path)

    def load_sample_data(self) -> bool:
        """Load sample data using the PostgreSQL-only sample data script"""
        if not self.config.get('load_sample_data', True):
            console.print("[yellow]Skipping sample data loading[/yellow]")
            return True
            
        console.print("\n[cyan]Loading sample data...[/cyan]")
        
        if not self.sample_data_script.exists():
            console.print(f"[red]Sample data script not found: {self.sample_data_script}[/red]")
            return False
        
        # Show what will be loaded
        if self.config.get('setup_staging'):
            console.print(f"[green]>>> LOADING CLAIMS INTO STAGING: smart_claims_staging[/green]")
            console.print("[blue]Claims will be loaded into public.claims table for processing workflow[/blue]")
        
        if self.config.get('setup_production'):
            console.print(f"[green]>>> LOADING REFERENCE DATA INTO PRODUCTION: smart_pro_claims[/green]")
            console.print("[blue]Facilities, providers, RVU data, and configuration will be loaded[/blue]")
        
        # Run sample data loader
        cmd = [sys.executable, str(self.sample_data_script)]
        
        if self.config.get('skip_claims_data'):
            cmd.append('--skip-claims')
        
        try:
            console.print(f"[blue]Running PostgreSQL-only sample data loader...[/blue]")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            console.print("[green]Sample data loaded successfully[/green]")
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
        table = Table(title="PostgreSQL Setup Summary")
        table.add_column("Database", style="cyan")
        table.add_column("Purpose", style="green")
        table.add_column("Status", style="bright_green")
        
        if self.config.get('setup_staging'):
            table.add_row(
                "smart_claims_staging",
                "Claims processing workflow",
                "Ready"
            )
        
        if self.config.get('setup_production'):
            table.add_row(
                "smart_pro_claims",
                "Production data & analytics",
                "Ready"
            )
        
        console.print(table)
        
        console.print(f"\n[cyan]Connection Details:[/cyan]")
        console.print(f"[blue]Host: {self.config['postgres_host']}:{self.config['postgres_port']}[/blue]")
        console.print(f"[blue]User: {self.config['postgres_user']}[/blue]")

    def run(self) -> bool:
        """Run the complete database setup"""
        self.show_banner()
        
        if not self.check_dependencies():
            return False
        
        success = True
        
        # Setup staging database if requested
        if not self.setup_staging_database():
            success = False
        
        # Setup production database if requested
        if not self.setup_production_database():
            success = False
        
        # Load sample data if requested and at least one database was set up successfully
        if success and (self.config.get('setup_staging') or self.config.get('setup_production')):
            if not self.load_sample_data():
                success = False
        
        if success:
            console.print("\n[bold green]SUCCESS: PostgreSQL database setup completed successfully![/bold green]")
            console.print("[dim]Both databases are ready for high-performance claims processing[/dim]")
            self.show_summary()
        else:
            console.print("\n[bold red]ERROR: Database setup failed[/bold red]")
        
        return success


def main():
    parser = argparse.ArgumentParser(
        description="Setup Smart Pro Claims PostgreSQL databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup both staging and production databases (RECOMMENDED)
  python scripts/setup_database_postgres_only.py
  
  # Setup only staging database for claims processing
  python scripts/setup_database_postgres_only.py --staging-only
  
  # Setup only production database for analytics
  python scripts/setup_database_postgres_only.py --production-only
  
  # Use custom environment file
  python scripts/setup_database_postgres_only.py --env config/.env
  
  # Skip sample data loading
  python scripts/setup_database_postgres_only.py --skip-sample-data
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
        '--staging-only', 
        action='store_true', 
        help='Setup only smart_claims_staging database'
    )
    parser.add_argument(
        '--production-only', 
        action='store_true', 
        help='Setup only smart_pro_claims database'
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
    if args.staging_only and args.production_only:
        parser.error("Cannot specify both --staging-only and --production-only")
    
    # Load environment configuration
    env_vars = load_env_file(args.env)
    if not env_vars:
        console.print("[red]ERROR: Failed to load environment configuration. Exiting.[/red]")
        console.print("[yellow]   Make sure config/.env exists (copy from config/.env.example)[/yellow]")
        sys.exit(1)
    
    # Build configuration from environment
    config = build_config_from_env(env_vars, args.staging_only, args.production_only)
    
    # Apply command line overrides
    if args.skip_sample_data:
        config['load_sample_data'] = False
    if args.skip_claims_data:
        config['skip_claims_data'] = True
    
    # Show configuration summary
    console.print("\n[cyan]PostgreSQL Setup Configuration:[/cyan]")
    console.print(f"[blue]Environment file: {args.env}[/blue]")
    console.print(f"[green]PostgreSQL: {config['postgres_host']}:{config['postgres_port']}[/green]")
    
    if config.get('setup_staging'):
        console.print(f"[blue]Staging Database: {config['staging_database']}[/blue]")
    if config.get('setup_production'):
        console.print(f"[blue]Production Database: {config['production_database']}[/blue]")
    
    if config.get('load_sample_data'):
        claims_note = " (config only)" if config.get('skip_claims_data') else " (with 100k claims)"
        console.print(f"[blue]Sample Data: Enabled{claims_note}[/blue]")
    
    setup = PostgreSQLSetup(config)
    success = setup.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
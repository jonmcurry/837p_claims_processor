#!/usr/bin/env python3
"""
Quick setup script for PostgreSQL claims processing database
This script sets up PostgreSQL with the claims staging schema and loads sample claims data
"""

import sys
import subprocess
from pathlib import Path

def main():
    print("🏥 PostgreSQL Claims Processing Database Setup")
    print("=" * 50)
    
    # Get database connection details
    print("Enter PostgreSQL connection details:")
    host = input("Host [localhost]: ").strip() or "localhost"
    port = input("Port [5432]: ").strip() or "5432"
    user = input("Username [claims_user]: ").strip() or "claims_user"
    password = input("Password: ").strip()
    database = input("Database name [claims_staging]: ").strip() or "claims_staging"
    
    if not password:
        print("❌ Password is required")
        sys.exit(1)
    
    print(f"\n📋 Configuration:")
    print(f"   Host: {host}:{port}")
    print(f"   User: {user}")
    print(f"   Database: {database}")
    
    confirm = input("\nProceed with setup? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Setup cancelled")
        sys.exit(0)
    
    # Build command for setup_database.py
    script_dir = Path(__file__).parent
    setup_script = script_dir / "setup_database.py"
    
    if not setup_script.exists():
        print(f"❌ Setup script not found: {setup_script}")
        sys.exit(1)
    
    cmd = [
        sys.executable,
        str(setup_script),
        "--postgres-host", host,
        "--postgres-port", port,
        "--postgres-user", user,
        "--postgres-password", password,
        "--postgres-database", database
    ]
    
    print(f"\n🚀 Running database setup...")
    print("This will:")
    print("  1. Create PostgreSQL database (if needed)")
    print("  2. Load claims processing schema")
    print("  3. Load sample claims data into public.claims table")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ PostgreSQL claims processing database setup complete!")
        print(f"🎯 Claims data loaded into {database}.public.claims")
        print("📊 Ready for claims processing workflow!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Setup failed with exit code: {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
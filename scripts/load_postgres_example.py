#!/usr/bin/env python3
"""
Example script showing how to load claims into PostgreSQL staging database
This ensures claims go to PostgreSQL public.claims table for processing workflow
"""

import subprocess
import sys

def main():
    print("🏥 PostgreSQL Claims Loading Example")
    print("=" * 50)
    
    # Example PostgreSQL connection strings
    examples = [
        {
            "name": "Local PostgreSQL (default)",
            "connection": "postgresql://claims_user:password@localhost:5432/claims_staging"
        },
        {
            "name": "Remote PostgreSQL",
            "connection": "postgresql://claims_user:password@your-server:5432/claims_staging"
        },
        {
            "name": "PostgreSQL with custom port",
            "connection": "postgresql://claims_user:password@localhost:5433/claims_staging"
        }
    ]
    
    print("Example PostgreSQL connection strings:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example['name']}")
        print(f"   {example['connection']}")
        print()
    
    print("To load claims into PostgreSQL staging database, run:")
    print(f"python3 load_sample_data.py --connection-string \"{examples[0]['connection']}\"")
    print()
    
    # Ask user if they want to run with example connection
    response = input("Run with example connection? (y/N): ").strip().lower()
    
    if response == 'y':
        connection_string = examples[0]['connection']
        print(f"\n🚀 Loading claims into PostgreSQL with connection:")
        print(f"   {connection_string}")
        print()
        
        # Run the load_sample_data.py script
        try:
            cmd = [sys.executable, "load_sample_data.py", "--connection-string", connection_string]
            subprocess.run(cmd, check=True)
            print("\n✅ Claims loaded successfully into PostgreSQL!")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Failed to load claims: {e}")
            sys.exit(1)
        except FileNotFoundError:
            print("\n❌ load_sample_data.py not found. Make sure you're in the scripts directory.")
            sys.exit(1)
    else:
        print("Manually run the script with your PostgreSQL connection string.")

if __name__ == "__main__":
    main()
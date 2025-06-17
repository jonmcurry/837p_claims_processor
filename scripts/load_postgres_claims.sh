#!/bin/bash

# Script to load claims data into PostgreSQL staging database
# This ensures claims are loaded into PostgreSQL public.claims table for processing workflow
# Usage: ./scripts/load_postgres_claims.sh [connection_string]

# Default PostgreSQL connection string
DEFAULT_CONNECTION="postgresql://claims_user:password@localhost:5432/claims_staging"

# Use provided connection string or default
CONNECTION_STRING="${1:-$DEFAULT_CONNECTION}"

echo "🏥 Loading Claims Data into PostgreSQL Staging Database"
echo "======================================================="
echo "Connection: ${CONNECTION_STRING}"
echo

# Validate connection string
if [[ ! "$CONNECTION_STRING" =~ ^postgresql:// ]] && [[ ! "$CONNECTION_STRING" =~ ^postgres:// ]]; then
    echo "❌ ERROR: Connection string must start with 'postgresql://' or 'postgres://'"
    echo "   Current: $CONNECTION_STRING"
    echo "   Example: postgresql://claims_user:password@localhost:5432/claims_staging"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is required but not installed."
    exit 1
fi

# Install required packages if needed
echo "📦 Installing required packages..."
pip3 install sqlalchemy psycopg2-binary faker

# Run the sample data loader
echo "🚀 Starting data load into PostgreSQL..."
echo ">>> Claims will be loaded into public.claims table"
python3 scripts/load_sample_data.py --connection-string "$CONNECTION_STRING"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Claims data loaded successfully into PostgreSQL!"
    echo "🎯 Claims are now in public.claims table for processing workflow!"
    echo "📊 Database ready for claims processing pipeline!"
else
    echo "❌ Failed to load claims data"
    exit 1
fi
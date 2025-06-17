#!/bin/bash

# Simple script to load claims data into PostgreSQL
# Usage: ./scripts/load_postgres_claims.sh [connection_string]

# Default PostgreSQL connection string
DEFAULT_CONNECTION="postgresql://claims_user:password@localhost:5432/claims_staging"

# Use provided connection string or default
CONNECTION_STRING="${1:-$DEFAULT_CONNECTION}"

echo "🏥 Loading Claims Data into PostgreSQL"
echo "======================================"
echo "Connection: ${CONNECTION_STRING}"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is required but not installed."
    exit 1
fi

# Install required packages if needed
echo "📦 Installing required packages..."
pip3 install sqlalchemy psycopg2-binary faker

# Run the sample data loader
echo "🚀 Starting data load..."
python3 scripts/load_sample_data.py --connection-string "$CONNECTION_STRING"

if [ $? -eq 0 ]; then
    echo "✅ Claims data loaded successfully into PostgreSQL!"
    echo "🎯 Ready for claims processing workflow!"
else
    echo "❌ Failed to load claims data"
    exit 1
fi
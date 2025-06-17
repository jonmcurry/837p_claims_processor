#!/bin/bash
# Example usage script for the claims processing pipeline runner
# This script demonstrates different ways to use run_pipeline.py

echo "=== Claims Processing Pipeline Runner - Usage Examples ==="
echo

echo "1. Basic usage with batch ID:"
echo "   python scripts/run_pipeline.py BATCH_001"
echo

echo "2. Using the --batch-id flag:"
echo "   python scripts/run_pipeline.py --batch-id BATCH_001"
echo

echo "3. Verbose logging for debugging:"
echo "   python scripts/run_pipeline.py --batch-id BATCH_001 --verbose"
echo

echo "4. Dry run (validate without processing):"
echo "   python scripts/run_pipeline.py --batch-id BATCH_001 --dry-run"
echo

echo "5. Configuration check:"
echo "   python scripts/run_pipeline.py --config-check"
echo

echo "6. Help information:"
echo "   python scripts/run_pipeline.py --help"
echo

echo "=== Running a test configuration check ==="
echo "python scripts/run_pipeline.py --config-check"
# Uncomment the line below to actually run the config check
# python scripts/run_pipeline.py --config-check

echo
echo "=== Notes ==="
echo "- Make sure to install dependencies first: pip install -r requirements.txt"
echo "- Configure your .env file with database connection settings"
echo "- Run from the project root directory"
echo "- Check pipeline_runner.log for detailed logs"
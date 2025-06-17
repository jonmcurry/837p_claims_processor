#!/usr/bin/env python3
"""
Simple Claims Processing Execution Script

This script demonstrates how to execute claims processing from PostgreSQL to SQL Server.
Run this script to process a batch of claims.

Usage:
    python run_claims_processing.py [batch_id]

Examples:
    python run_claims_processing.py BATCH_001
    python run_claims_processing.py
"""

import requests
import sys
import time
import asyncio

def process_claims_via_api(batch_id="BATCH_001", api_url="http://localhost:8000"):
    """Process claims via the API endpoint."""
    print(f"🔄 Processing claims batch: {batch_id}")
    print(f"📡 API URL: {api_url}")
    
    try:
        # Make POST request to process batch
        response = requests.post(f"{api_url}/process-batch/{batch_id}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Claims processing completed successfully!")
            print(f"📊 Results:")
            print(f"   • Batch ID: {result['batch_id']}")
            print(f"   • Total Claims: {result['total_claims']:,}")
            print(f"   • Processed: {result['processed_claims']:,}")
            print(f"   • Failed: {result['failed_claims']:,}")
            print(f"   • Processing Time: {result['processing_time']:.2f} seconds")
            print(f"   • Throughput: {result['throughput']:.1f} claims/sec")
            print(f"   • Status: {result['status']}")
            print(f"   • Message: {result['message']}")
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the API server is running:")
        print("   python simple_api.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_api_health(api_url="http://localhost:8000"):
    """Check if the API is running and healthy."""
    try:
        response = requests.get(f"{api_url}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✅ API is healthy - Uptime: {health['uptime']:.1f} seconds")
            return True
        else:
            return False
    except:
        return False

async def run_pipeline_directly(batch_id="BATCH_001"):
    """Run the processing pipeline directly (requires dependencies)."""
    try:
        print(f"🔄 Running pipeline directly for batch: {batch_id}")
        
        # This would require the full environment to be set up
        # from src.processing.batch_processor.pipeline import processing_pipeline
        # result = await processing_pipeline.process_batch(batch_id)
        
        print("⚠️  Direct pipeline execution requires full environment setup")
        print("   Use the API approach instead: python simple_api.py")
        return False
        
    except ImportError as e:
        print(f"⚠️  Missing dependencies for direct execution: {e}")
        print("   Use the API approach instead")
        return False

def main():
    print("🏥 Claims Processing System")
    print("=" * 50)
    
    # Get batch ID from command line or use default
    batch_id = sys.argv[1] if len(sys.argv) > 1 else "BATCH_001"
    
    print(f"📋 Processing batch: {batch_id}")
    print(f"🔄 Flow: PostgreSQL → Processing → SQL Server")
    print()
    
    # Check if API is running
    if check_api_health():
        print("🚀 Processing claims via API...")
        success = process_claims_via_api(batch_id)
        
        if success:
            print()
            print("🎉 Claims processing completed!")
            print("📊 Check the results in:")
            print("   • SQL Server production database")
            print("   • Claims UI at http://localhost:8050")
        else:
            print("❌ Claims processing failed")
    else:
        print("⚠️  API server not running. Starting instructions:")
        print()
        print("1. Start the API server:")
        print("   python simple_api.py")
        print()
        print("2. Then run this script again:")
        print(f"   python run_claims_processing.py {batch_id}")
        print()
        print("3. Or start the UI for monitoring:")
        print("   cd claims_ui && python run.py")

if __name__ == "__main__":
    main()
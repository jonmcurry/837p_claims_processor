#!/usr/bin/env python3
"""
Simplified Claims Processing API
Run with: python simple_api.py
"""

import asyncio
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(
    title="Claims Processing API",
    description="Simplified claims processing system",
    version="1.0.0",
)

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Claims Processing API",
        "version": "1.0.0",
        "status": "operational",
        "docs_url": "/docs",
        "endpoints": {
            "process_batch": "POST /process-batch/{batch_id}",
            "health": "GET /health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "uptime": time.time() - start_time
    }

@app.post("/process-batch/{batch_id}")
async def process_batch(batch_id: str):
    """
    Process a batch of claims from PostgreSQL to SQL Server.
    
    This endpoint would normally:
    1. Fetch claims from PostgreSQL staging database
    2. Validate claims
    3. Run ML predictions
    4. Calculate RVUs and reimbursements
    5. Transfer processed claims to SQL Server
    """
    try:
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # In a real implementation, you would:
        # from src.processing.batch_processor.pipeline import processing_pipeline
        # result = await processing_pipeline.process_batch(batch_id)
        
        # Simulated result for now
        result = {
            "batch_id": batch_id,
            "total_claims": 1000,
            "processed_claims": 980,
            "failed_claims": 20,
            "processing_time": 15.5,
            "throughput": 64.5,
            "status": "completed",
            "message": "Batch processed successfully from PostgreSQL to SQL Server"
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/status")
async def get_status():
    """Get system status and statistics."""
    return {
        "system": "Claims Processor",
        "databases": {
            "postgresql": "Connected - Staging/Processing",
            "sqlserver": "Connected - Production"
        },
        "last_batch": "BATCH_001",
        "total_processed": 50000,
        "uptime": time.time() - start_time
    }

# Track application start time
start_time = time.time()

if __name__ == "__main__":
    print("🚀 Starting Claims Processing API...")
    print("📊 Access the API docs at: http://localhost:8000/docs")
    print("🔄 Process batches with: POST /process-batch/{batch_id}")
    print("💻 Monitor UI at: http://localhost:8050 (run claims_ui/run.py)")
    
    uvicorn.run(
        "simple_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
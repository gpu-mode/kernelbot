#!/usr/bin/env python3
"""
Standalone script to run the Discord Cluster Manager Backend API.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from main import app

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    print("Starting Discord Cluster Manager Backend API...")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Debug: {debug}")
    print(f"Database URL: {os.getenv('DATABASE_URL', 'Not set')}")

    uvicorn.run(app, host=host, port=port, reload=debug, log_level="debug" if debug else "info")

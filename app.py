"""
Voltforge AI Microservice - Application Entry Point

This module provides backwards compatibility with legacy runners (e.g. Dockerfile).
All active routing and domain logic is modularized under /api and /engine.
"""

import os
import sys

# Ensure current directory is on Python path before importing local modules
AI_DIR = os.path.dirname(os.path.abspath(__file__))
if AI_DIR not in sys.path:
    sys.path.insert(0, AI_DIR)

import uvicorn
import config
from main import app

if __name__ == "__main__":
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=False)


#!/bin/bash
# Start the API server using the virtual environment

cd "$(dirname "$0")"
source .venv/bin/activate
.venv/bin/uvicorn api.main:app --reload --port 8000

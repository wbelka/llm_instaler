#!/bin/bash

# Universal model startup script
# This script is copied to each model's directory during installation

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if installation is complete
if [ ! -f ".install_complete" ]; then
    echo "Error: Model installation is not complete"
    echo "Please run: llm-installer install <model-id>"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found"
    echo "Model may not be properly installed"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Parse command line arguments
PORT=8000
HOST="0.0.0.0"
DTYPE="auto"
DEVICE="auto"
STREAM_MODE="false"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --load-lora)
            EXTRA_ARGS="$EXTRA_ARGS --load-lora $2"
            shift 2
            ;;
        --stream)
            STREAM_MODE="true"
            shift
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Check for LoRA adapter
if [ -f "./lora/adapter_model.bin" ]; then
    echo "Found LoRA adapter, loading with modifications..."
    EXTRA_ARGS="$EXTRA_ARGS --load-lora ./lora"
fi

# Get model name from model_info.json
MODEL_NAME=$(python -c "import json; print(json.load(open('model_info.json'))['model_id'])")

echo "Starting model: $MODEL_NAME"
echo "API will be available at: http://localhost:$PORT"

# Start the API server
echo "Starting API server..."
if [ "$STREAM_MODE" = "true" ]; then
    echo "Streaming mode enabled"
fi

python serve_api.py \
    --port "$PORT" \
    --host "$HOST" \
    --dtype "$DTYPE" \
    --device "$DEVICE" \
    --stream-mode "$STREAM_MODE" \
    $EXTRA_ARGS &

API_PID=$!

# Wait for server to start
echo "Waiting for server to start..."
for i in {1..30}; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    sleep 1
done

# Show API URL
echo ""
echo "API is running at: http://localhost:$PORT"
echo "Web UI available at: http://localhost:$PORT"
echo "API docs at: http://localhost:$PORT/docs"
echo ""
echo "Press Ctrl+C to stop the server"

# Wait for API process
wait $API_PID
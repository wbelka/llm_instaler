#!/bin/bash

# Universal model training script
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
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Parse command line arguments
DATA_PATH=""
OUTPUT_PATH="./lora"
METHOD="lora"
EPOCHS=""
BATCH_SIZE=""
LEARNING_RATE=""
LORA_R=""
LORA_ALPHA=""
CIRCULAR=false
RESUME=false
RESUME_FROM=""
NO_TENSORBOARD=false
TENSORBOARD_PORT=6006
PATIENCE=3
OVERFITTING_THRESHOLD=0.1
MIN_EVALUATIONS=5

while [[ $# -gt 0 ]]; do
    case $1 in
        --data)
            DATA_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --lora-r)
            LORA_R="$2"
            shift 2
            ;;
        --lora-alpha)
            LORA_ALPHA="$2"
            shift 2
            ;;
        --circular)
            CIRCULAR=true
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --resume-from)
            RESUME_FROM="$2"
            shift 2
            ;;
        --no-tensorboard)
            NO_TENSORBOARD=true
            shift
            ;;
        --tensorboard-port)
            TENSORBOARD_PORT="$2"
            shift 2
            ;;
        --patience)
            PATIENCE="$2"
            shift 2
            ;;
        --overfitting-threshold)
            OVERFITTING_THRESHOLD="$2"
            shift 2
            ;;
        --min-evaluations)
            MIN_EVALUATIONS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if data path is provided
if [ -z "$DATA_PATH" ]; then
    echo "Error: --data parameter is required"
    echo "Usage: $0 --data <path_to_training_data>"
    exit 1
fi

# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Training data file not found: $DATA_PATH"
    exit 1
fi

# Get model name from model_info.json
MODEL_NAME=$(python -c "import json; print(json.load(open('model_info.json'))['model_id'])")

echo "Training model: $MODEL_NAME"
echo "Training data: $DATA_PATH"
echo "Method: $METHOD"
echo "Output path: $OUTPUT_PATH"

# Create output directory
mkdir -p "$OUTPUT_PATH"
mkdir -p "logs/training"

# Start TensorBoard if not disabled
if [ "$NO_TENSORBOARD" = false ]; then
    echo "Starting TensorBoard on port $TENSORBOARD_PORT..."
    tensorboard --logdir ./logs/training --port "$TENSORBOARD_PORT" --bind_all > /dev/null 2>&1 &
    TB_PID=$!
    echo "TensorBoard: http://localhost:$TENSORBOARD_PORT"
fi

# For now, show a placeholder message
echo ""
echo "Note: Full training functionality will be implemented in Step 5"
echo "This script currently serves as a placeholder for the training interface"
echo ""
echo "Expected features:"
echo "- Automatic parameter detection"
echo "- LoRA/QLoRA fine-tuning"
echo "- Auto-learning algorithm to prevent overfitting"
echo "- Support for various data formats (Alpaca, ShareGPT, etc.)"
echo "- Progress monitoring with TensorBoard"
echo ""

# Clean up TensorBoard if started
if [ ! -z "$TB_PID" ]; then
    echo "Press Ctrl+C to stop..."
    wait $TB_PID
fi
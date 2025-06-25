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
USE_8BIT=false
USE_4BIT=false
MAX_SEQ_LENGTH=""
MODE=""
FORCE_EPOCHS=false

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
        --use-8bit)
            USE_8BIT=true
            shift
            ;;
        --use-4bit)
            USE_4BIT=true
            shift
            ;;
        --max-seq-length)
            MAX_SEQ_LENGTH="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --force-epochs)
            FORCE_EPOCHS=true
            shift
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
    echo "Examples:"
    echo "  $0 --data dataset.json                  # Single file"
    echo "  $0 --data 'data/*.json'                 # Multiple files (use quotes!)"
    echo "  $0 --data datasets/                     # Directory"
    echo "  $0 --data 'file1.json,file2.json'       # Comma-separated list"
    exit 1
fi

# Check if data exists (file, directory, or pattern)
# Handle comma-separated list
if [[ "$DATA_PATH" == *","* ]]; then
    # Split comma-separated paths and check each
    IFS=',' read -ra DATA_ARRAY <<< "$DATA_PATH"
    for path in "${DATA_ARRAY[@]}"; do
        path=$(echo "$path" | xargs)  # Trim whitespace
        if [ ! -f "$path" ] && [ ! -d "$path" ]; then
            echo "Error: Data file not found: $path"
            exit 1
        fi
    done
    echo "Using multiple data files: $DATA_PATH"
elif [[ "$DATA_PATH" == *"*"* ]]; then
    # Handle glob pattern
    # Check if any files match the pattern
    shopt -s nullglob
    files=($DATA_PATH)
    shopt -u nullglob
    if [ ${#files[@]} -eq 0 ]; then
        echo "Error: No files match pattern: $DATA_PATH"
        exit 1
    fi
    echo "Found ${#files[@]} files matching pattern: $DATA_PATH"
elif [ -d "$DATA_PATH" ]; then
    # Directory
    echo "Using directory: $DATA_PATH"
elif [ -f "$DATA_PATH" ]; then
    # Single file
    echo "Using file: $DATA_PATH"
else
    echo "Error: Data path not found: $DATA_PATH"
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

# Build command for train_lora.py
CMD="python train_lora.py --data \"$DATA_PATH\" --output \"$OUTPUT_PATH\""

# Add optional parameters
[ ! -z "$METHOD" ] && CMD="$CMD --method $METHOD"
[ ! -z "$EPOCHS" ] && CMD="$CMD --epochs $EPOCHS"
[ ! -z "$BATCH_SIZE" ] && CMD="$CMD --batch-size $BATCH_SIZE"
[ ! -z "$LEARNING_RATE" ] && CMD="$CMD --learning-rate $LEARNING_RATE"
[ ! -z "$LORA_R" ] && CMD="$CMD --lora-r $LORA_R"
[ ! -z "$LORA_ALPHA" ] && CMD="$CMD --lora-alpha $LORA_ALPHA"
[ "$CIRCULAR" = true ] && CMD="$CMD --circular"
[ "$RESUME" = true ] && CMD="$CMD --resume"
[ ! -z "$RESUME_FROM" ] && CMD="$CMD --resume-from \"$RESUME_FROM\""
[ ! -z "$PATIENCE" ] && CMD="$CMD --patience $PATIENCE"
[ ! -z "$OVERFITTING_THRESHOLD" ] && CMD="$CMD --overfitting-threshold $OVERFITTING_THRESHOLD"
[ "$USE_8BIT" = true ] && CMD="$CMD --use-8bit"
[ "$USE_4BIT" = true ] && CMD="$CMD --use-4bit"
[ ! -z "$MAX_SEQ_LENGTH" ] && CMD="$CMD --max-seq-length $MAX_SEQ_LENGTH"
[ ! -z "$MODE" ] && CMD="$CMD --mode $MODE"
[ "$FORCE_EPOCHS" = true ] && CMD="$CMD --force-epochs"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ ! -z "$TB_PID" ] && kill -0 $TB_PID 2>/dev/null; then
        kill $TB_PID 2>/dev/null || true
    fi
    exit 0
}

# Set trap for cleanup
trap cleanup INT TERM EXIT

# Run training
echo "Starting training..."
echo "Command: $CMD"
echo ""

# Execute training
eval $CMD
TRAINING_EXIT_CODE=$?

# Check if training completed successfully
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
    echo "LoRA adapter saved to: $OUTPUT_PATH"
    echo ""
    echo "To use the trained model:"
    echo "  ./start.sh"
    echo ""
    echo "The model will automatically load the LoRA adapter if present."
else
    echo ""
    echo "❌ Training failed with exit code: $TRAINING_EXIT_CODE"
fi

# Cleanup will be called automatically due to trap
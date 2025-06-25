#!/bin/bash

# Universal model training script
# This script is copied to each model's directory during installation

set -e

# Check if only tensorboard is requested
if [ "$1" == "--tensorboard-only" ]; then
    shift
    PORT="${1:-6006}"
    echo "Starting TensorBoard server..."
    
    # Check if logs directory exists
    if [ -d "./lora/logs" ]; then
        echo "Looking for logs in: ./lora/logs"
    elif [ -d "./logs/training" ]; then
        echo "Looking for logs in: ./logs/training"
        LOG_DIR="./logs/training"
    else
        echo "No training logs found. Run training first to generate logs."
        exit 1
    fi
    
    LOG_DIR="${LOG_DIR:-./lora/logs}"
    echo "TensorBoard: http://localhost:$PORT"
    echo ""
    echo "Press Ctrl+C to stop"
    tensorboard --logdir "$LOG_DIR" --port "$PORT" --bind_all
    exit 0
fi

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
MAX_CIRCULAR_EPOCHS=""
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
CIRCULAR_BATCH_MULTIPLIER=""
HELP=false

# Show help if requested
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    HELP=true
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            HELP=true
            shift
            ;;
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
        --max-circular-epochs)
            MAX_CIRCULAR_EPOCHS="$2"
            shift 2
            ;;
        --circular-batch-multiplier)
            CIRCULAR_BATCH_MULTIPLIER="$2"
            shift 2
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

# Show help if requested or no data provided
if [ "$HELP" = true ] || [ -z "$DATA_PATH" ]; then
    if [ -z "$DATA_PATH" ] && [ "$HELP" = false ]; then
        echo "Error: --data parameter is required"
        echo ""
    fi
    
    echo "Universal LoRA Training Script"
    echo "=============================="
    echo ""
    echo "Usage: $0 --data <path_to_training_data> [options]"
    echo ""
    echo "REQUIRED:"
    echo "  --data PATH                    Training data (file, directory, pattern, or comma-separated list)"
    echo ""
    echo "TRAINING MODES:"
    echo "  --mode MODE                    Training mode: quick, balanced, quality (default: auto-detect)"
    echo "                                 Also supports: fast, medium, slow (legacy names)"
    echo "  --method METHOD                Training method: lora, qlora (default: lora)"
    echo "  --circular                     Enable circular training (repeat dataset multiple times)"
    echo "  --force-epochs                 Force exact number of epochs (disable auto-stop)"
    echo ""
    echo "TRAINING PARAMETERS:"
    echo "  --epochs N                     Number of epochs (overrides mode default)"
    echo "  --batch-size N                 Batch size (auto-detected if not set)"
    echo "  --learning-rate RATE           Learning rate (auto-detected if not set)"
    echo "  --max-seq-length N             Maximum sequence length (model default if not set)"
    echo ""
    echo "LORA PARAMETERS:"
    echo "  --lora-r N                     LoRA rank (auto-detected: 8-64 based on model)"
    echo "  --lora-alpha N                 LoRA alpha (default: 2*r)"
    echo ""
    echo "CIRCULAR TRAINING OPTIONS:"
    echo "  --max-circular-epochs N        Maximum circular epochs (default: 100)"
    echo "  --circular-batch-multiplier X  Batch size change per cycle:"
    echo "                                 - 1.0 = no change (default)"
    echo "                                 - 0<X<2 = increment by X each cycle"
    echo "                                 - X>=2 = multiply by X each cycle"
    echo ""
    echo "AUTO-STOP PARAMETERS:"
    echo "  --patience N                   Early stopping patience (default: 3)"
    echo "  --overfitting-threshold N      Max train/val loss gap (default: 0.1)"
    echo "  --min-evaluations N            Min evals before stopping (default: 5)"
    echo ""
    echo "MEMORY OPTIMIZATION:"
    echo "  --use-8bit                     Use 8-bit quantization"
    echo "  --use-4bit                     Use 4-bit quantization (implies qlora)"
    echo ""
    echo "RESUME TRAINING:"
    echo "  --resume                       Resume from last checkpoint"
    echo "  --resume-from PATH             Resume from specific checkpoint"
    echo ""
    echo "OTHER OPTIONS:"
    echo "  --output PATH                  Output directory (default: ./lora)"
    echo "  --no-tensorboard               Disable TensorBoard"
    echo "  --tensorboard-port N           TensorBoard port (default: 6006)"
    echo "  --tensorboard-only [PORT]      Only start TensorBoard server"
    echo "  -h, --help                     Show this help"
    echo ""
    echo "EXAMPLES:"
    echo ""
    echo "Basic training:"
    echo "  $0 --data dataset.json"
    echo "  $0 --data 'data/*.json'                    # Multiple files (use quotes!)"
    echo "  $0 --data datasets/                        # Directory"
    echo ""
    echo "Quality modes:"
    echo "  $0 --data data.json --mode quick           # Fast training, lower quality"
    echo "  $0 --data data.json --mode balanced        # Balanced speed/quality"
    echo "  $0 --data data.json --mode quality         # Best quality, slower"
    echo ""
    echo "Custom parameters:"
    echo "  $0 --data data.json --epochs 10 --batch-size 4 --learning-rate 2e-4"
    echo "  $0 --data data.json --lora-r 32 --lora-alpha 64"
    echo ""
    echo "Circular training (for small datasets):"
    echo "  $0 --data small_data.json --circular"
    echo "  $0 --data small_data.json --circular --max-circular-epochs 20"
    echo "  $0 --data small_data.json --circular --circular-batch-multiplier 1  # +1 batch each cycle"
    echo "  $0 --data small_data.json --circular --circular-batch-multiplier 2  # 2x batch each cycle"
    echo ""
    echo "Memory optimization:"
    echo "  $0 --data data.json --use-8bit             # 8-bit quantization"
    echo "  $0 --data data.json --method qlora         # 4-bit quantization"
    echo "  $0 --data data.json --use-4bit             # Also enables qlora"
    echo ""
    echo "Force exact epochs:"
    echo "  $0 --data data.json --epochs 5 --force-epochs"
    echo "  $0 --data data.json --circular --max-circular-epochs 50 --force-epochs"
    echo ""
    echo "Resume training:"
    echo "  $0 --data data.json --resume               # From last checkpoint"
    echo "  $0 --data data.json --resume-from checkpoints/checkpoint-500"
    echo ""
    echo "Multiple datasets:"
    echo "  $0 --data 'train.json,valid.json,test.json'"
    echo "  $0 --data 'dataset1.json,dataset2.json,dataset3.json'"
    echo ""
    echo "TensorBoard:"
    echo "  $0 --data data.json --no-tensorboard       # Disable TensorBoard"
    echo "  $0 --data data.json --tensorboard-port 6007"
    echo "  $0 --tensorboard-only                      # View existing logs"
    echo "  $0 --tensorboard-only 6008                 # Custom port"
    echo ""
    echo "NOTES:"
    echo "- Auto-detection adapts parameters to your model and dataset"
    echo "- Circular training is recommended for datasets < 1000 examples"
    echo "- Use quotes around glob patterns like 'data/*.json'"
    echo "- TensorBoard runs at http://localhost:6006 by default"
    echo "- Training automatically stops when optimal (unless --force-epochs)"
    echo ""
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
[ ! -z "$MAX_CIRCULAR_EPOCHS" ] && CMD="$CMD --max-circular-epochs $MAX_CIRCULAR_EPOCHS"
[ ! -z "$CIRCULAR_BATCH_MULTIPLIER" ] && CMD="$CMD --circular-batch-multiplier $CIRCULAR_BATCH_MULTIPLIER"
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
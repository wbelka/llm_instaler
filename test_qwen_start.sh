#!/bin/bash
# Test starting Qwen2.5-VL model with updated handler

MODEL_DIR="/home/wblk/LLM/models/Qwen_Qwen2.5-VL-7B-Instruct"

echo "Testing Qwen2.5-VL model startup..."
echo "Model directory: $MODEL_DIR"

# Check if venv exists
if [ ! -d "$MODEL_DIR/.venv" ]; then
    echo "Error: Virtual environment not found at $MODEL_DIR/.venv"
    exit 1
fi

# Activate venv
echo "Activating virtual environment..."
source "$MODEL_DIR/.venv/bin/activate"

# Test Python import
echo -e "\nTesting Python imports..."
python3 -c "
import sys
print(f'Python: {sys.executable}')
try:
    import transformers
    print(f'✓ transformers version: {transformers.__version__}')
except ImportError as e:
    print(f'✗ transformers import failed: {e}')
    
try:
    from transformers import AutoProcessor, AutoModel
    print('✓ Can import AutoProcessor and AutoModel')
except ImportError as e:
    print(f'✗ AutoProcessor/AutoModel import failed: {e}')

try:
    from transformers import Qwen2VLForConditionalGeneration
    print('✓ Can import Qwen2VLForConditionalGeneration')
except ImportError:
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        print('✓ Can import Qwen2_5_VLForConditionalGeneration')
    except ImportError:
        print('✗ No Qwen VL model class found in transformers')
"

# Test model loading
echo -e "\nTesting model loading..."
cd "$MODEL_DIR"
python3 -c "
import os
import sys
import json

# Add current dir to path
sys.path.insert(0, '.')

# Load model info
with open('model_info.json', 'r') as f:
    model_info = json.load(f)

print(f'Model type: {model_info.get(\"model_type\")}')
print(f'Model family: {model_info.get(\"model_family\")}')

# Try loading with model_loader
try:
    from model_loader import load_model
    print('\\nAttempting to load model...')
    model, processor = load_model(
        model_path='./model',
        model_info=model_info,
        device='cuda',
        dtype='int8'
    )
    print('✓ Model loaded successfully!')
    print(f'Model class: {type(model).__name__}')
    print(f'Processor class: {type(processor).__name__}')
except Exception as e:
    print(f'✗ Failed to load model: {e}')
    import traceback
    traceback.print_exc()
"
#!/bin/bash
# Quick fix for already installed Janus model

MODEL_DIR="${1:-/home/wblk/LLM/models/deepseek-ai_Janus-Pro-7B}"

echo "Fixing Janus model installation at: $MODEL_DIR"

# Activate virtual environment
cd "$MODEL_DIR"
source .venv/bin/activate

# Install the Janus package
echo "Installing Janus package..."
pip install git+https://github.com/deepseek-ai/Janus.git

# Also ensure other dependencies are installed
echo "Ensuring other dependencies..."
pip install einops sentencepiece protobuf accelerate

# Update model_info.json to ensure proper handler is used
if [ -f "model_info.json" ]; then
    echo "Updating model_info.json..."
    python3 -c "
import json
with open('model_info.json', 'r') as f:
    info = json.load(f)
info['model_type'] = 'multi_modality'
info['model_family'] = 'multimodal'
info['handler_class'] = 'MultimodalHandler'
with open('model_info.json', 'w') as f:
    json.dump(info, f, indent=2)
"
fi

echo "✓ Fix applied. Try running ./start.sh again"
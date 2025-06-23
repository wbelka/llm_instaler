#!/bin/bash
# Upgrade torch to 2.6+ to fix security vulnerability

MODEL_DIR="${1:-/home/wblk/LLM/models/deepseek-ai_Janus-Pro-7B}"

echo "Upgrading PyTorch in: $MODEL_DIR"

cd "$MODEL_DIR"
source .venv/bin/activate

echo "Current torch version:"
python -c "import torch; print(torch.__version__)"

echo "Upgrading torch, torchvision, torchaudio..."
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "New torch version:"
python -c "import torch; print(torch.__version__)"

echo "✓ PyTorch upgraded successfully"
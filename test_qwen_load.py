#!/usr/bin/env python3
"""Test loading Qwen2.5-VL model with the new handler system."""

import sys
import os

# Add model directory to path
model_dir = "/home/wblk/LLM/models/Qwen_Qwen2.5-VL-7B-Instruct"
sys.path.insert(0, model_dir)

# Import model loader
from model_loader import load_model, get_model_config

# Load model info
import json
model_info_path = os.path.join(model_dir, "model_info.json")
with open(model_info_path, 'r') as f:
    model_info = json.load(f)
    
print(f"Model type: {model_info.get('model_type')}")
print(f"Model family: {model_info.get('model_family')}")
print(f"Architecture: {model_info.get('architecture_type')}")

# Try to load the model
print("\nAttempting to load model...")
try:
    model, processor = load_model(
        model_path=os.path.join(model_dir, "model"),
        model_info=model_info,
        device="cuda",
        dtype="int8"
    )
    print("✓ Model loaded successfully!")
    print(f"Model type: {type(model).__name__}")
    print(f"Processor type: {type(processor).__name__}")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
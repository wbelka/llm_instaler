#!/usr/bin/env python3
"""Test script to explore Qwen2.5-VL model structure."""

import sys
import os
import json

model_dir = "/home/wblk/LLM/models/Qwen_Qwen2.5-VL-7B-Instruct"
sys.path.insert(0, model_dir)

# Activate venv
activate_this = os.path.join(model_dir, '.venv/bin/activate_this.py')
if os.path.exists(activate_this):
    exec(open(activate_this).read(), {'__file__': activate_this})

from transformers import AutoModel, AutoProcessor

print("Loading model...")
model = AutoModel.from_pretrained(
    os.path.join(model_dir, "model"),
    trust_remote_code=True,
    device_map="auto"
)

print(f"\nModel type: {type(model).__name__}")
print(f"Model class: {model.__class__}")

# Explore model attributes
print("\nModel attributes:")
for attr in dir(model):
    if not attr.startswith('_') and not callable(getattr(model, attr, None)):
        print(f"  .{attr}: {type(getattr(model, attr)).__name__}")

# Check for generate method
print(f"\nHas generate: {hasattr(model, 'generate')}")
print(f"Has forward: {hasattr(model, 'forward')}")

# Check language_model
if hasattr(model, 'language_model'):
    lm = model.language_model
    print(f"\nlanguage_model type: {type(lm).__name__}")
    print(f"language_model has generate: {hasattr(lm, 'generate')}")
    
    # Check language_model attributes
    print("\nlanguage_model attributes:")
    for attr in dir(lm):
        if not attr.startswith('_') and not callable(getattr(lm, attr, None)):
            val = getattr(lm, attr)
            if hasattr(val, '__class__'):
                print(f"  .{attr}: {type(val).__name__}")

# Look for model with generate capability
print("\nSearching for generate method...")
def find_generate(obj, path="model", visited=None):
    if visited is None:
        visited = set()
    
    if id(obj) in visited:
        return
    visited.add(id(obj))
    
    if hasattr(obj, 'generate') and callable(getattr(obj, 'generate')):
        print(f"Found generate at: {path}")
        return path
    
    for attr in dir(obj):
        if attr.startswith('_'):
            continue
        try:
            val = getattr(obj, attr)
            if hasattr(val, '__module__') and val.__module__ and 'transformers' in val.__module__:
                result = find_generate(val, f"{path}.{attr}", visited)
                if result:
                    return result
        except:
            pass

result = find_generate(model)
print(f"Generate method location: {result}")

# Try the official way
print("\n\nTrying official Qwen2.5-VL approach...")
try:
    from transformers import Qwen2VLForConditionalGeneration
    print("Found Qwen2VLForConditionalGeneration!")
    
    # Try loading with this class
    model2 = Qwen2VLForConditionalGeneration.from_pretrained(
        os.path.join(model_dir, "model"),
        trust_remote_code=True,
        device_map="auto"
    )
    print(f"Model2 type: {type(model2).__name__}")
    print(f"Model2 has generate: {hasattr(model2, 'generate')}")
except ImportError as e:
    print(f"Qwen2VLForConditionalGeneration not available: {e}")
except Exception as e:
    print(f"Error loading with Qwen2VLForConditionalGeneration: {e}")
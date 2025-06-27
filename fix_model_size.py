#!/usr/bin/env python3
"""Fix model_size_gb in existing model installations."""

import json
import sys
from pathlib import Path

def fix_model_size(model_path: str):
    """Fix model_size_gb in model_info.json."""
    model_dir = Path(model_path)
    model_info_path = model_dir / "model_info.json"
    
    if not model_info_path.exists():
        print(f"Error: {model_info_path} not found")
        return False
    
    # Load model info
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    # Calculate actual model size
    model_files = list((model_dir / "model").glob("*.safetensors")) + \
                  list((model_dir / "model").glob("*.bin"))
    
    total_size_gb = sum(f.stat().st_size for f in model_files) / (1024**3)
    
    print(f"Current model_size_gb: {model_info.get('model_size_gb', 'Not set')}")
    print(f"Actual model size: {total_size_gb:.2f} GB")
    
    # Update model info
    model_info['model_size_gb'] = total_size_gb
    
    # Save back
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"Updated model_size_gb to {total_size_gb:.2f} GB")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_model_size.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    if fix_model_size(model_path):
        print("Success!")
    else:
        print("Failed!")
        sys.exit(1)
#!/usr/bin/env python3
"""Check model_info.json in an installed model."""

import json
import sys
from pathlib import Path


def check_model_info(model_path: Path):
    """Check and display model_info.json content."""
    
    model_info_path = model_path / "model_info.json"
    
    if not model_info_path.exists():
        print(f"Error: model_info.json not found at {model_info_path}")
        return
    
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    print("Current model_info.json content:")
    print(json.dumps(model_info, indent=2))
    
    print("\nKey fields for handler selection:")
    print(f"- model_type: {model_info.get('model_type', 'NOT SET')}")
    print(f"- model_family: {model_info.get('model_family', 'NOT SET')}")
    print(f"- handler_class: {model_info.get('handler_class', 'NOT SET')}")
    print(f"- primary_library: {model_info.get('primary_library', 'NOT SET')}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_model_info.py <model_path>")
        sys.exit(1)
    
    model_path = Path(sys.argv[1])
    check_model_info(model_path)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Check torch version in a model installation."""

import sys
import subprocess
from pathlib import Path


def check_torch_version(model_path: Path):
    """Check torch version in model's virtual environment."""
    
    venv_python = model_path / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = model_path / ".venv" / "Scripts" / "python.exe"
    
    if not venv_python.exists():
        print(f"Error: Virtual environment not found at {model_path}")
        return
    
    # Check torch version
    cmd = [str(venv_python), "-c", "import torch; print(f'PyTorch version: {torch.__version__}')"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error checking torch version: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_torch_version.py <model_path>")
        sys.exit(1)
    
    model_path = Path(sys.argv[1])
    check_torch_version(model_path)


if __name__ == "__main__":
    main()
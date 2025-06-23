#!/usr/bin/env python3
"""Direct torch upgrade for a model."""

import subprocess
import sys
from pathlib import Path


def upgrade_torch(model_path: Path):
    """Upgrade torch directly in model venv."""
    
    pip_path = model_path / ".venv" / "bin" / "pip"
    if not pip_path.exists():
        pip_path = model_path / ".venv" / "Scripts" / "pip.exe"
    
    print(f"Upgrading torch in: {model_path}")
    
    # First uninstall old versions
    print("Uninstalling old torch packages...")
    subprocess.run([str(pip_path), "uninstall", "-y", "torch", "torchvision", "torchaudio"], 
                   capture_output=True)
    
    # Install new versions
    print("Installing torch 2.6.0+...")
    cmd = [
        str(pip_path), "install", 
        "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        # Try without specific versions
        print("Trying with latest versions...")
        cmd = [
            str(pip_path), "install", 
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121",
            "--upgrade"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Failed: {result.stderr}")
            return False
    
    print("✓ Torch upgraded successfully")
    
    # Check version
    check_cmd = [str(model_path / ".venv" / "bin" / "python"), "-c", 
                 "import torch; print(f'New torch version: {torch.__version__}')"]
    subprocess.run(check_cmd)
    
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python direct_torch_upgrade.py <model_path>")
        sys.exit(1)
    
    model_path = Path(sys.argv[1])
    if not upgrade_torch(model_path):
        sys.exit(1)


if __name__ == "__main__":
    main()
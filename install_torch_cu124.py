#!/usr/bin/env python3
"""Install torch 2.6+ with CUDA 12.4 support."""

import subprocess
import sys
from pathlib import Path


def install_torch_cu124(model_path: Path):
    """Install torch 2.6.0 with CUDA 12.4."""
    
    pip_path = model_path / ".venv" / "bin" / "pip"
    python_path = model_path / ".venv" / "bin" / "python"
    
    if not pip_path.exists():
        pip_path = model_path / ".venv" / "Scripts" / "pip.exe"
        python_path = model_path / ".venv" / "Scripts" / "python.exe"
    
    print(f"Installing torch 2.6.0 with CUDA 12.4 in: {model_path}")
    
    # First uninstall existing torch
    print("Uninstalling existing torch packages...")
    subprocess.run([str(pip_path), "uninstall", "-y", "torch", "torchvision", "torchaudio"], 
                   capture_output=True)
    
    # Install torch 2.6.0 with CUDA 12.4
    print("Installing torch 2.6.0+cu124...")
    cmd = [
        str(pip_path), "install", 
        "torch==2.6.0", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu124"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    
    print("✓ Torch 2.6.0 with CUDA 12.4 installed successfully")
    
    # Verify installation
    check_cmd = [str(python_path), "-c", 
                 "import torch; print(f'Torch version: {torch.__version__}'); "
                 "print(f'CUDA available: {torch.cuda.is_available()}'); "
                 "print(f'CUDA version: {torch.version.cuda}')"]
    subprocess.run(check_cmd)
    
    # Remove the environment variable workaround
    env_file = model_path / ".env"
    if env_file.exists():
        print("\nRemoving TRANSFORMERS_ALLOW_UNSAFE_TORCH workaround...")
        env_file.unlink()
    
    # Update start.sh to remove the workaround
    start_sh = model_path / "start.sh"
    if start_sh.exists():
        content = start_sh.read_text()
        lines = [line for line in content.split('\n') 
                 if "TRANSFORMERS_ALLOW_UNSAFE_TORCH" not in line]
        start_sh.write_text('\n'.join(lines))
        print("✓ Removed workaround from start.sh")
    
    print("\n✓ Installation complete. The model should now work with proper security.")
    print("Note: CUDA 12.4 drivers are backward compatible with CUDA 12.1 applications.")
    
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python install_torch_cu124.py <model_path>")
        sys.exit(1)
    
    model_path = Path(sys.argv[1])
    if not install_torch_cu124(model_path):
        sys.exit(1)


if __name__ == "__main__":
    main()
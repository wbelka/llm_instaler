#!/usr/bin/env python3
"""Quick fix script for Janus model installation."""

import json
import subprocess
import sys
from pathlib import Path

def main():
    if len(sys.argv) > 1:
        model_path = Path(sys.argv[1])
    else:
        model_path = Path("/home/wblk/LLM/models/deepseek-ai_Janus-Pro-7B")
    
    if not model_path.exists():
        print(f"Error: Model path {model_path} does not exist")
        return 1
    
    # Update model_info.json
    model_info_path = model_path / "model_info.json"
    if model_info_path.exists():
        print(f"Updating {model_info_path}")
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        
        # Update model type and family
        model_info["model_type"] = "multi_modality"
        model_info["model_family"] = "multimodal"
        model_info["architecture_type"] = "vision-language"
        
        # Save updated info
        with open(model_info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print("✓ Updated model_info.json")
    
    # Install Janus package
    venv_pip = model_path / ".venv" / "bin" / "pip"
    if not venv_pip.exists():
        venv_pip = model_path / ".venv" / "Scripts" / "pip.exe"
    
    if venv_pip.exists():
        print("\nInstalling Janus package...")
        try:
            subprocess.run([
                str(venv_pip), "install", 
                "git+https://github.com/deepseek-ai/Janus.git"
            ], check=True)
            print("✓ Janus package installed")
        except subprocess.CalledProcessError as e:
            print(f"Error installing Janus: {e}")
            return 1
    else:
        print(f"Error: pip not found at {venv_pip}")
        return 1
    
    print("\nDone! The model should now work properly.")
    print("Try running ./start.sh again")
    return 0

if __name__ == "__main__":
    sys.exit(main())
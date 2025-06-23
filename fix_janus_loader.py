#!/usr/bin/env python3
"""Fix script to help load Deepseek Janus models properly."""

import json
import sys
from pathlib import Path


def create_janus_loader_wrapper(model_path: Path):
    """Create a custom loader wrapper for Janus models."""
    
    wrapper_content = '''#!/usr/bin/env python3
"""Custom loader for Deepseek Janus model."""

import sys
import os
import json
import torch
from pathlib import Path

# Add model directory to Python path
model_dir = Path(__file__).parent
sys.path.insert(0, str(model_dir))

def load_janus_model(model_path, device='auto', dtype='auto', **kwargs):
    """Load Janus model with proper configuration."""
    
    # First, try to import Janus
    try:
        from janus.models import MultiModalityCausalLM, VLChatProcessor
        print("✓ Janus package found")
    except ImportError:
        print("✗ Janus package not found. Installing...")
        import subprocess
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "git+https://github.com/deepseek-ai/Janus.git"
        ], check=True)
        
        # Try importing again
        from janus.models import MultiModalityCausalLM, VLChatProcessor
    
    # Load processor
    print(f"Loading processor from {model_path}")
    processor = VLChatProcessor.from_pretrained(model_path)
    
    # Determine device and dtype
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if dtype == 'auto':
        dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    else:
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16
        }
        dtype = dtype_map.get(dtype, torch.float32)
    
    # Load model
    print(f"Loading model from {model_path}")
    model = MultiModalityCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True
    )
    
    if device == 'cuda':
        model = model.cuda()
    
    model.eval()
    print("✓ Model loaded successfully")
    
    return model, processor


if __name__ == "__main__":
    # Test loading
    model_path = Path(__file__).parent / "model"
    try:
        model, processor = load_janus_model(str(model_path))
        print("Model loaded successfully!")
        print(f"Model type: {type(model)}")
        print(f"Processor type: {type(processor)}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
'''
    
    # Write the wrapper
    wrapper_path = model_path / "janus_loader.py"
    wrapper_path.write_text(wrapper_content)
    wrapper_path.chmod(0o755)
    
    print(f"✓ Created Janus loader wrapper at {wrapper_path}")
    
    # Also update model_loader.py to handle multi_modality type
    model_loader_path = model_path / "model_loader.py"
    if model_loader_path.exists():
        content = model_loader_path.read_text()
        
        # Add special handling for multi_modality type
        if "multi_modality" not in content:
            # Find the load_model function
            lines = content.split('\n')
            insert_idx = -1
            
            for i, line in enumerate(lines):
                if "def load_model(" in line:
                    # Find where to insert the check
                    for j in range(i, len(lines)):
                        if "model_info =" in lines[j] and "json.load" in lines[j]:
                            insert_idx = j + 3  # After loading model_info
                            break
                    break
            
            if insert_idx > 0:
                # Insert special handling
                special_handling = '''
    # Special handling for Janus models
    if model_info.get('model_type') == 'multi_modality':
        print("Detected Janus multi-modality model")
        try:
            # Import the custom loader
            from janus_loader import load_janus_model
            return load_janus_model(model_path, device=device, dtype=dtype, **kwargs)
        except ImportError:
            # Fallback: try installing Janus and loading directly
            print("Installing Janus package...")
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "git+https://github.com/deepseek-ai/Janus.git"], check=True)
            
            from janus.models import MultiModalityCausalLM, VLChatProcessor
            processor = VLChatProcessor.from_pretrained(model_path)
            model = MultiModalityCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )
            if device == 'cuda' and torch.cuda.is_available():
                model = model.cuda()
            return model, processor
'''
                lines.insert(insert_idx, special_handling)
                
                # Write back
                model_loader_path.write_text('\n'.join(lines))
                print("✓ Updated model_loader.py with Janus handling")


def main():
    """Main function."""
    if len(sys.argv) > 1:
        model_path = Path(sys.argv[1])
    else:
        print("Usage: python fix_janus_loader.py <model_path>")
        print("Example: python fix_janus_loader.py /home/wblk/LLM/models/deepseek-ai_Janus-Pro-7B")
        sys.exit(1)
    
    if not model_path.exists():
        print(f"Error: Model path {model_path} does not exist")
        sys.exit(1)
    
    # Check if it's a Janus model
    config_path = model_path / "model" / "config.json"
    if not config_path.exists():
        config_path = model_path / "config.json"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if config.get('model_type') != 'multi_modality':
            print("This doesn't appear to be a Janus model (model_type != 'multi_modality')")
            sys.exit(1)
    
    # Create the wrapper
    create_janus_loader_wrapper(model_path)
    
    print("\nNext steps:")
    print("1. cd " + str(model_path))
    print("2. source .venv/bin/activate")
    print("3. pip install git+https://github.com/deepseek-ai/Janus.git")
    print("4. ./start.sh")


if __name__ == "__main__":
    main()
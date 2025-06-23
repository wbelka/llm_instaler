#!/usr/bin/env python3
"""Patch the model_loader.py in an installed Janus model to properly load it."""

import sys
from pathlib import Path


def patch_model_loader(model_path: Path):
    """Patch the model_loader.py to handle Janus models correctly."""
    
    model_loader_path = model_path / "model_loader.py"
    
    if not model_loader_path.exists():
        print(f"Error: model_loader.py not found at {model_loader_path}")
        return False
    
    # Read the current content
    content = model_loader_path.read_text()
    
    # Find where to insert the Janus handling
    lines = content.split('\n')
    
    # Find the load_model function
    insert_idx = -1
    for i, line in enumerate(lines):
        if "def load_model(" in line:
            # Find where model_info is loaded
            for j in range(i, min(i + 50, len(lines))):
                if "json.load(f)" in lines[j] and "model_info" in lines[j]:
                    insert_idx = j + 1
                    break
            break
    
    if insert_idx == -1:
        print("Error: Could not find where to insert Janus handling")
        return False
    
    # Insert the Janus handling code
    janus_handling = '''
    # Special handling for Janus multi_modality models
    if model_info.get('model_type') == 'multi_modality' or 'janus' in model_id.lower():
        logger.info("Detected Janus multi-modality model, using special loader")
        try:
            from janus.models import MultiModalityCausalLM, VLChatProcessor
            
            # Load processor
            processor = VLChatProcessor.from_pretrained(model_path)
            
            # Determine torch dtype
            if dtype == 'auto':
                import torch
                torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            else:
                import torch
                dtype_map = {
                    'float32': torch.float32,
                    'float16': torch.float16,
                    'bfloat16': torch.bfloat16
                }
                torch_dtype = dtype_map.get(dtype, torch.float32)
            
            # Load model
            model = MultiModalityCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True
            )
            
            # Move to device if needed
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device == 'cuda' and torch.cuda.is_available():
                model = model.cuda()
            
            model.eval()
            logger.info("Successfully loaded Janus model")
            return model, processor
            
        except ImportError as e:
            logger.error(f"Failed to import Janus: {e}")
            logger.error("Please ensure Janus is installed: pip install git+https://github.com/deepseek-ai/Janus.git")
            raise
        except Exception as e:
            logger.error(f"Failed to load Janus model: {e}")
            raise
'''
    
    # Insert the code
    lines.insert(insert_idx, janus_handling)
    
    # Write back
    model_loader_path.write_text('\n'.join(lines))
    print(f"✓ Patched model_loader.py at {model_loader_path}")
    
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python patch_janus_loader.py <model_path>")
        sys.exit(1)
    
    model_path = Path(sys.argv[1])
    if not model_path.exists():
        print(f"Error: Model path {model_path} does not exist")
        sys.exit(1)
    
    if patch_model_loader(model_path):
        print("\nModel loader patched successfully!")
        print("Try running ./start.sh again")
    else:
        print("\nFailed to patch model loader")
        sys.exit(1)


if __name__ == "__main__":
    main()
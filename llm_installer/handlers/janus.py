"""
Handler for Janus multimodal models with visual processors
"""

from typing import List, Dict, Any
from .base import BaseHandler
import logging

logger = logging.getLogger(__name__)


class JanusHandler(BaseHandler):
    """Handler for Janus multimodal models"""
    
    @property
    def name(self) -> str:
        return "JanusHandler"
    
    def can_handle(self, model_info: Dict[str, Any]) -> bool:
        """Check if this is a Janus model"""
        model_id = model_info.get('model_id', '').lower()
        
        # Check model ID
        if 'janus' in model_id:
            return True
        
        # Check config
        config = model_info.get('config', {})
        model_type = config.get('model_type', '').lower()
        if model_type == 'janus':
            return True
        
        # Check architectures
        architectures = config.get('architectures', [])
        if any('janus' in arch.lower() for arch in architectures):
            return True
        
        # Check for visual processor indicators
        if all(key in config for key in ['vision_config', 'text_config']):
            if 'janus' in str(config).lower():
                return True
        
        return False
    
    def get_dependencies(self, model_info: Dict[str, Any]) -> List[str]:
        """Get multimodal-specific dependencies"""
        deps = [
            "torch>=2.0.0",
            "transformers>=4.35.0",
            "accelerate>=0.25.0",
            "safetensors>=0.3.1",
            "pillow>=10.0.0",  # For image processing
            "torchvision>=0.15.0",  # For vision transforms
            "opencv-python>=4.8.0",  # For advanced image operations
        ]
        
        # Vision-specific requirements
        config = model_info.get('config', {})
        vision_config = config.get('vision_config', {})
        
        # Check for specific vision model requirements
        if vision_config.get('model_type') == 'clip_vision_model':
            deps.append("clip>=1.0")
        
        # Janus-specific kernels if mentioned
        if 'janus-kernels' in str(config).lower():
            deps.append("janus-kernels>=0.1.0")
        
        # Video processing support
        if 'video' in model_info.get('tags', []):
            deps.extend([
                "av>=10.0.0",  # PyAV for video processing
                "decord>=0.6.0",  # Efficient video reader
            ])
        
        # Flash attention for efficiency
        deps.append("flash-attn>=2.0.0")
        
        return deps
    
    def get_environment_vars(self, model_info: Dict[str, Any]) -> Dict[str, str]:
        """Get multimodal-specific environment variables"""
        env_vars = {
            # Cache settings
            'TRANSFORMERS_CACHE': './cache',
            'HF_HOME': './cache',
            'TORCH_HOME': './cache/torch',
            
            # Vision model settings
            'VISION_CACHE_DIR': './cache/vision',
            
            # OpenCV settings
            'OPENCV_IO_MAX_IMAGE_PIXELS': str(2**31 - 1),  # Allow large images
            
            # Memory settings for multimodal
            'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:256',
        }
        
        return env_vars
    
    def get_optimization_options(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get multimodal-specific optimization options"""
        options = {
            'torch_dtype': 'float16',  # FP16 for efficiency
            'device_map': 'auto',      # Automatic device mapping
            'low_cpu_mem_usage': True,
            'use_flash_attention_2': True,
        }
        
        # Check model size for additional optimizations
        if model_info.get('size_gb', 0) > 10:
            options['load_in_8bit'] = True
            options['bnb_4bit_compute_dtype'] = 'float16'
        
        # Vision-specific settings
        config = model_info.get('config', {})
        if 'vision_config' in config:
            options['vision_feature_layer'] = -2  # Use second-to-last layer features
            options['vision_feature_select_strategy'] = 'default'
        
        return options
    
    def post_install_setup(self, install_path: str, model_info: Dict[str, Any]) -> None:
        """Create multimodal test scripts"""
        import os
        
        # Create image inference test script
        test_script = f"""#!/bin/bash
# Test multimodal inference with image

cd "$(dirname "$0")"
source .venv/bin/activate

echo "Testing multimodal inference..."
python -c "
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import requests

# Load model and processor
processor = AutoProcessor.from_pretrained('./model')
model = AutoModelForVision2Seq.from_pretrained('./model', device_map='auto')

# Load a test image
url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# Prepare prompt
prompt = 'Describe this image:'
inputs = processor(images=image, text=prompt, return_tensors='pt')

# Generate
outputs = model.generate(**inputs, max_new_tokens=100)
result = processor.decode(outputs[0], skip_special_tokens=True)
print(f'Result: {{result}}')
"

echo "Test complete!"
"""
        
        script_path = os.path.join(install_path, 'scripts', 'test_multimodal.sh')
        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        
        with open(script_path, 'w') as f:
            f.write(test_script)
        
        os.chmod(script_path, 0o755)
        logger.info(f"Created multimodal test script: {script_path}")
        
        # Create a README for multimodal usage
        readme_content = f"""# Multimodal Model Usage

This is a multimodal model that can process both text and images.

## Quick Start

1. Activate the environment:
   ```bash
   source .venv/bin/activate
   ```

2. Test multimodal inference:
   ```bash
   ./scripts/test_multimodal.sh
   ```

3. Run the API server:
   ```bash
   ./start.sh
   ```

## API Usage

Send requests with both text and image:
```python
import requests
import base64

# Encode image
with open('image.jpg', 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode()

# Send request
response = requests.post('http://localhost:8000/api/generate', json={{
    'prompt': 'Describe this image:',
    'image': image_base64
}})
```

## Model Information
- Model ID: {model_info.get('model_id')}
- Type: Multimodal (Vision + Language)
- Size: {model_info.get('size_gb')} GB
"""
        
        readme_path = os.path.join(install_path, 'MULTIMODAL_README.md')
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        logger.info(f"Created multimodal README: {readme_path}")
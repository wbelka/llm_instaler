# Handler Development Guide

## Overview

The LLM Installer uses a handler system to abstract model-specific logic. Each model type (transformer, diffusion, multimodal, etc.) has its own handler that implements a common interface.

## Architecture

```
handlers/
├── base.py           # Abstract base class defining the interface
├── registry.py       # Handler registration and discovery
├── transformer.py    # For text generation models (GPT, LLaMA, etc.)
├── diffusion.py      # For image/video generation (Stable Diffusion, etc.)
├── multimodal.py     # For vision-language models
├── janus.py          # Specific handler for Janus models
├── audio.py          # For audio models (Whisper, TTS, etc.)
├── vision.py         # For computer vision models
├── embedding.py      # For embedding models
└── specialized.py    # For other specialized models
```

## Creating a New Handler

### Step 1: Create Handler File

Create a new file in `handlers/` directory:

```python
"""Handler for [Model Type] models.

This module provides specific handling for [description].
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

from handlers.base import BaseHandler

logger = logging.getLogger(__name__)


class MyModelHandler(BaseHandler):
    """Handler for [Model Type] models."""
    
    def __init__(self, model_info: Dict[str, Any]):
        """Initialize handler."""
        super().__init__(model_info)
        # Add any model-specific initialization
```

### Step 2: Implement Required Abstract Methods

All handlers MUST implement these methods:

```python
def get_dependencies(self) -> List[str]:
    """Get Python dependencies."""
    return [
        'torch>=2.0.0',
        'transformers>=4.30.0',
        # Add model-specific dependencies
    ]

def get_system_dependencies(self) -> List[str]:
    """Get system dependencies."""
    return [
        'cuda>=11.7',  # If GPU required
        # Add system-level dependencies
    ]

def analyze(self) -> 'ModelRequirements':
    """Analyze model and return requirements."""
    from core.checker import ModelRequirements
    
    requirements = ModelRequirements()
    requirements.model_type = self.model_type
    requirements.model_family = self.model_family
    requirements.primary_library = "transformers"  # or diffusers, etc.
    requirements.base_dependencies = self.get_dependencies()
    requirements.special_dependencies = []  # e.g., ['flash-attn']
    requirements.optional_dependencies = []  # e.g., ['xformers']
    
    # Memory requirements
    model_size_gb = self._estimate_model_size()
    requirements.memory_requirements = {
        "min": model_size_gb * 2,
        "recommended": model_size_gb * 3,
        "gpu_min": model_size_gb * 1.5,
        "gpu_recommended": model_size_gb * 2
    }
    
    return requirements

def load_model(self, model_path: str, **kwargs):
    """Load model with optimal parameters."""
    # Implementation depends on model type
    pass

def get_inference_params(self) -> Dict[str, Any]:
    """Get default inference parameters."""
    return {
        'temperature': 0.7,
        'max_tokens': 512,
        # Add model-specific parameters
    }

def get_training_params(self) -> Dict[str, Any]:
    """Get default training parameters."""
    return {
        'learning_rate': 2e-5,
        'batch_size': 4,
        # Add training parameters
    }

def validate_model_files(self, model_path: str) -> Tuple[bool, Optional[str]]:
    """Validate model files exist."""
    from pathlib import Path
    
    model_path = Path(model_path)
    required_files = ['config.json']  # Add required files
    
    for file in required_files:
        if not (model_path / file).exists():
            return False, f"Missing required file: {file}"
    
    return True, None
```

### Step 3: Implement Generation/Processing Methods

Choose which methods to implement based on model capabilities:

#### For Text Generation:
```python
def generate_text(self, prompt: str = None, messages: List[Dict] = None,
                 model=None, tokenizer=None, **kwargs) -> Dict[str, Any]:
    """Generate text response."""
    # Implement text generation logic
    return {
        'text': generated_text,
        'usage': {
            'prompt_tokens': input_tokens,
            'completion_tokens': output_tokens,
            'total_tokens': total_tokens
        }
    }
```

#### For Streaming Text Generation:
```python
async def generate_stream(self, prompt: str = None, messages: List[Dict] = None,
                         model=None, tokenizer=None, **kwargs) -> Dict[str, Any]:
    """Generate text response as a stream of events."""
    # Must be an async generator
    # Yields dictionaries with a specific format
    
    # 1. YIELD 'text' chunks for each token
    # for token in streaming_logic:
    #     yield {
    #         "type": "text",
    #         "token": token
    #     }
    
    # 2. YIELD 'done' message at the end with usage stats
    # yield {
    #     "type": "done",
    #     "full_text": final_text,
    #     "usage": {
    #         "prompt_tokens": ...,
    #         "completion_tokens": ...,
    #         "total_tokens": ...
    #     }
    # }
```

#### For Image Generation:
```python
def generate_image(self, prompt: str, negative_prompt: str = None,
                  model=None, **kwargs) -> Dict[str, Any]:
    """Generate image from text."""
    # Implement image generation logic
    return {
        'image': base64_image_data,
        'metadata': {
            'width': width,
            'height': height,
            'format': 'png',
            'steps': num_inference_steps
        }
    }
```

#### For Multimodal Processing:
```python
def process_multimodal(self, text: str = None, images: List[str] = None,
                      model=None, processor=None, **kwargs) -> Dict[str, Any]:
    """Process multimodal inputs."""
    # Implement multimodal logic
    return {
        'text': response_text,
        'type': 'multimodal_response'
    }
```

### Step 4: Add Mode Support

```python
def get_supported_modes(self) -> List[str]:
    """Get supported generation modes."""
    return ['auto', 'chat', 'complete', 'instruct']

def get_mode_descriptions(self) -> Dict[str, str]:
    """Get mode descriptions."""
    return {
        'auto': 'Automatic mode selection',
        'chat': 'Conversational mode',
        'complete': 'Text completion mode',
        'instruct': 'Instruction following mode'
    }
```

### Step 5: Add Special Installation Instructions

```python
def get_installation_notes(self) -> Dict[str, str]:
    """Get special installation instructions."""
    return {
        'special-package': """
Special instructions for installing this package...
""",
        'git+https://github.com/...': """
Instructions for git-based dependencies...
"""
    }
```

### Step 6: Register Handler

Add to `handlers/registry.py`:

```python
# In HANDLER_MAPPING
'my_model_type': MyModelHandler,

# Or in get_handler_for_model() for complex logic
if 'special_model' in model_id:
    return MyModelHandler
```

## Best Practices

### 1. Memory Management
```python
# Clear CUDA cache when needed
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Use context managers
with torch.no_grad():
    # Inference code
```

### 2. Error Handling
```python
try:
    # Model operations
except ImportError as e:
    logger.error(f"Missing dependency: {e}")
    raise ValueError(f"Please install required packages: {e}")
except Exception as e:
    logger.error(f"Generation failed: {e}")
    # Clean up resources
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    raise
```

### 3. Input Validation
```python
def validate_inputs(self, inputs: Dict[str, Any], task: str = None) -> Tuple[bool, Optional[str]]:
    """Validate inputs before processing."""
    if task == 'image' and 'prompt' not in inputs:
        return False, "Prompt required for image generation"
    return True, None
```

### 4. Capability Declaration
```python
def get_model_capabilities(self) -> Dict[str, Any]:
    """Declare what the model can do."""
    return {
        'supports_streaming': True,
        'supports_system_prompt': True,
        'supports_multimodal': False,
        'max_context_length': 4096,
        'input_modalities': ['text'],
        'output_modalities': ['text'],
        'supported_modes': self.get_supported_modes()
    }
```

## Testing Your Handler

1. **Unit Test** (create `tests/test_my_handler.py`):
```python
def test_my_handler_dependencies():
    handler = MyModelHandler({'model_id': 'test/model'})
    deps = handler.get_dependencies()
    assert 'torch' in str(deps)
```

2. **Integration Test**:
```bash
# Install a model using your handler
./llm-installer install test/model

# Test the model
cd /path/to/installed/model
./start.sh
```

3. **API Test**:
```python
# Test through API
response = requests.post('http://localhost:8000/generate', json={
    'prompt': 'Test prompt',
    'mode': 'your_mode'
})
```

## Common Patterns

### 1. Model Size Estimation
```python
def _estimate_model_size(self) -> float:
    """Estimate model size in GB."""
    if 'model_size_gb' in self.model_info:
        return self.model_info['model_size_gb']
    
    # Pattern matching
    model_id_lower = self.model_id.lower()
    if '7b' in model_id_lower:
        return 14.0
    elif '13b' in model_id_lower:
        return 26.0
    
    return 10.0  # Default
```

### 2. Device Selection
```python
def get_optimal_device(self, available_devices: List[str]) -> str:
    """Select best device."""
    for device in available_devices:
        if 'cuda' in device:
            return device
    return 'cpu'
```

### 3. Quantization Support
```python
if load_in_8bit or load_in_4bit:
    from transformers import BitsAndBytesConfig
    model_kwargs['quantization_config'] = BitsAndBytesConfig(
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit
    )
```

## Debugging Tips

1. **Enable Debug Logging**:
```python
logger.debug(f"Loading model from {model_path}")
logger.debug(f"Model config: {model_config}")
```

2. **Add Checkpoints**:
```python
logger.info("Step 1: Loading tokenizer...")
# Code
logger.info("Step 2: Loading model weights...")
# Code
```

3. **Resource Monitoring**:
```python
import psutil
import GPUtil

# Log memory usage
logger.info(f"CPU Memory: {psutil.virtual_memory().percent}%")
if torch.cuda.is_available():
    logger.info(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

## Handler Checklist

- [ ] Inherits from BaseHandler
- [ ] Implements all abstract methods
- [ ] Has proper error handling
- [ ] Includes logging
- [ ] Handles CUDA memory properly
- [ ] Returns standardized responses
- [ ] Has installation notes for special dependencies
- [ ] Registered in registry.py
- [ ] Tested with actual model
- [ ] Documentation in docstrings
- [ ] Supports relevant generation methods
- [ ] Declares capabilities correctly

## Example: Complete Minimal Handler

```python
"""Handler for MyModel models."""

import logging
from typing import List, Dict, Any, Optional, Tuple

from handlers.base import BaseHandler

logger = logging.getLogger(__name__)


class MyModelHandler(BaseHandler):
    """Handler for MyModel."""
    
    def get_dependencies(self) -> List[str]:
        return ['torch>=2.0.0', 'transformers>=4.30.0']
    
    def get_system_dependencies(self) -> List[str]:
        return ['cuda>=11.7'] if self.model_info.get('requires_gpu', True) else []
    
    def analyze(self) -> 'ModelRequirements':
        from core.checker import ModelRequirements
        
        requirements = ModelRequirements()
        requirements.model_type = self.model_type
        requirements.model_family = self.model_family
        requirements.primary_library = "transformers"
        requirements.base_dependencies = self.get_dependencies()
        
        model_size_gb = 10.0  # Estimate
        requirements.memory_requirements = {
            "min": 16,
            "recommended": 24,
            "gpu_min": 12,
            "gpu_recommended": 16
        }
        
        return requirements
    
    def load_model(self, model_path: str, **kwargs):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            **kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        return model, tokenizer
    
    def get_inference_params(self) -> Dict[str, Any]:
        return {'temperature': 0.7, 'max_tokens': 512}
    
    def get_training_params(self) -> Dict[str, Any]:
        return {'learning_rate': 2e-5, 'batch_size': 4}
    
    def validate_model_files(self, model_path: str) -> Tuple[bool, Optional[str]]:
        from pathlib import Path
        
        if not (Path(model_path) / 'config.json').exists():
            return False, "Missing config.json"
        
        return True, None
    
    def generate_text(self, prompt: str = None, messages: List[Dict] = None,
                     model=None, tokenizer=None, **kwargs) -> Dict[str, Any]:
        # Basic implementation
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, **kwargs)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {'text': text}
```

This guide provides a complete framework for creating new handlers. Follow these patterns to ensure consistency across the codebase.
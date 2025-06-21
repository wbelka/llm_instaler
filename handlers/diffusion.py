"""Handler for diffusion models (Stable Diffusion, DALL-E, etc.).

This handler manages image and video generation models that use
diffusion techniques.
"""

from typing import Dict, Any, List
from handlers.base import BaseHandler, ModelRequirements


class DiffusionHandler(BaseHandler):
    """Handler for diffusion-based generative models."""
    
    def __init__(self, model_info: Dict[str, Any]):
        """Initialize the diffusion handler.
        
        Args:
            model_info: Model information from HuggingFace.
        """
        super().__init__(model_info)
    
    def analyze(self) -> ModelRequirements:
        """Analyze diffusion model requirements.
        
        Returns:
            ModelRequirements object with analyzed data.
        """
        requirements = ModelRequirements()
        
        # Set model type and family
        requirements.model_type = "diffusion"
        requirements.model_family = "image-generation"
        requirements.primary_library = "diffusers"
        
        # Check if it's a video model
        model_id = self.model_info.get('model_id', '').lower()
        if 'video' in model_id or 'text2video' in model_id:
            requirements.model_family = "video-generation"
        
        # Determine architecture
        config = self.model_info.get('config', {})
        if '_class_name' in config:
            requirements.architecture_type = config['_class_name']
        else:
            requirements.architecture_type = "diffusion"
        
        # Base dependencies
        requirements.base_dependencies = [
            "torch",
            "torchvision",
            "diffusers",
            "transformers",  # Many diffusion models need this
            "accelerate",
            "pillow",
            "numpy",
            "safetensors"
        ]
        
        # Check for specific model requirements
        if 'controlnet' in model_id:
            requirements.base_dependencies.append("opencv-python")
            requirements.capabilities["controlnet"] = True
        
        if 'sdxl' in model_id:
            requirements.capabilities["sdxl"] = True
            requirements.special_dependencies.append("invisible-watermark")
        
        # Optional dependencies
        requirements.optional_dependencies = [
            "xformers",  # For memory efficient attention
            "triton",    # For some optimizations
            "compel",    # For better prompt handling
        ]
        
        # Memory requirements
        model_size = self._estimate_model_size()
        requirements.disk_space_gb = model_size
        
        # Estimate memory requirements based on model size
        requirements.memory_requirements = {
            "min": model_size * 0.3,    # Minimum for int4
            "recommended": model_size * 1.2,  # For float16
            "optimal": model_size * 2.4,     # For float32
        }
        
        # Training memory requirements
        requirements.memory_requirements["training"] = {
            "lora": model_size * 1.3,
            "full": model_size * 4,
        }
        
        # Capabilities
        requirements.capabilities.update({
            "batch_inference": True,
            "streaming": False,
            "quantization": ["int8", "int4"],
            "lora": True,
            "native_dtype": "float16"
        })
        
        # Special configurations
        requirements.special_config = {
            "supports_fp16": True,
            "supports_bf16": True,
            "supports_cpu_offload": True,
            "supports_sequential_cpu_offload": True,
            "requires_safety_checker": True,
        }
        
        return requirements
    
    def _estimate_model_size(self) -> float:
        """Estimate model size in GB.
        
        Returns:
            Estimated size in GB.
        """
        # Check if we have size information
        if 'model_size' in self.model_info:
            return self.model_info['model_size']
        
        # Otherwise estimate from files
        total_size = 0
        siblings = self.model_info.get('siblings', [])
        
        for file_info in siblings:
            if isinstance(file_info, dict) and 'size' in file_info:
                # Check if it's a model weight file
                filename = file_info.get('filename', '')
                if any(filename.endswith(ext) for ext in ['.bin', '.safetensors', '.ckpt', '.pt']):
                    total_size += file_info['size']
        
        # Convert to GB
        size_gb = total_size / (1024**3)
        
        # If no size found, use defaults based on model type
        if size_gb == 0:
            model_id = self.model_info.get('model_id', '').lower()
            if 'sdxl' in model_id:
                size_gb = 6.5
            elif 'sd-1.5' in model_id or 'sd-v1-5' in model_id:
                size_gb = 4.0
            elif 'sd-2' in model_id:
                size_gb = 5.0
            else:
                size_gb = 5.0  # Default for unknown diffusion models
        
        return size_gb
"""Handler for diffusion models (Stable Diffusion, DALL-E, etc.).

This handler manages image and video generation models that use
diffusion techniques.
"""

from typing import Dict, Any, List, Tuple, Optional
from handlers.base import BaseHandler
from core.checker import ModelRequirements


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
        config = self.model_info.get('config', {})
        
        # Check various indicators for video models
        if any(indicator in model_id for indicator in ['video', 'text2video', 'text-to-video', 't2v']):
            requirements.model_family = "video-generation"
        elif '_class_name' in config and 'video' in config['_class_name'].lower():
            requirements.model_family = "video-generation"
        
        # Determine architecture
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
    
    def get_dependencies(self) -> List[str]:
        """Get Python dependencies for diffusion models.
        
        Returns:
            List of required pip packages.
        """
        return self.analyze().base_dependencies
    
    def get_system_dependencies(self) -> List[str]:
        """Get system-level dependencies.
        
        Returns:
            List of system packages or requirements.
        """
        # Most diffusion models need CUDA
        return ["cuda>=11.0"]
    
    def get_inference_params(self) -> Dict[str, Any]:
        """Get optimal inference parameters.
        
        Returns:
            Dictionary of inference parameters.
        """
        return {
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "negative_prompt": "",
            "width": 512,
            "height": 512,
            "seed": None,
        }
    
    def get_training_params(self) -> Dict[str, Any]:
        """Get optimal training parameters.
        
        Returns:
            Dictionary of training parameters.
        """
        return {
            "learning_rate": 1e-4,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "num_train_epochs": 100,
            "checkpointing_steps": 500,
            "validation_epochs": 10,
            "enable_xformers_memory_efficient_attention": True,
        }
    
    def load_model(self, model_path: str, **kwargs):
        """Load diffusion model.
        
        Args:
            model_path: Path to the model directory.
            **kwargs: Additional loading parameters.
            
        Returns:
            Loaded model (pipeline) and tokenizer/processor (None for diffusion).
        """
        from model_loader import _load_diffusers_model
        return _load_diffusers_model(self.model_info, model_path, **kwargs)
    
    def validate_model_files(self, model_path: str) -> Tuple[bool, Optional[str]]:
        """Validate model files.
        
        Args:
            model_path: Path to the model directory.
            
        Returns:
            Tuple of (is_valid, error_message).
        """
        from pathlib import Path
        model_dir = Path(model_path)
        
        # Check for model_index.json
        if not (model_dir / "model_index.json").exists():
            return False, "Missing model_index.json file"
        
        # Check for essential components
        required_components = ["unet", "vae", "text_encoder", "tokenizer", "scheduler"]
        missing = []
        
        for component in required_components:
            component_path = model_dir / component
            if not component_path.exists():
                # Some models might have it in config only
                continue
                
        if missing:
            return False, f"Missing components: {', '.join(missing)}"
            
        return True, None
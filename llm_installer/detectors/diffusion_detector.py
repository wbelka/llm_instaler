"""
Detector for diffusion models (Stable Diffusion, etc.)
"""

from typing import Dict, List, Optional, Any
from .base import BaseDetector, ModelProfile


class DiffusionDetector(BaseDetector):
    """Detector for diffusion-based image generation models"""

    def detect(self, model_id: str, config: Dict[str, Any],
               files: List[str]) -> Optional[ModelProfile]:
        """
        Detect diffusion models like Stable Diffusion
        """
        # Check for model_index.json (diffusers indicator)
        if '_diffusers_version' in config:
            return self._create_diffusion_profile(model_id, config, files)

        # Check for diffusion-specific files
        diffusion_files = [
            'model_index.json',
            'unet/config.json',
            'vae/config.json',
            'scheduler/scheduler_config.json'
        ]

        if any(df in files for df in diffusion_files):
            return self._create_diffusion_profile(model_id, config, files)

        # Check config for diffusion indicators
        if self._is_diffusion_config(config):
            return self._create_diffusion_profile(model_id, config, files)

        return None

    def _is_diffusion_config(self, config: Dict[str, Any]) -> bool:
        """Check if config indicates a diffusion model"""
        # Check for diffusion-specific fields
        diffusion_indicators = [
            '_class',
            'diffusion',
            'unet',
            'vae',
            'scheduler',
            'tokenizer',
            'text_encoder'
        ]

        for indicator in diffusion_indicators:
            if indicator in config:
                value = str(config[indicator]).lower()
                if 'diffusion' in value or 'unet' in value or 'vae' in value:
                    return True

        # Check if it's a pipeline config
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, dict) and '_class' in value:
                    class_name = value['_class'].lower()
                    if 'diffusion' in class_name or 'pipeline' in class_name:
                        return True

        return False

    def _create_diffusion_profile(self, model_id: str, config: Dict[str, Any],
                                  files: List[str]) -> ModelProfile:
        """Create a profile for diffusion model"""
        profile = ModelProfile(
            model_type="diffusion",
            model_id=model_id,
            library="diffusers",
            task="text-to-image",
            is_multimodal=True  # Text + Image
        )

        # Detect specific diffusion model type
        if 'stable-diffusion' in model_id.lower() or 'sd' in model_id.lower():
            profile.architecture = "StableDiffusion"

            # Check SD version
            if 'xl' in model_id.lower() or 'sdxl' in model_id.lower():
                profile.architecture = "StableDiffusionXL"
                profile.estimated_size_gb = 6.5
                profile.estimated_memory_gb = 16.0
            elif '2' in model_id:
                profile.architecture = "StableDiffusion2"
                profile.estimated_size_gb = 5.0
                profile.estimated_memory_gb = 12.0
            else:
                profile.architecture = "StableDiffusion1"
                profile.estimated_size_gb = 4.0
                profile.estimated_memory_gb = 10.0

        # Check for ControlNet
        if 'controlnet' in model_id.lower():
            profile.architecture = "ControlNet"
            profile.task = "image-to-image"

        # Check for inpainting models
        if 'inpaint' in model_id.lower():
            profile.task = "image-inpainting"

        # Check for depth models
        if 'depth' in model_id.lower():
            profile.task = "depth-estimation"

        # Set requirements
        profile.special_requirements = [
            'torch',
            'diffusers',
            'transformers',
            'accelerate',
            'xformers',  # For memory efficient attention
            'opencv-python',
            'Pillow'
        ]

        # Check for specific optimizations
        if any('onnx' in f for f in files):
            profile.special_requirements.append('onnxruntime')

        if any('safetensors' in f for f in files):
            profile.special_requirements.append('safetensors')

        # Most diffusion models support TensorRT optimization
        profile.supports_tensorrt = True

        return profile

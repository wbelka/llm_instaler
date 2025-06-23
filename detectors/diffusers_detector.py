"""Detector for diffusers-based models.

This detector identifies models that use the diffusers library,
typically for image/video generation tasks.
"""

import logging
from typing import Dict, Any
from pathlib import Path

from detectors.base import BaseDetector

logger = logging.getLogger(__name__)


class DiffusersDetector(BaseDetector):
    """Detector for diffusers models."""
    
    @property
    def name(self) -> str:
        """Get detector name."""
        return "DiffusersDetector"
    
    @property
    def priority(self) -> int:
        """Get detector priority."""
        # High priority for diffusers models
        return 90
    
    def matches(self, model_info: Dict[str, Any]) -> bool:
        """Check if this is a diffusers model.
        
        Args:
            model_info: Model metadata
            
        Returns:
            True if this is a diffusers model
        """
        # Check for model_index.json in config_data
        config_data = model_info.get('config_data', {})
        if 'model_index.json' in config_data:
            return True
        
        # Check for diffusers-specific files
        files = model_info.get('files', [])
        file_names = [f.get('path', '') for f in files if isinstance(f, dict)]
        
        # Look for diffusers indicators
        diffusers_indicators = [
            'model_index.json',
            'unet/config.json',
            'vae/config.json',
            'scheduler/scheduler_config.json',
            'text_encoder/config.json'
        ]
        
        for indicator in diffusers_indicators:
            if any(indicator in fname for fname in file_names):
                return True
        
        # Check library name
        if model_info.get('library_name') == 'diffusers':
            return True
        
        return False
    
    def analyze(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze diffusers model.
        
        Args:
            model_info: Model metadata
            
        Returns:
            Analysis results
        """
        config_data = model_info.get('config_data', {})
        model_index = config_data.get('model_index.json', {})
        
        # Determine model type from pipeline class
        pipeline_class = model_index.get('_class_name', '')
        
        if 'Video' in pipeline_class:
            model_family = 'video-generation'
            model_type = 'text-to-video'
        elif 'Audio' in pipeline_class:
            model_family = 'audio-generation'
            model_type = 'audio-diffusion'
        elif 'Inpaint' in pipeline_class:
            model_family = 'image-generation'
            model_type = 'inpainting'
        elif 'ControlNet' in pipeline_class:
            model_family = 'image-generation'
            model_type = 'controlnet'
        else:
            model_family = 'image-generation'
            model_type = 'text-to-image'
        
        # Get components
        components = {}
        for key, value in model_index.items():
            if isinstance(value, list) and len(value) == 2:
                components[key] = value[0]  # Component class name
        
        analysis = {
            'model_type': model_type,
            'model_family': model_family,
            'architecture_type': pipeline_class,
            'primary_library': 'diffusers',
            'pipeline_class': pipeline_class,
            'components': components,
            'trust_remote_code': model_index.get('_custom_pipeline', None) is not None,
        }
        
        # Extract scheduler info
        if 'scheduler' in components:
            analysis['scheduler_type'] = components['scheduler']
        
        # Check for VAE
        if 'vae' in components:
            analysis['has_vae'] = True
        
        # Check for safety checker
        if 'safety_checker' in model_index:
            analysis['has_safety_checker'] = True
        
        # Capabilities
        analysis['capabilities'] = {
            'supports_streaming': False,
            'supports_batch': True,
            'supports_img2img': 'img2img' in pipeline_class.lower(),
            'supports_inpainting': 'inpaint' in pipeline_class.lower(),
            'supports_controlnet': 'controlnet' in pipeline_class.lower(),
            'is_video_model': model_family == 'video-generation',
            'native_dtype': 'float16',  # Most diffusers models use fp16
        }
        
        return analysis
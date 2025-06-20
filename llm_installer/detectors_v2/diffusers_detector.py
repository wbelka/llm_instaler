"""
Detector for diffusers library models
"""

import logging
from typing import Dict, Any
from .base import BaseDetector, ModelInfo
from .diffusers_utils import (
    estimate_component_parameters,
    parameters_to_size_gb
)

logger = logging.getLogger(__name__)


class DiffusersDetector(BaseDetector):
    """Detector for diffusion models using diffusers library"""

    def can_handle(self, info: ModelInfo) -> bool:
        """Check if this is a diffusers model"""
        # Explicit library
        if info.library_name == 'diffusers':
            return True

        # Has diffusers tag
        if 'diffusers' in info.tags:
            return True

        # Has model_index.json (diffusers specific)
        if 'model_index.json' in info.files:
            return True

        # Pipeline tags for image generation
        if info.pipeline_tag in ['text-to-image', 'image-to-image',
                                  'inpainting']:
            return True

        return False

    def detect(self, info: ModelInfo) -> ModelInfo:
        """Detect diffusers-specific information"""
        info.model_type = 'diffusion'
        info.is_multimodal = True  # Text + Image

        # Detect architecture from model_index
        if info.model_index and '_class_name' in info.model_index:
            pipeline_class = info.model_index['_class_name']
            info.architecture = pipeline_class

            # Detect specific type
            if 'StableDiffusion' in pipeline_class:
                if 'XL' in pipeline_class:
                    info.metadata['variant'] = 'SDXL'
                elif '3' in pipeline_class:
                    info.metadata['variant'] = 'SD3'
                else:
                    info.metadata['variant'] = 'SD'
                    # Check for optimized variants
                    model_lower = info.model_id.lower()
                    if 'turbo' in model_lower:
                        info.metadata['variant'] = 'SD-Turbo'
                        info.metadata['optimized'] = True
                    elif 'lcm' in model_lower:
                        info.metadata['variant'] = 'SD-LCM'
                        info.metadata['optimized'] = True
            elif 'Flux' in pipeline_class:
                info.metadata['variant'] = 'FLUX'

        # Task detection
        if info.pipeline_tag:
            info.task = info.pipeline_tag
        else:
            # Infer from architecture
            if 'inpaint' in str(info.architecture).lower():
                info.task = 'inpainting'
            elif 'img2img' in str(info.architecture).lower():
                info.task = 'image-to-image'
            else:
                info.task = 'text-to-image'

        # Component analysis
        if info.model_index:
            components = {}
            component_types = []
            
            # Extract component info
            for key, value in info.model_index.items():
                if isinstance(value, list) and len(value) >= 2:
                    if not key.startswith('_'):
                        # Skip null components
                        if value[0] is None or value[1] is None:
                            continue
                        component_name = key
                        component_class = value[1]
                        components[component_name] = component_class
                        component_types.append(component_name)

            info.metadata['components'] = components
            info.metadata['component_types'] = component_types

            # Estimate component sizes for all diffusion models
            if info.size_gb > 0 and component_types:
                # Try to get component configs for better estimation
                component_configs = self._fetch_component_configs(
                    info.model_id, components
                )
                
                if component_configs:
                    # Use config-based estimation
                    component_sizes = self._estimate_sizes_from_configs(
                        component_configs, info.size_gb, component_types
                    )
                else:
                    # Fall back to distribution-based estimation
                    component_sizes = self._estimate_component_sizes(
                        info.size_gb, component_types, info.architecture
                    )
                    
                info.metadata['component_sizes'] = component_sizes

        # Get scheduler info
        if 'scheduler' in info.files:
            scheduler_config = self._get_scheduler_config(info)
            if scheduler_config:
                info.metadata['scheduler'] = scheduler_config.get('_class')

        # Requirements
        info.special_requirements = [
            'diffusers',
            'transformers',
            'torch',
            'accelerate',
            'pillow',
            'opencv-python'
        ]

        # Add xformers for memory efficiency
        if 'xformers' in info.tags or info.size_gb > 2:
            info.special_requirements.append('xformers')

        # Safetensors support
        if any(f.endswith('.safetensors') for f in info.files):
            info.special_requirements.append('safetensors')

        # Quantization support (limited for diffusion)
        info.supports_quantization = ['fp32', 'fp16']

        # Default dtype - diffusion models typically use fp16
        if not info.default_dtype and info.config:
            info.default_dtype = 'float16'

        # vLLM doesn't support diffusion models
        info.metadata['supports_vllm'] = False

        # TensorRT support for diffusion models
        info.metadata['supports_tensorrt'] = True

        return info

    def _get_scheduler_config(self, info: ModelInfo) -> dict:
        """Try to get scheduler config"""
        # This would need actual file reading, for now return empty
        return {}

    def _estimate_component_sizes(self, total_size: float, components: list,
                                  architecture: str) -> dict:
        """Estimate component sizes for diffusion models"""
        component_sizes = {}

        # Different architectures have different distributions
        if 'Flux' in str(architecture):
            # FLUX models distribution
            distributions = {
                'transformer': 0.70,
                'text_encoder_2': 0.20,  # T5
                'vae': 0.05,
                'text_encoder': 0.05  # CLIP
            }
        elif 'StableDiffusionXL' in str(architecture):
            # SDXL distribution
            distributions = {
                'unet': 0.60,
                'text_encoder_2': 0.20,  # OpenCLIP
                'text_encoder': 0.10,  # CLIP
                'vae': 0.10
            }
        elif 'StableDiffusion3' in str(architecture):
            # SD3 distribution
            distributions = {
                'transformer': 0.65,
                'text_encoder_3': 0.15,  # T5
                'text_encoder': 0.10,  # CLIP
                'text_encoder_2': 0.05,  # CLIP-G
                'vae': 0.05
            }
        else:
            # Default SD 1.x/2.x distribution
            distributions = {
                'unet': 0.70,
                'text_encoder': 0.20,
                'vae': 0.10,
                'safety_checker': 0.05,
                'scheduler': 0.01,  # Very small
                'tokenizer': 0.01,  # Very small
                'feature_extractor': 0.01  # Very small
            }

        # First, filter distributions to only include components that exist
        actual_distributions = {}
        for comp in components:
            if comp in distributions:
                actual_distributions[comp] = distributions[comp]
        
        # If we have distributions for actual components, normalize them
        if actual_distributions:
            total_weight = sum(actual_distributions.values())
            # Normalize to sum to 1.0
            for comp, weight in actual_distributions.items():
                normalized_weight = weight / total_weight
                component_sizes[comp] = round(
                    total_size * normalized_weight, 1
                )
        
        # For any remaining components without distribution
        assigned_size = sum(component_sizes.values())
        remaining = total_size - assigned_size
        unassigned = [c for c in components if c not in component_sizes]
        
        if unassigned and remaining > 0:
            size_per_component = remaining / len(unassigned)
            for comp in unassigned:
                component_sizes[comp] = round(size_per_component, 1)

        return component_sizes
    
    def _fetch_component_configs(self, model_id: str, 
                                components: Dict[str, str]) -> Dict[str, Any]:
        """Try to fetch config.json for each component"""
        from ..utils import fetch_model_config
        
        configs = {}
        for comp_name, comp_class in components.items():
            # Try to fetch component config
            config_path = f"{comp_name}/config.json"
            config = fetch_model_config(model_id, config_path)
            if config:
                configs[comp_name] = config
                logger.debug(f"Fetched config for {comp_name}")
        
        return configs
    
    def _estimate_sizes_from_configs(self, configs: Dict[str, Any], 
                                    total_size: float,
                                    component_types: list) -> Dict[str, float]:
        """Estimate component sizes based on their configs"""
        component_params = {}
        component_sizes = {}
        
        # Estimate parameters for each component
        for comp_name, config in configs.items():
            params = estimate_component_parameters(comp_name, config)
            if params:
                component_params[comp_name] = params
                # Estimate size (assuming float16 storage)
                size_gb = parameters_to_size_gb(params, 'float16')
                component_sizes[comp_name] = size_gb
        
        # Handle components without configs
        for comp in component_types:
            if comp not in component_sizes:
                # Use minimal size for scheduler/tokenizer/etc
                if comp in ['scheduler', 'tokenizer', 'feature_extractor']:
                    component_sizes[comp] = 0.1
                else:
                    # We'll distribute remaining size to these
                    pass
        
        # If total estimated size differs significantly from actual,
        # scale proportionally
        estimated_total = sum(component_sizes.values())
        if estimated_total > 0 and abs(estimated_total - total_size) > 1:
            # Scale to match actual size
            scale_factor = total_size / estimated_total
            for comp in component_sizes:
                component_sizes[comp] = round(
                    component_sizes[comp] * scale_factor, 1
                )
        
        # Ensure we don't exceed total
        current_total = sum(component_sizes.values())
        if current_total > total_size:
            # Reduce largest component
            largest = max(component_sizes.keys(), 
                         key=lambda k: component_sizes[k])
            component_sizes[largest] -= (current_total - total_size)
            component_sizes[largest] = max(0.1, component_sizes[largest])
        
        return component_sizes

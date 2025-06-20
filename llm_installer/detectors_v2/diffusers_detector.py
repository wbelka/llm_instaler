"""
Detector for diffusers library models
"""

from .base import BaseDetector, ModelInfo


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
                        component_name = key
                        component_class = value[1]
                        components[component_name] = component_class
                        component_types.append(component_name)

            info.metadata['components'] = components
            info.metadata['component_types'] = component_types

            # Estimate component sizes for all diffusion models
            if info.size_gb > 0 and component_types:
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
                'safety_checker': 0.05
            }

        # Calculate sizes based on distribution
        for component in components:
            if component in distributions:
                component_sizes[component] = round(
                    total_size * distributions[component], 1
                )
            else:
                # For unknown components, use equal distribution of remaining
                remaining = total_size - sum(component_sizes.values())
                unknown_count = len([c for c in components
                                     if c not in distributions])
                if unknown_count > 0:
                    component_sizes[component] = round(
                        remaining / unknown_count, 1
                    )

        return component_sizes

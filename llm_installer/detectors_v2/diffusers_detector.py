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
        if info.pipeline_tag in ['text-to-image', 'image-to-image', 'inpainting']:
            return True

        return False

    def detect(self, info: ModelInfo) -> ModelInfo:
        """Detect diffusers-specific information"""
        info.model_type = 'diffusion'
        info.is_multimodal = True  # Text + Image

        # Detect architecture from model_index
        if info.model_index and '_class' in info.model_index:
            pipeline_class = info.model_index['_class']
            info.architecture = pipeline_class

            # Detect specific type
            if 'StableDiffusion' in pipeline_class:
                if 'XL' in pipeline_class:
                    info.metadata['variant'] = 'SDXL'
                elif '3' in pipeline_class:
                    info.metadata['variant'] = 'SD3'
                else:
                    info.metadata['variant'] = 'SD'

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
            components = []
            for key, value in info.model_index.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], str):
                        # Component name
                        components.append(key)
            info.metadata['components'] = components

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

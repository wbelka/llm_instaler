"""
Detector for FLUX models (Black Forest Labs)
"""

from .base import BaseDetector, ModelInfo


class FluxDetector(BaseDetector):
    """Detector for FLUX diffusion models"""

    def can_handle(self, info: ModelInfo) -> bool:
        """Check if this is a FLUX model"""
        # Check model ID
        if 'flux' in info.model_id.lower():
            return True

        # Check for FluxPipeline in model_index
        if (info.model_index and
                info.model_index.get('_class_name') == 'FluxPipeline'):
            return True

        # Check tags
        if any('flux' in tag.lower() for tag in info.tags):
            return True

        return False

    def detect(self, info: ModelInfo) -> ModelInfo:
        """Detect FLUX-specific information"""
        info.model_type = 'diffusion'
        info.task = 'text-to-image'
        info.is_multimodal = True
        info.architecture = 'FluxPipeline'

        # FLUX variant detection
        model_lower = info.model_id.lower()
        if 'schnell' in model_lower:
            info.metadata['variant'] = 'FLUX.1-schnell'
            info.metadata['description'] = 'Fast variant optimized for speed'
        elif 'dev' in model_lower:
            info.metadata['variant'] = 'FLUX.1-dev'
            info.metadata['description'] = \
                'Development variant with full quality'
        elif 'pro' in model_lower:
            info.metadata['variant'] = 'FLUX.1-pro'
            info.metadata['description'] = 'Professional variant (API only)'

        # Component analysis is handled by DiffusersDetector
        # We just add FLUX-specific metadata here

        # Model capabilities
        info.metadata['capabilities'] = [
            'text-to-image',
            'high-resolution',
            'photorealistic'
        ]

        # FLUX uses flow matching
        info.metadata['scheduler_type'] = 'flow-matching'

        # Requirements
        info.special_requirements = [
            'diffusers>=0.30.0',
            'transformers',
            'torch>=2.0',
            'accelerate',
            'safetensors',
            'sentencepiece',  # For T5
            'pillow',
            'opencv-python'
        ]

        # Memory optimization
        if info.size_gb > 30:
            info.special_requirements.append('xformers')

        # Quantization support
        info.supports_quantization = ['fp32', 'fp16', 'bf16']

        # Default dtype
        if info.config and 'torch_dtype' in info.config:
            info.default_dtype = info.config['torch_dtype']
        else:
            # FLUX models typically use float16
            info.default_dtype = 'float16'

        # FLUX doesn't support vLLM (it's for diffusion)
        info.metadata['supports_vllm'] = False

        # TensorRT can optimize diffusion models
        info.metadata['supports_tensorrt'] = True

        # Add guidance scale info
        info.metadata['guidance_scale'] = {
            'recommended': 3.5,
            'range': [1.0, 10.0]
        }

        # Add resolution info
        info.metadata['resolution'] = {
            'default': 1024,
            'supported': [512, 768, 1024, 1280, 1536]
        }

        return info

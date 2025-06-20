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
        info.is_multimodal = True
        
        # Respect pipeline_tag from API if available
        if info.pipeline_tag:
            info.task = info.pipeline_tag
        else:
            # Default to text-to-image for standard FLUX
            info.task = 'text-to-image'
        
        # Check for ControlNet
        if 'controlnet' in info.model_id.lower() or 'controlnet' in info.tags:
            info.architecture = 'FluxControlNetPipeline'
            # ControlNet models are always image-to-image
            if not info.pipeline_tag:
                info.task = 'image-to-image'
        else:
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

        # Model capabilities based on type
        if 'controlnet' in info.model_id.lower() or 'controlnet' in info.tags:
            # ControlNet capabilities
            capabilities = ['image-to-image']
            
            # Check for specific ControlNet types
            if 'upscaler' in info.model_id.lower() or 'upscaler' in info.tags:
                capabilities.extend(['super-resolution', 'upscaling'])
            elif 'depth' in info.model_id.lower() or 'depth' in info.tags:
                capabilities.extend(['depth-conditioning', 'structure-preserving'])
            elif 'canny' in info.model_id.lower() or 'canny' in info.tags:
                capabilities.extend(['edge-conditioning', 'line-following'])
            elif 'pose' in info.model_id.lower() or 'pose' in info.tags:
                capabilities.extend(['pose-conditioning', 'human-pose'])
            
            # Check for resolution info
            if 'super-resolution' in info.tags or 'upscaler' in info.tags:
                info.metadata['upscale_factor'] = 4  # Common default
                
            info.metadata['capabilities'] = capabilities
        else:
            # Standard FLUX capabilities
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
        upscaler_check = ('upscaler' in info.model_id.lower() or
                          'super-resolution' in info.tags)
        if upscaler_check:
            # For upscalers, input can be variable, output is higher
            info.metadata['resolution'] = {
                'input': 'Variable (low resolution)',
                'output': 'Up to 4x input resolution',
                'recommended_input': [256, 512, 768]
            }
        else:
            # Standard FLUX resolution
            info.metadata['resolution'] = {
                'default': 1024,
                'supported': [512, 768, 1024, 1280, 1536]
            }

        return info

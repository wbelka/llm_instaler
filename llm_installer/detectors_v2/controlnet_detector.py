"""
Detector for ControlNet models
"""

from .base import BaseDetector, ModelInfo


class ControlNetDetector(BaseDetector):
    """Detector for ControlNet conditioning models"""

    def can_handle(self, info: ModelInfo) -> bool:
        """Check if this is a ControlNet model"""
        # Check tags
        if 'controlnet' in info.tags or 'ControlNet' in info.tags:
            return True

        # Check model ID
        if 'controlnet' in info.model_id.lower():
            return True

        # Check for control-related tags
        control_tags = ['control', 'conditioning', 'guided-generation']
        if any(tag in info.tags for tag in control_tags):
            # Additional check for model name
            if 'control' in info.model_id.lower():
                return True

        return False

    def detect(self, info: ModelInfo) -> ModelInfo:
        """Detect ControlNet-specific information"""
        info.model_type = 'controlnet'
        info.is_multimodal = True  # Image + conditioning

        # Determine task from pipeline_tag or infer
        if info.pipeline_tag:
            info.task = info.pipeline_tag
        else:
            # ControlNet is typically image-to-image
            info.task = 'image-to-image'

        # Detect ControlNet variant/type
        model_lower = info.model_id.lower()
        tags_lower = [t.lower() for t in info.tags]
        all_text = model_lower + ' ' + ' '.join(tags_lower)

        # Detect conditioning type
        conditioning_type = 'general'
        capabilities = ['image-to-image', 'controlled-generation']

        upscale_terms = ['upscaler', 'upscale', 'super-resolution']
        if any(x in all_text for x in upscale_terms):
            conditioning_type = 'upscaler'
            capabilities.extend(['super-resolution', 'upscaling'])
            info.metadata['upscale_factor'] = 4  # Common default

        elif 'canny' in all_text:
            conditioning_type = 'canny'
            capabilities.extend(['edge-detection', 'edge-guided'])

        elif 'depth' in all_text:
            conditioning_type = 'depth'
            capabilities.extend(['depth-guided', '3d-aware'])

        elif 'pose' in all_text or 'openpose' in all_text:
            conditioning_type = 'pose'
            capabilities.extend(['pose-guided', 'human-pose'])

        elif 'mlsd' in all_text:
            conditioning_type = 'mlsd'
            capabilities.extend(['line-detection', 'architectural'])

        elif 'scribble' in all_text:
            conditioning_type = 'scribble'
            capabilities.extend(['sketch-guided', 'freehand-drawing'])

        elif 'seg' in all_text or 'segmentation' in all_text:
            conditioning_type = 'segmentation'
            capabilities.extend(['segmentation-guided', 'region-based'])

        elif 'normal' in all_text:
            conditioning_type = 'normal'
            capabilities.extend(['normal-map-guided', 'surface-aware'])

        elif 'inpaint' in all_text:
            conditioning_type = 'inpainting'
            capabilities.extend(['inpainting', 'mask-guided'])
            info.task = 'inpainting'

        info.metadata['conditioning_type'] = conditioning_type
        info.metadata['capabilities'] = capabilities

        # Detect base model architecture
        if 'sd-controlnet' in model_lower or 'stable-diffusion' in all_text:
            info.architecture = 'StableDiffusionControlNetPipeline'
            base_model = 'stable-diffusion'
        elif 'sdxl' in model_lower:
            info.architecture = 'StableDiffusionXLControlNetPipeline'
            base_model = 'stable-diffusion-xl'
        elif 'flux' in model_lower:
            info.architecture = 'FluxControlNetPipeline'
            base_model = 'flux'
        else:
            info.architecture = 'ControlNetModel'
            base_model = 'unknown'

        info.metadata['base_model'] = base_model

        # Requirements
        info.special_requirements = [
            'diffusers',
            'transformers',
            'torch',
            'accelerate',
            'opencv-python',  # For preprocessing
            'pillow'
        ]

        # Add specific requirements based on conditioning type
        if conditioning_type == 'pose':
            info.special_requirements.append('controlnet-aux')  # For OpenPose

        # Memory optimization
        if info.size_gb > 2:
            info.special_requirements.append('xformers')

        # Quantization support
        info.supports_quantization = ['fp32', 'fp16']
        if base_model == 'flux':
            info.supports_quantization.append('bf16')

        # Default dtype
        if info.config and 'torch_dtype' in info.config:
            info.default_dtype = info.config['torch_dtype']
        else:
            info.default_dtype = 'float16'

        # ControlNet models don't support vLLM
        info.metadata['supports_vllm'] = False

        # TensorRT support depends on base model
        info.metadata['supports_tensorrt'] = base_model in [
            'stable-diffusion', 'stable-diffusion-xl'
        ]

        # Add preprocessing requirements
        info.metadata['preprocessing'] = {
            'canny': 'Canny edge detection',
            'depth': 'Depth estimation model (e.g., MiDaS)',
            'pose': 'OpenPose detection',
            'mlsd': 'M-LSD line detection',
            'scribble': 'Scribble/sketch preprocessing',
            'segmentation': 'Segmentation model',
            'normal': 'Normal map estimation',
            'upscaler': 'Low resolution input image'
        }.get(conditioning_type, 'Conditioning image')

        return info

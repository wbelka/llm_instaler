"""
Detector for multi-modality models (complex multimodal architectures)
"""

import logging
from typing import Dict, List, Optional, Any
from .base import BaseDetector, ModelProfile

logger = logging.getLogger(__name__)


class MultiModalityDetector(BaseDetector):
    """Detector for advanced multi-modality models like Janus"""

    def detect(self, model_id: str, config: Dict[str, Any],
               files: List[str]) -> Optional[ModelProfile]:
        """
        Detect multi-modality models with complex architectures
        """
        # Check if it's a multi_modality model
        if not config or config.get('model_type') != 'multi_modality':
            return None

        # Create profile
        profile = ModelProfile(
            model_type="multi_modality",
            model_id=model_id,
            library="transformers",
            task="multimodal-generation",
            is_multimodal=True,
            metadata={}
        )

        # Extract modalities from config
        modalities = ['text']  # Always has text

        # Check for vision components
        if 'vision_config' in config:
            modalities.append('vision-understanding')

        # Check for generation vision components
        if 'gen_vision_config' in config:
            modalities.append('image-generation')

        # Use file-based estimation if available
        from ..utils import estimate_size_from_files
        file_sizes = config.get('_file_sizes', {})
        file_size = estimate_size_from_files(files, file_sizes)

        if file_size > 0:
            # Use actual file size from API
            profile.estimated_size_gb = file_size
        else:
            # No size information available
            logger.warning(f"Could not determine size for {model_id}")
            profile.estimated_size_gb = 0.0  # Unknown

        # Get torch dtype
        torch_dtype = config.get('torch_dtype', 'float32')
        if 'language_config' in config:
            # Prefer language config dtype
            torch_dtype = config['language_config'].get('torch_dtype', torch_dtype)
        profile.metadata['torch_dtype'] = torch_dtype

        # Store modalities info
        profile.metadata['modalities'] = modalities

        # Architecture info
        if 'language_config' in config:
            lang_config = config['language_config']
            num_layers = lang_config.get('num_hidden_layers', 0)
            arch_name = f"{lang_config.get('model_type', 'unknown').title()}-{num_layers}L"
            profile.architecture = arch_name

        # Hardware requirements - only if we know the size
        if profile.estimated_size_gb > 0:
            profile.min_ram_gb = max(16.0, profile.estimated_size_gb * 1.5)
            profile.min_vram_gb = max(8.0, profile.estimated_size_gb * 1.2)
            profile.recommended_ram_gb = max(32.0, profile.estimated_size_gb * 2.0)
            profile.recommended_vram_gb = max(16.0, profile.estimated_size_gb * 1.3)
        else:
            # Unknown size - use conservative defaults
            profile.min_ram_gb = 16.0
            profile.min_vram_gb = 8.0
            profile.recommended_ram_gb = 32.0
            profile.recommended_vram_gb = 16.0

        # Quantization support
        profile.supports_quantization = ['fp32', 'fp16', '8bit', '4bit']

        # Special requirements
        profile.special_requirements = [
            'torch>=2.0.0',
            'transformers>=4.33.0',
            'accelerate',
            'safetensors',
            'pillow',
            'torchvision',
            'einops',  # Often needed for vision components
        ]

        # These models may support specialized features
        if 'janus' in model_id.lower():
            profile.metadata['supports_image_generation'] = True
            profile.metadata['supports_image_understanding'] = True
            profile.task = 'multimodal-generation'
            profile.metadata['capabilities'] = [
                'text-generation',
                'image-understanding',
                'image-generation',
                'visual-question-answering'
            ]

        return profile

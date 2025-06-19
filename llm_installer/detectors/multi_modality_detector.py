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
        components = {}

        # Check for vision components
        if 'vision_config' in config:
            modalities.append('vision-understanding')
            vision_config = config['vision_config']
            # Vision encoder component
            if 'model_name' in vision_config.get('params', {}):
                model_name = vision_config['params']['model_name']
                if 'siglip_large' in model_name:
                    components['vision_encoder'] = 0.9  # SigLIP large ~900MB
                elif 'clip_large' in model_name:
                    components['vision_encoder'] = 0.9
                else:
                    components['vision_encoder'] = 0.6  # Default vision encoder

        # Check for generation vision components
        if 'gen_vision_config' in config:
            modalities.append('image-generation')
            # VQ tokenizer for image generation
            components['vision_tokenizer'] = 0.5

        # Check for aligner components
        if 'aligner_config' in config:
            components['aligner'] = 0.2  # MLP projector

        if 'gen_aligner_config' in config:
            components['gen_aligner'] = 0.1  # Smaller MLP

        # Check for generation head
        if 'gen_head_config' in config:
            components['gen_head'] = 0.3  # Vision generation head

        # Calculate language model size
        if 'language_config' in config:
            lang_config = config['language_config']
            num_layers = lang_config.get('num_hidden_layers', 0)

            # Estimate LLM size based on layers
            # Most modern models use ~0.4-0.5GB per layer in fp16/bf16
            # This is a rough estimate when we don't have actual file sizes
            components['language_model'] = max(5.0, num_layers * 0.45)

        # Calculate total size
        calculated_size = sum(components.values())

        # Use file-based estimation if available
        from ..utils import estimate_size_from_files
        # Get file sizes from config if available
        file_sizes = config.get('_file_sizes', {})
        file_size = estimate_size_from_files(files, file_sizes)

        if file_size > 0:
            # Use actual file size
            profile.estimated_size_gb = file_size
            # Adjust component sizes proportionally if needed
            if calculated_size > 0:
                ratio = file_size / calculated_size
                components = {k: round(v * ratio, 1) for k, v in components.items()}
        else:
            # No real size available
            if calculated_size > 0:
                profile.estimated_size_gb = round(calculated_size, 1)
            else:
                logger.warning(f"Could not determine size for {model_id}")
                profile.estimated_size_gb = 0.0  # Unknown

        # Get torch dtype
        torch_dtype = config.get('torch_dtype', 'float32')
        if 'language_config' in config:
            # Prefer language config dtype
            torch_dtype = config['language_config'].get('torch_dtype', torch_dtype)
        profile.metadata['torch_dtype'] = torch_dtype

        # Store component info
        profile.metadata['modalities'] = modalities
        profile.metadata['component_sizes'] = components

        # Architecture info
        if 'language_config' in config:
            lang_config = config['language_config']
            arch_name = f"{lang_config.get('model_type', 'unknown').title()}-{num_layers}L"
            profile.architecture = arch_name

        # Hardware requirements
        # These models need significant memory for multimodal processing
        profile.min_ram_gb = max(16.0, profile.estimated_size_gb * 1.5)
        profile.min_vram_gb = max(8.0, max(components.values()) * 1.2)
        profile.recommended_ram_gb = max(32.0, profile.estimated_size_gb * 2.0)
        profile.recommended_vram_gb = max(16.0, profile.estimated_size_gb * 1.3)

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

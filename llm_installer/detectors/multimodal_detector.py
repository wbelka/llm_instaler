"""
Detector for multimodal models (vision-language, audio-language, etc.)
"""

import logging
from typing import Dict, List, Optional, Any
from .base import BaseDetector, ModelProfile

logger = logging.getLogger(__name__)


class MultimodalDetector(BaseDetector):
    """Detector for multimodal models that combine multiple modalities"""

    def detect(self, model_id: str, config: Dict[str, Any],
               files: List[str]) -> Optional[ModelProfile]:
        """
        Detect multimodal models like Ming-Lite-Omni, CLIP, BLIP, etc.
        """
        # Check for multimodal indicators
        is_multimodal = False
        modalities = []

        # Check for vision config
        if any(key in config for key in ['vision_config', 'visual_config', 'image_config']):
            is_multimodal = True
            modalities.append('vision')

        # Check for audio config
        if any(key in config for key in ['audio_config', 'whisper_config', 'speech_config']):
            is_multimodal = True
            modalities.append('audio')

        # Check for LLM/text config
        if any(key in config for key in ['llm_config', 'text_config', 'language_config']):
            modalities.append('text')

        # Check model type
        model_type = config.get('model_type', '').lower()
        if model_type in ['bailingmm', 'ming', 'janus', 'blip', 'clip', 'align']:
            is_multimodal = True

        # Check architectures
        architectures = config.get('architectures', [])
        for arch in architectures:
            arch_lower = arch.lower()
            if any(mm in arch_lower for mm in ['multimodal', 'mm', 'vision', 'clip', 'blip']):
                is_multimodal = True
                break

        if not is_multimodal or len(modalities) < 2:
            return None

        # Create profile
        profile = ModelProfile(
            model_type="multimodal",
            model_id=model_id,
            library="transformers",
            architecture=architectures[0] if architectures else None,
            task=self._determine_task(modalities),
            is_multimodal=True,
            metadata={}
        )

        # Extract torch_dtype - prefer component configs over main config
        torch_dtype = None

        # For multimodal models, the component dtypes are more important
        # LLM config usually has the actual inference dtype
        if 'llm_config' in config and config['llm_config'].get('torch_dtype'):
            torch_dtype = config['llm_config'].get('torch_dtype')
        elif 'vision_config' in config and config['vision_config'].get('torch_dtype'):
            torch_dtype = config['vision_config'].get('torch_dtype')
        elif config.get('torch_dtype'):
            # Fallback to main config (but it's often misleading for multimodal)
            torch_dtype = config.get('torch_dtype')

        if torch_dtype:
            profile.metadata['torch_dtype'] = torch_dtype

        # Set modalities in metadata
        profile.metadata['modalities'] = modalities

        # Calculate model size
        file_sizes = config.get('_file_sizes', {})
        model_size = self._calculate_multimodal_size(config, files, model_id, file_sizes)
        profile.estimated_size_gb = model_size

        # Calculate memory requirements for multimodal models
        # Components can run on different devices
        component_sizes = self._calculate_component_sizes(config, files, model_id)

        # Minimum: largest component in VRAM, others can be in RAM
        largest_component = max(component_sizes.values())
        total_components = sum(component_sizes.values())

        # For inference: largest component needs to fit in VRAM
        profile.min_vram_gb = round(largest_component * 1.2, 1)  # 20% overhead
        profile.min_ram_gb = round(total_components * 1.2, 1)  # All components in RAM

        # Recommended: all components in VRAM for best performance
        profile.recommended_vram_gb = round(total_components * 1.2, 1)
        profile.recommended_ram_gb = round(total_components * 1.5, 1)  # Extra headroom

        # Store component info in metadata
        profile.metadata['component_sizes'] = component_sizes

        # Determine quantization support
        profile.supports_quantization = self._determine_quantization_support(config, model_size)

        # Special requirements
        requirements = ['torch', 'transformers', 'pillow']

        # Add vision requirements
        if 'vision' in modalities:
            requirements.extend(['torchvision', 'opencv-python'])

        # Add audio requirements
        if 'audio' in modalities:
            requirements.extend(['librosa', 'soundfile'])
            if 'whisper_config' in config:
                requirements.append('openai-whisper')

        # Add acceleration libraries
        requirements.extend(['accelerate', 'safetensors'])

        # Check for specific model requirements
        if model_type == 'bailingmm':
            requirements.extend(['einops', 'timm'])

        profile.special_requirements = list(dict.fromkeys(requirements))

        return profile

    def _determine_task(self, modalities: List[str]) -> str:
        """Determine task based on modalities"""
        if 'vision' in modalities and 'text' in modalities:
            return 'visual-question-answering'
        elif 'audio' in modalities and 'text' in modalities:
            return 'audio-text-to-text'
        elif 'video' in modalities and 'text' in modalities:
            return 'video-text-to-text'
        else:
            return 'multimodal'

    def _calculate_multimodal_size(self, config: Dict[str, Any],
                                   files: List[str], model_id: str,
                                   file_sizes: Optional[Dict[str, float]] = None) -> float:
        """Calculate size for multimodal models"""
        # First try to get size from files (most accurate)
        from ..utils import estimate_size_from_files
        file_based_size = estimate_size_from_files(files, file_sizes)

        if file_based_size > 0:
            return file_based_size

        # Otherwise calculate from config
        total_size = 0.0

        # Calculate LLM component size
        if 'llm_config' in config:
            llm_config = config['llm_config']
            llm_size = self._calculate_model_size(llm_config, files, model_id)
            total_size += llm_size

        # Calculate vision component size
        if 'vision_config' in config:
            vision_config = config['vision_config']
            # Vision models are typically smaller
            hidden_size = vision_config.get('hidden_size', 768)
            num_layers = vision_config.get('depth', vision_config.get('num_hidden_layers', 12))
            vision_size = (hidden_size * hidden_size * num_layers * 12) / 1e9  # Rough estimate
            total_size += vision_size

        # Calculate audio component size
        if 'audio_config' in config or 'whisper_config' in config:
            # Audio encoders are typically 0.5-2GB
            total_size += 1.0

        # If we couldn't calculate, return 0 (unknown)
        if total_size == 0:
            logger.debug("Could not calculate multimodal model size from config")

        return round(total_size, 1)

    def _calculate_component_sizes(self, config: Dict[str, Any],
                                   files: List[str], model_id: str) -> Dict[str, float]:
        """Calculate individual component sizes"""
        # Get total model size first
        total_size = self._calculate_multimodal_size(config, files, model_id)

        components = {}

        # For multimodal models, components are often integrated rather than separate
        # Use generic distributions based on detected modalities

        has_vision = 'vision_config' in config
        has_audio = 'audio_config' in config or 'whisper_config' in config
        has_talker = 'talker_config' in config

        # Count modalities
        num_modalities = 1  # Always have text/LLM
        if has_vision:
            num_modalities += 1
        if has_audio:
            num_modalities += 1
        if has_talker:
            num_modalities += 1

        # For integrated multimodal models, the LLM backbone is shared
        # Additional modalities add encoders/decoders but share representations
        if has_vision and has_audio:
            # Vision + Audio + Text model
            components['llm'] = round(total_size * 0.60, 1)  # 60% shared backbone
            components['vision'] = round(total_size * 0.25, 1)  # 25% vision
            components['audio'] = round(total_size * 0.15, 1)  # 15% audio
            if has_talker:
                # Adjust for talker
                components['llm'] = round(total_size * 0.58, 1)
                components['vision'] = round(total_size * 0.24, 1)
                components['audio'] = round(total_size * 0.14, 1)
                components['talker'] = round(total_size * 0.04, 1)
        elif has_vision:
            # Vision + Text model (like CLIP, BLIP)
            components['llm'] = round(total_size * 0.65, 1)  # 65% text/shared
            components['vision'] = round(total_size * 0.35, 1)  # 35% vision
        elif has_audio:
            # Audio + Text model
            components['llm'] = round(total_size * 0.70, 1)  # 70% text/shared
            components['audio'] = round(total_size * 0.30, 1)  # 30% audio
        else:
            # Fallback - shouldn't happen for multimodal
            components['llm'] = total_size

        # Ensure components sum to total size (handle rounding)
        component_sum = sum(components.values())
        if abs(component_sum - total_size) > 0.1:
            # Adjust the largest component
            largest_comp = max(components, key=components.get)
            components[largest_comp] = round(
                components[largest_comp] + (total_size - component_sum), 1
            )

        return components

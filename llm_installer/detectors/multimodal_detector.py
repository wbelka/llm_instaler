"""
Detector for multimodal models (vision-language, audio-language, etc.)
"""

from typing import Dict, List, Optional, Any
from .base import BaseDetector, ModelProfile


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
        model_size = self._calculate_multimodal_size(config, files, model_id)
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
                                   files: List[str], model_id: str) -> float:
        """Calculate size for multimodal models"""
        # First try to get size from files (most accurate)
        from ..utils import estimate_size_from_files
        file_based_size = estimate_size_from_files(files)

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

        # If we couldn't calculate, use default based on architecture
        if total_size == 0:
            if 'ming' in model_id.lower():
                total_size = 8.0  # Ming-Lite is around 8GB
            else:
                total_size = 10.0  # Default for multimodal

        return round(total_size, 1)

    def _calculate_component_sizes(self, config: Dict[str, Any],
                                   files: List[str], model_id: str) -> Dict[str, float]:
        """Calculate individual component sizes"""
        # Get total model size first
        total_size = self._calculate_multimodal_size(config, files, model_id)

        components = {}
        calculated_total = 0.0

        # Calculate vision component size
        if 'vision_config' in config:
            vision_config = config['vision_config']
            hidden_size = vision_config.get('hidden_size', 768)
            num_layers = vision_config.get('depth', vision_config.get('num_hidden_layers', 12))
            vision_size = (hidden_size * hidden_size * num_layers * 12) / 1e9
            components['vision'] = round(vision_size, 1)
            calculated_total += vision_size

        # Calculate audio component size
        if 'audio_config' in config:
            audio_config = config['audio_config']
            # Check for encoder size
            if 'audio_encoder_output_size' in audio_config:
                components['audio'] = 1.0  # Typical audio encoder
            else:
                components['audio'] = 0.5
            calculated_total += components['audio']
        elif 'whisper_config' in config:
            whisper_config = config['whisper_config']
            if 'whisper_encoder_config' in whisper_config:
                enc_config = whisper_config['whisper_encoder_config']
                n_state = enc_config.get('n_state', 1280)
                n_layer = enc_config.get('n_layer', 32)
                # Whisper size estimation
                components['audio'] = round((n_state * n_state * n_layer * 12) / 1e9, 1)
            else:
                components['audio'] = 1.5  # Default Whisper size
            calculated_total += components['audio']

        # Calculate talker/decoder component if present
        if 'talker_config' in config:
            components['talker'] = 0.5  # Typically small
            calculated_total += components['talker']

        # LLM component is the remainder (largest part)
        if 'llm_config' in config or total_size > calculated_total:
            components['llm'] = round(total_size - calculated_total, 1)
            if components['llm'] < 1.0:
                # Minimum size for LLM
                components['llm'] = 1.0

        # If no components calculated, estimate from total
        if not components:
            if 'ming' in model_id.lower():
                # Ming-Lite distribution for 80GB model
                ratio = total_size / 80.0  # Scale to actual size
                components = {
                    'llm': round(65.0 * ratio, 1),  # ~81% for LLM
                    'vision': round(10.0 * ratio, 1),  # ~12.5% for vision
                    'audio': round(4.0 * ratio, 1),  # ~5% for audio
                    'talker': round(1.0 * ratio, 1)  # ~1.5% for talker
                }
            else:
                # Generic multimodal distribution
                components = {
                    'llm': round(total_size * 0.7, 1),  # 70% for LLM
                    'vision': round(total_size * 0.2, 1),  # 20% for vision
                    'audio': round(total_size * 0.1, 1)  # 10% for audio
                }

        return components

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

        # Extract torch_dtype - check main config and sub-configs
        torch_dtype = config.get('torch_dtype')
        if not torch_dtype and 'llm_config' in config:
            torch_dtype = config['llm_config'].get('torch_dtype')
        if not torch_dtype and 'vision_config' in config:
            torch_dtype = config['vision_config'].get('torch_dtype')

        if torch_dtype:
            profile.metadata['torch_dtype'] = torch_dtype

        # Set modalities in metadata
        profile.metadata['modalities'] = modalities

        # Calculate model size
        model_size = self._calculate_multimodal_size(config, files, model_id)
        profile.estimated_size_gb = model_size

        # Calculate memory requirements
        mem_reqs = self._calculate_memory_requirements(model_size, None)
        profile.min_ram_gb = mem_reqs['min_ram_gb']
        profile.min_vram_gb = mem_reqs['min_vram_gb']
        profile.recommended_ram_gb = mem_reqs['recommended_ram_gb']
        profile.recommended_vram_gb = mem_reqs['recommended_vram_gb']

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

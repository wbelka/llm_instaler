"""
Detector for BAGEL (ByteDance) models
"""

from .base import BaseDetector, ModelInfo


class BagelDetector(BaseDetector):
    """Detector for BAGEL models from ByteDance"""

    def can_handle(self, info: ModelInfo) -> bool:
        """Check if this is a BAGEL model"""
        # Library name
        if info.library_name == 'bagel-mot':
            return True

        # Model ID pattern
        if 'bagel' in info.model_id.lower():
            return True

        # Tags
        if any('bagel' in tag.lower() for tag in info.tags):
            return True

        return False

    def detect(self, info: ModelInfo) -> ModelInfo:
        """Detect BAGEL-specific information"""
        # BAGEL models are multimodal text generation models
        info.model_type = 'text-generation'
        info.task = 'text-generation'

        # Check if we have llm_config
        if info.config and 'architectures' in info.config:
            # Get architecture from config
            info.architecture = info.config['architectures'][0]

            # If it's Qwen2 based
            if 'Qwen2' in info.architecture:
                info.metadata['base_model'] = 'Qwen2'

        # BAGEL models are multimodal (MoT = Mixture of Texts)
        info.is_multimodal = True
        info.metadata['modalities'] = ['text']

        # Model variant detection from name
        model_lower = info.model_id.lower()
        if '7b' in model_lower:
            info.metadata['parameters'] = '7B'
        elif '14b' in model_lower:
            info.metadata['parameters'] = '14B'

        # MoT = Mixture of Texts capability
        if 'mot' in model_lower:
            info.metadata['capabilities'] = [
                'mixture-of-texts',
                'multi-style-generation'
            ]

        # Requirements
        info.special_requirements = [
            'transformers',
            'torch',
            'accelerate',
            'sentencepiece',  # For tokenization
            'protobuf'
        ]

        # Get dtype from config
        if info.config and 'torch_dtype' in info.config:
            info.default_dtype = info.config['torch_dtype']
        else:
            info.default_dtype = 'bfloat16'  # Common for newer models

        # Quantization support - standard transformers quantizations
        info.supports_quantization = [
            'fp32', 'fp16', 'bfloat16', '8bit', '4bit'
        ]

        # vLLM support - yes for text generation models
        info.metadata['supports_vllm'] = True

        # TensorRT support
        info.metadata['supports_tensorrt'] = True

        # Add sliding window info if present
        if info.config:
            if 'sliding_window' in info.config:
                info.metadata['sliding_window'] = \
                    info.config['sliding_window']
            if 'max_position_embeddings' in info.config:
                info.metadata['max_length'] = \
                    info.config['max_position_embeddings']

        return info

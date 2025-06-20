"""
Detector for transformers library models
"""

from typing import List, Optional
from .base import BaseDetector, ModelInfo


class TransformersDetector(BaseDetector):
    """Detector for models using transformers library"""

    def can_handle(self, info: ModelInfo) -> bool:
        """Check if this is a transformers model"""
        # Explicit library
        if info.library_name == 'transformers':
            return True

        # Has transformers tag
        if 'transformers' in info.tags:
            return True

        # Has config.json with transformers-specific fields
        if info.config and 'model_type' in info.config:
            return True

        # Default for many models without explicit library
        if not info.library_name and info.config:
            return True

        return False

    def detect(self, info: ModelInfo) -> ModelInfo:
        """Detect transformers-specific information"""
        # Model type from config
        if info.config:
            info.model_type = info.config.get('model_type', 'transformer')

            # Architecture
            if 'architectures' in info.config and info.config['architectures']:
                info.architecture = info.config['architectures'][0]
            else:
                info.architecture = info.model_type

            # Default dtype
            info.default_dtype = info.config.get('torch_dtype')

            # Task from config or pipeline_tag
            info.task = info.pipeline_tag or self._infer_task_from_architecture(info.architecture)

        # Check if multimodal
        info.is_multimodal = self._check_multimodal(info)

        # Quantization detection
        info.quantization = self._detect_quantization_from_files(info.files)

        # Supported quantizations
        info.supports_quantization = self._get_supported_quantizations(info)

        # Requirements
        info.special_requirements = self._get_requirements(info)

        # Add metadata
        info.metadata = self._extract_metadata(info)

        # Check vLLM and TensorRT support
        info.metadata['supports_vllm'] = self._check_vllm_support(info)
        info.metadata['supports_tensorrt'] = self._check_tensorrt_support(info)

        return info

    def _check_multimodal(self, info: ModelInfo) -> bool:
        """Check if model is multimodal"""
        # Check tags
        multimodal_tags = ['vision', 'multimodal', 'multi_modality', 'image-text',
                          'visual-question-answering', 'image-to-text']
        if any(tag in info.tags for tag in multimodal_tags):
            return True

        # Check config for vision components
        if info.config:
            if 'vision_config' in info.config or 'visual_config' in info.config:
                return True
            if info.config.get('model_type') in ['vision-encoder-decoder', 'clip',
                                                 'multi_modality', 'git', 'blip']:
                return True

        # Check architecture
        if info.architecture:
            arch_lower = info.architecture.lower()
            if any(x in arch_lower for x in ['vision', 'clip', 'vit', 'visual']):
                return True

        return False

    def _infer_task_from_architecture(self, architecture: Optional[str]) -> Optional[str]:
        """Infer task from architecture name"""
        if not architecture:
            return 'text-generation'  # Safe default

        arch_lower = architecture.lower()

        # Classification models
        if 'forsequenceclassification' in arch_lower:
            return 'text-classification'
        elif 'fortokenclassification' in arch_lower:
            return 'token-classification'
        elif 'forquestionanswering' in arch_lower:
            return 'question-answering'

        # Generation models
        elif 'forcausallm' in arch_lower or 'forgpt' in arch_lower:
            return 'text-generation'
        elif 'forconditionalgeneration' in arch_lower:
            return 'text2text-generation'

        # Vision models
        elif 'forimageclassification' in arch_lower:
            return 'image-classification'
        elif 'forobjectdetection' in arch_lower:
            return 'object-detection'

        # Default
        return 'text-generation'

    def _get_supported_quantizations(self, info: ModelInfo) -> List[str]:
        """Get supported quantization methods"""
        quants = ['fp32', 'fp16']

        # Check tags for explicit support
        if '8bit' in info.tags or 'int8' in info.tags:
            quants.append('8bit')
        if '4bit' in info.tags or 'int4' in info.tags:
            quants.append('4bit')

        # GPTQ models
        if 'gptq' in info.tags or any('gptq' in f.lower() for f in info.files):
            if 'gptq' not in quants:
                quants.append('gptq')

        # AWQ models
        if 'awq' in info.tags or any('awq' in f.lower() for f in info.files):
            if 'awq' not in quants:
                quants.append('awq')

        return quants

    def _get_requirements(self, info: ModelInfo) -> List[str]:
        """Get special requirements"""
        reqs = ['transformers', 'torch', 'accelerate']

        # Multimodal requirements
        if info.is_multimodal:
            reqs.extend(['pillow', 'torchvision'])
            if 'clip' in (info.model_type or '').lower():
                reqs.append('ftfy')

        # Specific model requirements
        if info.config:
            # Mamba models
            if info.config.get('model_type') == 'mamba':
                reqs.append('mamba-ssm')
                reqs.append('causal-conv1d')

            # MoE models
            if 'num_experts' in info.config or 'moe' in info.tags:
                reqs.append('megablocks')

        # Flash attention
        if 'flash-attention' in info.tags or 'flash-attn' in info.tags:
            reqs.append('flash-attn')

        # Remove duplicates
        return list(set(reqs))

    def _extract_metadata(self, info: ModelInfo) -> dict:
        """Extract additional metadata"""
        metadata = {}

        if info.config:
            # Model size info
            if 'num_parameters' in info.config:
                metadata['parameters'] = info.config['num_parameters']
            elif 'n_params' in info.config:
                metadata['parameters'] = info.config['n_params']

            # Training info
            if 'max_position_embeddings' in info.config:
                metadata['max_length'] = info.config['max_position_embeddings']
            elif 'max_seq_len' in info.config:
                metadata['max_length'] = info.config['max_seq_len']

            # Vocab size
            if 'vocab_size' in info.config:
                metadata['vocab_size'] = info.config['vocab_size']

            # Hidden size
            if 'hidden_size' in info.config:
                metadata['hidden_size'] = info.config['hidden_size']

            # Layers
            if 'num_hidden_layers' in info.config:
                metadata['num_layers'] = info.config['num_hidden_layers']

        return metadata

    def _check_vllm_support(self, info: ModelInfo) -> bool:
        """Check if model supports vLLM"""
        # Only text generation models
        if info.task not in ['text-generation', None]:
            return False

        # Check explicit tags
        if 'vllm' in info.tags or 'text-generation-inference' in info.tags:
            return True

        # Don't guess - return False
        return False

    def _check_tensorrt_support(self, info: ModelInfo) -> bool:
        """Check if model supports TensorRT"""
        # Most transformer models can be optimized with TensorRT
        # But only if not quantized with GGUF
        if info.quantization and 'gguf' in info.quantization:
            return False

        # Check explicit tags
        if 'tensorrt' in info.tags or 'trt' in info.tags:
            return True

        # Transformers models generally support TensorRT
        return True

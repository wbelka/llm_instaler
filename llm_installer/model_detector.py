"""
Simplified model detector using HuggingFace model_info API
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Model information from HuggingFace API"""
    model_id: str
    library_name: Optional[str] = None
    pipeline_tag: Optional[str] = None
    tags: List[str] = None
    config: Dict[str, Any] = None
    size_gb: float = 0.0
    files: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.config is None:
            self.config = {}
        if self.files is None:
            self.files = []


class ModelDetector:
    """Detect model type and requirements using HuggingFace API"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def detect(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Detect model type and requirements

        Returns:
            Dict with model profile information
        """
        try:
            # Get model info from API
            model_info = self._fetch_model_info(model_id)
            if not model_info:
                return None

            # Detect default torch dtype
            default_dtype = None
            if model_info.config and 'torch_dtype' in model_info.config:
                default_dtype = model_info.config['torch_dtype']
            
            # Create base profile
            profile = {
                'model_id': model_id,
                'model_type': self._determine_model_type(model_info),
                'library': model_info.library_name or 'transformers',
                'task': model_info.pipeline_tag or 'text-generation',
                'estimated_size_gb': model_info.size_gb,
                'is_multimodal': self._is_multimodal(model_info),
                'quantization': self._detect_quantization(model_info),
                'architecture': self._detect_architecture(model_info),
                'special_requirements': self._get_requirements(model_info),
                'metadata': {'torch_dtype': default_dtype} if default_dtype else {}
            }
            
            # Store default dtype but don't calculate memory here
            # Let the hardware module handle memory calculations
            profile['estimated_memory_gb'] = 0.0  # Will be calculated based on dtype

            # Add hardware requirements
            hw_reqs = self._calculate_hardware_requirements(model_info, profile)
            profile.update(hw_reqs)

            # Add capabilities
            profile['supports_vllm'] = self._supports_vllm(model_info)
            profile['supports_tensorrt'] = self._supports_tensorrt(model_info)
            profile['supports_quantization'] = self._get_quantization_options(model_info)

            return profile

        except Exception as e:
            self.logger.error(f"Failed to detect model {model_id}: {e}")
            return None

    def _fetch_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Fetch model information from HuggingFace API"""
        try:
            from huggingface_hub import model_info as hf_model_info
            from .utils import get_hf_token, fetch_model_files_with_sizes

            token = get_hf_token()

            # Get model metadata
            info = hf_model_info(model_id, token=token)

            # Get file sizes
            file_sizes = fetch_model_files_with_sizes(model_id)
            total_size = round(sum(file_sizes.values()), 1)

            # Get file list
            files = list(file_sizes.keys()) if file_sizes else []

            # Get config from API
            config = {}
            if hasattr(info, 'config') and info.config:
                config = info.config
            
            # Also fetch config.json directly for more details
            from .utils import fetch_model_config
            detailed_config = fetch_model_config(model_id, "config.json")
            if detailed_config:
                config.update(detailed_config)

            return ModelInfo(
                model_id=model_id,
                library_name=getattr(info, 'library_name', None),
                pipeline_tag=getattr(info, 'pipeline_tag', None),
                tags=getattr(info, 'tags', []),
                config=config,
                size_gb=total_size,
                files=files
            )

        except Exception as e:
            self.logger.error(f"Failed to fetch model info for {model_id}: {e}")
            return None

    def _determine_model_type(self, info: ModelInfo) -> str:
        """Determine model type from info"""
        # Check tags first
        if 'gguf' in info.tags:
            return 'gguf'

        if 'sentence-transformers' in info.tags or info.library_name == 'sentence-transformers':
            return 'sentence-transformer'

        if 'diffusers' in info.tags or info.library_name == 'diffusers':
            return 'diffusion'

        if 'multi_modality' in info.tags:
            return 'multi_modality'

        if 'moe' in info.tags or 'mixture-of-experts' in info.tags:
            return 'moe'

        # Check pipeline tag
        if info.pipeline_tag:
            if 'image' in info.pipeline_tag:
                return 'diffusion'
            if 'sentence' in info.pipeline_tag or 'embedding' in info.pipeline_tag:
                return 'sentence-transformer'
            if info.pipeline_tag == 'any-to-any':
                return 'multi_modality'

        # Default to transformer
        return 'transformer'

    def _is_multimodal(self, info: ModelInfo) -> bool:
        """Check if model is multimodal"""
        multimodal_indicators = [
            'multi_modality', 'multimodal', 'any-to-any',
            'text-to-image', 'image-to-text', 'visual-question-answering'
        ]

        # Check tags
        if any(tag in info.tags for tag in multimodal_indicators):
            return True

        # Check pipeline tag
        if info.pipeline_tag and any(ind in info.pipeline_tag for ind in multimodal_indicators):
            return True

        return False

    def _detect_quantization(self, info: ModelInfo) -> Optional[str]:
        """Detect quantization type"""
        # Check GGUF files
        if any(f.endswith('.gguf') for f in info.files):
            # Try to detect GGUF quantization type
            for file in info.files:
                file_lower = file.lower()
                quant_patterns = ['q4_k_m', 'q5_k_m', 'q8_0', 'q4_0', 'q5_0', 'q6_k']
                for pattern in quant_patterns:
                    if pattern in file_lower:
                        return f'gguf-{pattern.upper()}'
            return 'gguf'

        # Check for other quantization indicators
        if '8bit' in info.tags or 'int8' in info.tags:
            return '8bit'
        if '4bit' in info.tags or 'int4' in info.tags:
            return '4bit'

        # Check config for torch_dtype
        if info.config and 'torch_dtype' in info.config:
            dtype = info.config['torch_dtype']
            if dtype == 'float16':
                return 'fp16'
            elif dtype == 'bfloat16':
                return 'bf16'

        return None

    def _detect_architecture(self, info: ModelInfo) -> Optional[str]:
        """Detect model architecture"""
        # From config
        if info.config:
            if 'model_type' in info.config:
                model_type = info.config['model_type']
                # Don't return generic types as architecture
                if model_type not in ['multi_modality', 'transformer']:
                    return model_type
            if 'architectures' in info.config and info.config['architectures']:
                return info.config['architectures'][0]

        # Just return model_type from config if available
        # Don't try to guess or extract architecture names

        return None

    def _get_requirements(self, info: ModelInfo) -> List[str]:
        """Get special requirements based on model type"""
        requirements = []

        # Base library
        if info.library_name:
            requirements.append(info.library_name)
        else:
            requirements.append('transformers')

        # Common requirements
        requirements.extend(['torch', 'accelerate'])

        # Library-specific requirements
        if info.library_name == 'diffusers':
            requirements.extend(['pillow', 'opencv-python', 'xformers'])
        elif info.library_name == 'sentence-transformers':
            requirements.extend(['numpy', 'scikit-learn', 'faiss-cpu'])
        elif info.library_name == 'timm':
            requirements.extend(['torchvision', 'pillow'])

        # GGUF models
        if 'gguf' in info.tags or any(f.endswith('.gguf') for f in info.files):
            requirements.append('llama-cpp-python')

        # Multimodal models
        if self._is_multimodal(info):
            requirements.extend(['pillow', 'torchvision', 'einops'])

        # MoE models
        if 'moe' in info.tags or 'mixture-of-experts' in info.tags:
            requirements.append('megablocks')

        # Mamba models
        if 'mamba' in info.tags or (info.config and info.config.get('model_type') == 'mamba'):
            requirements.append('mamba-ssm')

        # Remove duplicates
        return list(set(requirements))

    def _calculate_hardware_requirements(self, info: ModelInfo, profile: Dict) -> Dict[str, Any]:
        """Calculate hardware requirements"""
        size_gb = info.size_gb

        # If size is unknown, return zeros
        if size_gb == 0:
            return {
                'min_ram_gb': 0.0,
                'min_vram_gb': 0.0,
                'recommended_ram_gb': 0.0,
                'recommended_vram_gb': 0.0,
                'supports_cpu': True,
                'supports_cuda': True,
                'supports_metal': True,
                'estimated_memory_gb': 0.0
            }

        # Calculate based on size
        quantization = profile.get('quantization')

        # Simple memory calculation without hardcoded multipliers
        # Just use size * 1.2 for all models
        multiplier = 1.2

        # Adjust for quantization
        if quantization:
            if '4bit' in quantization:
                size_gb *= 0.25
            elif '8bit' in quantization:
                size_gb *= 0.5
            elif 'fp16' in quantization or 'bf16' in quantization:
                size_gb *= 0.5

        estimated_memory = size_gb * multiplier

        return {
            'min_ram_gb': round(estimated_memory, 1),
            'min_vram_gb': round(estimated_memory * 0.8, 1),
            'recommended_ram_gb': round(estimated_memory * 1.5, 1),
            'recommended_vram_gb': round(estimated_memory * 1.2, 1),
            'supports_cpu': True,
            'supports_cuda': True,
            'supports_metal': True,
            'estimated_memory_gb': round(estimated_memory, 1)
        }

    def _supports_vllm(self, info: ModelInfo) -> bool:
        """Check if model supports vLLM"""
        # vLLM primarily supports text generation models
        if info.pipeline_tag not in ['text-generation', None]:
            return False

        # Don't guess vLLM support - return False unless explicitly tagged
        if 'vllm' in info.tags or 'text-generation-inference' in info.tags:
            return True

        return False

    def _supports_tensorrt(self, info: ModelInfo) -> bool:
        """Check if model supports TensorRT"""
        # Most transformer and diffusion models can support TensorRT
        if info.library_name in ['transformers', 'diffusers']:
            return True

        # GGUF models don't support TensorRT
        if 'gguf' in info.tags:
            return False

        return False

    def _get_quantization_options(self, info: ModelInfo) -> List[str]:
        """Get supported quantization options"""
        options = ['fp32', 'fp16']

        # GGUF models have their own quantization
        if 'gguf' in info.tags:
            return ['gguf']

        # Only add quantization options if explicitly supported in tags/config
        if '8bit' in info.tags or 'int8' in info.tags:
            options.append('8bit')
        if '4bit' in info.tags or 'int4' in info.tags:
            options.append('4bit')

        return options

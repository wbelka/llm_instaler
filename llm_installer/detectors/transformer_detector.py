"""
Detector for standard transformer models
"""

from typing import Dict, List, Optional, Any
from .base import BaseDetector, ModelProfile


class TransformerDetector(BaseDetector):
    """Detector for standard transformer-based language models"""

    def detect(self, model_id: str, config: Dict[str, Any],
               files: List[str]) -> Optional[ModelProfile]:
        """
        Detect standard transformer models (default fallback)
        This detector handles most HuggingFace transformers models
        """
        # This is the default detector, so we always return a profile
        # unless it's clearly not a transformer model

        # Skip if it's clearly a diffusion model
        if 'diffusion' in str(config.get('_class', '')).lower():
            return None

        # Skip if model_index.json indicates diffusion pipeline
        if '_diffusers_version' in config:
            return None

        # Create base profile
        profile = ModelProfile(
            model_type="transformer",
            model_id=model_id,
            library="transformers",
            architecture=self._get_architecture(config),
            task=self._extract_task(config)
        )

        # Check for quantization
        quant_config = config.get('quantization_config', {})
        if quant_config:
            if quant_config.get('load_in_4bit'):
                profile.quantization = '4bit'
            elif quant_config.get('load_in_8bit'):
                profile.quantization = '8bit'
            elif 'quant_method' in quant_config:
                profile.quantization = quant_config['quant_method']

        # Check for special model types and requirements
        model_type = config.get('model_type', '').lower()

        # Mamba models
        if model_type == 'mamba':
            profile.special_requirements = ['mamba-ssm', 'causal-conv1d']
            profile.architecture = 'MambaForCausalLM'

        # Multimodal models
        if any(key in config for key in ['vision_config', 'visual_config', 'image_size']):
            profile.is_multimodal = True
            if 'janus' in model_type:
                profile.architecture = 'JanusMultiModalModel'
                profile.special_requirements = profile.special_requirements or []
                profile.special_requirements.append('janus-kernels')

        # Check VLLM support
        profile.supports_vllm = self._check_vllm_support(model_type, config)

        # Check TensorRT support (mainly for NVIDIA models)
        if 'nemotron' in model_id.lower() or 'nvidia' in model_id.lower():
            profile.supports_tensorrt = True
            profile.special_requirements = profile.special_requirements or []
            if 'tensorrt' not in profile.special_requirements:
                profile.special_requirements.append('tensorrt')

        # Add common requirements
        base_requirements = ['torch', 'transformers']

        # Add accelerate for quantized models
        if profile.quantization:
            base_requirements.extend(['accelerate', 'bitsandbytes'])

        # Merge with special requirements
        all_requirements = base_requirements + (profile.special_requirements or [])
        profile.special_requirements = list(dict.fromkeys(all_requirements))  # Remove duplicates

        return profile

    def _get_architecture(self, config: Dict[str, Any]) -> Optional[str]:
        """Extract architecture from config"""
        architectures = config.get('architectures', [])
        if architectures and isinstance(architectures, list):
            return architectures[0]

        # Fallback to model_type
        model_type = config.get('model_type', '')
        if model_type:
            # Convert model_type to architecture format
            arch_mapping = {
                'llama': 'LlamaForCausalLM',
                'gpt2': 'GPT2LMHeadModel',
                'gpt_neox': 'GPTNeoXForCausalLM',
                'opt': 'OPTForCausalLM',
                'bloom': 'BloomForCausalLM',
                'falcon': 'FalconForCausalLM',
                'mistral': 'MistralForCausalLM',
                'mixtral': 'MixtralForCausalLM',
                'qwen': 'QWenLMHeadModel',
                'qwen2': 'Qwen2ForCausalLM',
                'phi': 'PhiForCausalLM',
                'stablelm': 'StableLMForCausalLM',
            }

            for key, arch in arch_mapping.items():
                if key in model_type.lower():
                    return arch

        return None

    def _check_vllm_support(self, model_type: str, config: Dict[str, Any]) -> bool:
        """Check if model supports VLLM inference"""
        # Models known to work well with vLLM
        vllm_supported_types = [
            'llama', 'mistral', 'mixtral', 'gpt2', 'gpt_neox',
            'opt', 'bloom', 'falcon', 'qwen', 'phi', 'gemma',
            'internlm', 'yi', 'chatglm', 'baichuan'
        ]

        # Check model type
        if any(supported in model_type.lower() for supported in vllm_supported_types):
            return True

        # Check architecture
        arch = config.get('architectures', [''])[0].lower()
        if any(supported in arch for supported in vllm_supported_types):
            return True

        return False

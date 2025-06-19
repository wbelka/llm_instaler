"""
Detector for Mixture of Experts (MoE) models
"""

from typing import Dict, List, Optional, Any
from .base import BaseDetector, ModelProfile


class MoEDetector(BaseDetector):
    """Detector for Mixture of Experts models"""

    def detect(self, model_id: str, config: Dict[str, Any],
               files: List[str]) -> Optional[ModelProfile]:
        """
        Detect MoE models like Mixtral, BAGEL-MoT, etc.
        """
        # Check if it's a MoE model
        is_moe = False

        # Check model ID for MoE indicators
        model_id_lower = model_id.lower()
        if any(indicator in model_id_lower for indicator in ['moe', 'mot', 'mixtral', 'mixture']):
            is_moe = True

        # Check config for MoE indicators
        if config:
            # Check for num_experts
            if 'num_experts' in config or 'num_local_experts' in config:
                is_moe = True

            # Check architectures
            architectures = config.get('architectures', [])
            for arch in architectures:
                arch_lower = arch.lower()
                if any(moe_indicator in arch_lower for moe_indicator in
                       ['moe', 'mixtral', 'mixture']):
                    is_moe = True
                    break

        # Check for MoE-specific files
        for file in files:
            if 'llm_config.json' in file:
                # BAGEL-MoT style models have llm_config.json
                is_moe = True
                break

        if not is_moe:
            return None

        # Create profile
        profile = ModelProfile(
            model_type="moe",
            model_id=model_id,
            library="transformers",
            architecture=config.get('architectures', [None])[0] if config else None,
            task="text-generation",
            metadata={}
        )

        # Check for llm_config.json and fetch it
        llm_config = None
        if any('llm_config.json' in f for f in files):
            # Try to fetch llm_config.json
            from ..utils import fetch_model_config
            llm_config = fetch_model_config(model_id, "llm_config.json")
            if llm_config:
                # Use llm_config as primary config
                config = llm_config
                profile.architecture = llm_config.get('architectures', [None])[0]

        # Extract metadata
        if config:
            # Get torch dtype
            torch_dtype = config.get('torch_dtype', 'float32')
            profile.metadata['torch_dtype'] = torch_dtype

            # Get MoE specific parameters
            num_experts = config.get('num_experts', config.get('num_local_experts', 0))
            if num_experts:
                profile.metadata['num_experts'] = num_experts

            # Get model dimensions for size calculation
            hidden_size = config.get('hidden_size', 0)
            num_layers = config.get('num_hidden_layers', 0)
            intermediate_size = config.get('intermediate_size', 0)
            vocab_size = config.get('vocab_size', 0)

            # Calculate size based on architecture
            if hidden_size and num_layers:
                # MoE models are larger due to multiple experts
                # Base transformer size
                base_params = (
                    vocab_size * hidden_size +  # Embeddings
                    num_layers * (
                        4 * hidden_size * hidden_size +  # Attention
                        3 * hidden_size * intermediate_size  # FFN
                    )
                ) / 1e9  # Convert to billions

                # MoE multiplier (each expert adds to the model)
                if num_experts > 1:
                    # Typically only FFN is replicated per expert
                    expert_params = (num_layers * 2 * hidden_size * intermediate_size *
                                     (num_experts - 1) / 1e9)
                    total_params = base_params + expert_params
                else:
                    total_params = base_params

                # Convert to GB based on dtype
                dtype_multipliers = {
                    'float32': 4,
                    'float16': 2,
                    'bfloat16': 2,
                    'int8': 1,
                    'int4': 0.5
                }
                bytes_per_param = dtype_multipliers.get(torch_dtype, 4)
                size_gb = total_params * bytes_per_param

                profile.estimated_size_gb = round(size_gb, 1)

        # If no size calculated, use file-based estimation
        if profile.estimated_size_gb == 1.0:
            from ..utils import estimate_size_from_files
            file_size = estimate_size_from_files(files)
            if file_size > 0:
                profile.estimated_size_gb = file_size

        # MoE models typically support these quantizations
        profile.supports_quantization = ['fp32', 'fp16', '8bit', '4bit']

        # Set hardware requirements
        # MoE models need more memory due to expert routing
        profile.min_ram_gb = max(16.0, profile.estimated_size_gb * 1.5)
        profile.min_vram_gb = max(8.0, profile.estimated_size_gb * 1.2)
        profile.recommended_ram_gb = max(32.0, profile.estimated_size_gb * 2.0)
        profile.recommended_vram_gb = max(16.0, profile.estimated_size_gb * 1.5)

        # Special requirements for MoE
        profile.special_requirements = [
            'torch',
            'transformers>=4.36.0',  # MoE support
            'accelerate',
            'safetensors'
        ]

        # MoE models may support specialized inference
        if 'mixtral' in model_id.lower():
            profile.supports_vllm = True

        return profile

"""Handler for Llama 4 multimodal models.

This module provides specialized handling for Meta's Llama 4 models
that support text and image inputs with mixture-of-experts architecture.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

from handlers.multimodal import MultimodalHandler
from core.checker import ModelRequirements
from core.quantization_config import QuantizationConfig

logger = logging.getLogger(__name__)


class Llama4Handler(MultimodalHandler):
    """Handler for Llama 4 multimodal models."""

    def __init__(self, model_info: Dict[str, Any]):
        """Initialize Llama 4 handler.

        Args:
            model_info: Model information dictionary.
        """
        super().__init__(model_info)
        self.is_llama4_model = self._check_if_llama4_model()

    def _check_if_llama4_model(self) -> bool:
        """Check if the model is a Llama 4 multimodal model.

        Returns:
            True if it's a Llama 4 multimodal model, False otherwise.
        """
        model_id = self.model_id.lower()
        model_type = self.model_info.get('model_type', '').lower()
        config = self.model_info.get('config', {})

        # Check for Llama 4 models
        is_llama4 = (
            'llama-4' in model_id or
            'llama4' in model_id or
            (('llama' in model_id) and ('4-' in model_id or '4_' in model_id))
        )

        # Check for Scout or Maverick variants
        is_scout_or_maverick = (
            'scout' in model_id or
            'maverick' in model_id
        )

        # Check if it's multimodal (has vision capabilities)
        has_vision = (
            'vision' in str(config).lower() or
            'image' in str(config).lower() or
            'multimodal' in model_type or
            hasattr(config, 'vision_config') or
            is_scout_or_maverick  # Scout and Maverick are multimodal by default
        )

        return is_llama4 and has_vision

    def get_dependencies(self) -> List[str]:
        """Get required Python dependencies.

        Returns:
            List of pip package specifications.
        """
        base_deps = [
            'torch>=2.6.0',
            'transformers>=4.51.0',  # Llama 4 requires transformers 4.51.0+
            'Pillow>=9.0.0',
            'numpy',
            'einops',
            'sentencepiece',
            'protobuf',
            'accelerate>=0.20.0',
            'safetensors>=0.4.0',
            'tokenizers>=0.15.0'
        ]

        # Add quantization support through centralized config
        from core.quantization_config import QuantizationConfig
        quant_deps = QuantizationConfig.get_quantization_dependencies(
            self.model_type, self.model_family
        )
        base_deps.extend(quant_deps)

        return base_deps

    def analyze(self) -> 'ModelRequirements':
        """Analyze model and return requirements.

        Returns:
            ModelRequirements object.
        """
        from core.checker import ModelRequirements

        # Get model size for memory estimation
        model_size_gb = self._estimate_model_size()

        # Base requirements
        requirements = ModelRequirements()
        requirements.model_type = self.model_type
        requirements.model_family = "llama4"
        requirements.primary_library = "transformers"
        requirements.base_dependencies = self.get_dependencies()
        requirements.special_dependencies = []
        requirements.optional_dependencies = ['flash-attn==2.7.2.post1']
        requirements.disk_space_gb = model_size_gb * 3
        requirements.memory_requirements = {
            "min": max(24, model_size_gb * 2),
            "recommended": max(32, model_size_gb * 3),
            "gpu_min": max(16, model_size_gb * 1.5),
            "gpu_recommended": max(24, model_size_gb * 2)
        }
        from core.quantization_config import QuantizationConfig
        
        supports_quant = QuantizationConfig.supports_quantization(
            self.model_type, self.model_family, self.model_info
        )
        
        requirements.capabilities = {
            "supports_text_generation": True,
            "supports_image_generation": False,  # Llama 4 doesn't generate images
            "supports_chat": True,
            "supports_cpu_inference": False,  # Multimodal models need GPU
            "supports_quantization": supports_quant,
            "supported_quantization": ["int4", "fp8"] if supports_quant else [],
            "requires_gpu": True
        }
        
        # Determine model variant and context length
        is_scout = 'scout' in self.model_id.lower()
        is_maverick = 'maverick' in self.model_id.lower()
        
        requirements.special_config = {
            "is_llama4_model": True,
            "max_context_length": 10000000 if is_scout else 1000000,  # 10M for Scout, 1M for Maverick
            "supported_gpus": ['nvidia_a100', 'nvidia_a6000', 'nvidia_rtx_4090',
                               'nvidia_rtx_3090', 'nvidia_v100', 'nvidia_h100'],
            "supports_moe": True,  # Mixture of Experts
            "num_experts": 16 if is_scout else 128,
            "supported_languages": [
                "Arabic", "English", "French", "German", "Hindi", 
                "Indonesian", "Italian", "Portuguese", "Spanish", 
                "Tagalog", "Thai", "Vietnamese"
            ],
            "max_images": 5,  # Tested for up to 5 input images
            "knowledge_cutoff": "2024-08"
        }

        return requirements

    def _estimate_model_size(self) -> float:
        """Estimate model size in GB based on Llama 4 variants.

        Returns:
            Estimated size in GB.
        """
        model_id_lower = self.model_id.lower()

        # Scout: 17B activated, 109B total
        if 'scout' in model_id_lower:
            if 'fp8' in model_id_lower:
                return 54.5  # FP8 quantized
            return 109  # BF16
            
        # Maverick: 17B activated, 400B total
        elif 'maverick' in model_id_lower:
            if 'fp8' in model_id_lower:
                return 200  # FP8 quantized
            return 400  # BF16

        # Default estimation based on parameter count
        if '17b' in model_id_lower:
            return 34  # 17B params in BF16
        
        return 50  # Conservative default

    def load_model(
        self,
        model_path: str,
        device: str = "auto",
        dtype: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs
    ) -> Tuple[Any, Any]:
        """Load Llama 4 model and processor.

        Args:
            model_path: Path to model directory.
            device: Device to load model on.
            dtype: Data type for model weights.
            load_in_8bit: Whether to load in 8-bit.
            load_in_4bit: Whether to load in 4-bit.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (model, processor).
        """
        try:
            # Import Llama 4 specific classes
            from transformers import (
                AutoProcessor,
                Llama4ForConditionalGeneration,
                BitsAndBytesConfig
            )
            import torch

            # Get processor
            processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
            
            # Log processor info
            logger.info(f"Processor type: {type(processor)}")

            # Check if model has native dtype preference
            config_path = Path(model_path) / 'config.json'
            native_dtype = None
            if config_path.exists():
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    native_dtype = config.get('torch_dtype', 'bfloat16')
                    logger.info(f"Model native dtype: {native_dtype}")
            
            # Use base handler's quantization config
            quant_config, torch_dtype = self.get_quantization_config(dtype, load_in_8bit, load_in_4bit)
            
            # Llama 4 prefers bfloat16
            if native_dtype == 'bfloat16' and dtype == 'auto':
                torch_dtype = torch.bfloat16
                logger.info("Using bfloat16 as model's native dtype")
            
            # Prepare model loading arguments
            model_kwargs = {
                'pretrained_model_name_or_path': model_path,
                'torch_dtype': torch_dtype,
                'trust_remote_code': True,
                'attn_implementation': 'flex_attention'  # Llama 4 uses flex_attention
            }

            # Merge quantization config
            model_kwargs.update(quant_config)

            # Add device map and memory optimization
            if device == 'auto':
                model_kwargs['device_map'] = 'auto'
                # Add offload settings for better memory management
                model_kwargs['offload_folder'] = 'offload'
                model_kwargs['offload_state_dict'] = True

                # For large models, use sequential device map
                if self._estimate_model_size() > 50:  # 50GB+
                    model_kwargs['device_map'] = 'sequential'
                    # Adjust memory limits based on model variant
                    if 'scout' in self.model_id.lower():
                        model_kwargs['max_memory'] = {0: '40GiB', 'cpu': '100GiB'}
                    else:  # Maverick
                        model_kwargs['max_memory'] = {0: '80GiB', 'cpu': '200GiB'}

            # Load model with low CPU memory usage
            model_kwargs['low_cpu_mem_usage'] = True

            # Load the model
            model = Llama4ForConditionalGeneration.from_pretrained(**model_kwargs)
            logger.info(f"Using attention implementation: {model_kwargs.get('attn_implementation', 'default')}")

            # Move to device if not using device_map='auto'
            if device != 'auto' and not (load_in_8bit or load_in_4bit):
                if device == 'cuda' and not torch.cuda.is_available():
                    device = 'cpu'
                    logger.warning("CUDA not available, falling back to CPU")
                model = model.to(device)

            # Set to eval mode
            model.eval()

            logger.info(f"Successfully loaded Llama 4 model from {model_path}")
            return model, processor

        except ImportError as e:
            logger.error(f"Failed to import required dependencies: {e}")
            logger.info(
                "Please ensure transformers>=4.51.0 is installed for Llama 4 support"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load Llama 4 model: {e}")
            # Fallback to base multimodal handler
            logger.info("Attempting fallback to standard multimodal loading")
            return super().load_model(
                model_path,
                device=device,
                dtype=dtype,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                **kwargs
            )

    def get_inference_params(self) -> Dict[str, Any]:
        """Get default inference parameters for Llama 4.

        Returns:
            Dictionary of inference parameters.
        """
        return {
            'temperature': 0.7,
            'max_new_tokens': 2048,
            'do_sample': True,
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.0,
            'use_cache': True
        }

    def get_model_capabilities(self) -> Dict[str, Any]:
        """Get Llama 4 model capabilities.

        Returns:
            Dictionary of capabilities.
        """
        model_id_lower = self.model_id.lower()

        # Base context length depends on model variant
        if 'scout' in model_id_lower:
            max_context = 10000000  # 10M tokens
        elif 'maverick' in model_id_lower:
            max_context = 1000000  # 1M tokens
        else:
            max_context = 128000  # Default fallback

        capabilities = {
            'supports_streaming': True,
            'supports_reasoning': True,
            'supports_system_prompt': True,
            'supports_multimodal': True,
            'supports_batch_inference': True,
            'max_context_length': max_context,
            'max_output_length': 8192,
            'input_modalities': ['text', 'image'],
            'output_modalities': ['text', 'code'],
            'supported_image_formats': ['JPEG', 'PNG', 'GIF', 'BMP'],
            'max_images_per_request': 5,
            'supports_moe': True,  # Mixture of Experts
            'supports_flex_attention': True
        }

        return capabilities
    
    def get_training_parameters(self) -> Dict[str, Any]:
        """Get Llama 4 specific training parameters.
        
        Returns:
            Dictionary with training parameter overrides.
        """
        params = super().get_training_parameters()
        
        # Llama 4 specific modules for LoRA
        params['lora_target_modules'] = [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        
        # MoE specific settings
        params['moe_training'] = True
        params['expert_selection'] = 'top-2'  # Select top-2 experts
        
        # Llama 4 prefers bfloat16
        params['training_precision'] = 'bf16'
        
        # Context-aware training
        if 'scout' in self.model_id.lower():
            params['max_seq_length'] = 8192  # Safe limit for Scout
        else:
            params['max_seq_length'] = 4096  # Safe limit for Maverick
        
        # Dataset formats - Llama 4 works well with all formats
        params['dataset_formats'] = [
            'alpaca', 'sharegpt', 'openai', 'chat', 
            'completion', 'qa', 'vision_qa'
        ]
        
        # Flash attention not compatible with LoRA + MoE
        params['supports_flash_attention'] = False
        
        return params

    def get_supported_modes(self) -> List[str]:
        """Get supported generation modes.

        Returns:
            List of supported mode names.
        """
        return [
            'auto',
            'chat',
            'multimodal',  # For text+image input, text output
            'code',
            'analyze',
            'creative',
            'reasoning'
        ]

    def get_installation_notes(self) -> str:
        """Get installation notes specific to Llama 4.

        Returns:
            Installation notes as a string.
        """
        notes = super().get_installation_notes()
        notes += """
## Llama 4 Specific Notes:

1. **Model Variants**:
   - Scout: 17B active params, 109B total, 16 experts, 10M context
   - Maverick: 17B active params, 400B total, 128 experts, 1M context

2. **Quantization**:
   - Scout: Supports int4 quantization (fits on single H100)
   - Maverick: Released with FP8 weights option

3. **System Requirements**:
   - Minimum: H100 GPU for Scout with int4
   - Recommended: Multiple H100s for Maverick
   - Flex attention requires latest PyTorch

4. **Supported Languages**:
   Arabic, English, French, German, Hindi, Indonesian, 
   Italian, Portuguese, Spanish, Tagalog, Thai, Vietnamese

5. **Image Support**:
   - Tested up to 5 images per request
   - Automatic image normalization to 896x896

6. **System Prompt**:
   Use provided system prompt template for best results
"""
        return notes

    def validate_model_files(self, model_path: str) -> Tuple[bool, List[str]]:
        """Validate Llama 4 model files.

        Args:
            model_path: Path to model directory.

        Returns:
            Tuple of (is_valid, missing_files).
        """
        required_files = [
            'config.json',
            'tokenizer_config.json',
            'processor_config.json'
        ]

        # Check for either safetensors or bin files
        model_path_obj = Path(model_path)
        has_safetensors = any(model_path_obj.glob('*.safetensors'))
        has_bin = any(model_path_obj.glob('*.bin'))

        if not (has_safetensors or has_bin):
            required_files.append('model.safetensors or pytorch_model.bin')

        missing = []
        for file in required_files:
            if ' or ' in file:
                # Special case for model files
                continue
            if not (model_path_obj / file).exists():
                missing.append(file)

        return len(missing) == 0, missing

    def process_multimodal(
        self,
        model: Any,
        processor: Any,
        text: Optional[str] = None,
        images: Optional[List[Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process multimodal input through Llama 4.

        Args:
            model: The loaded model.
            processor: The loaded processor.
            text: Optional text input.
            images: Optional list of images.
            **kwargs: Additional generation parameters.

        Returns:
            Dictionary with generated text and metadata.
        """
        # Llama 4 follows similar pattern to other multimodal models
        # but with flex_attention and potential MoE considerations
        
        # Log if we're using a quantized model
        if hasattr(model, 'config') and hasattr(model.config, 'quantization_config'):
            logger.info("Using quantized Llama 4 model")
        
        # For Llama 4, we can use the base multimodal processing
        # with the specific attention implementation already set
        return super().process_multimodal(
            model, processor, text, images, **kwargs
        )
"""Handler for multimodal models.

This module provides handlers for multimodal models that combine
multiple input/output modalities such as vision-language models,
including the Deepseek Janus models.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from handlers.base import BaseHandler
from handlers.transformer import TransformerHandler
from core.checker import ModelRequirements

logger = logging.getLogger(__name__)


class MultimodalHandler(BaseHandler):
    """Handler for multimodal models including vision-language types."""

    def __init__(self, model_info: Dict[str, Any]):
        """Initialize multimodal handler.

        Args:
            model_info: Model information dictionary.
        """
        super().__init__(model_info)
        self.is_janus_model = self._check_if_janus_model()

    def _check_if_janus_model(self) -> bool:
        """Check if the model is a Deepseek Janus model.

        Returns:
            True if it's a Janus model, False otherwise.
        """
        model_id = self.model_id.lower()
        model_type = self.model_info.get('model_type', '').lower()

        return (
            'janus' in model_id or
            'deepseek' in model_id and 'janus' in model_id or
            model_type == 'multi_modality'
        )

    def get_dependencies(self) -> List[str]:
        """Get required Python dependencies.

        Returns:
            List of pip package specifications.
        """
        base_deps = [
            'torch>=2.6.0',
            'transformers>=4.36.0',
            'Pillow>=9.0.0',
            'numpy',
            'einops',
            'sentencepiece',
            'protobuf',
            'accelerate>=0.20.0'
        ]

        if self.is_janus_model:
            # Add Janus dependency for Janus models
            base_deps.append('git+https://github.com/deepseek-ai/Janus.git')

        # Check for specific vision encoders
        config = self.model_info.get('config', {})

        # Add vision-specific dependencies
        if 'clip' in str(config).lower() or 'siglip' in str(config).lower():
            base_deps.extend([
                'ftfy',
                'regex',
                'tqdm'
            ])

        # Add video dependencies if needed
        if 'video' in self.model_type.lower():
            base_deps.extend([
                'opencv-python',
                'av',
                'decord'
            ])

        return base_deps

    def get_system_dependencies(self) -> List[str]:
        """Get required system dependencies.

        Returns:
            List of system requirements.
        """
        deps = []

        # CUDA support for GPU acceleration
        if self.model_info.get('requires_gpu', True):
            deps.append('cuda>=11.7')
        
        # Git is required for Janus models
        if self.is_janus_model:
            deps.append('git')

        # Video processing system deps
        if 'video' in self.model_type.lower():
            deps.extend(['ffmpeg', 'libavcodec-dev', 'libavformat-dev'])

        return deps
    
    def get_installation_notes(self) -> Dict[str, str]:
        """Get special installation instructions for dependencies.
        
        Returns:
            Dictionary mapping dependency names to installation instructions.
        """
        notes = {}
        
        if self.is_janus_model:
            notes['git+https://github.com/deepseek-ai/Janus.git'] = """
Janus is required for Deepseek/FreedomIntelligence Janus multimodal models.

This package provides:
- VLChatProcessor for multimodal processing
- MultiModalityCausalLM model architecture
- Special handling for image generation with CFG

If installation fails:
1. Ensure git is installed: sudo apt-get install git (Ubuntu) or brew install git (macOS)
2. Try manual installation:
   cd {model_dir}
   source .venv/bin/activate
   pip install git+https://github.com/deepseek-ai/Janus.git

Note: This is a large package and may take some time to install.
"""
        
        return notes

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
        requirements.model_family = self.model_family
        requirements.primary_library = "transformers"
        requirements.base_dependencies = self.get_dependencies()
        requirements.special_dependencies = ['janus @ git+https://github.com/deepseek-ai/Janus.git'] if self.is_janus_model else []
        requirements.optional_dependencies = []
        requirements.disk_space_gb = model_size_gb * 3
        requirements.memory_requirements = {
            "min": max(16, model_size_gb * 2),
            "recommended": max(24, model_size_gb * 3),
            "gpu_min": max(12, model_size_gb * 1.5),
            "gpu_recommended": max(16, model_size_gb * 2)
        }
        requirements.capabilities = {
            "supports_text_generation": True,
            "supports_image_generation": True,
            "supports_chat": True,
            "supports_cpu_inference": False,
            "supports_quantization": True,
            "supported_quantization": ["int8", "int4"],
            "requires_gpu": True
        }
        requirements.special_config = {
            "is_janus_model": self.is_janus_model,
            "supported_gpus": ['nvidia_a100', 'nvidia_a6000', 'nvidia_rtx_4090',
                              'nvidia_rtx_3090', 'nvidia_v100']
        }

        return requirements

    def _estimate_model_size(self) -> float:
        """Estimate model size in GB.

        Returns:
            Estimated size in GB.
        """
        # Check if we have actual size info
        if 'model_size_gb' in self.model_info:
            return self.model_info['model_size_gb']

        # Estimate based on parameter count
        param_count = self.model_info.get('parameter_count', 0)
        if param_count > 0:
            # Rough estimation: 2 bytes per parameter for fp16
            return (param_count * 2) / (1024 ** 3)

        # Default estimates based on model name
        model_id_lower = self.model_id.lower()
        if '7b' in model_id_lower:
            return 14.0
        elif '1b' in model_id_lower:
            return 2.0
        elif '13b' in model_id_lower:
            return 26.0
        else:
            return 10.0  # Conservative default

    def load_model(
        self,
        model_path: str,
        device: str = 'auto',
        dtype: str = 'auto',
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs
    ) -> Tuple[Any, Any]:
        """Load multimodal model with appropriate configuration.

        Args:
            model_path: Path to model files.
            device: Device to load on.
            dtype: Data type for model.
            load_in_8bit: Whether to use 8-bit quantization.
            load_in_4bit: Whether to use 4-bit quantization.
            **kwargs: Additional loading arguments.

        Returns:
            Tuple of (model, processor/tokenizer).
        """
        if self.is_janus_model:
            return self._load_janus_model(
                model_path, device, dtype,
                load_in_8bit, load_in_4bit, **kwargs
            )

        # For other multimodal models, try standard transformers loading
        return self._load_transformers_multimodal(
            model_path, device, dtype,
            load_in_8bit, load_in_4bit, **kwargs
        )

    def _load_janus_model(
        self,
        model_path: str,
        device: str,
        dtype: str,
        load_in_8bit: bool,
        load_in_4bit: bool,
        **kwargs
    ) -> Tuple[Any, Any]:
        """Load Deepseek Janus model specifically.

        Args:
            model_path: Path to model files.
            device: Device to load on.
            dtype: Data type.
            load_in_8bit: 8-bit quantization.
            load_in_4bit: 4-bit quantization.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (model, processor).
        """
        try:
            # Try importing Janus-specific classes
            from janus.models import VLChatProcessor
            from transformers import AutoModelForCausalLM
            import torch

            # Load processor
            processor = VLChatProcessor.from_pretrained(model_path)

            # Determine torch dtype
            if dtype == 'auto':
                torch_dtype = (
                    torch.bfloat16 if torch.cuda.is_available()
                    else torch.float32
                )
            else:
                dtype_map = {
                    'float32': torch.float32,
                    'float16': torch.float16,
                    'bfloat16': torch.bfloat16
                }
                torch_dtype = dtype_map.get(dtype, torch.float32)

            # Load model with trust_remote_code
            model_kwargs = {
                'pretrained_model_name_or_path': model_path,
                'trust_remote_code': True,
                'torch_dtype': torch_dtype
            }

            # Add quantization config if needed
            if load_in_8bit or load_in_4bit:
                from transformers import BitsAndBytesConfig
                model_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_8bit=load_in_8bit,
                    load_in_4bit=load_in_4bit,
                    bnb_4bit_compute_dtype=(
                        torch_dtype if load_in_4bit else None
                    )
                )

            # Load model - try to use the Janus-specific model class if available
            try:
                # For Janus models, we might need to use their specific model class
                from janus.models import MultiModalityCausalLM
                model = MultiModalityCausalLM.from_pretrained(**model_kwargs)
            except ImportError:
                # Fallback to AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

            # Move to device if not quantized
            if not (load_in_8bit or load_in_4bit):
                if device == 'auto':
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model = model.to(device)

            # Set to eval mode
            model.eval()

            logger.info(f"Successfully loaded Janus model from {model_path}")
            return model, processor

        except ImportError as e:
            logger.error(f"Failed to import Janus dependencies: {e}")
            logger.info(
                "Please install the Janus package: "
                "pip install git+https://github.com/deepseek-ai/Janus.git"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load Janus model: {e}")
            raise

    def _load_transformers_multimodal(
        self,
        model_path: str,
        device: str,
        dtype: str,
        load_in_8bit: bool,
        load_in_4bit: bool,
        **kwargs
    ) -> Tuple[Any, Any]:
        """Load general multimodal model using transformers.

        Args:
            model_path: Path to model files.
            device: Device to load on.
            dtype: Data type.
            load_in_8bit: 8-bit quantization.
            load_in_4bit: 4-bit quantization.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (model, processor).
        """
        # Use transformer handler as fallback for standard multimodal models
        transformer_handler = TransformerHandler(self.model_info)
        return transformer_handler.load_model(
            model_path, device, dtype,
            load_in_8bit, load_in_4bit, **kwargs
        )

    def get_inference_params(self) -> Dict[str, Any]:
        """Get default inference parameters.

        Returns:
            Dictionary of inference parameters.
        """
        params = {
            'temperature': 0.7,
            'max_new_tokens': 1024,
            'do_sample': True,
            'top_p': 0.95,
            'repetition_penalty': 1.1
        }

        if self.is_janus_model:
            params.update({
                'max_new_tokens': 4096,  # Janus supports longer generation
                'image_size': 384,  # Janus uses 384x384 images
            })

        return params

    def get_training_params(self) -> Dict[str, Any]:
        """Get default training parameters.

        Returns:
            Dictionary of training parameters.
        """
        return {
            'learning_rate': 2e-5,
            'per_device_train_batch_size': 1,
            'gradient_accumulation_steps': 16,
            'warmup_steps': 100,
            'num_train_epochs': 3,
            'fp16': True,
            'gradient_checkpointing': True,
            'optim': 'adamw_torch',
            'lr_scheduler_type': 'cosine'
        }

    def validate_model_files(
        self, model_path: str
    ) -> Tuple[bool, Optional[str]]:
        """Validate required model files exist.

        Args:
            model_path: Path to model directory.

        Returns:
            Tuple of (is_valid, error_message).
        """
        required_files = ['config.json']
        model_path_obj = Path(model_path)

        # Check for basic files
        for file in required_files:
            if not (model_path_obj / file).exists():
                return False, f"Missing required file: {file}"

        # Check for model weights
        has_weights = any(
            model_path_obj.glob(pattern)
            for pattern in ['*.bin', '*.safetensors', '*.pth', '*.pt']
        )

        if not has_weights:
            return False, "No model weight files found"

        # For Janus models, check for processor config
        if self.is_janus_model:
            processor_files = [
                'preprocessor_config.json', 'tokenizer_config.json'
            ]
            missing_processor = []
            for file in processor_files:
                if not (model_path_obj / file).exists():
                    missing_processor.append(file)

            if missing_processor:
                logger.warning(
                    f"Missing processor files for Janus model: "
                    f"{missing_processor}"
                )

        return True, None

    def get_model_capabilities(self) -> Dict[str, Any]:
        """Get model capabilities.

        Returns:
            Dictionary of capabilities.
        """
        capabilities = {
            'supports_streaming': True,
            'supports_reasoning': False,  # Janus doesn't support o1-style reasoning
            # Note: Janus Pro models use role names with angle brackets: <|User|> and <|Assistant|>
            'supports_system_prompt': True,
            'supports_multimodal': True,
            'supports_batch_inference': True,
            'max_context_length': 4096,
            'input_modalities': ['text', 'image'],
            'output_modalities': ['text']
        }

        if self.is_janus_model:
            capabilities.update({
                'max_context_length': 32768,  # Janus supports longer context
                # Janus can generate images
                'output_modalities': ['text', 'image'],
                'supports_image_generation': True,
                'image_resolution': 384,
                # Janus Pro models use specific role format
                'role_format': {
                    'user': '<|User|>',
                    'assistant': '<|Assistant|>',
                    'note': 'Janus Pro models require angle brackets in role names'
                }
            })

        # Add video support if applicable
        if 'video' in self.model_type.lower():
            capabilities['input_modalities'].append('video')

        return capabilities

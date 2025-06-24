"""Handler for Gemma 3 multimodal models.

This module provides specialized handling for Google's Gemma 3 multimodal models
that support both text and image inputs with text output.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

from handlers.multimodal import MultimodalHandler
from core.checker import ModelRequirements

logger = logging.getLogger(__name__)


class Gemma3Handler(MultimodalHandler):
    """Handler for Gemma 3 multimodal models."""

    def __init__(self, model_info: Dict[str, Any]):
        """Initialize Gemma 3 handler.

        Args:
            model_info: Model information dictionary.
        """
        super().__init__(model_info)
        self.is_gemma3_model = self._check_if_gemma3_model()

    def _check_if_gemma3_model(self) -> bool:
        """Check if the model is a Gemma 3 multimodal model.

        Returns:
            True if it's a Gemma 3 multimodal model, False otherwise.
        """
        model_id = self.model_id.lower()
        model_type = self.model_info.get('model_type', '').lower()
        config = self.model_info.get('config', {})

        # Check for Gemma 3 models (multimodal variants)
        is_gemma3 = (
            'gemma-3' in model_id or
            'gemma3' in model_id or
            (('gemma' in model_id) and ('3-' in model_id or '3_' in model_id))
        )

        # Check if it's multimodal (has vision capabilities)
        has_vision = (
            'vision' in str(config).lower() or
            'image' in str(config).lower() or
            'multimodal' in model_type or
            'vlm' in model_type or
            hasattr(config, 'vision_config')
        )

        return is_gemma3 and has_vision

    def get_dependencies(self) -> List[str]:
        """Get required Python dependencies.

        Returns:
            List of pip package specifications.
        """
        base_deps = [
            'torch>=2.6.0',
            'transformers>=4.50.0',  # Gemma 3 requires latest transformers
            'Pillow>=9.0.0',
            'numpy',
            'einops',
            'sentencepiece',
            'protobuf',
            'accelerate>=0.20.0',
            'safetensors>=0.4.0',
            'tokenizers>=0.15.0'
        ]

        # Add quantization support
        if self.model_info.get('supports_quantization', True):
            base_deps.append('bitsandbytes>=0.41.0')

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
        requirements.model_family = "gemma3"
        requirements.primary_library = "transformers"
        requirements.base_dependencies = self.get_dependencies()
        requirements.special_dependencies = []
        requirements.optional_dependencies = ['flash-attn>=2.0.0']
        requirements.disk_space_gb = model_size_gb * 3
        requirements.memory_requirements = {
            "min": max(16, model_size_gb * 2),
            "recommended": max(24, model_size_gb * 3),
            "gpu_min": max(12, model_size_gb * 1.5),
            "gpu_recommended": max(16, model_size_gb * 2)
        }
        requirements.capabilities = {
            "supports_text_generation": True,
            "supports_image_generation": False,  # Gemma 3 doesn't generate images
            "supports_chat": True,
            "supports_cpu_inference": False,  # Multimodal models need GPU
            "supports_quantization": True,
            "supported_quantization": ["int8", "int4"],
            "requires_gpu": True
        }
        requirements.special_config = {
            "is_gemma3_model": True,
            "max_context_length": 128000,  # 128K context for 4B, 12B, 27B
            "supported_gpus": ['nvidia_a100', 'nvidia_a6000', 'nvidia_rtx_4090',
                               'nvidia_rtx_3090', 'nvidia_v100', 'nvidia_h100']
        }

        # Adjust for 1B model
        if '1b' in self.model_id.lower():
            requirements.special_config['max_context_length'] = 32000  # 32K for 1B

        return requirements

    def _estimate_model_size(self) -> float:
        """Estimate model size in GB based on Gemma 3 variants.

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

        # Default estimates based on Gemma 3 model sizes
        model_id_lower = self.model_id.lower()
        if '27b' in model_id_lower:
            return 54.0  # 27B model
        elif '12b' in model_id_lower:
            return 24.0  # 12B model
        elif '4b' in model_id_lower:
            return 8.0   # 4B model
        elif '1b' in model_id_lower:
            return 2.0   # 1B model
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
        """Load Gemma 3 multimodal model with appropriate configuration.

        Args:
            model_path: Path to model files.
            device: Device to load on.
            dtype: Data type for model.
            load_in_8bit: Whether to use 8-bit quantization.
            load_in_4bit: Whether to use 4-bit quantization.
            **kwargs: Additional loading arguments.

        Returns:
            Tuple of (model, processor).
        """
        try:
            from transformers import (
                AutoProcessor,
                Gemma3ForConditionalGeneration,
                BitsAndBytesConfig
            )
            import torch

            # Load processor with use_fast=True to avoid warning
            processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

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

            # Prepare model loading arguments
            model_kwargs = {
                'pretrained_model_name_or_path': model_path,
                'torch_dtype': torch_dtype,
                'trust_remote_code': True  # Gemma 3 may require this
            }

            # Add quantization config if needed
            if load_in_8bit or load_in_4bit:
                model_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_8bit=load_in_8bit,
                    load_in_4bit=load_in_4bit,
                    bnb_4bit_compute_dtype=torch_dtype if load_in_4bit else None,
                    bnb_4bit_use_double_quant=True if load_in_4bit else None
                )

            # Add device map and memory optimization
            if device == 'auto':
                model_kwargs['device_map'] = 'auto'
                # Add offload settings for better memory management
                model_kwargs['offload_folder'] = 'offload'
                model_kwargs['offload_state_dict'] = True

                # For large models, use sequential device map
                if self._estimate_model_size() > 20:  # 20GB+
                    model_kwargs['device_map'] = 'sequential'
                    model_kwargs['max_memory'] = {0: '20GiB', 'cpu': '100GiB'}

            # Load model with low CPU memory usage
            model_kwargs['low_cpu_mem_usage'] = True

            # Try to use Flash Attention 2 if available
            try:
                model_kwargs['attn_implementation'] = 'flash_attention_2'
                model = Gemma3ForConditionalGeneration.from_pretrained(**model_kwargs)
                logger.info("Using Flash Attention 2 for better performance")
            except Exception:
                # Fallback to standard attention
                model_kwargs.pop('attn_implementation', None)
                model = Gemma3ForConditionalGeneration.from_pretrained(**model_kwargs)
                logger.info("Using standard attention implementation")

            # Move to device if not using device_map='auto'
            if device != 'auto' and not (load_in_8bit or load_in_4bit):
                if device == 'cuda' and not torch.cuda.is_available():
                    device = 'cpu'
                    logger.warning("CUDA not available, falling back to CPU")
                model = model.to(device)

            # Set to eval mode
            model.eval()

            logger.info(f"Successfully loaded Gemma 3 model from {model_path}")
            return model, processor

        except ImportError as e:
            logger.error(f"Failed to import required dependencies: {e}")
            logger.info(
                "Please ensure transformers>=4.50.0 is installed for Gemma 3 support"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load Gemma 3 model: {e}")
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
        """Get default inference parameters for Gemma 3.

        Returns:
            Dictionary of inference parameters.
        """
        return {
            'temperature': 0.7,
            'max_new_tokens': 2048,
            'do_sample': True,
            'top_p': 0.95,
            'top_k': 40,
            'repetition_penalty': 1.1,
            'use_cache': True
        }

    def get_model_capabilities(self) -> Dict[str, Any]:
        """Get Gemma 3 model capabilities.

        Returns:
            Dictionary of capabilities.
        """
        model_id_lower = self.model_id.lower()

        # Base context length depends on model size
        if '1b' in model_id_lower:
            max_context = 32768  # 32K for 1B
        else:
            max_context = 131072  # 128K for 4B, 12B, 27B

        capabilities = {
            'supports_streaming': True,
            'supports_reasoning': False,
            'supports_system_prompt': True,
            'supports_multimodal': True,
            'supports_batch_inference': True,
            'max_context_length': max_context,
            'max_output_length': 8192,
            'input_modalities': ['text', 'image'],
            'output_modalities': ['text'],
            'supports_multiple_images': True,  # Can handle multiple images
            'image_resolution': 896,  # Images normalized to 896x896
            'image_tokens_per_image': 256,  # Each image encoded to 256 tokens
            'supports_multilingual': True,
            'supported_languages': 140,  # Supports over 140 languages
            'chat_template': 'gemma3',  # Uses specific Gemma 3 chat template
            'special_features': [
                'Long context window (up to 128K tokens)',
                'Multilingual support (140+ languages)',
                'Efficient image encoding (256 tokens per image)',
                'Optimized for both single and multi-turn conversations'
            ]
        }

        return capabilities

    def get_supported_modes(self) -> List[str]:
        """Get supported generation modes for Gemma 3."""
        return ['auto', 'chat', 'vision', 'multimodal', 'analyze']

    def get_mode_descriptions(self) -> Dict[str, str]:
        """Get descriptions for supported modes."""
        return {
            'auto': 'Automatic mode selection',
            'chat': 'Text conversation mode',
            'vision': 'Image understanding and analysis',
            'multimodal': 'Combined text and image processing',
            'analyze': 'Detailed image analysis with reasoning'
        }

    def generate_text(self, prompt: str, model=None, tokenizer=None,
                      processor=None, **kwargs) -> Dict[str, Any]:
        """Generate text from a text-only prompt.

        Args:
            prompt: Text prompt
            model: The loaded model
            tokenizer: The loaded tokenizer (or processor)
            processor: The loaded processor (preferred over tokenizer)
            **kwargs: Additional generation arguments

        Returns:
            Dictionary with generated text and usage statistics
        """
        # For text-only generation, delegate to process_multimodal without images
        return self.process_multimodal(
            text=prompt,
            images=None,
            model=model,
            processor=processor or tokenizer,
            **kwargs
        )

    def process_multimodal(self, text: str = None, images: List[Union[str, Any]] = None,
                           audio: str = None, video: str = None,
                           model=None, processor=None, **kwargs) -> Dict[str, Any]:
        """Process multimodal inputs for Gemma 3.

        Args:
            text: Input text
            images: List of images (base64 strings or PIL Images)
            audio: Not supported by Gemma 3
            video: Not supported by Gemma 3
            model: The loaded model
            processor: The loaded processor
            **kwargs: Additional generation arguments

        Returns:
            Dictionary with generated text and usage statistics
        """
        if not model or not processor:
            raise ValueError("Model and processor required for multimodal processing")

        if audio or video:
            logger.warning("Gemma 3 does not support audio or video inputs")

        try:
            import torch
            from PIL import Image
            import base64
            from io import BytesIO

            # Clear GPU cache for memory efficiency
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Enable memory efficient attention if available
            if hasattr(model, 'config'):
                model.config.use_cache = True
                if hasattr(model.config, 'use_memory_efficient_attention'):
                    model.config.use_memory_efficient_attention = True

            # Process images if provided
            pil_images = []
            if images:
                for img in images:
                    if isinstance(img, str):
                        # Decode base64
                        img_data = base64.b64decode(img)
                        pil_images.append(Image.open(BytesIO(img_data)))
                    elif isinstance(img, Image.Image):
                        pil_images.append(img)

            # Prepare messages in Gemma 3 format
            messages = kwargs.get('messages', [])

            if not messages:
                # Build messages from text/images if not provided
                messages = []

                # Add system message if needed
                if kwargs.get('system_prompt'):
                    messages.append({
                        "role": "system",
                        "content": kwargs['system_prompt']
                    })

                # Build user message
                if text:
                    messages.append({
                        "role": "user",
                        "content": text
                    })
            else:
                # Convert messages to simple format if needed
                simple_messages = []
                for msg in messages:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    
                    # Handle different content formats
                    if isinstance(content, str):
                        simple_messages.append({
                            "role": role,
                            "content": content
                        })
                    elif isinstance(content, list):
                        # Extract text from structured content
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                text_parts.append(item.get('text', ''))
                            elif isinstance(item, str):
                                text_parts.append(item)
                        simple_messages.append({
                            "role": role,
                            "content": ' '.join(text_parts)
                        })
                    else:
                        simple_messages.append({
                            "role": role,
                            "content": str(content)
                        })
                messages = simple_messages

            # Handle multimodal input with images
            if pil_images:
                # For multimodal input, we need to prepare it differently
                # Gemma 3 expects images to be processed separately
                inputs = processor(
                    text=messages,
                    images=pil_images,
                    return_tensors="pt",
                    padding=True
                )
            else:
                # Text-only input - apply chat template
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                )

            # Move to model device
            if hasattr(model, 'device'):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Prepare generation parameters
            generation_params = {
                'max_new_tokens': min(kwargs.get('max_tokens', 2048), 8192),
                'temperature': kwargs.get('temperature', 0.7),
                'top_p': kwargs.get('top_p', 0.95),
                'top_k': kwargs.get('top_k', 40),
                'do_sample': kwargs.get('temperature', 0.7) > 0,
                'use_cache': True,
                'pad_token_id': processor.tokenizer.pad_token_id,
                'eos_token_id': processor.tokenizer.eos_token_id
            }

            # Generate
            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_params)

            # Decode response
            input_length = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
            generated_tokens = outputs[0][input_length:]
            response = processor.decode(generated_tokens, skip_special_tokens=True)

            # Calculate usage
            usage = {
                'prompt_tokens': input_length,
                'completion_tokens': len(generated_tokens),
                'total_tokens': outputs.shape[1]
            }

            # Clean up to free memory
            del outputs
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {
                'text': response.strip(),
                'usage': usage,
                'model': 'gemma-3-multimodal'
            }

        except Exception as e:
            logger.error(f"Error in Gemma 3 multimodal processing: {e}")
            # Fallback to parent implementation
            return super().process_multimodal(
                text, images, audio, video,
                model, processor, **kwargs
            )

    def validate_model_files(self, model_path: str) -> Tuple[bool, Optional[str]]:
        """Validate required model files exist for Gemma 3.

        Args:
            model_path: Path to model directory.

        Returns:
            Tuple of (is_valid, error_message).
        """
        required_files = [
            'config.json',
            'tokenizer_config.json',
            'tokenizer.json'  # Gemma 3 uses fast tokenizer
        ]

        model_path_obj = Path(model_path)

        # Check for basic files
        for file in required_files:
            if not (model_path_obj / file).exists():
                return False, f"Missing required file: {file}"

        # Check for model weights (Gemma 3 typically uses safetensors)
        has_weights = any(
            model_path_obj.glob(pattern)
            for pattern in ['*.safetensors', '*.bin', '*.pth', '*.pt']
        )

        if not has_weights:
            return False, "No model weight files found"

        # Check for processor config (important for multimodal)
        processor_files = ['preprocessor_config.json']
        missing_processor = []
        for file in processor_files:
            if not (model_path_obj / file).exists():
                missing_processor.append(file)

        if missing_processor:
            logger.warning(
                f"Missing processor files for Gemma 3 model: {missing_processor}"
            )

        return True, None

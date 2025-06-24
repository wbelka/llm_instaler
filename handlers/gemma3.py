"""Handler for Gemma 3 multimodal models.

This module provides specialized handling for Google's Gemma 3 multimodal models
that support both text and image inputs with text output.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

from handlers.multimodal import MultimodalHandler
from core.checker import ModelRequirements
from core.quantization_config import QuantizationConfig

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
        requirements.model_family = "gemma3"
        requirements.primary_library = "transformers"
        requirements.base_dependencies = self.get_dependencies()
        requirements.special_dependencies = []
        requirements.optional_dependencies = ['flash-attn==2.7.2.post1']
        requirements.disk_space_gb = model_size_gb * 3
        requirements.memory_requirements = {
            "min": max(16, model_size_gb * 2),
            "recommended": max(24, model_size_gb * 3),
            "gpu_min": max(12, model_size_gb * 1.5),
            "gpu_recommended": max(16, model_size_gb * 2)
        }
        from core.quantization_config import QuantizationConfig
        
        supports_quant = QuantizationConfig.supports_quantization(
            self.model_type, self.model_family, self.model_info
        )
        
        requirements.capabilities = {
            "supports_text_generation": True,
            "supports_image_generation": False,  # Gemma 3 doesn't generate images
            "supports_chat": True,
            "supports_cpu_inference": False,  # Multimodal models need GPU
            "supports_quantization": supports_quant,
            "supported_quantization": ["int8", "int4"] if supports_quant else [],
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
            
            # Log processor info
            logger.info(f"Processor type: {type(processor)}")
            if hasattr(processor, 'image_token'):
                logger.info(f"Processor image token: {processor.image_token}")
            if hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'image_token'):
                logger.info(f"Tokenizer image token: {processor.tokenizer.image_token}")

            # Check if model has native dtype preference
            config_path = Path(model_path) / 'config.json'
            native_dtype = None
            if config_path.exists():
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    native_dtype = config.get('torch_dtype', None)
                    if native_dtype:
                        logger.info(f"Model native dtype: {native_dtype}")
            
            # Use base handler's quantization config
            quant_config, torch_dtype = self.get_quantization_config(dtype, load_in_8bit, load_in_4bit)
            
            # Override with native dtype for quantization if available and compatible
            special_config = QuantizationConfig.get_special_config(self.model_family)
            preferred_dtype = special_config.get('preferred_compute_dtype', native_dtype)
            
            if preferred_dtype == 'bfloat16' and (load_in_8bit or load_in_4bit):
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    logger.info(f"Using {preferred_dtype} for quantization compute dtype")
                    torch_dtype = torch.bfloat16
                    if 'quantization_config' in quant_config:
                        quant_config['quantization_config'].bnb_4bit_compute_dtype = torch.bfloat16
            
            # Log quantization config for debugging
            if 'quantization_config' in quant_config:
                logger.info(f"Quantization config: {quant_config['quantization_config']}")

            # Prepare model loading arguments
            model_kwargs = {
                'pretrained_model_name_or_path': model_path,
                'torch_dtype': torch_dtype,
                'trust_remote_code': True  # Gemma 3 may require this
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
                if self._estimate_model_size() > 20:  # 20GB+
                    model_kwargs['device_map'] = 'sequential'
                    model_kwargs['max_memory'] = {0: '20GiB', 'cpu': '100GiB'}

            # Load model with low CPU memory usage
            model_kwargs['low_cpu_mem_usage'] = True

            # Try to use Flash Attention 2 if available (but not with quantization)
            if not (load_in_8bit or load_in_4bit):
                try:
                    model_kwargs['attn_implementation'] = 'flash_attention_2'
                    model = Gemma3ForConditionalGeneration.from_pretrained(**model_kwargs)
                    logger.info("Using Flash Attention 2 for better performance")
                except Exception:
                    # Fallback to standard attention
                    model_kwargs.pop('attn_implementation', None)
                    model = Gemma3ForConditionalGeneration.from_pretrained(**model_kwargs)
                    logger.info("Using standard attention implementation")
            else:
                # For quantized models, use standard attention
                model = Gemma3ForConditionalGeneration.from_pretrained(**model_kwargs)
                logger.info("Using standard attention implementation (quantized model)")

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

        # Debug logging of incoming data
        logger.info("=== DEBUGGING INCOMING DATA ===")
        logger.info(f"Text input: {text}")
        logger.info(f"Images provided: {len(images) if images else 0}")
        logger.info(f"Mode: {kwargs.get('mode', 'not specified')}")
        logger.info(f"Messages in kwargs: {'messages' in kwargs}")
        if 'messages' in kwargs:
            messages = kwargs.get('messages', [])
            logger.info(f"Number of messages: {len(messages)}")
            for i, msg in enumerate(messages[:3]):  # Log first 3 messages
                logger.info(f"Message {i}: role={msg.get('role')}, content type={type(msg.get('content'))}")
                if isinstance(msg.get('content'), str):
                    logger.info(f"  Content (first 100 chars): {msg.get('content')[:100]}")
                elif isinstance(msg.get('content'), list):
                    logger.info(f"  Content list length: {len(msg.get('content'))}")
        logger.info("===============================")

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
            
            # Also check for images in message content
            if not pil_images and kwargs.get('messages'):
                for msg in kwargs.get('messages', []):
                    content = msg.get('content', '')
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'image':
                                # Handle image in message content
                                if 'image' in item:
                                    img_data = item['image']
                                    if isinstance(img_data, str):
                                        try:
                                            # Check if it's base64 or URL
                                            if img_data.startswith('http'):
                                                # It's a URL - download it
                                                import requests
                                                response = requests.get(img_data)
                                                pil_images.append(Image.open(BytesIO(response.content)))
                                            else:
                                                # Assume base64
                                                img_bytes = base64.b64decode(img_data)
                                                pil_images.append(Image.open(BytesIO(img_bytes)))
                                        except Exception as e:
                                            logger.warning(f"Failed to process image: {e}")
                                    elif isinstance(img_data, Image.Image):
                                        pil_images.append(img_data)
                                elif 'url' in item:
                                    # Handle URL format
                                    try:
                                        import requests
                                        response = requests.get(item['url'])
                                        pil_images.append(Image.open(BytesIO(response.content)))
                                    except Exception as e:
                                        logger.warning(f"Failed to download image from URL: {e}")

            # Check if we should use simple prompt mode
            use_simple_prompt = kwargs.get('use_simple_prompt', False) or (not kwargs.get('messages') and text and not pil_images)
            
            if use_simple_prompt and text:
                logger.info("Using simple prompt mode (no chat template)")
                # Simple tokenization without chat template
                try:
                    tokenizer = getattr(processor, 'tokenizer', processor)
                    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=8192)
                    logger.info(f"Simple tokenization successful, shape: {inputs['input_ids'].shape}")
                except Exception as e:
                    logger.error(f"Simple tokenization failed: {e}")
                    raise
            elif not use_simple_prompt:
                # Prepare messages in Gemma 3 format
                messages = kwargs.get('messages', [])
                
                # Debug: log incoming messages format
                logger.debug(f"Incoming messages: {messages}")

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
                    if text or pil_images:
                        # For vision mode with images, use structured content format
                        if pil_images:
                            content = []
                            # Add images first
                            for _ in pil_images:
                                content.append({"type": "image"})
                            # Add text
                            if not text:
                                text = "What is in this image?"
                            content.append({"type": "text", "text": text})
                            
                            messages.append({
                                "role": "user",
                                "content": content
                            })
                        else:
                            # Text only
                            messages.append({
                                "role": "user",
                                "content": text
                            })
                else:
                    # For text-only chat, ensure messages are in the correct format
                    # Gemma 3 expects content to be a list of dicts with 'type' and 'text' keys
                    processed_messages = []
                    for msg in messages:
                        role = msg.get('role', 'user')
                        content = msg.get('content', '')
                        
                        # Convert content to expected format
                        if isinstance(content, str):
                            # Convert string to structured format
                            processed_messages.append({
                                "role": role,
                                "content": [{"type": "text", "text": content}]
                            })
                        elif isinstance(content, list):
                            # Ensure list items are in correct format
                            processed_content = []
                            for item in content:
                                if isinstance(item, dict) and 'type' in item:
                                    if item['type'] == 'text':
                                        processed_content.append({
                                            'type': 'text',
                                            'text': item.get('text', '')
                                        })
                                    # Skip non-text items in text-only mode
                                elif isinstance(item, str):
                                    processed_content.append({
                                        'type': 'text',
                                        'text': item
                                    })
                            
                            if processed_content:
                                processed_messages.append({
                                    "role": role,
                                    "content": processed_content
                                })
                        else:
                            # Fallback - convert to text
                            processed_messages.append({
                                "role": role,
                                "content": [{"type": "text", "text": str(content)}]
                            })
                    messages = processed_messages

                # Handle multimodal input with images
                if pil_images:
                    # For multimodal input, we need to prepare it differently
                    # Convert messages to a text prompt for the processor
                    if isinstance(messages, list):
                        # Extract text from messages
                        text_parts = []
                        for msg in messages:
                            role = msg.get('role', 'user')
                            content = msg.get('content', '')
                            if isinstance(content, str):
                                if role != 'system' or content:  # Include system messages only if they have content
                                    text_parts.append(f"{role}: {content}")
                            elif isinstance(content, list):
                                # Extract text from structured content
                                for item in content:
                                    if isinstance(item, dict) and item.get('type') == 'text':
                                        text_parts.append(f"{role}: {item.get('text', '')}")
                        
                        # Join into a single prompt
                        text_prompt = "\n".join(text_parts) if text_parts else "What is in this image?"
                        if text_prompt and not text_prompt.endswith(":"):
                            text_prompt += "\nAssistant:"
                    else:
                        text_prompt = str(messages)
                
                    # Process with images
                    logger.info(f"Processing multimodal input with {len(pil_images)} images")
                    
                    # For Gemma3, we need to format messages properly with images
                    # Convert text prompt back to messages format for proper processing
                    if isinstance(messages, list) and messages:
                        # Use the original messages format with apply_chat_template
                        try:
                            logger.info("Using apply_chat_template for multimodal input")
                            
                            # Ensure messages have proper format
                            formatted_messages = []
                            for msg in messages:
                                role = msg.get('role', 'user')
                                content = msg.get('content')
                                
                                # For user message with images, format properly
                                if role == 'user' and pil_images:
                                    # Create content with image placeholder
                                    formatted_messages.append({
                                        'role': role,
                                        'content': content  # Processor will handle image token insertion
                                    })
                                else:
                                    formatted_messages.append(msg)
                            
                            # Apply chat template and process with images together
                            # For Gemma3, we need to pass both messages and images to processor
                            try:
                                # First approach: apply_chat_template with tokenization and images
                                inputs = processor.apply_chat_template(
                                    formatted_messages,
                                    images=pil_images,
                                    add_generation_prompt=True,
                                    tokenize=True,
                                    return_dict=True,
                                    return_tensors="pt"
                                )
                                logger.info("Successfully processed with apply_chat_template and images")
                            except Exception as e1:
                                logger.warning(f"apply_chat_template with images failed: {e1}")
                                # Second approach: get text template then process with images
                                try:
                                    text = processor.apply_chat_template(
                                        formatted_messages, 
                                        tokenize=False, 
                                        add_generation_prompt=True
                                    )
                                    logger.debug(f"Chat template output: {text}")
                                    
                                    inputs = processor(
                                        text=text,
                                        images=pil_images,
                                        return_tensors="pt",
                                        padding=True
                                    )
                                except TypeError as te:
                                    # Third approach: positional arguments
                                    logger.warning(f"Standard processing failed: {te}")
                                    inputs = processor(
                                        text,
                                        pil_images,
                                        return_tensors="pt",
                                        padding=True
                                    )
                            logger.info("Successfully processed multimodal input with chat template")
                        except Exception as e:
                            logger.warning(f"Chat template failed, using fallback: {e}")
                            # Fallback to simple format
                            text_prompt = text_prompt if text_prompt else "What is in this image?"
                    else:
                        text_prompt = text_prompt if text_prompt else "What is in this image?"
                    
                    # Process fallback if not already processed
                    if 'inputs' not in locals():
                        try:
                            # Let the processor handle image token insertion automatically
                            # Gemma3 processor will add the correct number of tokens internally
                            logger.info(f"Processing {len(pil_images)} images with text: {text_prompt[:50]}...")
                            
                            # Try different parameter formats for Gemma3
                            try:
                                inputs = processor(
                                    text=text_prompt,
                                    images=pil_images,
                                    return_tensors="pt",
                                    padding=True
                                )
                            except TypeError:
                                # Try positional arguments
                                inputs = processor(
                                    text_prompt,
                                    pil_images,
                                    return_tensors="pt",
                                    padding=True
                                )
                        except Exception as e:
                            logger.error(f"Failed to process multimodal input: {e}")
                            # Try alternative approaches
                            try:
                                # Some processors might expect pixel_values directly
                                if hasattr(processor, 'image_processor'):
                                    pixel_values = processor.image_processor(pil_images, return_tensors="pt").pixel_values
                                    text_inputs = processor.tokenizer(text_prompt, return_tensors="pt", padding=True)
                                    inputs = {
                                        **text_inputs,
                                        'pixel_values': pixel_values
                                    }
                                    logger.info("Using separate image and text processing")
                                else:
                                    raise ValueError("No image processor found")
                            except Exception as e2:
                                logger.error(f"Alternative approach also failed: {e2}")
                                # Last resort - try without images
                                inputs = processor(
                                    text=text_prompt,
                                    return_tensors="pt",
                                    padding=True
                                )
                                logger.warning("Processing without images as fallback")
                else:
                    # Text-only input - prepare for Gemma 3 format
                    # First try to use apply_chat_template as shown in the example
                    try:
                        logger.debug("Trying apply_chat_template for text-only input")
                        inputs = processor.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt"
                        )
                        logger.debug(f"Success! Inputs shape: {inputs['input_ids'].shape}")
                        logger.debug(f"Keys in inputs: {list(inputs.keys())}")
                    except Exception as e:
                        logger.warning(f"Failed to process with processor: {e}")
                        
                        # Try to get tokenizer from processor
                        tokenizer = getattr(processor, 'tokenizer', None)
                        if tokenizer is None:
                            tokenizer = getattr(processor, '_tokenizer', None)
                        
                        if tokenizer is not None:
                            try:
                                logger.info("Using tokenizer directly")
                                # Convert messages to simple text prompt
                                if messages and isinstance(messages, list):
                                    prompt_parts = []
                                    for msg in messages:
                                        role = msg.get('role', 'user')
                                        content = msg.get('content', '')
                                        if isinstance(content, str):
                                            prompt_parts.append(f"{role}: {content}")
                                        elif isinstance(content, list):
                                            # Extract text from structured content
                                            for item in content:
                                                if isinstance(item, dict) and item.get('type') == 'text':
                                                    prompt_parts.append(f"{role}: {item.get('text', '')}")
                                    prompt = "\n".join(prompt_parts)
                                else:
                                    prompt = text or "Hello"
                                
                                inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=8192)
                            except Exception as e2:
                                logger.error(f"Tokenizer also failed: {e2}")
                                # Last resort - create minimal inputs
                                import torch
                                inputs = {
                                    'input_ids': torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long),  # Dummy input
                                    'attention_mask': torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long)
                                }
                                logger.warning("Using dummy inputs as last resort")
                        else:
                            logger.error("No tokenizer found in processor")
                            raise ValueError("Cannot find tokenizer in processor")
            
            # If we still don't have inputs at this point, something went wrong
            if not use_simple_prompt and 'inputs' not in locals():
                logger.error("Failed to create inputs through any method")
                raise ValueError("Could not process input data")

            # Validate inputs
            if 'input_ids' in inputs:
                input_ids = inputs['input_ids']
                logger.debug(f"Input IDs shape: {input_ids.shape}, min: {input_ids.min().item()}, max: {input_ids.max().item()}")
                
                # Check for invalid tokens
                # Try different attributes for vocab size
                vocab_size = None
                if hasattr(model, 'config'):
                    if hasattr(model.config, 'vocab_size'):
                        vocab_size = model.config.vocab_size
                    elif hasattr(model.config, 'vocabulary_size'):
                        vocab_size = model.config.vocabulary_size
                    elif hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'vocab_size'):
                        vocab_size = model.config.text_config.vocab_size
                
                if vocab_size is None:
                    vocab_size = 256000  # Gemma 3 default
                    logger.debug(f"Using default vocab size: {vocab_size}")
                
                if input_ids.max().item() >= vocab_size:
                    logger.error(f"Invalid token ID {input_ids.max().item()} >= vocab size {vocab_size}")
                    # Clamp tokens to valid range
                    inputs['input_ids'] = torch.clamp(input_ids, 0, vocab_size - 1)
            
            # Remove token_type_ids if present (Gemma doesn't use them)
            if 'token_type_ids' in inputs:
                logger.warning("Removing token_type_ids from inputs (Gemma models don't use them)")
                del inputs['token_type_ids']
            
            # Move to model device
            if hasattr(model, 'device'):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Prepare generation parameters
            # For quantized models, use greedy decoding to avoid numerical issues
            is_quantized = hasattr(model, 'config') and (
                getattr(model.config, 'quantization_config', None) is not None or
                getattr(model, 'is_quantized', False)
            )
            
            if is_quantized:
                logger.info("Using special parameters for quantized Gemma model")
                # Try different approach - use greedy with beam search
                generation_params = {
                    'max_new_tokens': min(kwargs.get('max_tokens', 2048), 1024),  # Limit output
                    'do_sample': False,  # Disable sampling for stability
                    'num_beams': 2,  # Small beam search for better quality
                    'temperature': 1.0,  # Ignored with do_sample=False
                    'early_stopping': True,
                    'repetition_penalty': 1.1,
                }
            else:
                # For non-quantized models, use normal sampling
                temperature = kwargs.get('temperature', 0.7)
                temperature = max(0.01, min(temperature, 2.0))  # Clamp between 0.01 and 2.0
                
                generation_params = {
                    'max_new_tokens': min(kwargs.get('max_tokens', 2048), 8192),
                    'temperature': temperature,
                    'top_p': kwargs.get('top_p', 0.95),
                    'top_k': kwargs.get('top_k', 40),
                    'do_sample': temperature > 0.01,
                }
            
            # Add common generation parameters
            generation_params.update({
                'use_cache': True,
                'num_beams': 1,  # Disable beam search
                'repetition_penalty': 1.0,  # No repetition penalty
                'length_penalty': 1.0,  # No length penalty
                'early_stopping': False,
                'no_repeat_ngram_size': 0,  # Disable n-gram blocking
                'bad_words_ids': None,
                'force_words_ids': None,
                'constraints': None,
                'forced_bos_token_id': None,
                'forced_eos_token_id': None,
                'remove_invalid_values': True  # IMPORTANT: Remove inf/nan from logits
            })
            
            # Get tokenizer from processor
            tokenizer = getattr(processor, 'tokenizer', None)
            if tokenizer is None:
                # Try to get tokenizer attribute
                tokenizer = getattr(processor, '_tokenizer', processor)
            
            # Set pad and eos tokens
            if hasattr(tokenizer, 'pad_token_id'):
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                generation_params['pad_token_id'] = tokenizer.pad_token_id
                generation_params['eos_token_id'] = tokenizer.eos_token_id
            else:
                # Fallback values for Gemma
                generation_params['pad_token_id'] = 0  # Gemma uses 0 for padding
                generation_params['eos_token_id'] = 1  # Gemma uses 1 for EOS

            # Debug logging before generation
            logger.info("--- DEBUGGING MODEL INPUTS ---")
            logger.info(f"Model device: {model.device if hasattr(model, 'device') else 'unknown'}")
            logger.info(f"Model dtype: {model.dtype if hasattr(model, 'dtype') else 'unknown'}")
            logger.info(f"Is quantized: {is_quantized}")
            
            for key, tensor in inputs.items():
                logger.info(f"Input key: '{key}' | Shape: {tensor.shape} | DType: {tensor.dtype} | Device: {tensor.device}")
                # Log first and last 5 values for input_ids
                if key == 'input_ids':
                    logger.info(f"Input IDs (first 5): {tensor[0, :5].tolist()}")
                    logger.info(f"Input IDs (last 5): {tensor[0, -5:].tolist()}")
                    logger.info(f"Min/Max Input ID: {tensor.min().item()} / {tensor.max().item()}")
                elif key == 'attention_mask':
                    logger.info(f"Attention mask (first 5): {tensor[0, :5].tolist()}")
                    logger.info(f"Attention mask (last 5): {tensor[0, -5:].tolist()}")
                    logger.info(f"Attention mask unique values: {torch.unique(tensor).tolist()}")
            
            # Check for invalid token values
            if 'input_ids' in inputs:
                # Get vocab size from model config
                vocab_size = None
                if hasattr(model, 'config'):
                    if hasattr(model.config, 'vocab_size'):
                        vocab_size = model.config.vocab_size
                    elif hasattr(model.config, 'vocabulary_size'):
                        vocab_size = model.config.vocabulary_size
                    elif hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'vocab_size'):
                        vocab_size = model.config.text_config.vocab_size
                
                if vocab_size:
                    logger.info(f"Model vocab size: {vocab_size}")
                    if inputs['input_ids'].max().item() >= vocab_size:
                        logger.error(f"FATAL: Invalid token ID detected! Max ID: {inputs['input_ids'].max().item()}, Vocab Size: {vocab_size}")
                        # Show which tokens are invalid
                        invalid_mask = inputs['input_ids'] >= vocab_size
                        invalid_tokens = inputs['input_ids'][invalid_mask]
                        logger.error(f"Invalid tokens: {invalid_tokens.tolist()}")
                else:
                    logger.warning("Could not determine vocab size from model config")
            
            # Log generation parameters
            logger.info(f"Generation params: {generation_params}")
            logger.info("---------------------------------")
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_params)

            # Decode response
            input_length = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
            generated_tokens = outputs[0][input_length:]
            
            # Get proper decoder - try multiple sources
            decoder = None
            
            # First try tokenizer from processor
            if hasattr(processor, 'tokenizer'):
                decoder = processor.tokenizer
                logger.debug("Using processor.tokenizer for decoding")
            elif hasattr(processor, '_tokenizer'):
                decoder = processor._tokenizer
                logger.debug("Using processor._tokenizer for decoding")
            elif tokenizer is not None:
                decoder = tokenizer
                logger.debug("Using tokenizer for decoding")
            else:
                # Last resort - use processor itself
                decoder = processor
                logger.debug("Using processor for decoding")
            
            # Decode the response
            response = None
            
            # Ensure we're working with CPU tensors
            if hasattr(generated_tokens, 'cpu'):
                generated_tokens = generated_tokens.cpu()
            
            # Convert tensor to list first (most decoders expect lists)
            token_list = generated_tokens.tolist()
            
            try:
                # Try decode with list
                if hasattr(decoder, 'decode'):
                    response = decoder.decode(token_list, skip_special_tokens=True)
                    logger.debug("List decode successful")
            except Exception as e:
                logger.warning(f"List decode failed: {e}")
                
                # Try with tensor directly (some decoders might accept tensors)
                try:
                    response = decoder.decode(generated_tokens, skip_special_tokens=True)
                    logger.debug("Tensor decode successful")
                except Exception as e2:
                    logger.warning(f"Tensor decode also failed: {e2}")
                    
                    # Try decoding the full output instead of just generated tokens
                    try:
                        full_output = outputs[0].cpu().tolist()
                        response = decoder.decode(full_output, skip_special_tokens=True)
                        # Remove the prompt from response
                        if hasattr(decoder, 'decode'):
                            prompt_decoded = decoder.decode(inputs['input_ids'][0].cpu().tolist(), skip_special_tokens=True)
                            if response.startswith(prompt_decoded):
                                response = response[len(prompt_decoded):].strip()
                        logger.debug("Full output decode successful")
                    except Exception as e3:
                        logger.error(f"All decode attempts failed: {e3}")
                        # Show token IDs for debugging
                        response = f"[DECODE ERROR] Tokens: {token_list[:20]}..."

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
    
    def get_training_parameters(self) -> Dict[str, Any]:
        """Get Gemma 3 specific training parameters.
        
        Returns:
            Dictionary with training parameter overrides.
        """
        params = super().get_training_parameters()
        
        # Gemma 3 specific modules for LoRA
        params['lora_target_modules'] = [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        
        # Gemma 3 prefers bfloat16
        params['training_precision'] = 'bf16'
        
        # Context length based on model size
        if '1b' in self.model_id.lower():
            params['max_seq_length'] = 8192  # 1B model limit
        else:
            params['max_seq_length'] = 8192  # Safe limit for all
        
        # Dataset formats - multimodal support
        params['dataset_formats'] = [
            'alpaca', 'sharegpt', 'openai', 'chat',
            'completion', 'qa', 'vision', 'vision_qa'
        ]
        
        # Gemma 3 doesn't support flash attention with quantization
        params['supports_flash_attention'] = False
        
        # Special tokens for Gemma
        params['special_tokens'] = {
            'image_token': '<image_soft_token>'
        }
        
        return params

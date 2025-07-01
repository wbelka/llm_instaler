"""Handler for Gemma 3n multimodal models with extended capabilities.

This module provides specialized handling for Google's Gemma 3n multimodal models
that extend standard Gemma 3 with support for audio and video inputs in addition
to text and images.
"""

import logging
import io
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

from handlers.gemma3 import Gemma3Handler
from core.checker import ModelRequirements
from core.quantization_config import QuantizationConfig

logger = logging.getLogger(__name__)


class Gemma3nHandler(Gemma3Handler):
    """Handler for Gemma 3n multimodal models with audio/video support.

    Extends the base Gemma3Handler with additional capabilities for
    audio and video processing.
    """

    def __init__(self, model_info: Dict[str, Any]):
        """Initialize Gemma 3n handler.

        Args:
            model_info: Model information dictionary.
        """
        super().__init__(model_info)
        self.is_gemma3n_model = self._check_if_gemma3n_model()

    def _check_if_gemma3n_model(self) -> bool:
        """Check if the model is a Gemma 3n multimodal model.

        Returns:
            True if it's a Gemma 3n multimodal model, False otherwise.
        """
        model_id = self.model_id.lower()
        model_type = self.model_info.get('model_type', '').lower()
        config = self.model_info.get('config', {})

        # Check for Gemma 3n models specifically
        is_gemma3n = '3n' in model_id and ('gemma' in model_id)

        # Check if it's multimodal (has vision/audio/video capabilities)
        has_multimodal = (
            'vision' in str(config).lower() or
            'image' in str(config).lower() or
            'audio' in str(config).lower() or
            'video' in str(config).lower() or
            'multimodal' in model_type or
            'vlm' in model_type or
            hasattr(config, 'vision_config')
        )

        return is_gemma3n and has_multimodal

    def get_dependencies(self) -> List[str]:
        """Get required Python dependencies.

        Returns:
            List of pip package specifications.
        """
        base_deps = [
            'torch>=2.6.0',
            'transformers>=4.50.0',  # Gemma 3n requires latest transformers
            'Pillow>=9.0.0',
            'numpy',
            'einops',
            'sentencepiece',
            'protobuf',
            'accelerate>=0.20.0',
            'safetensors>=0.4.0',
            'tokenizers>=0.15.0',
            'timm>=0.9.0',  # Required for Gemma 3n
            'soundfile',  # For audio processing
            'librosa',  # For audio processing
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
        requirements.model_family = "gemma3n"  # Use gemma3n as family
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

        # Gemma 3n has limited quantization support
        supports_quant = QuantizationConfig.supports_quantization(
            self.model_type, self.model_family, self.model_info
        )

        requirements.capabilities = {
            "supports_text_generation": True,
            "supports_image_generation": False,
            "supports_chat": True,
            "supports_cpu_inference": False,  # Multimodal models need GPU
            "supports_quantization": supports_quant,
            "supported_quantization": ["int8"] if supports_quant else [],  # Only 8-bit
            "requires_gpu": True
        }
        requirements.special_config = {
            "is_gemma3n_model": True,
            "max_context_length": 32768,  # 32K context for Gemma 3n
            "max_output_length": 32768,  # 32K output
            "supported_gpus": ['nvidia_a100', 'nvidia_a6000', 'nvidia_rtx_4090',
                               'nvidia_rtx_3090', 'nvidia_v100', 'nvidia_h100'],
            "audio_sample_rate": 16000,  # 16kHz audio
            "audio_tokens_per_second": 6.25,
            "supports_selective_activation": True,  # 2B/4B effective params
        }

        return requirements

    def _estimate_model_size(self) -> float:
        """Estimate model size in GB based on Gemma 3n variants.

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

        # Default estimates for Gemma 3n models
        # Gemma 3n models are typically smaller due to selective activation
        return 4.0  # Conservative default for Gemma 3n

    def load_model(
        self,
        model_path: str,
        device: str = 'auto',
        dtype: str = 'auto',
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs
    ) -> Tuple[Any, Any]:
        """Load Gemma 3n multimodal model with appropriate configuration.

        Args:
            model_path: Path to model files.
            device: Device to load on.
            dtype: Data type for model.
            load_in_8bit: Whether to use 8-bit quantization.
            load_in_4bit: Whether to use 4-bit quantization (will fallback to 8-bit).
            **kwargs: Additional loading arguments.

        Returns:
            Tuple of (model, processor).
        """
        try:
            from transformers import AutoProcessor, Gemma3nForConditionalGeneration
            import torch

            # Gemma3n has issues with 4-bit quantization due to altup layer
            if load_in_4bit:
                logger.warning(
                    "Gemma3n models have compatibility issues with 4-bit quantization. Falling back to 8-bit.")
                load_in_4bit = False
                load_in_8bit = True

            # Load processor with use_fast=True to avoid warning
            processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

            # Log processor info
            logger.info(f"Processor type: {type(processor)}")
            if hasattr(processor, 'image_token'):
                # Removed token logging
                pass
            if hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'image_token'):
                # Removed token logging
                pass

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
                logger.info("Quantization config applied")
                # Enable CPU offloading for 8-bit quantization to handle large models
                if load_in_8bit:
                    quant_config['quantization_config'].llm_int8_enable_fp32_cpu_offload = True
                    logger.info("Enabled FP32 CPU offloading for 8-bit quantization")

            # Prepare model loading arguments
            model_kwargs = {
                'pretrained_model_name_or_path': model_path,
                'torch_dtype': torch_dtype,
                'trust_remote_code': True  # Gemma 3n may require this
            }

            # Merge quantization config
            model_kwargs.update(quant_config)

            # Add device map and memory optimization
            if device == 'auto':
                # Check available GPU memory
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
                    available_memory = gpu_memory * 0.85  # Use 85% of available memory
                    logger.info(f"GPU memory: {gpu_memory:.1f}GB, using up to {available_memory:.1f}GB")
                else:
                    available_memory = 0
                
                # For quantized models, use balanced device map
                if load_in_8bit:
                    model_kwargs['device_map'] = 'balanced'
                    # Allow CPU offloading for large quantized models
                    # Adjust GPU memory based on actual availability
                    gpu_limit = f"{int(available_memory - 2)}GiB" if available_memory > 4 else "2GiB"
                    model_kwargs['max_memory'] = {0: gpu_limit, 'cpu': '100GiB'}
                    logger.info(f"Using balanced device map with GPU limit: {gpu_limit}")
                else:
                    model_kwargs['device_map'] = 'auto'
                    # Add offload settings for better memory management
                    model_kwargs['offload_folder'] = 'offload'
                    model_kwargs['offload_state_dict'] = True
                    
                    # For Gemma3n models, always enable CPU offloading due to multimodal requirements
                    logger.info("Enabling CPU offloading for Gemma3n multimodal model")
                    model_kwargs['device_map'] = 'balanced'
                    gpu_limit = f"{int(available_memory - 3)}GiB" if available_memory > 6 else "3GiB"
                    model_kwargs['max_memory'] = {0: gpu_limit, 'cpu': '100GiB'}
                    model_kwargs['offload_buffers'] = True

            # Load model with low CPU memory usage
            model_kwargs['low_cpu_mem_usage'] = True

            # For Gemma 3n, we use standard attention (no Flash Attention with quantization)
            model = Gemma3nForConditionalGeneration.from_pretrained(**model_kwargs)
            logger.info("Using standard attention implementation for Gemma 3n")

            # Move to device if not using device_map='auto'
            if device != 'auto' and not load_in_8bit:
                if device == 'cuda' and not torch.cuda.is_available():
                    device = 'cpu'
                    logger.warning("CUDA not available, falling back to CPU")
                model = model.to(device)

            # Set to eval mode
            model.eval()

            logger.info(f"Successfully loaded Gemma 3n model from {model_path}")
            return model, processor

        except ImportError as e:
            logger.error(f"Failed to import required dependencies: {e}")
            logger.info(
                "Please ensure transformers>=4.50.0 and timm>=0.9.0 are installed for Gemma 3n support"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load Gemma 3n model: {e}")
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
        """Get default inference parameters for Gemma 3n.

        Returns:
            Dictionary of inference parameters.
        """
        # Gemma 3n specific parameters
        params = {
            'temperature': 0.7,
            'max_new_tokens': 32768,  # Can generate up to 32K tokens
            'do_sample': True,
            'top_p': 0.95,
            'top_k': 40,
            'repetition_penalty': 1.1,
            'use_cache': True
        }

        return params

    def get_model_capabilities(self) -> Dict[str, Any]:
        """Get Gemma 3n model capabilities.

        Returns:
            Dictionary of capabilities.
        """
        capabilities = {
            'stream': True,
            'supports_streaming': True,
            'supports_reasoning': False,
            'supports_system_prompt': True,
            'supports_multimodal': True,
            'supports_batch_inference': True,
            'max_context_length': 32768,  # 32K for Gemma 3n
            'max_output_length': 32768,  # 32K output
            'input_modalities': ['text', 'image', 'audio', 'video'],
            'output_modalities': ['text'],
            'supports_multiple_images': True,
            'image_resolutions': [256, 512, 768],  # Multiple resolutions
            'image_tokens_per_image': 256,
            'audio_tokens_per_second': 6.25,
            'audio_sample_rate': 16000,  # 16kHz
            'supports_multilingual': True,
            'supported_languages': 140,
            'chat_template': 'gemma3',
            'special_features': [
                'Context window: 32768 tokens',
                'Multilingual support (140+ languages)',
                'Efficient image encoding (256 tokens per image)',
                'Audio encoding: 6.25 tokens per second',
                'Video input support',
                'Selective parameter activation (2B/4B effective params)',
                'Optimized for low-resource devices',
                'Multiple image resolutions: 256x256, 512x512, 768x768'
            ]
        }

        return capabilities

    def get_supported_modes(self) -> List[str]:
        """Get supported generation modes for Gemma 3n."""
        return ['auto', 'chat', 'vision', 'multimodal', 'analyze', 'audio', 'video']

    def get_mode_descriptions(self) -> Dict[str, str]:
        """Get descriptions for supported modes."""
        return {
            'auto': 'Automatic mode selection',
            'chat': 'Text conversation mode',
            'vision': 'Image understanding and analysis',
            'multimodal': 'Combined text and image processing',
            'analyze': 'Detailed image analysis with reasoning',
            'audio': 'Audio understanding and transcription',
            'video': 'Video understanding and analysis'
        }
    
    def _prepare_generation_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare generation parameters from kwargs.
        
        Args:
            kwargs: Raw keyword arguments
            
        Returns:
            Filtered generation parameters
        """
        # Define valid generation parameters
        valid_params = {
            'temperature', 'max_tokens', 'max_new_tokens', 'top_p', 'top_k',
            'do_sample', 'num_beams', 'repetition_penalty', 'length_penalty',
            'early_stopping', 'num_return_sequences', 'stop_sequences'
        }
        
        # Extract only valid generation parameters
        generation_params = {}
        
        # Handle max_tokens vs max_new_tokens
        if 'max_tokens' in kwargs and kwargs['max_tokens']:
            generation_params['max_new_tokens'] = kwargs['max_tokens']
        
        # Set other parameters
        for key, value in kwargs.items():
            if key in valid_params and value is not None:
                if key == 'stop_sequences':
                    # Handle stop sequences
                    generation_params['stopping_criteria'] = value
                elif key != 'max_tokens':  # Skip max_tokens as we handle it above
                    generation_params[key] = value
        
        # Set defaults if not provided
        if 'temperature' not in generation_params:
            generation_params['temperature'] = 0.7
        if 'max_new_tokens' not in generation_params:
            generation_params['max_new_tokens'] = 512
        if 'do_sample' not in generation_params:
            generation_params['do_sample'] = generation_params.get('temperature', 0.7) > 0
        
        return generation_params

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
        import torch
        
        if not model or not (processor or tokenizer):
            raise ValueError("Model and processor/tokenizer required for text generation")
        
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.debug(f"generate_text called with prompt: {prompt[:100] if prompt else 'None'}...")
            logger.debug(f"kwargs keys: {list(kwargs.keys())}")
            
            # Use tokenizer for text-only generation (avoid multimodal processing)
            tokenizer_to_use = processor.tokenizer if processor and hasattr(processor, 'tokenizer') else (tokenizer or processor)
            # Removed tokenizer type logging
            
            # Handle messages format if provided
            messages = kwargs.get('messages', [])
            if messages and not prompt:
                # Extract text content from messages
                full_prompt = ""
                for msg in messages:
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        full_prompt += f"{role}: {content}\n"
                    elif isinstance(content, list):
                        # Extract text from structured content
                        text_parts = [item['text'] for item in content if isinstance(item, dict) and item.get('type') == 'text']
                        if text_parts:
                            full_prompt += f"{role}: {' '.join(text_parts)}\n"
                if full_prompt:
                    prompt = full_prompt.strip()
            
            # Ensure we have a prompt
            if not prompt:
                raise ValueError("No prompt provided for text generation")
            
            # Apply chat template for text-only generation
            if hasattr(tokenizer_to_use, 'apply_chat_template'):
                # Create simple text messages without structured content
                if messages:
                    simple_messages = []
                    for msg in messages:
                        role = msg.get('role', 'user')
                        content = msg.get('content', '')
                        if isinstance(content, str):
                            simple_messages.append({"role": role, "content": content})
                        elif isinstance(content, list):
                            # Extract text from structured content
                            text_parts = [item['text'] for item in content if isinstance(item, dict) and item.get('type') == 'text']
                            if text_parts:
                                simple_messages.append({"role": role, "content": ' '.join(text_parts)})
                    
                    if simple_messages:
                        formatted_prompt = tokenizer_to_use.apply_chat_template(
                            simple_messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        prompt = formatted_prompt
                else:
                    formatted_prompt = tokenizer_to_use.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    prompt = formatted_prompt
            
            logger.debug(f"Text generation prompt: {prompt[:200]}...")
            
            # Tokenize the prompt
            # Check if we're dealing with a processor that might expect multimodal input
            if hasattr(tokenizer_to_use, '__class__') and 'Processor' in tokenizer_to_use.__class__.__name__:
                # For processors, we need to be more careful
                try:
                    # Try to use the tokenizer directly if available
                    if hasattr(tokenizer_to_use, 'tokenizer'):
                        inputs = tokenizer_to_use.tokenizer(prompt, return_tensors="pt", truncation=True)
                    else:
                        # Fallback: try to call it with text parameter explicitly
                        inputs = tokenizer_to_use(text=prompt, return_tensors="pt", truncation=True)
                except Exception as e:
                    logger.warning(f"Processor failed: {e}. Trying alternative approach.")
                    # Last resort: get the tokenizer from the processor
                    if hasattr(tokenizer_to_use, 'tokenizer'):
                        inputs = tokenizer_to_use.tokenizer(prompt, return_tensors="pt", truncation=True)
                    else:
                        raise ValueError("Could not find a way to tokenize the input")
            else:
                # Standard tokenizer
                inputs = tokenizer_to_use(prompt, return_tensors="pt", truncation=True)
            
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']
            
            # Move inputs to the correct device
            if hasattr(model, 'device') and model.device.type != 'meta':
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            elif hasattr(model, 'hf_device_map'):
                # For models with device_map, move inputs to first device (usually GPU)
                first_device = next(iter(model.hf_device_map.values()))
                if isinstance(first_device, int):
                    first_device = f'cuda:{first_device}'
                elif first_device == 'cpu':
                    first_device = 'cpu'
                inputs = {k: v.to(first_device) for k, v in inputs.items()}
                logger.info(f"Moving inputs to device: {first_device}")
            elif torch.cuda.is_available():
                # Fallback: if GPU is available, use it
                inputs = {k: v.cuda() for k, v in inputs.items()}
                logger.info("Moving inputs to CUDA (fallback)")
            
            # Prepare generation parameters
            generation_params = self._prepare_generation_params(kwargs)
            
            # Generate response
            with torch.no_grad():
                # Get pad_token_id and eos_token_id from the actual tokenizer
                if hasattr(tokenizer_to_use, 'tokenizer'):
                    # It's a processor, get tokens from the embedded tokenizer
                    pad_token_id = tokenizer_to_use.tokenizer.pad_token_id
                    eos_token_id = tokenizer_to_use.tokenizer.eos_token_id
                else:
                    # It's a tokenizer
                    pad_token_id = tokenizer_to_use.pad_token_id
                    eos_token_id = tokenizer_to_use.eos_token_id
                
                outputs = model.generate(
                    **inputs,
                    **generation_params,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id
                )
            
            # Decode the generated text
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            
            # Use the correct tokenizer for decoding
            if hasattr(tokenizer_to_use, 'tokenizer'):
                # It's a processor, use the embedded tokenizer
                generated_text = tokenizer_to_use.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            else:
                # It's a tokenizer
                generated_text = tokenizer_to_use.decode(generated_tokens, skip_special_tokens=True)
            
            # Calculate usage statistics
            usage = {
                "prompt_tokens": inputs['input_ids'].shape[1],
                "completion_tokens": generated_tokens.shape[0],
                "total_tokens": outputs[0].shape[0]
            }
            
            return {
                "text": generated_text.strip(),
                "usage": usage,
                "finish_reason": "stop"
            }
            
        except Exception as e:
            logger.error(f"Error in text generation: {str(e)}", exc_info=True)
            raise

    def process_audio(self, audio_data: bytes) -> Any:
        """Process audio data for Gemma 3n.

        Args:
            audio_data: Raw audio bytes

        Returns:
            Processed audio suitable for model input
        """
        try:
            import soundfile as sf
            import librosa
            from io import BytesIO

            # Load audio and resample to 16kHz if necessary
            audio_np, samplerate = sf.read(BytesIO(audio_data))

            if samplerate != 16000:
                audio_np = librosa.resample(audio_np, orig_sr=samplerate, target_sr=16000)
                samplerate = 16000

            logger.info(f"Processed audio: shape={audio_np.shape}, samplerate={samplerate}")
            return audio_np

        except Exception as e:
            logger.error(f"Failed to process audio: {e}")
            return None

    def process_video(self, video_data: bytes) -> Any:
        """Process video data for Gemma 3n.

        Args:
            video_data: Raw video bytes or list of frame images

        Returns:
            Processed video frames suitable for model input
        """
        # This is a placeholder implementation
        # In production, you would use a video processing library like OpenCV or PyAV
        logger.warning("Video processing is not fully implemented. Returning raw data.")
        return video_data

    def _prepare_multimodal_messages(self, text: str = None, images: List[Any] = None, messages: List[Dict] = None) -> List[Dict]:
        """Prepare a structured message list for multimodal input."""
        if messages:
            # If messages are provided, ensure they are in the correct format
            # and add image placeholders if they are missing.
            processed_messages = []
            for i, msg in enumerate(messages):
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                new_content = []
                if isinstance(content, str):
                    new_content.append({'type': 'text', 'text': content})
                elif isinstance(content, list):
                    # Handle cases where content is already a list (e.g., from API)
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            new_content.append(item)
                        # The presence of images is handled globally below
                
                # Add image placeholders to the last user message if not already present
                is_last_user_msg = (i == len(messages) - 1 and role == 'user')
                if is_last_user_msg and images:
                    # Add one placeholder for each image
                    for _ in images:
                        new_content.append({'type': 'image'})

                processed_messages.append({'role': role, 'content': new_content})
            return processed_messages
        else:
            # If no messages are provided, create a new one from text and images
            content = []
            if text:
                content.append({'type': 'text', 'text': text})
            if images:
                for _ in images:
                    content.append({'type': 'image'})
            return [{'role': 'user', 'content': content}]

    def process_multimodal(self, text: str = None, images: List[Union[str, Any]] = None,
                           audio: str = None, video: str = None,
                           model=None, processor=None, **kwargs) -> Dict[str, Any]:
        """Process multimodal inputs for Gemma 3n using the standard transformers workflow."""
        if audio or video:
            logger.warning("Gemma 3n models do not support audio or video input.")
        if not model or not processor:
            raise ValueError("Model and processor are required for multimodal processing.")

        try:
            import torch
            from PIL import Image
            import base64
            from io import BytesIO

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 1. Decode images from base64 to PIL
            pil_images = []
            if images:
                for img in images:
                    if isinstance(img, str):
                        try:
                            img_data = base64.b64decode(img)
                            pil_images.append(Image.open(BytesIO(img_data)))
                        except Exception as e:
                            logger.error(f"Failed to decode base64 image: {e}")
                    elif isinstance(img, Image.Image):
                        pil_images.append(img)
            
            # 2. Prepare structured messages for the chat template
            messages = self._prepare_multimodal_messages(text, pil_images, kwargs.get('messages'))
            
            # 3. Apply the chat template to create the prompt text
            # The template will correctly place image tokens based on the structured messages.
            prompt_text = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
            
            # 4. Use the processor to prepare inputs for the model
            # This combines the tokenized text and processed images.
            inputs = processor(
                text=prompt_text,
                images=pil_images,
                return_tensors="pt",
                padding=True
            )

            # 5. Clean up inputs and move to the correct device
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']
            if 'num_crops' in inputs: # Parameter not used by Gemma 3
                del inputs['num_crops']

            if hasattr(model, 'device') and model.device.type != 'meta':
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            elif torch.cuda.is_available():
                # Fallback for device_map='auto'
                first_device = next(iter(model.hf_device_map.values())) if hasattr(model, 'hf_device_map') else 'cuda:0'
                if isinstance(first_device, int):
                    first_device = f'cuda:{first_device}'
                inputs = {k: v.to(first_device) for k, v in inputs.items()}

            # 6. Set and sanitize generation parameters
            generation_params = self.get_inference_params()
            generation_params.update(kwargs)

            # Whitelist of valid generation parameters to avoid passing unsupported args
            VALID_GENERATE_KWARGS = {
                'max_new_tokens', 'temperature', 'top_p', 'top_k', 'do_sample',
                'num_beams', 'repetition_penalty', 'length_penalty', 'early_stopping',
                'num_return_sequences', 'pad_token_id', 'eos_token_id', 'use_cache',
                'remove_invalid_values'
            }

            # Handle the alias 'max_tokens'
            if 'max_tokens' in generation_params:
                generation_params['max_new_tokens'] = generation_params.pop('max_tokens')

            # Filter the parameters to only include those valid for the generate method
            sanitized_generation_params = {
                key: value for key, value in generation_params.items()
                if key in VALID_GENERATE_KWARGS
            }

            tokenizer = getattr(processor, 'tokenizer', processor)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            sanitized_generation_params['pad_token_id'] = tokenizer.pad_token_id
            sanitized_generation_params['eos_token_id'] = tokenizer.eos_token_id

            # 7. Generate response
            with torch.no_grad():
                outputs = model.generate(**inputs, **sanitized_generation_params)

            # 8. Decode and return the response
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = processor.decode(generated_tokens, skip_special_tokens=True)

            usage = {
                'prompt_tokens': input_length,
                'completion_tokens': len(generated_tokens),
                'total_tokens': input_length + len(generated_tokens)
            }

            return {
                'text': response.strip(),
                'type': 'text_generation',
                'usage': usage,
                'model': self.model_id
            }

        except Exception as e:
            logger.error(f"Fatal error in Gemma 3n multimodal processing: {e}", exc_info=True)
            # Fallback to the base handler's processing as a last resort
            return super().process_multimodal(
                text=text, images=images, model=model, processor=processor, **kwargs
            )

    def validate_model_files(self, model_path: str) -> Tuple[bool, Optional[str]]:
        """Validate required model files exist for Gemma 3n.

        Args:
            model_path: Path to model directory.

        Returns:
            Tuple of (is_valid, error_message).
        """
        required_files = [
            'config.json',
            'tokenizer_config.json',
            'tokenizer.json'  # Gemma 3n uses fast tokenizer
        ]

        model_path_obj = Path(model_path)

        # Check for basic files
        for file in required_files:
            if not (model_path_obj / file).exists():
                return False, f"Missing required file: {file}"

        # Check for model weights (Gemma 3n typically uses safetensors)
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
                f"Missing processor files for Gemma 3n model: {missing_processor}"
            )

        return True, None

    def get_training_parameters(self) -> Dict[str, Any]:
        """Get Gemma 3n specific training parameters.

        Returns:
            Dictionary with training parameter overrides.
        """
        params = super().get_training_parameters()

        # Gemma 3n specific modules for LoRA
        params['lora_target_modules'] = [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]

        # Gemma 3n prefers bfloat16
        params['training_precision'] = 'bf16'

        # Context length for Gemma 3n
        params['max_seq_length'] = 8192  # Safe limit

        # Dataset formats - multimodal support including audio/video
        params['dataset_formats'] = [
            'alpaca', 'sharegpt', 'openai', 'chat',
            'completion', 'qa', 'vision', 'vision_qa',
            'audio', 'video', 'multimodal'
        ]

        # Gemma 3n doesn't support flash attention with quantization
        params['supports_flash_attention'] = False

        # Special tokens for Gemma 3n
        params['special_tokens'] = {
            'image_token': '<image_soft_token>',
            'audio_token': '<audio_soft_token>',
            'video_token': '<video_soft_token>'
        }

        return params

    async def generate_stream(self, prompt: str = None, messages: List[Dict[str, str]] = None,
                              model=None, tokenizer=None, processor=None, images=None,
                              audio=None, video=None, **kwargs):
        """Stream text generation for Gemma 3n models.

        This is an async generator that yields chunks of generated text.

        Args:
            prompt: Text prompt (optional if messages provided)
            messages: List of messages for chat format
            model: The loaded model
            tokenizer: The loaded tokenizer
            processor: The loaded processor (preferred over tokenizer)
            images: Images for multimodal input
            audio: Audio for multimodal input
            video: Video for multimodal input
            **kwargs: Additional generation arguments

        Yields:
            Dictionary chunks with streaming data
        """
        import asyncio
        from transformers import TextIteratorStreamer
        from threading import Thread

        try:
            # Use processor if available, otherwise tokenizer
            tokenizer_to_use = processor or tokenizer
            if not tokenizer_to_use:
                raise ValueError("No tokenizer or processor available")

            # Get the actual tokenizer from processor if needed
            if hasattr(tokenizer_to_use, 'tokenizer'):
                actual_tokenizer = tokenizer_to_use.tokenizer
            else:
                actual_tokenizer = tokenizer_to_use

            # Prepare messages
            if not messages and prompt:
                messages = [{"role": "user", "content": prompt}]
            elif not messages:
                raise ValueError("Either prompt or messages must be provided")

            # Check if this is multimodal request
            is_multimodal = (images is not None and len(images) > 0) or audio is not None or video is not None

            if is_multimodal:
                # For multimodal, delegate to process_multimodal
                logger.info(
                    f"[STREAM] Detected multimodal request - images: {len(images) if images else 0}, "
                    f"audio: {bool(audio)}, video: {bool(video)}")
                result = self.process_multimodal(
                    text=prompt,
                    images=images,
                    audio=audio,
                    video=video,
                    model=model,
                    processor=processor or tokenizer,
                    messages=messages,
                    **kwargs
                )
                # Convert single response to streaming format
                yield {
                    "type": "text",
                    "token": result.get('text', ''),
                    "text": result.get('text', ''),
                    "finished": True,
                    "usage": result.get('usage', {})
                }
                return

            # For text-only streaming, convert messages to proper format
            processed_messages = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')

                if isinstance(content, str):
                    processed_messages.append({
                        "role": role,
                        "content": [{"type": "text", "text": content}]
                    })
                else:
                    processed_messages.append(msg)

            # Apply chat template
            if hasattr(tokenizer_to_use, 'apply_chat_template'):
                text = tokenizer_to_use.apply_chat_template(
                    processed_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback to simple format
                text = "\n".join([
                    f"{msg['role']}: "
                    f"{msg['content'][0]['text'] if isinstance(msg['content'], list) else msg['content']}"
                    for msg in processed_messages])

            # Tokenize
            inputs = actual_tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)

            # Move to model device
            if hasattr(model, 'device') and model.device.type != 'meta':
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            elif hasattr(model, 'hf_device_map'):
                # For models with device_map='auto', move inputs to first device
                first_device = next(iter(model.hf_device_map.values()))
                if isinstance(first_device, int):
                    first_device = f'cuda:{first_device}'
                elif first_device == 'cpu':
                    first_device = 'cpu'
                inputs = {k: v.to(first_device) for k, v in inputs.items()}

            # Create streamer
            streamer = TextIteratorStreamer(
                actual_tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            # Prepare generation kwargs
            generation_kwargs = {
                **inputs,
                'streamer': streamer,
                'max_new_tokens': kwargs.get('max_tokens', 2048),
                'temperature': kwargs.get('temperature', 0.7),
                'top_p': kwargs.get('top_p', 0.95),
                'top_k': kwargs.get('top_k', 40),
                'do_sample': kwargs.get('temperature', 0.7) > 0,
                'pad_token_id': actual_tokenizer.pad_token_id or actual_tokenizer.eos_token_id,
                'eos_token_id': actual_tokenizer.eos_token_id,
            }

            # For models with device_map, we need to handle cache differently
            if hasattr(model, 'hf_device_map') and model.hf_device_map:
                # Don't use cache for models with device_map to avoid the error
                generation_kwargs['use_cache'] = False

            # Start generation in a separate thread
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            # Stream tokens
            generated_text = ""
            for new_text in streamer:
                if new_text:
                    generated_text += new_text
                    yield {
                        "type": "text",
                        "token": new_text,
                        "text": generated_text,
                        "finished": False
                    }
                    await asyncio.sleep(0)  # Yield control

            # Wait for generation to complete
            thread.join()

            # Calculate token usage
            prompt_tokens = len(inputs['input_ids'][0])
            completion_tokens = len(actual_tokenizer.encode(generated_text))

            # Final yield with usage stats
            yield {
                "type": "done",
                "full_text": generated_text,
                "finished": True,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }

        except Exception as e:
            logger.error(f"Error in Gemma 3n streaming: {e}", exc_info=True)
            yield {
                "type": "error",
                "error": str(e),
                "finished": True
            }

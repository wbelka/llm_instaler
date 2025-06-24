"""Base handler class for all model type handlers.

This module defines the abstract base class that all specific model handlers
must inherit from. It provides the interface for model loading, dependency
management, and parameter configuration.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BaseHandler(ABC):
    """Abstract base class for model handlers.

    Each model type (transformer, diffusion, etc.) must implement a handler
    that inherits from this class and provides the required functionality.
    """

    def __init__(self, model_info: Dict[str, Any]):
        """Initialize handler with model information.

        Args:
            model_info: Dictionary containing model metadata from check phase.
        """
        self.model_info = model_info
        self.model_id = model_info.get('model_id', '')
        self.model_type = model_info.get('model_type', '')
        self.model_family = model_info.get('model_family', '')

    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """Get list of Python dependencies required for this model.

        Returns:
            List of pip package specifications (e.g., ['transformers>=4.30.0']).
        """
        raise NotImplementedError("Subclasses must implement get_dependencies()")

    @abstractmethod
    def get_system_dependencies(self) -> List[str]:
        """Get list of system dependencies required for this model.

        Returns:
            List of system packages or requirements (e.g., ['cuda>=11.0']).
        """
        raise NotImplementedError("Subclasses must implement get_system_dependencies()")

    @abstractmethod
    def analyze(self):
        """Analyze model and return its requirements.

        Returns:
            ModelRequirements object containing all detected requirements.
        """
        raise NotImplementedError("Subclasses must implement analyze()")

    @abstractmethod
    def load_model(self, model_path: str, **kwargs):
        """Load model from the specified path with optimal parameters.

        Args:
            model_path: Path to the model directory.
            **kwargs: Additional parameters like device, dtype, quantization.

        Returns:
            Loaded model instance and any additional components (tokenizer, etc.).
        """
        raise NotImplementedError("Subclasses must implement load_model()")

    @abstractmethod
    def get_inference_params(self) -> Dict[str, Any]:
        """Get default parameters for model inference.

        Returns:
            Dictionary of inference parameters specific to this model type.
        """
        raise NotImplementedError("Subclasses must implement get_inference_params()")

    @abstractmethod
    def get_training_params(self) -> Dict[str, Any]:
        """Get default parameters for model training/fine-tuning.

        Returns:
            Dictionary of training parameters specific to this model type.
        """
        raise NotImplementedError("Subclasses must implement get_training_params()")

    @abstractmethod
    def validate_model_files(self, model_path: str) -> Tuple[bool, Optional[str]]:
        """Validate that all required model files are present.

        Args:
            model_path: Path to the model directory.

        Returns:
            Tuple of (is_valid, error_message). error_message is None if valid.
        """
        raise NotImplementedError("Subclasses must implement validate_model_files()")
    
    def get_quantization_config(self, dtype: str, load_in_8bit: bool = False, 
                               load_in_4bit: bool = False) -> Tuple[Dict[str, Any], Any]:
        """Get quantization configuration based on dtype.
        
        This is a common method that can be used by all handlers to setup
        quantization consistently.
        
        Args:
            dtype: Requested dtype ('auto', 'int8', 'int4', 'float16', 'float32', 'bfloat16')
            load_in_8bit: Force 8-bit quantization
            load_in_4bit: Force 4-bit quantization
            
        Returns:
            Tuple of (model_kwargs, torch_dtype) where model_kwargs contains
            quantization config if needed
        """
        model_kwargs = {}
        
        # Import torch only when needed
        try:
            import torch
        except ImportError:
            # Return defaults if torch not available
            return model_kwargs, 'float16'
        
        # Handle quantization dtype
        if dtype == 'int8' or load_in_8bit:
            load_in_8bit = True
            # Check if model prefers bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float16
            logger.info("Enabling 8-bit quantization for memory efficiency")
        elif dtype == 'int4' or load_in_4bit:
            load_in_4bit = True  
            # Check if model prefers bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float16
            logger.info("Enabling 4-bit quantization for memory efficiency")
        elif dtype == 'auto':
            # Auto-detect best dtype
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            elif torch.cuda.is_available():
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
        else:
            # Explicit dtype mapping
            dtype_map = {
                'float32': torch.float32,
                'float16': torch.float16,
                'bfloat16': torch.bfloat16
            }
            torch_dtype = dtype_map.get(dtype, torch.float16)
        
        # Add quantization config if needed
        if load_in_8bit or load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                if load_in_4bit:
                    model_kwargs['quantization_config'] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",  # Use NormalFloat4 quantization
                        bnb_4bit_compute_dtype=torch_dtype,
                        bnb_4bit_use_double_quant=True,  # Nested quantization for more memory savings
                    )
                else:
                    model_kwargs['quantization_config'] = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_4bit_compute_dtype=None,
                    )
                # Add bitsandbytes to dependencies if not present
                from core.quantization_config import QuantizationConfig
                if QuantizationConfig.should_include_bitsandbytes(
                    self.model_type, self.model_family, self.get_dependencies()
                ):
                    logger.warning("Model uses quantization but bitsandbytes not in dependencies")
            except ImportError:
                logger.warning("BitsAndBytesConfig not available, quantization disabled")
                load_in_8bit = False
                load_in_4bit = False
        
        return model_kwargs, torch_dtype
    
    def get_memory_config(self, device: str = 'auto', memory_fraction: float = 0.95) -> Dict[str, Any]:
        """Get memory configuration for model loading.
        
        Args:
            device: Device to use ('auto', 'cuda', 'cpu', etc.)
            memory_fraction: Fraction of GPU memory to use (0.0-1.0)
            
        Returns:
            Dictionary with device_map and max_memory settings
        """
        config = {}
        
        if device == 'auto':
            config['device_map'] = 'auto'
            # Calculate available GPU memory
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory
                    usable_memory = int(gpu_memory * memory_fraction)
                    memory_gb = usable_memory / (1024**3)
                    config['max_memory'] = {0: f"{memory_gb:.1f}GB"}
                    logger.info(f"Using {memory_gb:.1f}GB of GPU memory for model loading")
            except ImportError:
                pass
        elif device != 'cpu':
            config['device_map'] = {'': device}
            
        return config
    
    def extract_assistant_response(self, generated_text: str) -> str:
        """Extract only the assistant's response from generated text.
        
        This is useful for models that output the full conversation format.
        
        Args:
            generated_text: Full generated text that may include conversation format
            
        Returns:
            Clean assistant response without role markers
        """
        # Look for assistant role markers
        for marker in ['assistant\n', 'assistant:', 'Assistant\n', 'Assistant:']:
            if marker in generated_text:
                parts = generated_text.split(marker)
                if len(parts) > 1:
                    # Take everything after the last occurrence
                    generated_text = parts[-1].strip()
                    break
        
        # Remove any remaining role markers at the start
        for marker in ['user\n', 'system\n', 'User\n', 'System\n', 'user:', 'system:', 'User:', 'System:']:
            if generated_text.startswith(marker):
                generated_text = generated_text[len(marker):].strip()
        
        # Remove any role marker that appears later (indicating next turn)
        for marker in ['\nuser:', '\nsystem:', '\nassistant:', '\nUser:', '\nSystem:', '\nAssistant:']:
            if marker in generated_text:
                generated_text = generated_text.split(marker)[0].strip()
        
        return generated_text

    def get_model_capabilities(self) -> Dict[str, Any]:
        """Get model capabilities for UI/API adaptation.

        Returns:
            Dictionary describing what this model can do.
        """
        # Base implementation - subclasses can override
        return {
            'supports_streaming': False,
            'supports_reasoning': False,
            'supports_system_prompt': False,
            'supports_multimodal': False,
            'supports_batch_inference': True,
            'max_context_length': None,
            'input_modalities': ['text'],
            'output_modalities': ['text']
        }

    def get_memory_requirements(self, dtype: str = 'float16') -> Dict[str, float]:
        """Estimate memory requirements for different operations.

        Args:
            dtype: Data type for model weights.

        Returns:
            Dictionary with memory estimates in GB.
        """
        # Base implementation with conservative estimates
        model_size_gb = self.model_info.get('model_size_gb', 10.0)

        dtype_multipliers = {
            'float32': 1.0,
            'float16': 0.5,
            'bfloat16': 0.5,
            'int8': 0.25,
            'int4': 0.125
        }

        multiplier = dtype_multipliers.get(dtype, 0.5)
        base_memory = model_size_gb * multiplier

        return {
            'inference': base_memory * 1.5,  # Model + activations
            'training_lora': base_memory * 1.8,  # Model + LoRA + gradients
            'training_full': base_memory * 4.0   # Model + optimizer states
        }

    def get_installation_notes(self) -> Dict[str, str]:
        """Get special installation instructions for dependencies.
        
        Returns:
            Dictionary mapping dependency names to installation instructions.
        """
        return {}
    
    def get_quantization_options(self) -> List[Dict[str, Any]]:
        """Get available quantization options for this model.

        Returns:
            List of quantization configurations.
        """
        # Base implementation - subclasses can provide more specific options
        return [
            {
                'name': 'int8',
                'description': '8-bit quantization',
                'memory_reduction': 0.5,
                'quality_impact': 'minimal'
            },
            {
                'name': 'int4',
                'description': '4-bit quantization',
                'memory_reduction': 0.75,
                'quality_impact': 'moderate'
            }
        ]

    def get_optimal_device(self, available_devices: List[str]) -> str:
        """Determine optimal device for running this model.

        Args:
            available_devices: List of available devices (e.g., ['cpu', 'cuda:0']).

        Returns:
            Optimal device string.
        """
        # Prefer GPU for most models
        for device in available_devices:
            if 'cuda' in device or 'mps' in device:
                return device

        return 'cpu'

    def prepare_for_training(self, model, training_method: str = 'lora') -> Any:
        """Prepare model for training with specified method.

        Args:
            model: The loaded model instance.
            training_method: Training method ('lora', 'qlora', 'full').

        Returns:
            Prepared model ready for training.
        """
        raise NotImplementedError(
            "Subclasses must implement prepare_for_training() if they support training"
        )

    def __repr__(self) -> str:
        """String representation of the handler."""
        return f"{self.__class__.__name__}(model_id='{self.model_id}')"
    
    # Generation interface methods - override in subclasses as needed
    
    def generate_text(self, prompt: str = None, messages: List[Dict] = None, 
                     model=None, tokenizer=None, **kwargs) -> Dict[str, Any]:
        """Generate text response.
        
        Args:
            prompt: Text prompt (for completion models)
            messages: Chat messages (for chat models)
            model: Model instance
            tokenizer: Tokenizer/processor instance
            **kwargs: Generation parameters (temperature, max_tokens, etc.)
            
        Returns:
            Dictionary with generated text and metadata
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support text generation")
    
    def generate_image(self, prompt: str, negative_prompt: str = None,
                      model=None, **kwargs) -> Dict[str, Any]:
        """Generate image from text prompt.
        
        Args:
            prompt: Text description of image to generate
            negative_prompt: What to avoid in the image
            model: Model instance
            **kwargs: Generation parameters (size, steps, guidance_scale, etc.)
            
        Returns:
            Dictionary with image data (base64) and metadata
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support image generation")
    
    def generate_audio(self, text: str = None, prompt: str = None,
                      model=None, **kwargs) -> Dict[str, Any]:
        """Generate audio (TTS or music generation).
        
        Args:
            text: Text to convert to speech (for TTS)
            prompt: Description for music generation
            model: Model instance
            **kwargs: Generation parameters (voice, sample_rate, duration, etc.)
            
        Returns:
            Dictionary with audio data (base64) and metadata
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support audio generation")
    
    def generate_video(self, prompt: str, image: str = None,
                      model=None, **kwargs) -> Dict[str, Any]:
        """Generate video from text or image prompt.
        
        Args:
            prompt: Text description of video to generate
            image: Base64 image for image-to-video generation
            model: Model instance
            **kwargs: Generation parameters (fps, duration, size, etc.)
            
        Returns:
            Dictionary with video data (base64) and metadata
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support video generation")
    
    def process_image(self, image: str, task: str = "classify",
                     model=None, processor=None, **kwargs) -> Dict[str, Any]:
        """Process image for various vision tasks.
        
        Args:
            image: Base64-encoded image
            task: Task type (classify, detect, segment, etc.)
            model: Model instance
            processor: Image processor instance
            **kwargs: Task-specific parameters
            
        Returns:
            Dictionary with results and metadata
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support image processing")
    
    def process_audio(self, audio: str, task: str = "transcribe",
                     model=None, processor=None, **kwargs) -> Dict[str, Any]:
        """Process audio for various audio tasks.
        
        Args:
            audio: Base64-encoded audio
            task: Task type (transcribe, classify, separate, etc.)
            model: Model instance
            processor: Audio processor instance
            **kwargs: Task-specific parameters
            
        Returns:
            Dictionary with results and metadata
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support audio processing")
    
    def process_multimodal(self, text: str = None, images: List[str] = None,
                          audio: str = None, video: str = None,
                          model=None, processor=None, **kwargs) -> Dict[str, Any]:
        """Process multimodal inputs.
        
        Args:
            text: Text input
            images: List of base64-encoded images
            audio: Base64-encoded audio
            video: Base64-encoded video
            model: Model instance
            processor: Multimodal processor instance
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with processed output and metadata
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support multimodal processing")
    
    def embed_text(self, texts: List[str], model=None, tokenizer=None,
                   **kwargs) -> Dict[str, Any]:
        """Generate text embeddings.
        
        Args:
            texts: List of texts to embed
            model: Model instance
            tokenizer: Tokenizer instance
            **kwargs: Embedding parameters (normalize, pooling, etc.)
            
        Returns:
            Dictionary with embeddings and metadata
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support text embeddings")
    
    def embed_image(self, images: List[str], model=None, processor=None,
                   **kwargs) -> Dict[str, Any]:
        """Generate image embeddings.
        
        Args:
            images: List of base64-encoded images
            model: Model instance
            processor: Image processor instance
            **kwargs: Embedding parameters
            
        Returns:
            Dictionary with embeddings and metadata
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support image embeddings")
    
    def embed_multimodal(self, text: str = None, image: str = None,
                        model=None, processor=None, **kwargs) -> Dict[str, Any]:
        """Generate multimodal embeddings (e.g., CLIP).
        
        Args:
            text: Text input
            image: Base64-encoded image
            model: Model instance
            processor: Multimodal processor instance
            **kwargs: Embedding parameters
            
        Returns:
            Dictionary with embeddings and metadata
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support multimodal embeddings")
    
    # Mode and configuration methods
    
    def get_supported_modes(self) -> List[str]:
        """Get list of supported generation modes.
        
        Returns:
            List of mode names (e.g., ['chat', 'complete', 'image'])
        """
        return ['auto']
    
    def get_mode_descriptions(self) -> Dict[str, str]:
        """Get descriptions for each supported mode.
        
        Returns:
            Dictionary mapping mode names to descriptions
        """
        return {'auto': 'Automatic mode selection'}
    
    def apply_mode_settings(self, mode: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mode-specific settings to parameters.
        
        Args:
            mode: Selected mode
            params: Current parameters
            
        Returns:
            Updated parameters
        """
        return params
    
    def prepare_inputs(self, inputs: Dict[str, Any], model=None, 
                      processor=None) -> Dict[str, Any]:
        """Prepare inputs for model processing.
        
        Args:
            inputs: Raw inputs
            model: Model instance
            processor: Processor/tokenizer instance
            
        Returns:
            Prepared inputs ready for model
        """
        return inputs
    
    def postprocess_outputs(self, outputs: Any, task: str = None) -> Dict[str, Any]:
        """Postprocess model outputs.
        
        Args:
            outputs: Raw model outputs
            task: Task type for task-specific postprocessing
            
        Returns:
            Processed outputs
        """
        return {'raw_outputs': outputs}
    
    def validate_inputs(self, inputs: Dict[str, Any], task: str = None) -> Tuple[bool, Optional[str]]:
        """Validate inputs before processing.
        
        Args:
            inputs: Input dictionary
            task: Task type for task-specific validation
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        return True, None
    
    def get_generation_config(self, task: str = "text") -> Dict[str, Any]:
        """Get default generation configuration for a task.
        
        Args:
            task: Task type (text, image, audio, video)
            
        Returns:
            Default configuration dictionary
        """
        configs = {
            'text': {
                'temperature': 0.7,
                'max_tokens': 512,
                'top_p': 0.9,
                'top_k': 50
            },
            'image': {
                'num_inference_steps': 50,
                'guidance_scale': 7.5,
                'width': 512,
                'height': 512
            },
            'audio': {
                'sample_rate': 22050,
                'duration': 10.0
            },
            'video': {
                'fps': 8,
                'num_frames': 16,
                'width': 256,
                'height': 256
            }
        }
        return configs.get(task, {})
    
    def get_training_parameters(self) -> Dict[str, Any]:
        """Get model-specific training parameters.
        
        Returns:
            Dictionary with training parameter overrides.
        """
        # Base parameters that can be overridden by specific handlers
        params = {
            'lora_target_modules': None,  # Auto-detect
            'lora_modules_to_save': None,
            'training_precision': 'auto',  # auto, fp16, bf16, fp32
            'gradient_checkpointing': True,
            'max_seq_length': None,  # Auto-detect
            'supports_flash_attention': True,
            'special_tokens': {},
            'dataset_formats': ['alpaca', 'chat', 'completion', 'text'],
            'recommended_batch_size': None,  # Auto-detect
            'recommended_learning_rate': None,  # Auto-detect
        }
        
        return params
    
    def prepare_model_for_training(self, model: Any, training_config: Dict[str, Any]) -> Any:
        """Prepare model for training (e.g., enable gradient checkpointing).
        
        Args:
            model: The loaded model.
            training_config: Training configuration.
            
        Returns:
            Prepared model.
        """
        # Enable gradient checkpointing if supported
        if hasattr(model, 'gradient_checkpointing_enable') and training_config.get('gradient_checkpointing', True):
            model.gradient_checkpointing_enable()
        
        # Set model to training mode
        if hasattr(model, 'train'):
            model.train()
        
        return model
    
    def get_tokenizer_config(self) -> Dict[str, Any]:
        """Get tokenizer configuration for training.
        
        Returns:
            Dictionary with tokenizer settings.
        """
        return {
            'padding_side': 'right',
            'truncation_side': 'right',
            'add_eos_token': True,
            'add_bos_token': True,
            'pad_to_multiple_of': 8,
        }

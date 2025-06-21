"""Base handler class for all model type handlers.

This module defines the abstract base class that all specific model handlers
must inherit from. It provides the interface for model loading, dependency
management, and parameter configuration.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


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

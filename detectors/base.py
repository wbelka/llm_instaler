"""Base detector class for model type detection.

This module defines the abstract base class that all model detectors
must inherit from. Detectors analyze model metadata to determine the
appropriate handler for a model.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class BaseDetector(ABC):
    """Abstract base class for model type detectors.

    Each detector specializes in identifying specific types of models
    based on their configuration, files, and metadata.
    """

    @abstractmethod
    def matches(self, model_info: Dict[str, Any]) -> bool:
        """Check if this detector can handle the given model.

        Args:
            model_info: Dictionary containing model metadata, including:
                - config: Model configuration (config.json contents)
                - files: List of files in the model repository
                - tags: HuggingFace model tags
                - library_name: Primary library for the model

        Returns:
            True if this detector can handle the model, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement matches()")

    @abstractmethod
    def analyze(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the model and return additional information.

        This method is called after matches() returns True. It should
        extract detailed information about the model that will be used
        by the corresponding handler.

        Args:
            model_info: Dictionary containing model metadata.

        Returns:
            Dictionary with additional model information, including:
                - model_type: Specific type identifier
                - model_family: General family (e.g., 'language-model')
                - architecture_type: Architecture details
                - special_requirements: Any special dependencies
                - capabilities: What the model can do
        """
        raise NotImplementedError("Subclasses must implement analyze()")

    @property
    @abstractmethod
    def priority(self) -> int:
        """Get the priority of this detector.

        Higher priority detectors are checked first. This allows more
        specific detectors to take precedence over general ones.

        Returns:
            Priority value (higher = checked first).
        """
        raise NotImplementedError("Subclasses must implement priority property")

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of this detector.

        Returns:
            Human-readable name for logging and debugging.
        """
        raise NotImplementedError("Subclasses must implement name property")

    def get_handler_class(self) -> str:
        """Get the handler class name for models detected by this detector.

        Returns:
            Handler class name (e.g., 'TransformerHandler').
        """
        # Default implementation - can be overridden
        # Assumes detector name follows pattern: ConfigDetector -> ConfigHandler
        detector_name = self.__class__.__name__
        if detector_name.endswith('Detector'):
            return detector_name[:-8] + 'Handler'
        return 'BaseHandler'

    def extract_model_size(self, model_info: Dict[str, Any]) -> Optional[float]:
        """Extract model size in billions of parameters.

        Args:
            model_info: Model metadata dictionary.

        Returns:
            Model size in billions of parameters, or None if unknown.
        """
        config = model_info.get('config', {})

        # Try common parameter count fields
        param_count = None
        for field in ['num_parameters', 'n_params', 'total_params']:
            if field in config:
                param_count = config[field]
                break

        # Try to calculate from architecture
        if param_count is None and 'hidden_size' in config:
            # Very rough estimate - subclasses should override
            hidden_size = config['hidden_size']
            num_layers = config.get('num_hidden_layers', config.get('n_layers', 12))
            param_count = (hidden_size * hidden_size * num_layers * 4) / 1e9

        if param_count:
            # Convert to billions if needed
            if param_count > 1000:
                param_count = param_count / 1e9
            return param_count

        return None

    def extract_special_requirements(self, model_info: Dict[str, Any]) -> List[str]:
        """Extract special requirements from model configuration.

        Args:
            model_info: Model metadata dictionary.

        Returns:
            List of special requirement identifiers.
        """
        requirements = []
        config = model_info.get('config', {})

        # Check for Flash Attention
        if config.get('_attn_implementation') == 'flash_attention_2':
            requirements.append('flash-attn')
        elif config.get('use_flash_attn', False):
            requirements.append('flash-attn')

        # Check for other special features
        if config.get('use_cache', True):
            requirements.append('kv-cache')

        return requirements

    def __repr__(self) -> str:
        """String representation of the detector."""
        return f"{self.__class__.__name__}(priority={self.priority})"

"""Handler registry for model type handlers.

This module provides a registry system for managing and retrieving
model handlers based on model information.
"""

from typing import Dict, Any, Type, Optional
from handlers.base import BaseHandler
from handlers import get_handler_class


class HandlerRegistry:
    """Registry for managing model handlers.

    This class provides a centralized way to register and retrieve
    handlers for different model types.
    """

    def __init__(self):
        """Initialize the handler registry."""
        self._handlers: Dict[str, Type[BaseHandler]] = {}
        self._initialize_default_handlers()

    def _initialize_default_handlers(self):
        """Initialize default handlers.

        This sets up the mapping between model families/types
        and their corresponding handler classes.
        """
        # Import available handlers
        try:
            from handlers.transformer import TransformerHandler
            self._handlers['transformer'] = TransformerHandler
            self._handlers['language-model'] = TransformerHandler
            self._handlers['text-classifier'] = TransformerHandler
        except ImportError:
            pass
            
        try:
            from handlers.diffusion import DiffusionHandler
            self._handlers['diffusion'] = DiffusionHandler
            self._handlers['image-generation'] = DiffusionHandler
            self._handlers['text-to-image'] = DiffusionHandler
            self._handlers['text-to-video'] = DiffusionHandler
        except ImportError:
            pass
            
        try:
            from handlers.embedding import EmbeddingHandler
            self._handlers['embedding'] = EmbeddingHandler
            self._handlers['text-embedding'] = EmbeddingHandler
        except ImportError:
            pass
            
        try:
            from handlers.vision import VisionHandler
            self._handlers['vision'] = VisionHandler
            self._handlers['image-classification'] = VisionHandler
            self._handlers['object-detection'] = VisionHandler
        except ImportError:
            pass
            
        try:
            from handlers.audio import AudioHandler
            self._handlers['audio'] = AudioHandler
            self._handlers['speech-to-text'] = AudioHandler
            self._handlers['text-to-speech'] = AudioHandler
        except ImportError:
            pass
            
        try:
            from handlers.multimodal import MultimodalHandler
            self._handlers['multimodal'] = MultimodalHandler
            self._handlers['vision-language'] = MultimodalHandler
            # For Deepseek Janus models
            self._handlers['multi_modality'] = MultimodalHandler
        except ImportError:
            pass

    def register_handler(
        self,
        key: str,
        handler_class: Type[BaseHandler]
    ):
        """Register a handler for a specific key.

        Args:
            key: The key to register the handler under
                (e.g., model family or type).
            handler_class: The handler class to register.
        """
        if not issubclass(handler_class, BaseHandler):
            raise ValueError("Handler class must inherit from BaseHandler")

        self._handlers[key] = handler_class

    def get_handler_for_model(
        self,
        model_info: Dict[str, Any]
    ) -> Optional[Type[BaseHandler]]:
        """Get the appropriate handler class for a model.

        This method tries to find a handler in the following order:
        1. Check for explicitly registered handlers by model family
        2. Check for explicitly registered handlers by model type
        3. Fall back to dynamic loading using get_handler_class

        Args:
            model_info: Model information dictionary containing
                analysis results.

        Returns:
            Handler class (not instance) or None if no suitable
            handler found.
        """
        model_family = model_info.get('model_family', '')
        model_type = model_info.get('model_type', '')

        # Try to find in registered handlers
        if model_family and model_family in self._handlers:
            return self._handlers[model_family]

        if model_type and model_type in self._handlers:
            return self._handlers[model_type]

        # Fall back to dynamic loading
        try:
            return get_handler_class(model_info)
        except ValueError:
            # Return None instead of raising to match expected behavior
            return None

    def list_handlers(self) -> Dict[str, Type[BaseHandler]]:
        """List all registered handlers.

        Returns:
            Dictionary mapping keys to handler classes.
        """
        return self._handlers.copy()

    def has_handler(self, key: str) -> bool:
        """Check if a handler is registered for a given key.

        Args:
            key: The key to check.

        Returns:
            True if a handler is registered, False otherwise.
        """
        return key in self._handlers


# Global registry instance
_registry = None


def get_handler_registry() -> HandlerRegistry:
    """Get the global handler registry instance.

    Returns:
        The global HandlerRegistry instance.
    """
    global _registry
    if _registry is None:
        _registry = HandlerRegistry()
    return _registry


__all__ = ['HandlerRegistry', 'get_handler_registry']

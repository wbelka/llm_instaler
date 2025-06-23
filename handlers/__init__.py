"""Handler package for different model types.

This package contains handlers for various model architectures and provides
a unified interface for loading and working with different model types.
"""

from typing import Dict, Any, Type
from handlers.base import BaseHandler


def get_handler_class(model_info: Dict[str, Any]) -> Type[BaseHandler]:
    """Get the appropriate handler class for a model.

    Args:
        model_info: Model information dictionary containing analysis results.

    Returns:
        Handler class (not instance).

    Raises:
        ValueError: If no suitable handler is found.
    """
    model_family = model_info.get('model_family', '')
    model_type = model_info.get('model_type', '')
    # handler_class_name = model_info.get('handler_class', '')  # Reserved for future use

    # Map model families to handler modules
    handler_map = {
        'language-model': 'transformer',
        'image-generation': 'diffusion',
        'multimodal': 'multimodal',
        'vision-language': 'multimodal',
        'text-classifier': 'transformer',
        'computer-vision': 'vision',
        'audio-model': 'audio',
        'embedding': 'embedding'
    }

    # Determine handler module
    handler_module = handler_map.get(model_family)

    # Special cases for specific model types
    if model_type in ['mamba', 'rwkv', 'jamba']:
        handler_module = 'specialized'

    if not handler_module:
        raise ValueError(
            f"No handler available for model family: {model_family}, "
            f"type: {model_type}"
        )

    # Import handler class
    try:
        module = __import__(f'handlers.{handler_module}', fromlist=[f'{handler_module.title()}Handler'])
        handler_class = getattr(module, f'{handler_module.title()}Handler')
        return handler_class
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Failed to load handler for {handler_module}: {e}")


def get_handler(model_info: Dict[str, Any]) -> BaseHandler:
    """Get an initialized handler instance for a model.

    Args:
        model_info: Model information dictionary containing analysis results.

    Returns:
        Initialized handler instance.
    """
    handler_class = get_handler_class(model_info)
    return handler_class(model_info)


__all__ = ['BaseHandler', 'get_handler', 'get_handler_class']

"""Handler package for different model types.

This package contains handlers for various model architectures and provides
a unified interface for loading and working with different model types.
"""

import importlib
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

    # Check for specific models
    model_id = model_info.get('model_id', '').lower()
    
    # Check for DeepSeek models (especially R1)
    if 'deepseek' in model_id:
        handler_module = 'deepseek'

    # Check for Gemma 3 multimodal models
    if (('gemma-3' in model_id or 'gemma3' in model_id) and
        (model_info.get('is_gemma3_multimodal', False) or
         model_family == 'multimodal' or
         'vision' in str(model_info.get('config', {})).lower())):
        handler_module = 'gemma3'
    # Check for PaliGemma models
    elif 'paligemma' in model_id:
        handler_module = 'gemma3'
    # Check for Qwen models
    elif 'qwen3' in model_id or model_type in ['qwen3', 'qwen-3']:
        handler_module = 'qwen3'
    elif 'qwen' in model_id and ('vl' in model_id or model_type == 'qwen2_5_vl'):
        handler_module = 'qwen_vl'

    # Check for reasoning models
    tags = model_info.get('tags', [])
    if 'o1' in model_id or 'reasoning' in tags or model_type in ['o1', 'reasoning-llm']:
        handler_module = 'specialized'

    # Check for code models
    if any(kw in model_id for kw in ['code', 'codegen', 'starcoder', 'codellama']):
        handler_module = 'specialized'

    if not handler_module:
        raise ValueError(
            f"No handler available for model family: {model_family}, "
            f"type: {model_type}"
        )

    # Import handler class using importlib for better security
    try:
        # Import the module securely
        module = importlib.import_module(f'handlers.{handler_module}')
        
        # Get the handler class name (capitalize first letter)
        handler_class_name = f'{handler_module.title()}Handler'
        
        # Get the handler class from the module
        if not hasattr(module, handler_class_name):
            raise AttributeError(f"Module {handler_module} does not have {handler_class_name}")
            
        handler_class = getattr(module, handler_class_name)
        
        # Validate it's actually a handler class
        if not issubclass(handler_class, BaseHandler):
            raise TypeError(f"{handler_class_name} is not a subclass of BaseHandler")
            
        return handler_class
    except (ImportError, AttributeError, TypeError) as e:
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

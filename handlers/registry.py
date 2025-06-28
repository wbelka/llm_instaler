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

        try:
            from handlers.trocr import TrOCRHandler
            self._handlers['ocr'] = TrOCRHandler
            self._handlers['trocr'] = TrOCRHandler # Also register by name
        except ImportError:
            pass

        try:
            from handlers.janus import JanusHandler
            # Register Janus handler for specific model IDs
            self._handlers['janus'] = JanusHandler
        except ImportError:
            pass

        try:
            from handlers.specialized import SpecializedHandler
            # Register specialized handler for unique model types
            self._handlers['specialized'] = SpecializedHandler
            self._handlers['reasoning'] = SpecializedHandler
            self._handlers['code-generation'] = SpecializedHandler
            self._handlers['o1'] = SpecializedHandler
        except ImportError:
            pass

        try:
            from handlers.qwen_vl import QwenVLHandler
            # Register Qwen VL handler
            self._handlers['qwen_vl'] = QwenVLHandler
            self._handlers['qwen2_vl'] = QwenVLHandler
            self._handlers['qwen2_5_vl'] = QwenVLHandler
        except ImportError:
            pass

        try:
            from handlers.qwen3 import Qwen3Handler
            # Register Qwen3 handler for thinking mode support
            self._handlers['qwen3'] = Qwen3Handler
            self._handlers['qwen-3'] = Qwen3Handler
        except ImportError:
            pass

        try:
            from handlers.gemma3 import Gemma3Handler
            # Register Gemma 3 multimodal handler
            self._handlers['gemma3'] = Gemma3Handler
            self._handlers['gemma-3'] = Gemma3Handler
            self._handlers['gemma3_vlm'] = Gemma3Handler
            self._handlers['paligemma'] = Gemma3Handler
        except ImportError as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to import Gemma3Handler: {e}")
            pass

        try:
            from handlers.deepseek import DeepseekHandler
            # Register DeepSeek handler for R1 reasoning models
            self._handlers['deepseek'] = DeepseekHandler
            self._handlers['deepseek_r1'] = DeepseekHandler
        except ImportError:
            pass
        
        try:
            from handlers.llama4 import Llama4Handler
            # Register Llama 4 multimodal handler
            self._handlers['llama4'] = Llama4Handler
            self._handlers['llama-4'] = Llama4Handler
            self._handlers['llama4_scout'] = Llama4Handler
            self._handlers['llama4_maverick'] = Llama4Handler
        except ImportError as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to import Llama4Handler: {e}")
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
        1. Check for model-specific handlers (e.g., Janus)
        2. Check for explicitly registered handlers by model family
        3. Check for explicitly registered handlers by model type
        4. Fall back to dynamic loading using get_handler_class

        Args:
            model_info: Model information dictionary containing
                analysis results.

        Returns:
            Handler class (not instance) or None if no suitable
            handler found.
        """
        model_id = model_info.get('model_id', '').lower()
        model_family = model_info.get('model_family', '')
        model_type = model_info.get('model_type', '')
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Registry: Looking for handler - model_id={model_id}, model_type={model_type}, model_family={model_family}")

        # Check for model-specific handlers
        # Check for DeepSeek models first (especially R1)
        if 'deepseek' in model_id:
            # Check if it's an R1 model
            if any(pattern in model_id for pattern in ['deepseek-r1', 'deepseek_r1', '-r1-', '_r1_']):
                if 'deepseek_r1' in self._handlers:
                    logger.info("Registry: Found DeepSeekHandler for R1 model")
                    return self._handlers['deepseek_r1']
            # General DeepSeek models
            if 'deepseek' in self._handlers:
                logger.info("Registry: Found DeepSeekHandler")
                return self._handlers['deepseek']
        
        if 'janus' in model_id:
            if 'janus' in self._handlers:
                return self._handlers['janus']

        # Check for Llama 4 models
        if 'llama-4' in model_id or 'llama4' in model_id or model_type == 'llama4':
            logger.info(f"Registry: Checking for llama4 handler")
            if 'llama4' in self._handlers:
                logger.info(f"Registry: Found Llama4Handler")
                return self._handlers['llama4']

        # Check for Qwen3 models first (before Qwen VL)
        if 'qwen3' in model_id or 'qwen-3' in model_id or model_type in ['qwen3', 'qwen-3']:
            if 'qwen3' in self._handlers:
                return self._handlers['qwen3']

        # Check for Gemma 3 multimodal models
        # Check model_type first as it's more specific
        if model_type == 'gemma3':
            logger.info(f"Registry: Checking for gemma3 handler by model_type")
            if 'gemma3' in self._handlers:
                logger.info(f"Registry: Found Gemma3Handler by model_type")
                return self._handlers['gemma3']
            else:
                logger.warning(f"Registry: gemma3 not in handlers: {list(self._handlers.keys())}")
        
        # Also check by model ID patterns
        if (('gemma-3' in model_id or 'gemma3' in model_id) and
            (model_info.get('is_gemma3_multimodal', False) or
             'vision' in str(model_info.get('config', {})).lower() or
             model_family == 'multimodal')):
            logger.info(f"Registry: Checking for gemma3 handler by model_id pattern")
            if 'gemma3' in self._handlers:
                logger.info(f"Registry: Found Gemma3Handler by model_id pattern")
                return self._handlers['gemma3']

        # Check for PaliGemma models
        if 'paligemma' in model_id:
            if 'paligemma' in self._handlers:
                return self._handlers['paligemma']

        # Check for Qwen VL models
        if 'qwen' in model_id and ('vl' in model_id or model_type == 'qwen2_5_vl'):
            if 'qwen_vl' in self._handlers:
                return self._handlers['qwen_vl']

        # Check for reasoning models (o1-style and DeepSeek-R1)
        if ('o1' in model_id or 
            'reasoning' in model_info.get('tags', []) or
            'deepseek-r1' in model_id or
            'deepseek_r1' in model_id or
            '-r1-' in model_id):
            if 'reasoning' in self._handlers:
                return self._handlers['reasoning']

        # Check for code models
        if any(kw in model_id for kw in ['code', 'codegen', 'starcoder', 'codellama']):
            if 'code-generation' in self._handlers:
                return self._handlers['code-generation']

        # Try to find in registered handlers
        # Check model_type first as it's more specific than model_family
        if model_type and model_type in self._handlers:
            logger.info(f"Registry: Found handler for model_type '{model_type}': {self._handlers[model_type].__name__}")
            return self._handlers[model_type]
        
        if model_family and model_family in self._handlers:
            logger.info(f"Registry: Found handler for model_family '{model_family}': {self._handlers[model_family].__name__}")
            return self._handlers[model_family]

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

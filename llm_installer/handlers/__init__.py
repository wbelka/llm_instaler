"""
Model-specific handlers for installation and configuration
"""

from .base import BaseHandler
from .general import GeneralHandler
from .nematron import NematronHandler
from .janus import JanusHandler

__all__ = [
    'BaseHandler',
    'GeneralHandler', 
    'NematronHandler',
    'JanusHandler',
]

# Handler registry
HANDLERS = [
    NematronHandler(),
    JanusHandler(),
    GeneralHandler(),  # Default fallback handler
]

def get_handler_for_model(model_info: dict) -> BaseHandler:
    """Get appropriate handler for a model"""
    for handler in HANDLERS:
        if handler.can_handle(model_info):
            return handler
    
    # Should never reach here as GeneralHandler handles everything
    return GeneralHandler()
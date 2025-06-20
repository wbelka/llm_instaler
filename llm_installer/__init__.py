"""
LLM Installer - Universal installer for HuggingFace models
"""

__version__ = "2.0.0"

from .checker import ModelChecker, check_model
from .model_profile import ModelProfile

__all__ = [
    'ModelChecker',
    'check_model',
    'ModelProfile',
]

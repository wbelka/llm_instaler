"""Core package for LLM Installer.

This package contains the main functionality for the LLM Installer,
including configuration management, CLI interface, and utility functions.
"""

from core.config import Config, get_config, ConfigError
from core.utils import setup_logging, check_system_requirements

__version__ = "2.0.0"

__all__ = [
    'Config',
    'get_config',
    'ConfigError',
    'setup_logging',
    'check_system_requirements',
    '__version__'
]

"""
LLM Installer - Universal installer for HuggingFace models
"""

__version__ = "2.0.0"

from .checker import ModelChecker
from .model_detector_v2 import ModelDetectorV2
from .detectors_v2.base import ModelInfo
from .installer import ModelInstaller, InstallConfig

__all__ = [
    'ModelChecker',
    'ModelDetectorV2',
    'ModelInfo',
    'ModelInstaller',
    'InstallConfig',
]

"""
Model detectors v2 - API-first approach with specialized parsers
"""

from .base import BaseDetector, ModelInfo
from .transformers_detector import TransformersDetector
from .diffusers_detector import DiffusersDetector
from .sentence_transformers_detector import SentenceTransformersDetector
from .gguf_detector import GGUFDetector
from .timm_detector import TimmDetector
from .audio_detector import AudioDetector

__all__ = [
    'BaseDetector',
    'ModelInfo',
    'TransformersDetector',
    'DiffusersDetector',
    'SentenceTransformersDetector',
    'GGUFDetector',
    'TimmDetector',
    'AudioDetector',
]

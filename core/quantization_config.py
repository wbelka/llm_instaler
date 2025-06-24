"""Centralized quantization configuration for LLM models."""

from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class QuantizationConfig:
    """Central configuration for model quantization support."""
    
    # Default bitsandbytes version
    DEFAULT_BITSANDBYTES_VERSION = "bitsandbytes>=0.41.0"
    
    # Models that support quantization by default
    QUANTIZATION_SUPPORTED_FAMILIES = {
        "transformer": True,
        "llama": True,
        "llama4": True,  # Llama 4 supports int4 and fp8
        "mistral": True,
        "mixtral": True,
        "qwen": True,
        "qwen3": True,
        "qwen_vl": True,
        "gemma": True,
        "gemma3": True,
        "phi": True,
        "deepseek": True,
        "multimodal": True,  # Most multimodal models support quantization
        "janus": True,
        "embedding": True,  # Transformer-based embeddings only
    }
    
    # Models that don't support quantization
    QUANTIZATION_UNSUPPORTED_FAMILIES = {
        "diffusion": False,
        "stable-diffusion": False,
        "sdxl": False,
        "kandinsky": False,
        "audio": False,
        "whisper": False,
        "bark": False,
        "musicgen": False,
        "vision": False,  # Pure vision models
        "clip": False,  # CLIP doesn't need quantization
        "siglip": False,
    }
    
    # Special quantization configurations for specific models
    SPECIAL_CONFIGS = {
        "gemma3": {
            "preferred_compute_dtype": "bfloat16",
            "supports_flash_attention_with_quantization": False,
        },
        "qwen3": {
            "preferred_compute_dtype": "float16",
            "supports_flash_attention_with_quantization": True,
        },
        "llama": {
            "preferred_compute_dtype": "float16",
            "supports_flash_attention_with_quantization": True,
        },
        "llama4": {
            "preferred_compute_dtype": "bfloat16",
            "supports_flash_attention_with_quantization": False,  # Uses flex_attention instead
            "supports_fp8": True,  # Llama 4 Maverick supports FP8
            "supports_int4": True,  # Llama 4 Scout fits on single H100 with int4
        }
    }
    
    @classmethod
    def get_bitsandbytes_version(cls, model_family: str = None) -> str:
        """Get the appropriate bitsandbytes version for a model family.
        
        Args:
            model_family: The model family name
            
        Returns:
            The bitsandbytes version string
        """
        # In the future, we can return different versions for different models
        return cls.DEFAULT_BITSANDBYTES_VERSION
    
    @classmethod
    def supports_quantization(cls, model_type: str, model_family: str, 
                            model_info: Dict[str, Any] = None) -> bool:
        """Check if a model supports quantization.
        
        Args:
            model_type: The model type
            model_family: The model family
            model_info: Additional model information
            
        Returns:
            True if the model supports quantization
        """
        # First check explicit model info
        if model_info and 'supports_quantization' in model_info:
            return model_info['supports_quantization']
        
        # Check unsupported families first
        if model_family in cls.QUANTIZATION_UNSUPPORTED_FAMILIES:
            return False
        
        # Check supported families
        if model_family in cls.QUANTIZATION_SUPPORTED_FAMILIES:
            return True
        
        # Check model type as fallback
        if model_type in cls.QUANTIZATION_UNSUPPORTED_FAMILIES:
            return False
        
        if model_type in cls.QUANTIZATION_SUPPORTED_FAMILIES:
            return True
        
        # Default to False for unknown models
        logger.warning(f"Unknown model family '{model_family}' and type '{model_type}' for quantization support")
        return False
    
    @classmethod
    def get_quantization_dependencies(cls, model_type: str, model_family: str) -> List[str]:
        """Get quantization-related dependencies for a model.
        
        Args:
            model_type: The model type
            model_family: The model family
            
        Returns:
            List of dependency strings
        """
        if not cls.supports_quantization(model_type, model_family):
            return []
        
        return [cls.get_bitsandbytes_version(model_family)]
    
    @classmethod
    def get_special_config(cls, model_family: str) -> Dict[str, Any]:
        """Get special quantization configuration for a model family.
        
        Args:
            model_family: The model family
            
        Returns:
            Dictionary with special configuration
        """
        return cls.SPECIAL_CONFIGS.get(model_family, {})
    
    @classmethod
    def should_include_bitsandbytes(cls, model_type: str, model_family: str,
                                  dependencies: List[str]) -> bool:
        """Check if bitsandbytes should be included in dependencies.
        
        Args:
            model_type: The model type
            model_family: The model family
            dependencies: Current list of dependencies
            
        Returns:
            True if bitsandbytes should be included
        """
        # Don't include if model doesn't support quantization
        if not cls.supports_quantization(model_type, model_family):
            return False
        
        # Check if already included
        bitsandbytes_included = any('bitsandbytes' in dep for dep in dependencies)
        
        return not bitsandbytes_included
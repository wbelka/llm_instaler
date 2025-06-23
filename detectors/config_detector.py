"""Configuration-based model detector.

This detector analyzes model configuration files to determine
the model type and appropriate handler.
"""

import logging
from typing import Dict, Any, List
from pathlib import Path

from detectors.base import BaseDetector
from detectors.architecture_detector import get_architecture_detector

logger = logging.getLogger(__name__)


class ConfigDetector(BaseDetector):
    """Detector that analyzes model configuration files."""
    
    def __init__(self):
        """Initialize the config detector."""
        self.architecture_detector = get_architecture_detector()
    
    @property
    def name(self) -> str:
        """Get detector name."""
        return "ConfigDetector"
    
    @property
    def priority(self) -> int:
        """Get detector priority."""
        # High priority as config.json is the most reliable source
        return 100
    
    def matches(self, model_info: Dict[str, Any]) -> bool:
        """Check if this detector can handle the model.
        
        Args:
            model_info: Model metadata
            
        Returns:
            True if config data is available
        """
        # Check if we have config data
        config_data = model_info.get('config_data', {})
        
        # We can handle if we have any config file
        return bool(config_data) and any(
            'config.json' in name or name == 'config.json'
            for name in config_data.keys()
        )
    
    def analyze(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model configuration.
        
        Args:
            model_info: Model metadata
            
        Returns:
            Analysis results
        """
        config_data = model_info.get('config_data', {})
        
        # Find the main config
        config = None
        for name, data in config_data.items():
            if 'config.json' in name or name == 'config.json':
                config = data
                break
        
        if not config:
            return {
                'model_type': 'unknown',
                'model_family': 'unknown',
                'architecture_type': 'unknown'
            }
        
        # Use architecture detector to analyze config
        model_type, model_family, additional_info = self.architecture_detector.detect_from_config(
            Path("dummy_config.json")  # We pass config directly
        )
        
        # Since we can't pass the config directly, we need to work with what we have
        model_type = config.get('model_type', 'unknown').lower()
        architectures = config.get('architectures', [])
        architecture = architectures[0] if architectures else 'unknown'
        
        # Use architecture detector's mappings
        if architecture in self.architecture_detector.ARCHITECTURE_HANDLERS:
            model_family = self.architecture_detector.ARCHITECTURE_HANDLERS[architecture]
        elif model_type in self.architecture_detector.MODEL_TYPE_MAPPING:
            model_family = self.architecture_detector.MODEL_TYPE_MAPPING[model_type]
        else:
            model_family = self._infer_family_from_config(config, model_type)
        
        # Build analysis results
        analysis = {
            'model_type': model_type,
            'model_family': model_family,
            'architecture_type': architecture,
            'architectures': architectures,
            'primary_library': 'transformers',  # Default for config.json models
            'trust_remote_code': self._requires_trust_remote_code(model_type, architecture, config),
        }
        
        # Add configuration details
        analysis['config'] = {
            'hidden_size': config.get('hidden_size'),
            'num_hidden_layers': config.get('num_hidden_layers'),
            'num_attention_heads': config.get('num_attention_heads'),
            'vocab_size': config.get('vocab_size'),
            'max_position_embeddings': config.get('max_position_embeddings'),
            'torch_dtype': config.get('torch_dtype', 'float32'),
        }
        
        # Special handling for multimodal models
        if model_family == 'multimodal':
            analysis['vision_config'] = config.get('vision_config', {})
            analysis['text_config'] = config.get('text_config', {})
            
            # Check for special tokens
            for key in ['image_token_id', 'video_token_id', 'vision_start_token_id']:
                if key in config:
                    analysis['config'][key] = config[key]
        
        # Extract capabilities
        analysis['capabilities'] = self._extract_capabilities(config, model_type, model_family)
        
        # Extract special requirements
        analysis['special_requirements'] = self.extract_special_requirements(model_info)
        
        # Model size estimation
        model_size = self.extract_model_size(model_info)
        if model_size:
            analysis['model_size_b'] = model_size
        
        return analysis
    
    def _infer_family_from_config(self, config: Dict[str, Any], model_type: str) -> str:
        """Infer model family from configuration.
        
        Args:
            config: Model configuration
            model_type: Detected model type
            
        Returns:
            Inferred model family
        """
        # Check for vision/multimodal indicators
        if any(key in config for key in ['vision_config', 'image_token_id', 'vision_tower']):
            return 'multimodal'
        
        # Check for audio indicators
        if any(key in config for key in ['audio_config', 'mel_bins', 'sample_rate']):
            return 'audio-model'
        
        # Check for diffusion indicators
        if 'diffusion' in model_type or 'unet' in str(config.get('architectures', [])).lower():
            return 'image-generation'
        
        # Default to language model for transformer-based models
        if any(key in config for key in ['vocab_size', 'hidden_size', 'num_hidden_layers']):
            return 'language-model'
        
        return 'unknown'
    
    def _requires_trust_remote_code(self, model_type: str, architecture: str, config: Dict[str, Any]) -> bool:
        """Check if model requires trust_remote_code.
        
        Args:
            model_type: Model type
            architecture: Model architecture
            config: Model configuration
            
        Returns:
            Whether trust_remote_code is required
        """
        # Use architecture detector's logic
        if hasattr(self.architecture_detector, '_requires_trust_remote_code'):
            return self.architecture_detector._requires_trust_remote_code(model_type, architecture)
        
        # Fallback logic
        trust_required = [
            'qwen2_5_vl', 'qwen2_vl', 'janus', 'multi_modality',
            'deepseek', 'phi3', 'starcoder', 'baichuan', 'chatglm', 'internlm'
        ]
        
        return any(t in model_type.lower() for t in trust_required) or \
               any(t in architecture.lower() for t in trust_required) or \
               config.get('auto_map', {}) != {}
    
    def _extract_capabilities(self, config: Dict[str, Any], model_type: str, model_family: str) -> Dict[str, Any]:
        """Extract model capabilities from configuration.
        
        Args:
            config: Model configuration
            model_type: Model type
            model_family: Model family
            
        Returns:
            Capabilities dictionary
        """
        capabilities = {
            'supports_streaming': model_family == 'language-model',
            'supports_system_prompt': model_family == 'language-model',
            'supports_reasoning': 'reasoning' in model_type or 'o1' in model_type,
            'max_context_length': config.get('max_position_embeddings', 
                                           config.get('max_sequence_length', 
                                                    config.get('n_positions', 2048))),
            'native_dtype': config.get('torch_dtype', 'float32'),
        }
        
        # Multimodal capabilities
        if model_family == 'multimodal':
            capabilities.update({
                'supports_images': True,
                'supports_multimodal': True,
                'supports_vision': 'vision_config' in config or 'image_token_id' in config,
                'supports_video': 'video_token_id' in config,
            })
        
        # Audio capabilities
        if model_family == 'audio-model':
            capabilities.update({
                'supports_audio': True,
                'sample_rate': config.get('sample_rate', 16000),
                'supports_streaming': config.get('supports_streaming', False),
            })
        
        return capabilities
    
    def get_handler_class(self) -> str:
        """Get the handler class name.
        
        Returns:
            Handler class name
        """
        # This will be determined by the handler registry based on model_family
        return "auto"  # Let the registry decide
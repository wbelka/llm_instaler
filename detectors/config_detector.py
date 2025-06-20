"""Detector that analyzes model configuration files.

This detector examines config.json and similar files to determine
model type based on architecture and configuration parameters.
"""

from typing import Dict, Any, List
from detectors.base import BaseDetector


class ConfigDetector(BaseDetector):
    """Detector that analyzes model configuration to determine type."""
    
    @property
    def name(self) -> str:
        """Get detector name."""
        return "Configuration-based Detector"
    
    @property
    def priority(self) -> int:
        """Get detector priority."""
        return 100  # High priority - check config first
    
    def matches(self, model_info: Dict[str, Any]) -> bool:
        """Check if model has analyzable configuration.
        
        Args:
            model_info: Model metadata.
            
        Returns:
            True if configuration can be analyzed.
        """
        config = model_info.get('config', {})
        
        # Must have some configuration
        if not config:
            return False
        
        # Check for standard architecture indicators
        has_architecture = any([
            'architectures' in config,
            'model_type' in config,
            '_class_name' in config,  # Diffusers models
            'model_class' in config
        ])
        
        return has_architecture
    
    def analyze(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model configuration.
        
        Args:
            model_info: Model metadata.
            
        Returns:
            Analysis results.
        """
        config = model_info.get('config', {})
        analysis = {}
        
        # Determine model type from architecture
        model_type, model_family = self._determine_model_type(config)
        analysis['model_type'] = model_type
        analysis['model_family'] = model_family
        
        # Determine architecture type
        analysis['architecture_type'] = self._determine_architecture_type(config)
        
        # Extract special requirements
        analysis['special_requirements'] = self._extract_special_requirements(config)
        
        # Determine primary library
        analysis['primary_library'] = self._determine_library(config, model_info)
        
        # Extract model size if possible
        model_size = self.extract_model_size(model_info)
        if model_size:
            analysis['model_size_b'] = model_size
        
        # Extract capabilities
        analysis['capabilities'] = self._extract_capabilities(config)
        
        return analysis
    
    def _determine_model_type(self, config: Dict[str, Any]) -> tuple[str, str]:
        """Determine model type and family from configuration.
        
        Args:
            config: Model configuration.
            
        Returns:
            Tuple of (model_type, model_family).
        """
        # Check architectures field
        architectures = config.get('architectures', [])
        if architectures:
            arch = architectures[0].lower()
            
            # Language models
            if 'causallm' in arch:
                return 'transformer', 'language-model'
            elif 'maskedlm' in arch:
                return 'transformer-encoder', 'language-model'
            elif 'seq2seq' in arch:
                return 'transformer-seq2seq', 'language-model'
            elif 'sequenceclassification' in arch:
                return 'transformer-classifier', 'text-classifier'
            
            # Vision models
            elif 'imageclassification' in arch:
                return 'vision-classifier', 'computer-vision'
            elif 'objectdetection' in arch:
                return 'object-detector', 'computer-vision'
            
            # Multimodal
            elif 'visiontext' in arch or 'multimodal' in arch:
                return 'multimodal', 'vision-language'
        
        # Check model_type field
        model_type = config.get('model_type', '').lower()
        
        # Transformer variants
        if model_type in ['gpt2', 'gpt_neo', 'gptj', 'llama', 'mistral', 'mixtral']:
            return model_type, 'language-model'
        
        # Special architectures
        elif model_type == 'mamba':
            return 'mamba', 'language-model'
        elif model_type == 'rwkv':
            return 'rwkv', 'language-model'
        elif model_type == 'whisper':
            return 'whisper', 'audio-model'
        elif model_type == 'clip':
            return 'clip', 'multimodal'
        
        # Check for vision + text configs (multimodal)
        if 'vision_config' in config and 'text_config' in config:
            return 'vision-language', 'multimodal'
        
        # Check for diffusion models
        if '_class_name' in config:
            class_name = config['_class_name'].lower()
            if 'diffusion' in class_name:
                return 'diffusion', 'image-generation'
        
        # Default fallback
        return 'unknown', 'unknown'
    
    def _determine_architecture_type(self, config: Dict[str, Any]) -> str:
        """Determine architecture type.
        
        Args:
            config: Model configuration.
            
        Returns:
            Architecture type string.
        """
        # Check if encoder-decoder
        if config.get('is_encoder_decoder', False):
            return 'encoder-decoder'
        
        # Check architectures
        architectures = config.get('architectures', [])
        if architectures:
            arch = architectures[0].lower()
            if 'encoder' in arch and 'decoder' not in arch:
                return 'encoder-only'
            elif 'decoder' in arch or 'causallm' in arch:
                return 'decoder-only'
        
        # Check for specific indicators
        if 'vision_config' in config and 'text_config' in config:
            return 'dual-encoder'
        
        return 'unknown'
    
    def _extract_special_requirements(self, config: Dict[str, Any]) -> List[str]:
        """Extract special requirements from configuration.
        
        Args:
            config: Model configuration.
            
        Returns:
            List of special requirements.
        """
        requirements = []
        
        # Check for Flash Attention
        if config.get('_attn_implementation') == 'flash_attention_2':
            requirements.append('flash-attn')
        elif config.get('use_flash_attention', False):
            requirements.append('flash-attn')
        elif config.get('use_flash_attn', False):
            requirements.append('flash-attn')
        
        # Check for special architectures
        model_type = config.get('model_type', '').lower()
        if model_type == 'mamba':
            requirements.extend(['mamba-ssm', 'causal-conv1d'])
        elif model_type == 'rwkv':
            requirements.append('rwkv')
        
        # Check for 8-bit/4-bit support
        if config.get('load_in_8bit', False):
            requirements.append('bitsandbytes')
        elif config.get('load_in_4bit', False):
            requirements.append('bitsandbytes')
        
        return requirements
    
    def _determine_library(self, config: Dict[str, Any], model_info: Dict[str, Any]) -> str:
        """Determine primary library for the model.
        
        Args:
            config: Model configuration.
            model_info: Full model information.
            
        Returns:
            Library name.
        """
        # Check library_name in model info
        library_name = model_info.get('library_name', '')
        if library_name:
            return library_name
        
        # Check for diffusers
        if '_class_name' in config or 'model_index.json' in model_info.get('files', []):
            return 'diffusers'
        
        # Check for sentence-transformers
        if 'sentence_transformers' in str(config.get('architectures', [])):
            return 'sentence-transformers'
        
        # Default to transformers for most models
        if 'architectures' in config:
            return 'transformers'
        
        return 'unknown'
    
    def _extract_capabilities(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model capabilities from configuration.
        
        Args:
            config: Model configuration.
            
        Returns:
            Dictionary of capabilities.
        """
        capabilities = {}
        
        # Check for reasoning models
        model_type = config.get('model_type', '').lower()
        if model_type in ['o1', 'reasoning-llm']:
            capabilities['supports_reasoning'] = True
        
        # Check context length
        max_length = config.get('max_position_embeddings',
                               config.get('max_sequence_length',
                                        config.get('n_positions')))
        if max_length:
            capabilities['max_context_length'] = max_length
        
        # Check for specific features
        if config.get('use_cache', True):
            capabilities['supports_caching'] = True
        
        if 'num_labels' in config:
            capabilities['num_labels'] = config['num_labels']
        
        return capabilities
    
    def get_handler_class(self) -> str:
        """Get appropriate handler class.
        
        Returns:
            Handler class name.
        """
        # This detector can identify multiple types
        # The actual handler is determined by model_family
        return 'auto'  # Special value indicating dynamic selection
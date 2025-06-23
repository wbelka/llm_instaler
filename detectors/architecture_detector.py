"""Architecture detection for various model types.

This module provides functionality to detect model architectures
from configuration files and determine appropriate handlers.
"""

import json
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class ArchitectureDetector:
    """Detects model architecture from configuration."""
    
    # Mapping of model types to model families
    MODEL_TYPE_MAPPING = {
        # Language models
        'llama': 'language-model',
        'mistral': 'language-model',
        'mixtral': 'language-model',
        'qwen2': 'language-model',
        'qwen3': 'language-model',
        'gemma': 'language-model',
        'gemma2': 'language-model',
        'phi': 'language-model',
        'phi3': 'language-model',
        'gpt2': 'language-model',
        'gpt_neox': 'language-model',
        'opt': 'language-model',
        'bloom': 'language-model',
        'falcon': 'language-model',
        'starcoder': 'language-model',
        'codegen': 'language-model',
        'codellama': 'language-model',
        'deepseek': 'language-model',
        'deepseek_v2': 'language-model',
        'mamba': 'language-model',
        'rwkv': 'language-model',
        'jamba': 'language-model',
        'qwen3': 'language-model',
        'qwen-3': 'language-model',
        
        # Multimodal models
        'llava': 'multimodal',
        'clip': 'multimodal',
        'blip': 'multimodal',
        'blip2': 'multimodal',
        'flamingo': 'multimodal',
        'idefics': 'multimodal',
        'kosmos': 'multimodal',
        'fuyu': 'multimodal',
        'qwen2_vl': 'multimodal',
        'qwen2_5_vl': 'multimodal',
        'janus': 'multimodal',
        'multi_modality': 'multimodal',
        
        # Vision models
        'vit': 'vision',
        'deit': 'vision',
        'swin': 'vision',
        'convnext': 'vision',
        'resnet': 'vision',
        'efficientnet': 'vision',
        
        # Audio models
        'whisper': 'audio-model',
        'wav2vec2': 'audio-model',
        'hubert': 'audio-model',
        'seamless_m4t': 'audio-model',
        'musicgen': 'audio-model',
        
        # Diffusion models
        'stable_diffusion': 'image-generation',
        'sdxl': 'image-generation',
        'dit': 'image-generation',
        'pixart': 'image-generation',
        'dalle': 'image-generation',
        'kandinsky': 'image-generation',
        
        # Video models
        'text_to_video': 'video-generation',
        'video_diffusion': 'video-generation',
        'animatediff': 'video-generation',
        
        # Embedding models
        'bert': 'embedding',
        'roberta': 'embedding',
        'sentence_transformers': 'embedding',
        'e5': 'embedding',
        'bge': 'embedding',
        'gte': 'embedding',
        
        # Reasoning models
        'o1': 'reasoning',
        'reasoning': 'reasoning',
    }
    
    # Architecture to handler mapping
    ARCHITECTURE_HANDLERS = {
        # Qwen2.5-VL specific
        'Qwen2_5_VLForConditionalGeneration': 'multimodal',
        'Qwen2VLForConditionalGeneration': 'multimodal',
        
        # Janus specific
        'MultiModalityPreTrainedModel': 'multimodal',
        
        # Standard transformers
        'LlamaForCausalLM': 'language-model',
        'MistralForCausalLM': 'language-model',
        'Qwen2ForCausalLM': 'language-model',
        'Qwen3ForCausalLM': 'language-model',
        'GemmaForCausalLM': 'language-model',
        'PhiForCausalLM': 'language-model',
        
        # Vision-language models
        'LlavaForConditionalGeneration': 'multimodal',
        'Blip2ForConditionalGeneration': 'multimodal',
        'ClipModel': 'multimodal',
        
        # Diffusion models
        'StableDiffusionPipeline': 'image-generation',
        'StableDiffusionXLPipeline': 'image-generation',
        'PixArtAlphaPipeline': 'image-generation',
        'TextToVideoSDPipeline': 'video-generation',
        
        # Audio models
        'WhisperForConditionalGeneration': 'audio-model',
        'Wav2Vec2ForCTC': 'audio-model',
        'MusicgenForConditionalGeneration': 'audio-model',
    }
    
    def detect_from_config(self, config_path: Path) -> Tuple[str, str, Dict[str, Any]]:
        """Detect model type and family from config.json.
        
        Args:
            config_path: Path to config.json file
            
        Returns:
            Tuple of (model_type, model_family, additional_info)
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Get model type from config
            model_type = config.get('model_type', '').lower()
            
            # Get architectures if available
            architectures = config.get('architectures', [])
            architecture = architectures[0] if architectures else None
            
            # Determine model family
            model_family = 'unknown'
            
            # First check architecture mapping
            if architecture and architecture in self.ARCHITECTURE_HANDLERS:
                model_family = self.ARCHITECTURE_HANDLERS[architecture]
            # Then check model type mapping
            elif model_type in self.MODEL_TYPE_MAPPING:
                model_family = self.MODEL_TYPE_MAPPING[model_type]
            # Special cases
            elif 'vision' in model_type and 'language' in model_type:
                model_family = 'multimodal'
            elif 'text-to-image' in model_type or 'diffusion' in model_type:
                model_family = 'image-generation'
            elif 'text-to-video' in model_type:
                model_family = 'video-generation'
            
            # Additional info
            additional_info = {
                'architectures': architectures,
                'hidden_size': config.get('hidden_size'),
                'num_hidden_layers': config.get('num_hidden_layers'),
                'vocab_size': config.get('vocab_size'),
                'max_position_embeddings': config.get('max_position_embeddings'),
                'torch_dtype': config.get('torch_dtype', 'float32'),
                'model_type': model_type,
                'trust_remote_code': self._requires_trust_remote_code(model_type, architecture),
            }
            
            # Special handling for multimodal models
            if model_family == 'multimodal':
                additional_info['vision_config'] = config.get('vision_config', {})
                additional_info['text_config'] = config.get('text_config', {})
                
                # Check for special tokens that indicate multimodal capability
                for key in ['image_token_id', 'video_token_id', 'vision_start_token_id']:
                    if key in config:
                        additional_info[key] = config[key]
            
            return model_type, model_family, additional_info
            
        except Exception as e:
            logger.error(f"Error detecting architecture from config: {e}")
            return 'unknown', 'unknown', {}
    
    def _requires_trust_remote_code(self, model_type: str, architecture: str) -> bool:
        """Determine if model requires trust_remote_code.
        
        Args:
            model_type: Model type string
            architecture: Architecture string
            
        Returns:
            Whether trust_remote_code is required
        """
        # Models that typically require trust_remote_code
        trust_required = [
            'qwen3',
            'qwen-3',
            'qwen2_5_vl',
            'qwen2_vl', 
            'janus',
            'multi_modality',
            'deepseek',
            'phi3',
            'starcoder',
            'baichuan',
            'chatglm',
            'internlm',
        ]
        
        # Check if model type requires trust
        if any(t in model_type.lower() for t in trust_required):
            return True
            
        # Check architecture
        if architecture and any(t in architecture.lower() for t in trust_required):
            return True
            
        return False
    
    def detect_from_model_files(self, model_path: Path) -> Tuple[str, str, Dict[str, Any]]:
        """Detect model type from model files in directory.
        
        Args:
            model_path: Path to model directory
            
        Returns:
            Tuple of (model_type, model_family, additional_info)
        """
        # Check for config.json
        config_path = model_path / 'config.json'
        if config_path.exists():
            return self.detect_from_config(config_path)
        
        # Check for model_index.json (diffusers)
        model_index_path = model_path / 'model_index.json'
        if model_index_path.exists():
            return self._detect_diffusers_model(model_index_path)
        
        # Check for specific model files
        return self._detect_by_file_patterns(model_path)
    
    def _detect_diffusers_model(self, model_index_path: Path) -> Tuple[str, str, Dict[str, Any]]:
        """Detect diffusers model type.
        
        Args:
            model_index_path: Path to model_index.json
            
        Returns:
            Tuple of (model_type, model_family, additional_info)
        """
        try:
            with open(model_index_path, 'r') as f:
                model_index = json.load(f)
            
            pipeline_class = model_index.get('_class_name', '')
            
            # Determine family from pipeline class
            if 'Video' in pipeline_class:
                model_family = 'video-generation'
                model_type = 'text-to-video'
            elif 'Audio' in pipeline_class:
                model_family = 'audio-model'
                model_type = 'audio-diffusion'
            else:
                model_family = 'image-generation'
                model_type = 'stable-diffusion'
            
            additional_info = {
                'pipeline_class': pipeline_class,
                'scheduler': model_index.get('scheduler', [None])[0],
                'requires_safety_checker': 'safety_checker' in model_index,
            }
            
            return model_type, model_family, additional_info
            
        except Exception as e:
            logger.error(f"Error detecting diffusers model: {e}")
            return 'unknown', 'unknown', {}
    
    def _detect_by_file_patterns(self, model_path: Path) -> Tuple[str, str, Dict[str, Any]]:
        """Detect model type by file patterns.
        
        Args:
            model_path: Path to model directory
            
        Returns:
            Tuple of (model_type, model_family, additional_info)
        """
        # Check for common patterns
        files = list(model_path.glob('*'))
        file_names = [f.name for f in files]
        
        # Diffusers models
        if 'unet' in file_names or 'vae' in file_names:
            if any('video' in f for f in file_names):
                return 'text-to-video', 'video-generation', {}
            else:
                return 'stable-diffusion', 'image-generation', {}
        
        # Transformers models
        if any(f.endswith('.safetensors') or f.endswith('.bin') for f in file_names):
            # Try to guess from file names
            for fname in file_names:
                fname_lower = fname.lower()
                if 'vision' in fname_lower or 'clip' in fname_lower:
                    return 'vision-language', 'multimodal', {}
                elif 'whisper' in fname_lower or 'wav2vec' in fname_lower:
                    return 'speech', 'audio-model', {}
            
            # Default to language model
            return 'transformer', 'language-model', {}
        
        return 'unknown', 'unknown', {}


# Singleton instance
_detector = None


def get_architecture_detector() -> ArchitectureDetector:
    """Get the architecture detector instance."""
    global _detector
    if _detector is None:
        _detector = ArchitectureDetector()
    return _detector
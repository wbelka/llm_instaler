"""
Utilities for diffusers model analysis
"""

import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


def estimate_component_parameters(component_type: str, config: Dict[str, Any]) -> Optional[int]:
    """
    Estimate parameter count for a diffusers component based on its config
    
    Returns:
        Estimated parameter count or None if unable to estimate
    """
    try:
        if component_type == 'unet':
            return estimate_unet_parameters(config)
        elif component_type == 'text_encoder':
            return estimate_text_encoder_parameters(config)
        elif component_type == 'vae':
            return estimate_vae_parameters(config)
        elif component_type == 'transformer':
            return estimate_transformer_parameters(config)
        else:
            return None
    except Exception as e:
        logger.debug(f"Failed to estimate parameters for {component_type}: {e}")
        return None


def estimate_unet_parameters(config: Dict[str, Any]) -> int:
    """Estimate UNet2D parameter count from config"""
    # Simplified estimation based on key architecture parameters
    # This is approximate but better than blind guessing
    
    in_channels = config.get('in_channels', 4)
    out_channels = config.get('out_channels', 4)
    
    # Get block channel counts
    block_out_channels = config.get('block_out_channels', [320, 640, 1280, 1280])
    layers_per_block = config.get('layers_per_block', 2)
    
    # Cross attention dimension
    cross_attention_dim = config.get('cross_attention_dim', 768)
    
    # Very rough estimation based on typical UNet architecture
    # Each block has conv layers + attention layers
    total_params = 0
    
    # Down blocks
    prev_channels = in_channels
    for channels in block_out_channels:
        # Conv layers
        total_params += prev_channels * channels * 9 * layers_per_block  # 3x3 conv
        # Self attention
        total_params += channels * channels * 4  # Q, K, V, O projections
        # Cross attention if enabled
        if cross_attention_dim:
            total_params += channels * cross_attention_dim * 4
        prev_channels = channels
    
    # Middle block
    total_params += prev_channels * prev_channels * 9 * 2
    
    # Up blocks (similar to down blocks)
    total_params *= 2  # Roughly double for up blocks
    
    # Add some overhead for normalization, embeddings, etc.
    total_params = int(total_params * 1.2)
    
    return total_params


def estimate_text_encoder_parameters(config: Dict[str, Any]) -> int:
    """Estimate text encoder (CLIP/BERT) parameter count from config"""
    hidden_size = config.get('hidden_size', 768)
    num_layers = config.get('num_hidden_layers', 12)
    num_heads = config.get('num_attention_heads', 12)
    intermediate_size = config.get('intermediate_size', 3072)
    vocab_size = config.get('vocab_size', 30000)
    max_position_embeddings = config.get('max_position_embeddings', 512)
    
    # Embeddings
    params = vocab_size * hidden_size  # Token embeddings
    params += max_position_embeddings * hidden_size  # Position embeddings
    
    # Each transformer layer
    layer_params = 0
    # Self attention
    layer_params += hidden_size * hidden_size * 4  # Q, K, V, O
    # FFN
    layer_params += hidden_size * intermediate_size * 2  # Up and down projection
    # Layer norms
    layer_params += hidden_size * 4  # Rough estimate for norms and biases
    
    params += layer_params * num_layers
    
    return int(params)


def estimate_vae_parameters(config: Dict[str, Any]) -> int:
    """Estimate VAE parameter count from config"""
    in_channels = config.get('in_channels', 3)
    out_channels = config.get('out_channels', 3)
    latent_channels = config.get('latent_channels', 4)
    
    # Get block channel counts
    block_out_channels = config.get('block_out_channels', [128, 256, 512, 512])
    layers_per_block = config.get('layers_per_block', 2)
    
    # Encoder
    total_params = 0
    prev_channels = in_channels
    for channels in block_out_channels:
        total_params += prev_channels * channels * 9 * layers_per_block
        prev_channels = channels
    
    # Latent space
    total_params += prev_channels * latent_channels * 9
    
    # Decoder (roughly symmetric)
    total_params *= 2
    
    # Add overhead
    total_params = int(total_params * 1.1)
    
    return total_params


def estimate_transformer_parameters(config: Dict[str, Any]) -> int:
    """Estimate transformer (for FLUX/SD3) parameter count from config"""
    hidden_size = config.get('hidden_size', 1024)
    num_layers = config.get('num_layers', 24)
    num_heads = config.get('num_attention_heads', 16)
    
    # Similar to text encoder but often larger
    params = hidden_size * hidden_size * 4 * num_layers  # Attention
    params += hidden_size * hidden_size * 8 * num_layers  # FFN (larger ratio)
    params += hidden_size * 8 * num_layers  # Norms and biases
    
    return int(params)


def parameters_to_size_gb(param_count: int, dtype: str = 'float32') -> float:
    """Convert parameter count to size in GB based on dtype"""
    bytes_per_param = {
        'float32': 4,
        'float16': 2,
        'bfloat16': 2,
        'int8': 1,
        'int4': 0.5
    }
    
    bytes_per = bytes_per_param.get(dtype, 4)
    size_bytes = param_count * bytes_per
    size_gb = size_bytes / (1024 ** 3)
    
    return round(size_gb, 2)
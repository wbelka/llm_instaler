"""LoRA utilities for model training and adaptation.

This module provides utility functions for LoRA (Low-Rank Adaptation) operations,
including finding suitable layers for LoRA application.
"""

import logging
from typing import List, Optional, Any
import torch.nn as nn

logger = logging.getLogger(__name__)


def find_all_linear_names(model: Any, quantization_config: Optional[Any] = None) -> List[str]:
    """Find all linear layer names in the model suitable for LoRA application.
    
    This function identifies all Linear layers that can be targeted for LoRA,
    excluding certain layers like lm_head and embeddings that shouldn't be adapted.
    
    Args:
        model: The model to analyze (PyTorch model).
        quantization_config: Quantization config (for 4-bit/8-bit models).
        
    Returns:
        List of linear layer names suitable for LoRA application.
        
    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b")
        >>> target_modules = find_all_linear_names(model)
        >>> print(target_modules)
        ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    """
    # Get the base model if it's wrapped (e.g., in PEFT)
    if hasattr(model, 'model'):
        base_model = model.model
    elif hasattr(model, 'base_model'):
        base_model = model.base_model
    else:
        base_model = model
    
    # Linear layer classes to look for
    linear_classes = [nn.Linear]
    
    # Add Conv1D for models like GPT2
    try:
        from transformers.pytorch_utils import Conv1D
        linear_classes.append(Conv1D)
    except ImportError:
        pass
    
    # Add quantized linear layers if using quantization
    if quantization_config is not None:
        try:
            from bitsandbytes.nn import Linear8bitLt, Linear4bit
            linear_classes.extend([Linear8bitLt, Linear4bit])
        except ImportError:
            logger.warning("bitsandbytes not available, skipping quantized layers")
    
    # Find all linear layer names
    linear_names = []
    seen_names = set()
    
    for name, module in base_model.named_modules():
        # Skip the base model name
        if name == '':
            continue
            
        # Check if this is a linear layer
        if any(isinstance(module, linear_class) for linear_class in linear_classes):
            # Skip output layers and embeddings
            excluded_keywords = [
                'lm_head',           # Language modeling head
                'embed_tokens',      # Token embeddings
                'wte',              # Word token embeddings (GPT-style)
                'wpe',              # Word position embeddings (GPT-style)
                'embeddings',       # General embeddings
                'classifier',       # Classification head
                'pooler',          # Pooling layer
                'ln_f',            # Final layer norm
                'score',           # Scoring head
                'output',          # Output layers
                'head'             # General head layers
            ]
            
            if any(excluded in name.lower() for excluded in excluded_keywords):
                logger.debug(f"Excluding layer: {name}")
                continue
            
            # Extract the projection name (e.g., 'q_proj' from 'model.layers.0.self_attn.q_proj')
            names = name.split('.')
            layer_name = names[-1]
            
            # Add to list if not already present
            if layer_name not in seen_names:
                linear_names.append(layer_name)
                seen_names.add(layer_name)
                logger.debug(f"Found linear layer: {layer_name}")
    
    # Common LoRA target modules for known architectures
    # This helps ensure we get the right modules even if naming is slightly different
    common_targets = {
        # Llama-style models
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj',
        # GPT-style models
        'c_attn', 'c_proj', 'c_fc',
        # BERT-style models
        'query', 'key', 'value', 'dense',
        # T5-style models
        'q', 'k', 'v', 'o', 'wi', 'wo',
        # Qwen-style models
        'c_attn', 'c_proj', 'w1', 'w2',
        # Mistral/Mixtral
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'w1', 'w2', 'w3',
    }
    
    # If we found very few layers, try to add common targets that exist
    if len(linear_names) < 3:
        logger.warning(f"Found only {len(linear_names)} linear layers, checking common targets")
        for name, module in base_model.named_modules():
            if any(isinstance(module, linear_class) for linear_class in linear_classes):
                layer_name = name.split('.')[-1]
                if layer_name in common_targets and layer_name not in seen_names:
                    linear_names.append(layer_name)
                    seen_names.add(layer_name)
    
    if not linear_names:
        logger.warning("No linear layers found for LoRA application!")
        # Fallback to common defaults
        linear_names = ["q_proj", "v_proj"]
    
    logger.info(f"Found {len(linear_names)} linear layers for LoRA: {linear_names}")
    return linear_names


def get_default_lora_target_modules(model_family: str) -> List[str]:
    """Get default LoRA target modules for a specific model family.
    
    Args:
        model_family: The model family name (e.g., 'llama', 'qwen', 'mistral').
        
    Returns:
        List of default target module names for the model family.
    """
    defaults = {
        'llama': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        'llama2': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        'llama3': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        'mistral': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        'mixtral': ["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3"],
        'qwen': ["c_attn", "c_proj", "w1", "w2"],
        'qwen2': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        'gpt2': ["c_attn", "c_proj", "c_fc"],
        'gptj': ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out"],
        'bert': ["query", "key", "value", "dense"],
        'roberta': ["query", "key", "value", "dense"],
        't5': ["q", "k", "v", "o", "wi", "wo"],
        'bart': ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        'opt': ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        'bloom': ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        'falcon': ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        'gemma': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        'phi': ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
        'stablelm': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }
    
    # Return defaults for the family, or common defaults
    return defaults.get(model_family.lower(), ["q_proj", "v_proj"])
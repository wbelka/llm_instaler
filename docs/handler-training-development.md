# Handler Training Development Guide

This guide explains how to implement training support in custom handlers for the LLM Installer.

## Overview

Handlers provide model-specific training configurations that the universal training system uses to optimize fine-tuning for each model type.

## Required Methods

To support training, handlers must implement these methods from `BaseHandler`:

### 1. `get_training_parameters()`

Returns model-specific training parameters:

```python
def get_training_parameters(self) -> Dict[str, Any]:
    """Get model-specific training parameters.
    
    Returns:
        Dictionary with training parameter overrides.
    """
    params = super().get_training_parameters()  # Get base parameters
    
    # Override with model-specific settings
    params.update({
        'lora_target_modules': ["q_proj", "v_proj", "k_proj", "o_proj"],
        'training_precision': 'bf16',  # or 'fp16', 'fp32', 'auto'
        'max_seq_length': 8192,
        'supports_flash_attention': True,
        'dataset_formats': ['alpaca', 'chat', 'vision_qa'],
        # Add any model-specific parameters
    })
    
    return params
```

#### Key Parameters:

- **`lora_target_modules`**: List of module names to apply LoRA to
- **`lora_modules_to_save`**: Additional modules to save (e.g., embeddings)
- **`training_precision`**: Recommended precision ('auto', 'fp16', 'bf16', 'fp32')
- **`gradient_checkpointing`**: Whether to enable gradient checkpointing
- **`max_seq_length`**: Maximum sequence length for training
- **`supports_flash_attention`**: Whether model supports flash attention
- **`dataset_formats`**: List of supported dataset formats
- **`special_tokens`**: Dictionary of special tokens (e.g., image tokens)

### 2. `prepare_model_for_training()`

Prepares the model for training:

```python
def prepare_model_for_training(self, model: Any, training_config: Dict[str, Any]) -> Any:
    """Prepare model for training.
    
    Args:
        model: The loaded model.
        training_config: Training configuration.
        
    Returns:
        Prepared model.
    """
    # Call parent implementation
    model = super().prepare_model_for_training(model, training_config)
    
    # Model-specific preparations
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    
    # Custom configurations
    if self.model_family == 'vision_language':
        # Freeze vision encoder for efficiency
        if hasattr(model, 'vision_model'):
            for param in model.vision_model.parameters():
                param.requires_grad = False
    
    return model
```

### 3. `get_tokenizer_config()`

Returns tokenizer configuration for training:

```python
def get_tokenizer_config(self) -> Dict[str, Any]:
    """Get tokenizer configuration for training.
    
    Returns:
        Dictionary with tokenizer settings.
    """
    config = super().get_tokenizer_config()
    
    # Model-specific tokenizer settings
    config.update({
        'padding_side': 'left',  # Some models need left padding
        'add_bos_token': False,  # Model-specific
        'max_length': 8192,
    })
    
    return config
```

## Implementation Examples

### Example 1: Transformer Model Handler

```python
class MyTransformerHandler(TransformerHandler):
    """Handler for a custom transformer model."""
    
    def get_training_parameters(self) -> Dict[str, Any]:
        params = super().get_training_parameters()
        
        # Specific architecture modules
        params['lora_target_modules'] = [
            "self_attn.q_proj",
            "self_attn.v_proj", 
            "self_attn.k_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj"
        ]
        
        # Model prefers bfloat16
        params['training_precision'] = 'bf16'
        
        # Context window
        params['max_seq_length'] = 4096
        
        # Supports standard formats
        params['dataset_formats'] = ['alpaca', 'sharegpt', 'openai', 'chat']
        
        return params
```

### Example 2: Multimodal Model Handler

```python
class MyMultimodalHandler(MultimodalHandler):
    """Handler for a vision-language model."""
    
    def get_training_parameters(self) -> Dict[str, Any]:
        params = super().get_training_parameters()
        
        # Only train language model parts
        params['lora_target_modules'] = [
            "language_model.q_proj",
            "language_model.v_proj",
            "cross_attention.q_proj",
            "cross_attention.v_proj"
        ]
        
        # Don't use flash attention with images
        params['supports_flash_attention'] = False
        
        # Special tokens
        params['special_tokens'] = {
            'image_token': '<image>',
            'video_token': '<video>'
        }
        
        # Supported formats
        params['dataset_formats'] = [
            'vision_qa',
            'image_caption',
            'video_qa',
            'alpaca'  # Also supports text-only
        ]
        
        return params
    
    def prepare_model_for_training(self, model: Any, training_config: Dict[str, Any]) -> Any:
        model = super().prepare_model_for_training(model, training_config)
        
        # Freeze vision encoder
        if hasattr(model, 'vision_tower'):
            for param in model.vision_tower.parameters():
                param.requires_grad = False
            print("Vision encoder frozen for training")
        
        # Enable gradient checkpointing only for language model
        if hasattr(model, 'language_model'):
            model.language_model.gradient_checkpointing_enable()
        
        return model
```

### Example 3: MoE Model Handler

```python
class MoEHandler(BaseHandler):
    """Handler for Mixture of Experts models."""
    
    def get_training_parameters(self) -> Dict[str, Any]:
        params = super().get_training_parameters()
        
        # MoE specific modules
        params['lora_target_modules'] = [
            "self_attn.q_proj",
            "self_attn.v_proj",
            "block_sparse_moe.gate",  # Train router
            "block_sparse_moe.experts.*.w1",  # All expert MLPs
            "block_sparse_moe.experts.*.w2",
            "block_sparse_moe.experts.*.w3"
        ]
        
        # MoE training settings
        params['moe_training'] = True
        params['expert_selection'] = 'top-2'
        params['load_balancing_loss_coef'] = 0.01
        
        # Larger batch size for MoE
        params['recommended_batch_size'] = 8
        
        return params
```

## Advanced Features

### Dynamic Parameter Adjustment

```python
def get_training_parameters(self) -> Dict[str, Any]:
    params = super().get_training_parameters()
    
    # Adjust based on model size
    model_size = self._estimate_model_size()
    
    if model_size < 1:  # < 1B
        params['lora_target_modules'] = ["q_proj", "v_proj"]  # Fewer modules
        params['recommended_learning_rate'] = 5e-4
    elif model_size < 7:  # 1-7B
        params['lora_target_modules'] = ["q_proj", "v_proj", "k_proj", "o_proj"]
        params['recommended_learning_rate'] = 2e-4
    else:  # > 7B
        # All attention and MLP modules
        params['lora_target_modules'] = [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        params['recommended_learning_rate'] = 1e-4
        params['gradient_checkpointing'] = True  # Memory efficiency
    
    return params
```

### Custom Dataset Support

```python
def get_training_parameters(self) -> Dict[str, Any]:
    params = super().get_training_parameters()
    
    # Support for specialized datasets
    params['dataset_formats'] = [
        'alpaca',
        'medical_qa',      # Custom format
        'code_instruct',   # Custom format
        'multi_choice'     # Custom format
    ]
    
    # Dataset-specific preprocessing hints
    params['dataset_preprocessing'] = {
        'medical_qa': {
            'add_context': True,
            'max_context_length': 2048
        },
        'code_instruct': {
            'syntax_highlighting': True,
            'language_tags': True
        }
    }
    
    return params
```

### Hardware-Aware Configuration

```python
def get_training_parameters(self) -> Dict[str, Any]:
    params = super().get_training_parameters()
    
    # Check available hardware
    import torch
    if torch.cuda.is_available():
        compute_capability = torch.cuda.get_device_capability()
        
        if compute_capability[0] >= 8:  # Ampere or newer
            params['training_precision'] = 'bf16'
            params['supports_flash_attention'] = True
        elif compute_capability[0] >= 7:  # Volta/Turing
            params['training_precision'] = 'fp16'
            params['supports_flash_attention'] = True
        else:
            params['training_precision'] = 'fp32'
            params['supports_flash_attention'] = False
    
    return params
```

## Testing Your Handler

### Unit Test Example

```python
def test_training_parameters():
    """Test handler training configuration."""
    from handlers.my_handler import MyHandler
    
    # Create handler
    model_info = {'model_id': 'test/model', 'model_size_gb': 7.0}
    handler = MyHandler(model_info)
    
    # Get training parameters
    params = handler.get_training_parameters()
    
    # Verify required fields
    assert 'lora_target_modules' in params
    assert isinstance(params['lora_target_modules'], list)
    assert len(params['lora_target_modules']) > 0
    
    # Verify supported formats
    assert 'dataset_formats' in params
    assert 'alpaca' in params['dataset_formats']
    
    print("Training parameters test passed!")
```

### Integration Test

```python
def test_training_integration():
    """Test full training integration."""
    import subprocess
    
    # Install test model
    result = subprocess.run([
        './llm-installer', 'install', 'test/small-model'
    ], capture_output=True)
    assert result.returncode == 0
    
    # Run training
    result = subprocess.run([
        './train.sh', 
        '--data', 'test_data.json',
        '--max-examples', '10',
        '--mode', 'fast'
    ], cwd='models/test_small-model', capture_output=True)
    
    assert result.returncode == 0
    assert Path('models/test_small-model/lora').exists()
```

## Best Practices

### 1. **Conservative Defaults**
Always provide sensible defaults that work for most cases:
```python
params = {
    'lora_target_modules': None,  # Auto-detect if None
    'gradient_checkpointing': True,  # Safe default
    'max_seq_length': 2048,  # Conservative default
}
```

### 2. **Document Special Requirements**
```python
def get_training_parameters(self) -> Dict[str, Any]:
    """Get training parameters.
    
    Note: This model requires special preprocessing for
    multi-turn conversations. See dataset_formats for
    supported formats.
    """
    params = super().get_training_parameters()
    params['special_requirements'] = {
        'min_gpu_memory': 16,  # GB
        'requires_flash_attn': True,
        'notes': 'Best results with ShareGPT format'
    }
    return params
```

### 3. **Validate Configuration**
```python
def prepare_model_for_training(self, model: Any, training_config: Dict[str, Any]) -> Any:
    # Validate configuration
    if training_config.get('max_seq_length', 0) > 8192:
        print("Warning: This model performs best with sequences <= 8192")
    
    if training_config.get('batch_size', 1) > 4 and training_config.get('use_4bit'):
        print("Warning: Large batch size with 4-bit may cause instability")
    
    return super().prepare_model_for_training(model, training_config)
```

### 4. **Provide Clear Error Messages**
```python
def get_training_parameters(self) -> Dict[str, Any]:
    params = super().get_training_parameters()
    
    # Check model variant
    if 'base' in self.model_id.lower():
        raise ValueError(
            "This appears to be a base model. "
            "Training is optimized for instruction-tuned variants. "
            "Consider using the -instruct or -chat version."
        )
    
    return params
```

## Common Patterns

### Pattern 1: Vision-Language Models
- Freeze vision encoder
- Only train cross-attention and language model
- Disable flash attention
- Support image-text pair formats

### Pattern 2: Code Models  
- Include all linear layers in LoRA
- Support longer context (8K+)
- Preserve code-specific tokens
- Support code-instruct formats

### Pattern 3: Small Models (<3B)
- Can train more modules
- Higher learning rates work well
- Full precision often better than mixed
- Support for all formats

### Pattern 4: Large Models (>30B)
- Minimal LoRA modules (attention only)
- Require gradient checkpointing
- Force 4-bit quantization
- Limited format support

## Debugging Tips

1. **Enable Verbose Logging**
   ```python
   import logging
   logger = logging.getLogger(__name__)
   
   def get_training_parameters(self):
       params = super().get_training_parameters()
       logger.info(f"Training params for {self.model_id}: {params}")
       return params
   ```

2. **Test Module Names**
   ```python
   def verify_target_modules(self, model):
       """Verify target modules exist in model."""
       for module_name in self.get_training_parameters()['lora_target_modules']:
           try:
               # Try to access module
               _ = model.get_submodule(module_name)
           except AttributeError:
               print(f"Warning: Module '{module_name}' not found in model")
   ```

3. **Memory Profiling**
   ```python
   def prepare_model_for_training(self, model, config):
       if config.get('debug_memory'):
           import torch
           print(f"Model memory before: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
       
       model = super().prepare_model_for_training(model, config)
       
       if config.get('debug_memory'):
           print(f"Model memory after: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
       
       return model
   ```

Remember: The training system relies on handlers to provide model-specific optimizations. Well-implemented training support in your handler ensures users get the best fine-tuning experience for your model type.
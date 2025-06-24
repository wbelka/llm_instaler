# Quantization Configuration System

## Overview

The LLM Installer now uses a centralized quantization configuration system to manage quantization support across all model handlers. This eliminates hardcoded values and provides a single source of truth for quantization settings.

## Key Components

### 1. QuantizationConfig Class (`core/quantization_config.py`)

Central configuration class that manages:
- Which model families support quantization
- BitsAndBytes version requirements
- Special configuration for specific models
- Compute dtype preferences

### 2. Configuration Structure

```python
QUANTIZATION_SUPPORTED_FAMILIES = {
    "transformer": True,
    "llama": True,
    "gemma3": True,
    # ... etc
}

SPECIAL_CONFIGS = {
    "gemma3": {
        "preferred_compute_dtype": "bfloat16",
        "supports_flash_attention_with_quantization": False,
    },
    # ... etc
}
```

### 3. Usage in Handlers

Handlers now use the centralized config instead of hardcoding:

```python
# Old way (hardcoded):
if 'bitsandbytes' not in str(deps):
    deps.append('bitsandbytes>=0.41.0')

# New way (centralized):
from core.quantization_config import QuantizationConfig
quant_deps = QuantizationConfig.get_quantization_dependencies(
    self.model_type, self.model_family
)
deps.extend(quant_deps)
```

## Benefits

1. **Single Source of Truth**: All quantization settings in one place
2. **Easy Updates**: Change BitsAndBytes version in one location
3. **Model-Specific Config**: Different models can have different settings
4. **Better Detection**: Centralized logic for determining quantization support
5. **Flexibility**: Easy to add new models or change configurations

## Adding New Models

To add quantization support for a new model:

1. Add the model family to `QUANTIZATION_SUPPORTED_FAMILIES` or `QUANTIZATION_UNSUPPORTED_FAMILIES`
2. If needed, add special configuration to `SPECIAL_CONFIGS`
3. The handler will automatically use the centralized configuration

## Special Configurations

### Gemma3
- Preferred compute dtype: `bfloat16`
- Flash attention disabled with quantization

### Qwen3
- Preferred compute dtype: `float16`
- Flash attention supported with quantization

## Future Improvements

1. **Dynamic Version Detection**: Detect optimal BitsAndBytes version based on CUDA version
2. **Model-Specific Versions**: Different BitsAndBytes versions for different models
3. **Quantization Method Selection**: Support for different quantization methods (NF4, FP4, etc.)
4. **Performance Profiles**: Pre-configured settings for different hardware
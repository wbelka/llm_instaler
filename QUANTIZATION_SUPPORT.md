# Quantization Support in LLM Installer

## Overview
This document describes quantization support across different model handlers in LLM Installer.

## Supported Handlers

### ✅ Full Quantization Support
These handlers fully support 8-bit and 4-bit quantization via `--dtype int8` or `--dtype int4`:

1. **TransformerHandler** - All transformer-based language models
   - Uses `get_quantization_config` from base handler
   - Supports BitsAndBytes quantization
   - Works with GPT, LLaMA, Mistral, etc.

2. **Qwen3Handler** - Qwen3 models with thinking mode
   - Inherits from TransformerHandler
   - Full quantization support
   - Includes bitsandbytes in dependencies

3. **QwenVLHandler** - Qwen vision-language models
   - Uses `get_quantization_config` from base handler
   - Supports quantization for large VL models

4. **MultimodalHandler** - General multimodal models
   - Updated to use `get_quantization_config`
   - Supports quantization for Janus and other multimodal models

5. **Gemma3Handler** - Google Gemma 3 multimodal models
   - Updated to use `get_quantization_config`
   - Full quantization support for memory efficiency

6. **EmbeddingHandler** - Embedding models (transformer-based only)
   - Added quantization support for transformer-based embeddings
   - Sentence transformers and CLIP models don't support quantization

7. **JanusHandler** - Deepseek Janus models
   - Inherits from MultimodalHandler
   - Full quantization support

8. **SpecializedHandler** - Reasoning and specialized models
   - Delegates to TransformerHandler
   - Full quantization support

## ❌ No Quantization Support
These handlers don't support quantization (by design):

1. **DiffusionHandler** - Image/video generation models
   - Diffusion models use different optimization techniques
   - Not compatible with BitsAndBytes quantization

2. **VisionHandler** - Computer vision models
   - Classification/detection models are already optimized
   - Usually small enough without quantization

3. **AudioHandler** - Audio processing models
   - Whisper, Bark, etc. use different architectures
   - Not compatible with standard quantization

## Usage

To use quantization with any supported model:

```bash
# 4-bit quantization (recommended for most cases)
./llm-installer install <model_id> --dtype int4

# 8-bit quantization
./llm-installer install <model_id> --dtype int8

# Start with quantization
./start.sh --dtype int4
```

## Requirements

- CUDA-capable GPU (quantization requires CUDA)
- bitsandbytes library (automatically installed when needed)
- Sufficient GPU memory for the quantized model

## Troubleshooting

If quantization is not working:

1. Check if the model handler supports quantization (see list above)
2. Ensure bitsandbytes is installed: `./llm-installer fix <model_dir> --fix-cuda`
3. Verify CUDA is available and working
4. Check model compatibility - some models may not support quantization

## Technical Details

All quantization support is implemented through:
- `BaseHandler.get_quantization_config()` - Standardized quantization configuration
- Automatic conversion of `--dtype int4/int8` to `load_in_4bit/load_in_8bit` flags
- BitsAndBytes library for actual quantization implementation
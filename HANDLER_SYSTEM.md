# Handler System Documentation

## Overview

The LLM Installer now uses a universal handler system that abstracts model-specific logic into dedicated handler classes. This makes the code more maintainable and allows for easy addition of new model types.

## Architecture

### Base Handler (handlers/base.py)

The `BaseHandler` class defines the interface that all handlers must implement:

**Core Methods:**
- `load_model()` - Load the model with optimal parameters
- `get_dependencies()` - Get Python dependencies
- `get_system_dependencies()` - Get system requirements
- `analyze()` - Analyze model requirements
- `validate_model_files()` - Validate model files
- `get_model_capabilities()` - Get model capabilities

**Generation Methods:**
- `generate_text()` - Generate text responses
- `generate_image()` - Generate images from text
- `generate_audio()` - Generate audio (TTS, music)
- `generate_video()` - Generate videos
- `process_image()` - Process images (classify, detect, etc.)
- `process_audio()` - Process audio (transcribe, etc.)
- `process_multimodal()` - Process multiple modalities
- `embed_text()` - Generate text embeddings
- `embed_image()` - Generate image embeddings
- `embed_multimodal()` - Generate multimodal embeddings

**Configuration Methods:**
- `get_supported_modes()` - Get list of supported modes
- `get_mode_descriptions()` - Get mode descriptions
- `apply_mode_settings()` - Apply mode-specific settings
- `prepare_inputs()` - Prepare inputs for model
- `postprocess_outputs()` - Postprocess model outputs
- `validate_inputs()` - Validate inputs
- `get_generation_config()` - Get default generation config

## Handler Types

### TransformerHandler
For transformer-based language models (GPT, LLaMA, Mistral, etc.)
- Supports: text generation, chat, completion, instruction following
- Modes: auto, chat, complete, instruct, creative, code, analyze, translate, summarize

### MultimodalHandler
For general multimodal models
- Supports: text + image processing
- Base class for more specific multimodal handlers

### JanusHandler (extends MultimodalHandler)
Specifically for Deepseek Janus models
- Supports: text generation, image generation, multimodal understanding
- Modes: auto, chat, image, multimodal, analyze
- Special features: Classifier-Free Guidance for image generation

### DiffusionHandler
For diffusion-based image/video generation models
- Supports: text-to-image, text-to-video generation

### AudioHandler
For audio processing models
- Supports: speech-to-text, text-to-speech

### VisionHandler
For computer vision models
- Supports: image classification, object detection, segmentation

### EmbeddingHandler
For embedding models
- Supports: text embeddings, multimodal embeddings

## Usage in serve_api.py

The API server now uses handlers when available:

```python
# Load model and handler
MODEL, TOKENIZER = load_model(MODEL_INFO, ...)
HANDLER = get_handler(MODEL_INFO)

# In generate endpoint
if HANDLER:
    if request.mode == "image":
        result = HANDLER.generate_image(...)
    elif request.images:
        result = HANDLER.process_multimodal(...)
    else:
        result = HANDLER.generate_text(...)
```

## Adding New Handlers

To add support for a new model type:

1. Create a new handler class in `handlers/` that extends `BaseHandler`
2. Implement the required abstract methods
3. Override generation methods as needed
4. Register the handler in `handlers/registry.py`

Example:
```python
class MyCustomHandler(BaseHandler):
    def generate_text(self, prompt, model, tokenizer, **kwargs):
        # Custom generation logic
        return {'text': generated_text}
    
    def get_supported_modes(self):
        return ['mode1', 'mode2']
```

## Benefits

1. **Modularity**: Model-specific code is isolated in handlers
2. **Universality**: Common interface for all model types
3. **Extensibility**: Easy to add new model support
4. **Maintainability**: Changes to model-specific logic don't affect core code
5. **Type Safety**: Clear interfaces and type hints

## Memory Optimizations

All handlers include CUDA memory optimizations:
- `PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'` to prevent fragmentation
- `torch.cuda.empty_cache()` calls at strategic points
- Proper error handling with memory cleanup
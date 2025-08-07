# Model Type Detection Algorithm in LLM Installer

## Overview

The detection system determines the model type NOT by its name, but by its structure and metadata. This allows automatic support for new models without code changes.

## Step-by-Step Algorithm

### Step 1: Load Metadata (ModelChecker.check_model)

```python
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"  # example
```

1. **HuggingFace API Request**:
   - Get model information (tags, pipeline_tag, library_name)
   - Get list of files in repository
   - Download configuration files

2. **Form Data Structure**:
```python
model_data = {
    "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
    "files": [
        {"path": "config.json", "size": 1234},
        {"path": "model.safetensors", "size": 15000000000},
        {"path": "tokenizer.json", "size": 5678},
        # ... other files
    ],
    "config_data": {
        "config.json": {
            "model_type": "qwen2_5_vl",
            "architectures": ["Qwen2_5_VLForConditionalGeneration"],
            "vision_config": {...},
            "hidden_size": 3584,
            # ... other fields
        },
        # If other configs exist (model_index.json for diffusers)
    },
    "tags": ["vision", "multimodal", "text-generation"],
    "pipeline_tag": "text-generation",  # or None
    "library_name": "transformers"  # or "diffusers", "timm", etc
}
```

### Step 2: Run Detector Chain

```python
# DetectorRegistry contains an ordered list of detectors
detectors = [
    ConfigDetector(priority=100),      # Highest priority
    DiffusersDetector(priority=90),    
    AudioDetector(priority=80),
    EmbeddingDetector(priority=70),
    # ... other detectors
]
```

### Step 3: Check Each Detector

For each detector in descending priority order:

```python
for detector in detectors:
    if detector.matches(model_data):
        analysis = detector.analyze(model_data)
        model_data.update(analysis)
        break  # First matching detector determines the type
```

### Step 4: Detector Logic

#### ConfigDetector (priority 100)
**matches()**: Checks for config.json in model_data["config_data"]

**analyze()**: 
1. Extracts config.json
2. Reads model_type and architectures
3. Checks mappings:
   ```python
   ARCHITECTURE_HANDLERS = {
       "Qwen2_5_VLForConditionalGeneration": "multimodal",
       "LlamaForCausalLM": "language-model",
       "StableDiffusionPipeline": "image-generation",
       # ...
   }
   
   MODEL_TYPE_MAPPING = {
       "qwen2_5_vl": "multimodal",
       "llama": "language-model",
       "whisper": "audio-model",
       # ...
   }
   ```
4. If not found in mappings, analyzes structure:
   - Has `vision_config` or `image_token_id` → multimodal
   - Has `audio_config` or `mel_bins` → audio-model
   - Has only `vocab_size` and `hidden_size` → language-model

#### DiffusersDetector (priority 90)
**matches()**: 
- Has model_index.json
- Or has unet/, vae/ folders
- Or library_name == "diffusers"

**analyze()**: 
- Reads pipeline class from model_index.json
- Determines type: text-to-image, text-to-video, inpainting, etc

#### AudioDetector (priority 80)
**matches()**: 
- Has feature_extractor_type in config
- Or has audio_processor
- Or model_type contains "whisper", "wav2vec2", etc

### Step 5: Analysis Result

Detector returns:
```python
{
    "model_type": "qwen2_5_vl",              # Exact model type
    "model_family": "multimodal",            # Family (for handler selection)
    "architecture_type": "Qwen2_5_VLForConditionalGeneration",
    "primary_library": "transformers",       # Primary library
    "trust_remote_code": True,               # Whether trust_remote_code is needed
    "capabilities": {
        "supports_images": True,
        "supports_vision": True,
        "max_context_length": 128000,
        # ...
    },
    "special_requirements": ["flash-attn"],  # Special dependencies
}
```

### Step 6: Fallback Mechanism

If no detector matched:
```python
def _infer_model_type(model_data):
    # Try to determine by pipeline_tag from HuggingFace
    pipeline_tag = model_data.get("pipeline_tag", "")
    
    if "text-generation" in pipeline_tag:
        return {
            "model_type": "transformer",
            "model_family": "language-model"
        }
    elif "text-to-image" in pipeline_tag:
        return {
            "model_type": "diffusion",
            "model_family": "image-generation"
        }
    # ... etc.
    
    # If nothing found at all
    return {
        "model_type": "unknown",
        "model_family": "unknown"
    }
```

## Example for Qwen2.5-VL

1. **Loading**: Get config.json with model_type="qwen2_5_vl"
2. **ConfigDetector.matches()**: Yes, has config.json → True
3. **ConfigDetector.analyze()**:
   - model_type = "qwen2_5_vl"
   - Check ARCHITECTURE_HANDLERS["Qwen2_5_VLForConditionalGeneration"] = "multimodal"
   - Find vision_config in config → confirm multimodal
   - trust_remote_code = True (for qwen models)
4. **Result**: model_family = "multimodal" → will use MultimodalHandler or QwenVLHandler

## Important Principles

1. **Detection NOT by name**: Don't check model name, only structure
2. **Priority-based**: More specific detectors have higher priority
3. **First match wins**: Uses first detector that returned matches()=True
4. **Extensibility**: New detectors are automatically registered through registry
5. **Fallback**: Always has fallback option through pipeline_tag

## Adding Support for a New Model

1. If model has unique structure → create new detector
2. If model is similar to existing → add to mappings
3. Create corresponding handler in handlers/
4. System will automatically start detecting and supporting new type
# Gemma3 Vision Support Notes

## Current Status
- ✅ **Text chat works** with bfloat16 quantization
- ❌ **Vision mode has issues** with image token handling

## The Problem
Gemma3 multimodal expects a special image token in the text prompt, but the processor doesn't automatically insert it. Error:
```
Prompt contained 0 image tokens but received 1 images
```

## What We Tried
1. Added logic to detect and insert image token
2. Used `apply_chat_template` for proper formatting
3. Added fallback processing methods

## Root Cause
The Gemma3 processor expects the image token to be in a specific format that matches the tokenizer's vocabulary. The exact token format may vary between model versions.

## Recommendations

### 1. Check Model Documentation
Look for the specific image token format in the model card:
- https://huggingface.co/google/gemma-3-12b-it

### 2. Use Direct Format
Try using the processor's expected format directly:
```python
# Example format that might work
messages = [
    {"role": "user", "content": "<image>\nWhat do you see in this image?"}
]
```

### 3. Alternative Models
If vision support is critical, consider:
- **Qwen-VL** models - have good vision support
- **LLaVA** models - designed for multimodal
- **Janus** models - work well with images

### 4. Debug the Processor
To find the correct image token:
```python
# Check processor attributes
print(dir(processor))
print(processor.image_token if hasattr(processor, 'image_token') else "No image_token")

# Check tokenizer
print(processor.tokenizer.special_tokens_map)
```

## Temporary Workaround
For now, use text-only mode which works perfectly:
```bash
./start.sh --dtype bfloat16 --stream
```

Vision support requires more investigation into the specific token format expected by this Gemma3 version.
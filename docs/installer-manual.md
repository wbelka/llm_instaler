# LLM Installer Manual

## Overview
LLM Installer is a command-line tool for installing and managing Large Language Models (LLMs) from HuggingFace Hub.

## Installation
```bash
git clone https://github.com/yourusername/llm_installer.git
cd llm_installer
./llm-installer --help
```

## Commands

### check
Check model compatibility without downloading.
```bash
./llm-installer check <model_id>
```

Example:
```bash
./llm-installer check meta-llama/Llama-2-7b-hf
```

### install
Install a model with all dependencies.
```bash
./llm-installer install <model_id> [options]
```

Options:
- `--device`: Device to use (auto/cuda/cpu/mps). Default: auto
- `--dtype`: Data type (auto/float16/float32/int8/int4). Default: auto
- `--quantization`: Quantization method (none/8bit/4bit). Default: none
- `-f, --force`: Force reinstall if model exists
- `--debug`: Enable debug logging

Examples:
```bash
# Install with default settings
./llm-installer install Qwen/Qwen3-8B

# Install with 4-bit quantization
./llm-installer install Qwen/Qwen3-8B --dtype int4

# Install with 8-bit quantization
./llm-installer install meta-llama/Llama-2-7b-hf --quantization 8bit

# Force reinstall
./llm-installer install mistralai/Mistral-7B-v0.1 --force
```

### list
List installed models.
```bash
./llm-installer list
```

### update
Update scripts and libraries in an installed model.
```bash
./llm-installer update <model_dir>
```

Example:
```bash
./llm-installer update /home/user/LLM/models/Qwen_Qwen3-8B
```

### fix
Fix dependencies in an installed model.
```bash
./llm-installer fix <model_dir> [options]
```

Options:
- `-r, --reinstall`: Reinstall all dependencies
- `--fix-torch`: Fix PyTorch version compatibility
- `--fix-cuda`: Fix CUDA dependencies (including bitsandbytes)

Examples:
```bash
# Check for issues
./llm-installer fix /home/user/LLM/models/Qwen_Qwen3-8B

# Fix CUDA dependencies (including bitsandbytes for quantization)
./llm-installer fix /home/user/LLM/models/Qwen_Qwen3-8B --fix-cuda

# Fix PyTorch version
./llm-installer fix /home/user/LLM/models/Qwen_Qwen3-8B --fix-torch

# Reinstall all dependencies
./llm-installer fix /home/user/LLM/models/Qwen_Qwen3-8B --reinstall
```

### doctor
Run system diagnostics to check environment.
```bash
./llm-installer doctor
```

### config
Show current configuration.
```bash
./llm-installer config
```

## Device and Data Type Options

### Device Options
- `auto`: Automatically detect best device (GPU if available, otherwise CPU)
- `cuda`: Force CUDA GPU usage
- `cpu`: Force CPU usage
- `mps`: Force Metal Performance Shaders (macOS)

### Data Type Options
- `auto`: Automatically select based on available memory
- `float32`: Full precision (most memory)
- `float16`: Half precision (recommended for GPU)
- `bfloat16`: Brain floating point (for newer GPUs)
- `int8`: 8-bit quantization (less memory, slight quality loss)
- `int4`: 4-bit quantization (least memory, more quality loss)

### Quantization Options
- `none`: No quantization
- `8bit`: 8-bit quantization using bitsandbytes
- `4bit`: 4-bit quantization using bitsandbytes

## Memory Requirements

Approximate memory usage by model size and quantization:

| Model Size | float32 | float16 | int8 | int4 |
|------------|---------|---------|------|------|
| 7B params  | 28 GB   | 14 GB   | 7 GB | 3.5 GB |
| 13B params | 52 GB   | 26 GB   | 13 GB| 6.5 GB |
| 70B params | 280 GB  | 140 GB  | 70 GB| 35 GB |

## Troubleshooting

### CUDA Out of Memory
If you get CUDA out of memory errors:
1. Use quantization: `--dtype int4` or `--quantization 4bit`
2. Reduce batch size in generation
3. Clear GPU cache between runs

### Quantization Not Working
If quantization doesn't reduce memory usage:
1. Install bitsandbytes: `./llm-installer fix <model_dir> --fix-cuda`
2. Check CUDA version compatibility
3. Ensure PyTorch has CUDA support

### Model Not Loading
1. Check model compatibility: `./llm-installer check <model_id>`
2. Update scripts: `./llm-installer update <model_dir>`
3. Fix dependencies: `./llm-installer fix <model_dir> --reinstall`

## Configuration

Configuration file location: `~/.config/llm-installer/config.yaml`

Example configuration:
```yaml
install_path: ~/LLM/models
huggingface_token: hf_...
default_device: auto
default_dtype: float16
cuda_version: "12.4"
```

## Environment Variables

- `HF_TOKEN`: HuggingFace API token
- `LLM_INSTALL_PATH`: Default installation path
- `CUDA_VISIBLE_DEVICES`: Control GPU visibility
- `PYTORCH_CUDA_ALLOC_CONF`: PyTorch memory configuration

## Model-Specific Notes

### Qwen3 Models
- Support thinking mode with `/think` and `/no_think` switches
- Native 32K context, 131K with YaRN scaling
- Recommended: `--dtype int4` for 8B model on consumer GPUs

### Vision Models
- Qwen-VL models support image input
- Adjust `max_image_size` in UI settings to prevent OOM
- Images are automatically resized if too large

### Multimodal Models
- Support text, image, and sometimes audio/video
- Higher memory requirements than text-only models
- May require additional dependencies

## Advanced Usage

### Custom Handler Development
Create custom handlers for new model types by extending `BaseHandler`:
```python
from handlers.base import BaseHandler

class MyCustomHandler(BaseHandler):
    def analyze(self):
        # Implementation
        pass
```

### Batch Installation
Install multiple models:
```bash
for model in "Qwen/Qwen3-8B" "mistralai/Mistral-7B-v0.1"; do
    ./llm-installer install "$model" --dtype int4
done
```

### Running Models
After installation:
```bash
cd /path/to/installed/model
./start.sh --dtype int4
```

Access the model:
- API: http://localhost:8000
- UI: http://localhost:8000/ui/terminal
- Docs: http://localhost:8000/docs
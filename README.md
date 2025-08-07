# LLM Installer

Automated installation and management of LLM models from HuggingFace Hub.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/llm_installer.git
cd llm_installer

# Setup
./setup.sh

# Check model compatibility
./llm-installer check meta-llama/Llama-3-8B

# Install model
./llm-installer install meta-llama/Llama-3-8B --quantization 4bit

# Run model
cd ~/LLM/models/meta-llama_Llama-3-8B
./start.sh
```

## 📋 Requirements

- Linux or macOS (Windows not supported)
- Python >= 3.8
- At least 50 GB free disk space
- CUDA-compatible GPU (optional)

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm_installer.git
cd llm_installer
```

2. Run the setup script:
```bash
./setup.sh
```

3. (Optional) Configure HuggingFace token:
```bash
export HF_TOKEN="your_token_here"
```

## 📚 Main Commands

### check - Compatibility Check
```bash
./llm-installer check <model_id>
```
Checks model compatibility with your system without downloading files.

### install - Install Model
```bash
./llm-installer install <model_id> [options]
```

Options:
- `--device [auto|cuda|cpu|mps]` - Device for execution
- `--dtype [auto|float16|float32|int8|int4]` - Data type
- `--quantization [none|8bit|4bit]` - Quantization method
- `-f, --force` - Force reinstall

### list - List Installed Models
```bash
./llm-installer list
```

### update - Update Scripts
```bash
./llm-installer update <model_dir>
```

### fix - Fix Dependencies
```bash
./llm-installer fix <model_dir> [options]
```

Options:
- `--fix-cuda` - Fix CUDA dependencies (including bitsandbytes)
- `--fix-torch` - Fix PyTorch version
- `-r, --reinstall` - Reinstall all dependencies

### doctor - System Diagnostics
```bash
./llm-installer doctor
```

## 🎯 Supported Models

- **Language Models**: Llama, Mistral, Qwen, Gemma, DeepSeek
- **Multimodal**: Qwen-VL, Janus, LLaVA
- **Image Generation**: Stable Diffusion, SDXL
- **Audio Models**: Whisper, MusicGen
- **Embeddings**: BERT, Sentence Transformers
- **Special**: Mamba, RWKV

## 📂 Installed Model Structure

```
~/LLM/models/model_name/
├── model/              # Model files
├── .venv/              # Virtual environment
├── model_info.json     # Model information
├── start.sh           # Startup script
├── serve_api.py       # API server
└── logs/              # Logs
```

## 💻 Using Installed Models

### Run via Terminal
```bash
cd ~/LLM/models/model_name
./start.sh
```

### Run with Parameters
```bash
# Run on different port with quantization
./start.sh --port 8080 --dtype int8

# Run with CPU offloading for large models
./start.sh --cpu-offload --dtype int4

# Run with streaming mode
./start.sh --stream
```

### API Server
After launch, the model is accessible via REST API:
- Web interface: http://localhost:8000
- API endpoint: http://localhost:8000/generate

## 🔍 Troubleshooting

### Out of Memory Error
```bash
# Use quantization
./start.sh --dtype int8
# or
./start.sh --dtype int4

# Enable CPU offloading for large models
./start.sh --cpu-offload --dtype int4
```

CPU offloading allows running models larger than GPU memory by automatically moving layers between GPU and CPU/RAM.

### CUDA Issues
```bash
./llm-installer fix <model_dir> --fix-cuda
```

### Update Scripts
```bash
./llm-installer update <model_dir>
```

## 📖 Documentation

### Core Documentation
- [**Installer Manual**](docs/installer-manual.md) - Complete command reference and usage examples
- [**Technical Requirements**](docs/TECHNICAL_REQUIREMENTS.md) - System requirements and architecture overview
- [**Changelog**](docs/CHANGELOG.md) - Version history and recent updates

### System Architecture
- [**Handler System**](docs/HANDLER_SYSTEM.md) - Universal handler architecture for model management
- [**Detection Algorithm**](docs/DETECTION_ALGORITHM.md) - How the installer detects and classifies model types
- [**Quantization Support**](docs/QUANTIZATION_SUPPORT.md) - Quantization capabilities for different model types
- [**Quantization System**](docs/quantization-system.md) - Centralized quantization configuration

### Development Guides
- [**Handler Development Guide**](docs/HANDLER_DEVELOPMENT_GUIDE.md) - Creating custom handlers for new model types
- [**Handler Training Development**](docs/handler-training-development.md) - Implementing training support in handlers
- [**Training Guide**](docs/training-guide.md) - Fine-tuning models with LoRA/QLoRA
- [**LLM Installer Reference**](docs/LLM_INSTALLER_REFERENCE.md) - Complete API and function reference

## 🤝 Contributing

We welcome contributions to the project! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
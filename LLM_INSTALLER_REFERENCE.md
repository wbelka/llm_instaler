# LLM Installer - Complete Reference Documentation

## Table of Contents
1. [CLI Commands Reference](#cli-commands-reference)
2. [Core Modules API](#core-modules-api)
3. [Handler System](#handler-system)
4. [Detector System](#detector-system)
5. [Training Scripts](#training-scripts)
6. [API Server](#api-server)
7. [Configuration](#configuration)

---

## CLI Commands Reference

### Global Options
```bash
./llm-installer [--debug] [--version] COMMAND [OPTIONS]
```
- `--debug` - Enable debug logging
- `--version` - Show version (2.0.0)

### Commands

#### `check` - Check model compatibility
```bash
./llm-installer check MODEL_ID [OPTIONS]
```
**Arguments:**
- `MODEL_ID` - HuggingFace model identifier (e.g., "meta-llama/Llama-2-7b-hf")

**Options:**
- `--token TEXT` - HuggingFace API token (overrides config)
- `--show-details` - Show detailed requirements analysis

**Example:**
```bash
./llm-installer check meta-llama/Llama-2-7b-hf --show-details
./llm-installer check deepseek-ai/Janus-Pro-7B --token hf_xxxxx
```

#### `install` - Install a model
```bash
./llm-installer install MODEL_ID [OPTIONS]
```
**Arguments:**
- `MODEL_ID` - HuggingFace model identifier

**Options:**
- `--token TEXT` - HuggingFace API token
- `--device [auto|cuda|cpu|mps]` - Device to use (default: auto)
- `--dtype [auto|float16|float32|int8|int4]` - Data type (default: auto)
- `--quantization [none|8bit|4bit]` - Quantization method (default: none)
- `-f, --force` - Force reinstall if exists
- `--target-modules TEXT` - Custom LoRA target modules (comma-separated)

**Examples:**
```bash
./llm-installer install meta-llama/Llama-2-7b-hf
./llm-installer install deepseek-ai/Janus-Pro-7B --device cuda --dtype int4
./llm-installer install microsoft/phi-2 --quantization 4bit --force
./llm-installer install meta-llama/Llama-2-7b-hf --target-modules q_proj,v_proj
```

#### `list` - List installed models
```bash
./llm-installer list
```
**Output:** Shows all installed models with paths and installation dates

#### `update` - Update scripts in installed model
```bash
./llm-installer update MODEL_DIR
```
**Arguments:**
- `MODEL_DIR` - Path to installed model directory

**Example:**
```bash
./llm-installer update /home/user/LLM/models/meta-llama_Llama-2-7b-hf
```

#### `fix` - Fix dependencies
```bash
./llm-installer fix MODEL_DIR [OPTIONS]
```
**Arguments:**
- `MODEL_DIR` - Path to installed model directory

**Options:**
- `-r, --reinstall` - Reinstall all dependencies
- `--fix-torch` - Fix PyTorch installation
- `--fix-cuda` - Fix CUDA dependencies (installs bitsandbytes)

**Examples:**
```bash
./llm-installer fix /home/user/LLM/models/phi-2 --fix-cuda
./llm-installer fix ./models/llama-7b --reinstall
```

#### `doctor` - Run system diagnostics
```bash
./llm-installer doctor
```
**Output:** System information, GPU status, disk space, memory

#### `config` - Show configuration
```bash
./llm-installer config
```
**Output:** Current configuration from config.yaml and defaults

---

## Core Modules API

### core.installer.ModelInstaller

**Constructor:**
```python
ModelInstaller(logger: Optional[logging.Logger] = None)
```

**Methods:**
```python
def install(
    self,
    model_id: str,
    model_info: Dict[str, Any],
    requirements: ModelRequirements,
    token: Optional[str] = None,
    device: str = "auto",
    dtype: str = "auto",
    quantization: str = "none",
    force: bool = False,
    progress_callback: Optional[Callable] = None,
    interrupt_event: Optional[Event] = None
) -> bool:
    """Install a model with all dependencies"""

def update_scripts(self, model_dir: str) -> bool:
    """Update scripts in existing installation"""

def fix_dependencies(
    self,
    model_dir: str,
    reinstall: bool = False,
    fix_torch: bool = False,
    fix_cuda: bool = False
) -> bool:
    """Fix or reinstall dependencies"""
```

### core.checker.ModelChecker

**Constructor:**
```python
ModelChecker(
    model_id: str,
    token: Optional[str] = None,
    logger: Optional[logging.Logger] = None
)
```

**Methods:**
```python
def check_compatibility(self) -> Tuple[ModelRequirements, CompatibilityResult]:
    """Check if model is compatible with system"""

def get_model_info(self) -> Dict[str, Any]:
    """Fetch model metadata from HuggingFace"""

def _analyze_model() -> ModelRequirements:
    """Analyze model requirements using detectors and handlers"""
```

### core.utils - Utility Functions

```python
def check_system_requirements() -> Dict[str, Any]:
    """Get system info: GPU, memory, disk space"""

def setup_logging(debug: bool = False) -> logging.Logger:
    """Configure logging with optional debug mode"""

def format_size(size_bytes: int) -> str:
    """Format bytes to human readable (e.g., 1.5 GB)"""

def get_console() -> Console:
    """Get Rich console instance for pretty output"""

def run_command(cmd: List[str], cwd: Optional[str] = None) -> Tuple[int, str, str]:
    """Run subprocess command safely"""

def is_docker() -> bool:
    """Check if running inside Docker container"""
```

### core.config.Config

**Loading:**
```python
config = Config()  # Loads from config.yaml or defaults
```

**Attributes:**
```python
config.hf_token: Optional[str]  # HuggingFace token
config.hf_cache: str  # Default: ~/.cache/huggingface
config.models_dir: str  # Default: ~/LLM/models
config.scripts_dir: str  # Scripts source directory
config.default_device: str  # Default: auto
config.default_dtype: str  # Default: auto
config.default_quantization: str  # Default: none
config.memory_buffer: float  # Default: 0.8
config.parallel_downloads: int  # Default: 4
config.telemetry_enabled: bool  # Default: False
```

---

## Handler System

### Base Handler Interface (handlers.base.BaseHandler)

**Abstract Methods (must implement):**
```python
@abstractmethod
def get_dependencies(self) -> List[str]:
    """Return list of required pip packages"""

@abstractmethod
def analyze(self, model_info: Dict[str, Any]) -> ModelRequirements:
    """Analyze model and return requirements"""

@abstractmethod
def load_model(self, device: str, dtype: str, quantization: str, **kwargs):
    """Load the model with specified configuration"""

@abstractmethod
def validate_model_files(self, model_path: str) -> Tuple[bool, List[str]]:
    """Validate model files exist"""
```

**Optional Methods (can override):**
```python
def generate_text(self, prompt: str, **kwargs) -> str:
    """Generate text from prompt"""

def generate_image(self, prompt: str, **kwargs) -> Any:
    """Generate image from prompt"""

def generate_audio(self, prompt: str, **kwargs) -> Any:
    """Generate audio from prompt"""

def process_multimodal(self, inputs: Dict[str, Any], **kwargs) -> Any:
    """Process multimodal inputs"""

def get_model_capabilities(self) -> Dict[str, bool]:
    """Return supported capabilities"""

def get_supported_modes(self) -> List[str]:
    """Return supported UI modes"""

def get_installation_notes(self) -> List[str]:
    """Return special installation instructions"""
```

### Available Handlers

1. **TransformerHandler** - Standard language models
2. **MultimodalHandler** - Vision-language models
3. **JanusHandler** - DeepSeek Janus models
4. **DiffusionHandler** - Image/video generation
5. **AudioHandler** - Speech models
6. **VisionHandler** - Computer vision
7. **EmbeddingHandler** - Embeddings
8. **SpecializedHandler** - O1-style reasoning
9. **DeepseekHandler** - DeepSeek R1
10. **QwenVLHandler** - Qwen VL models
11. **Gemma3Handler** - Gemma 3 multimodal
12. **Llama4Handler** - Llama 4 multimodal
13. **CogVideoXHandler** - Video generation
14. **CogVLMHandler** - CogVLM models

### Handler Registry (handlers.registry.py)

```python
def get_handler(model_type: str, model_family: str) -> Type[BaseHandler]:
    """Get handler class for model type/family"""

# Registry structure:
HANDLER_REGISTRY = {
    "transformer": {
        "default": TransformerHandler,
        "llama": TransformerHandler,
        "gpt": TransformerHandler,
        # ...
    },
    "multimodal": {
        "default": MultimodalHandler,
        "janus": JanusHandler,
        "qwen-vl": QwenVLHandler,
        # ...
    },
    # ...
}
```

---

## Detector System

### Base Detector Interface (detectors.base.BaseDetector)

```python
class BaseDetector(ABC):
    priority: int = 0  # Higher priority = checked first
    
    @abstractmethod
    def matches(self, model_info: Dict[str, Any], model_path: str) -> bool:
        """Check if detector can handle this model"""
    
    @abstractmethod
    def analyze(self, model_info: Dict[str, Any], model_path: str) -> Dict[str, Any]:
        """Analyze model and return type/family/requirements"""
```

### Available Detectors

1. **ConfigDetector** (priority: 100) - Analyzes config.json
2. **DiffusersDetector** (priority: 90) - Detects diffusion models
3. **AudioDetector** (priority: 80) - Detects audio models
4. **VisionDetector** (priority: 70) - Detects vision models
5. **TransformerDetector** (priority: 50) - Default transformer models

### Detector Registry (detectors.registry.py)

```python
def get_detectors() -> List[Type[BaseDetector]]:
    """Get all detectors sorted by priority"""

def detect_model_type(model_info: Dict[str, Any], model_path: str) -> Dict[str, Any]:
    """Run detection chain and return first match"""
```

---

## Training Scripts

### train_lora.py - LoRA Fine-tuning

**Usage:**
```bash
python train_lora.py --data DATA_FILE [OPTIONS]
```

**Required Arguments:**
- `--data PATH` - Training data (JSON/JSONL)

**Options:**
- `--mode [quick|balanced|quality]` - Training mode (default: balanced)
- `--epochs INT` - Number of epochs (overrides mode)
- `--batch-size INT` - Batch size (overrides mode)
- `--learning-rate FLOAT` - Learning rate (overrides mode)
- `--warmup-steps INT` - Warmup steps
- `--weight-decay FLOAT` - Weight decay
- `--gradient-accumulation INT` - Gradient accumulation steps
- `--max-length INT` - Maximum sequence length
- `--val-split FLOAT` - Validation split ratio (0.0-0.5)
- `--seed INT` - Random seed
- `--checkpoint-dir PATH` - Checkpoint directory
- `--patience INT` - Early stopping patience
- `--lora-r INT` - LoRA rank
- `--lora-alpha INT` - LoRA alpha
- `--lora-dropout FLOAT` - LoRA dropout
- `--use-flash-attention` - Enable flash attention
- `--gradient-checkpointing` - Enable gradient checkpointing
- `--output-dir PATH` - Output directory
- `--hub-model-id TEXT` - Push to HuggingFace Hub
- `--wandb-project TEXT` - W&B project name
- `--save-total-limit INT` - Max checkpoints to keep
- `--save-steps INT` - Save every N steps
- `--eval-steps INT` - Evaluate every N steps
- `--logging-steps INT` - Log every N steps
- `--target-modules TEXT` - Custom LoRA target modules (comma-separated)
- `--resume-from-checkpoint PATH` - Resume from checkpoint
- `--overfitting-threshold FLOAT` - Overfitting detection threshold

**Training Modes:**
- `quick`: 1 epoch, batch_size=8, lr=3e-4
- `balanced`: 3 epochs, batch_size=4, lr=2e-4
- `quality`: 5 epochs, batch_size=2, lr=1e-4

**Data Format:**
```json
[
  {"instruction": "...", "input": "...", "output": "..."},
  {"instruction": "...", "response": "..."},
  {"text": "..."}
]
```

---

## API Server

### server.py - REST API Server

**Usage:**
```bash
python server.py [--host HOST] [--port PORT]
```

**Endpoints:**

#### GET /
Health check
```json
{"status": "ok", "model": "model_name"}
```

#### POST /generate
Generate text
```json
Request:
{
  "prompt": "Hello, how are you?",
  "max_length": 100,
  "temperature": 0.7,
  "top_p": 0.9
}

Response:
{
  "text": "Generated response...",
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 20,
    "total_tokens": 25
  }
}
```

#### POST /multimodal
Process multimodal input
```json
Request:
{
  "text": "Describe this image",
  "image": "base64_encoded_image"
}

Response:
{
  "text": "Image description...",
  "type": "text"
}
```

#### GET /capabilities
Get model capabilities
```json
{
  "text_generation": true,
  "image_generation": false,
  "multimodal": true,
  "modes": ["chat", "multimodal"]
}
```

---

## Configuration

### config.yaml Structure
```yaml
# HuggingFace settings
hf_token: "hf_xxxxx"  # Optional
hf_cache: "~/.cache/huggingface"  # Cache directory

# Installation paths
models_dir: "~/LLM/models"  # Where to install models
scripts_dir: null  # Auto-detected

# Default settings
default_device: "auto"  # auto/cuda/cpu/mps
default_dtype: "auto"  # auto/float16/float32/int8/int4
default_quantization: "none"  # none/8bit/4bit

# System settings
memory_buffer: 0.8  # Use 80% of available memory
parallel_downloads: 4  # Concurrent downloads

# Features
telemetry:
  enabled: false  # Telemetry disabled by default
```

### Environment Variables
```bash
# Override config values
export HF_TOKEN="hf_xxxxx"
export LLM_MODELS_DIR="/custom/path/models"
export TRANSFORMERS_CACHE="/custom/cache"
export HF_HOME="/custom/hf_home"
export TORCH_HOME="/custom/torch_home"

# Debug mode
export LLM_DEBUG=1
```

### Model Installation Structure
```
~/LLM/models/
└── meta-llama_Llama-2-7b-hf/
    ├── venv/                 # Isolated Python environment
    ├── model/                # Model weights and config
    ├── lora/                 # LoRA adapters
    ├── scripts/              # Copied scripts
    │   ├── chat.py          # Terminal UI
    │   ├── server.py        # API server
    │   ├── train_lora.py    # Training script
    │   └── generate.py      # Generation script
    ├── requirements.txt      # Python dependencies
    └── .llm_installer.json   # Installation metadata
```
"""Constants used throughout LLM Installer.

This module centralizes magic numbers and configuration values
for better maintainability and clarity.
"""

# Memory calculation constants
MEMORY_MULTIPLIERS = {
    'float32': 1.0,    # Base multiplier
    'float16': 0.5,    # Half precision
    'int8': 0.25,      # 8-bit quantization
    'int4': 0.125,     # 4-bit quantization
}

# Memory overhead multipliers for different operations
MEMORY_OVERHEAD_MULTIPLIERS = {
    'inference': 1.2,      # 20% overhead for inference
    'training': 2.0,       # 100% overhead for training (gradients, optimizer states)
    'quantization': 1.5,   # 50% overhead for quantization
}

# Disk space requirements (GB)
DISK_SPACE = {
    'min_free_space': 3.0,           # Minimum free space after installation
    'venv_overhead': 3.0,            # Space for virtual environment
    'cache_overhead': 2.0,           # Space for caches and temp files
}

# Installation timeouts (seconds)
TIMEOUTS = {
    'pip_install': 600,              # Standard pip install timeout
    'pip_build': 1200,               # Timeout for packages that build from source
    'model_download': 3600,          # Model download timeout (1 hour)
    'subprocess_default': 300,       # Default subprocess timeout
    'file_lock': 300,                # File lock acquisition timeout
}

# Request limits for API
API_LIMITS = {
    'max_request_size': 10 * 1024 * 1024,    # 10MB max request size
    'max_image_size': 5 * 1024 * 1024,       # 5MB max image size  
    'max_text_length': 100000,               # 100k chars max text
    'max_batch_size': 10,                    # Max items in batch requests
    'rate_limit_requests': 60,               # Requests per minute
    'rate_limit_window': 60,                 # Window in seconds
}

# Model size thresholds (parameters)
MODEL_SIZE_THRESHOLDS = {
    'small': 1_000_000_000,          # < 1B parameters
    'medium': 10_000_000_000,        # < 10B parameters
    'large': 50_000_000_000,         # < 50B parameters
    'xlarge': float('inf'),          # >= 50B parameters
}

# GPU memory requirements (GB)
GPU_MEMORY_REQUIREMENTS = {
    'min_cuda': 4.0,                 # Minimum CUDA memory
    'recommended_cuda': 8.0,         # Recommended CUDA memory
    'min_mps': 8.0,                  # Minimum MPS (Apple Silicon) memory
    'recommended_mps': 16.0,         # Recommended MPS memory
}

# Batch size defaults
BATCH_SIZES = {
    'inference_default': 1,
    'inference_max': 32,
    'training_default': 4,
    'training_max': 16,
    'gradient_accumulation': 4,
}

# Learning rate defaults
LEARNING_RATES = {
    'default': 5e-5,
    'min': 1e-6,
    'max': 1e-3,
    'warmup_ratio': 0.03,
}

# Cache expiration times (seconds)
CACHE_EXPIRY = {
    'model_check': 24 * 60 * 60,     # 24 hours for model check cache
    'model_info': 7 * 24 * 60 * 60,  # 7 days for model info cache
    'handler_registry': 60 * 60,      # 1 hour for handler registry cache
}

# Default generation parameters
GENERATION_DEFAULTS = {
    'temperature': 0.7,
    'top_p': 0.9,
    'top_k': 50,
    'max_tokens': 4096,
    'repetition_penalty': 1.1,
}

# Image generation defaults
IMAGE_GENERATION_DEFAULTS = {
    'width': 512,
    'height': 512,
    'num_inference_steps': 50,
    'guidance_scale': 7.5,
    'negative_prompt': '',
}

# Video generation defaults
VIDEO_GENERATION_DEFAULTS = {
    'width': 256,
    'height': 256,
    'num_frames': 16,
    'fps': 8,
}

# Supported file formats
SUPPORTED_FORMATS = {
    'image': ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'],
    'audio': ['.wav', '.mp3', '.flac', '.ogg', '.m4a'],
    'video': ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
    'document': ['.txt', '.pdf', '.docx', '.md', '.json'],
}

# PyTorch index URLs
PYTORCH_INDEX_URLS = {
    'cuda118': 'https://download.pytorch.org/whl/cu118',
    'cuda121': 'https://download.pytorch.org/whl/cu121',
    'cuda124': 'https://download.pytorch.org/whl/cu124',
    'rocm': 'https://download.pytorch.org/whl/rocm5.6',
    'cpu': 'https://download.pytorch.org/whl/cpu',
}

# Trusted package sources
TRUSTED_DOMAINS = [
    'https://pypi.org/',
    'https://download.pytorch.org/',
    'https://pypi.python.org/',
    'https://test.pypi.org/',
    'https://huggingface.co/',
]

# Environment variable names
ENV_VARS = {
    'hf_token': 'HUGGING_FACE_HUB_TOKEN',
    'cuda_visible_devices': 'CUDA_VISIBLE_DEVICES',
    'pytorch_cuda_alloc_conf': 'PYTORCH_CUDA_ALLOC_CONF',
    'transformers_cache': 'TRANSFORMERS_CACHE',
    'hf_home': 'HF_HOME',
    'torch_home': 'TORCH_HOME',
}

# Logging configuration
LOGGING = {
    'format': '[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'max_log_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
}

# Model type priorities for detection
DETECTOR_PRIORITIES = {
    'config': 100,
    'diffusers': 90,
    'audio': 80,
    'vision': 70,
    'specialized': 60,
    'architecture': 50,
    'library': 40,
}
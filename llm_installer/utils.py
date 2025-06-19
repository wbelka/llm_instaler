"""
Utility functions for LLM Installer
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests
from huggingface_hub import hf_hub_url, list_repo_files, login
import yaml
import json


logger = logging.getLogger(__name__)

# Global token storage
_HF_TOKEN = None


def set_hf_token(token: Optional[str]) -> None:
    """Set HuggingFace token for authentication"""
    global _HF_TOKEN
    _HF_TOKEN = token
    if token:
        os.environ['HF_TOKEN'] = token
        os.environ['HUGGING_FACE_HUB_TOKEN'] = token
        try:
            login(token=token, add_to_git_credential=False)
            logger.info("Successfully authenticated with HuggingFace")
        except Exception as e:
            logger.warning(f"Failed to login with token: {e}")


def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from various sources"""
    global _HF_TOKEN

    # Priority order: explicit token > env var > config file
    if _HF_TOKEN:
        return _HF_TOKEN

    # Check environment variables
    for env_var in ['HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_TOKEN']:
        token = os.environ.get(env_var)
        if token:
            _HF_TOKEN = token
            return token

    # Check config file
    config_path = Path.home() / '.config' / 'llm-installer' / 'config.yaml'
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                token = config.get('huggingface_token')
                if token:
                    _HF_TOKEN = token
                    return token
        except Exception as e:
            logger.debug(f"Failed to load token from config: {e}")

    return None


def save_hf_token(token: str) -> bool:
    """Save HuggingFace token to config file"""
    config_dir = Path.home() / '.config' / 'llm-installer'
    config_path = config_dir / 'config.yaml'

    try:
        config_dir.mkdir(parents=True, exist_ok=True)

        # Load existing config or create new
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}

        # Update token
        config['huggingface_token'] = token

        # Save config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        # Set permissions to user-only
        config_path.chmod(0o600)

        logger.info(f"Token saved to {config_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save token: {e}")
        return False


def get_models_dir() -> Path:
    """Get the models directory path"""
    models_dir = os.environ.get('LLM_MODELS_DIR', os.path.expanduser('~/LLM/models'))
    return Path(models_dir)


def get_cache_dir() -> Path:
    """Get the cache directory path"""
    cache_dir = os.environ.get('LLM_CACHE_DIR', os.path.expanduser('~/LLM/cache'))
    return Path(cache_dir)


def sanitize_model_name(model_id: str) -> str:
    """
    Convert HuggingFace model ID to safe directory name
    Example: meta-llama/Llama-3-8B -> meta-llama_Llama-3-8B
    """
    return model_id.replace('/', '_')


def fetch_model_config(model_id: str, filename: str = "config.json") -> Optional[Dict[str, Any]]:
    """
    Fetch configuration file from HuggingFace without downloading the model

    Args:
        model_id: HuggingFace model ID
        filename: Config file to fetch (default: config.json)

    Returns:
        Parsed config dict or None if error
    """
    try:
        config_url = hf_hub_url(repo_id=model_id, filename=filename)

        # Prepare headers with token if available
        headers = {}
        token = get_hf_token()
        if token:
            headers['Authorization'] = f'Bearer {token}'

        response = requests.get(config_url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            logger.error(f"Authentication required for {model_id}. "
                         "Please provide a HuggingFace token.")
        elif e.response.status_code == 403:
            logger.error(f"Access forbidden for {model_id}. "
                         "Make sure you have accepted the model license.")
        else:
            logger.error(f"Failed to fetch {filename} for {model_id}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch {filename} for {model_id}: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse {filename} for {model_id}: {e}")
        return None


def fetch_model_files_list(model_id: str) -> List[str]:
    """
    Get list of files in the model repository without downloading

    Args:
        model_id: HuggingFace model ID

    Returns:
        List of file paths in the repository
    """
    try:
        # Set token if available
        token = get_hf_token()
        files = list_repo_files(repo_id=model_id, token=token)
        return list(files)
    except Exception as e:
        if "401" in str(e):
            logger.error(f"Authentication required for {model_id}. "
                         "Please provide a HuggingFace token.")
        elif "403" in str(e):
            logger.error(f"Access forbidden for {model_id}. "
                         "Make sure you have accepted the model license.")
        else:
            logger.error(f"Failed to list files for {model_id}: {e}")
        return []


def has_file_with_extension(files: List[str], extension: str) -> bool:
    """Check if any file in the list has the given extension"""
    return any(f.endswith(extension) for f in files)


def load_yaml_config(file_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file"""
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Failed to load YAML config from {file_path}: {e}")
        return {}


def save_yaml_config(data: Dict[str, Any], file_path: Path) -> bool:
    """Save data as YAML configuration file"""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        return True
    except Exception as e:
        logger.error(f"Failed to save YAML config to {file_path}: {e}")
        return False


def save_json_config(data: Dict[str, Any], file_path: Path) -> bool:
    """Save data as JSON configuration file"""
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON config to {file_path}: {e}")
        return False


def estimate_model_size(config: Dict[str, Any], files: List[str]) -> Dict[str, Any]:
    """
    Estimate model size and memory requirements based on config

    Returns:
        Dict with size_gb and estimated_memory_gb
    """
    # First try to estimate from actual files
    size_gb = estimate_size_from_files(files)

    # If no size from files, try to estimate from config
    if size_gb == 0 and config:
        # Get model dimensions
        hidden_size = config.get('hidden_size', 0)
        num_layers = config.get('num_hidden_layers', 0)
        vocab_size = config.get('vocab_size', 0)

        if hidden_size and num_layers:
            # Rough estimation: parameters * 4 bytes (fp32)
            params_millions = (hidden_size * hidden_size * num_layers * 4 +
                               vocab_size * hidden_size) / 1e6
            size_gb = params_millions / 250  # Very rough estimate

    # Default if still no estimate
    if size_gb == 0:
        size_gb = 8.0  # Default

    # Memory requirement is typically 1.2-1.5x model size for inference
    memory_gb = size_gb * 1.2

    # Check for quantized models
    if any('.gguf' in f for f in files):
        # GGUF models are typically quantized, already accounted in file size
        memory_gb = size_gb * 1.1  # Less overhead for quantized

    return {
        'size_gb': round(size_gb, 1),
        'estimated_memory_gb': round(memory_gb, 1)
    }


def estimate_size_from_files(files: List[str]) -> float:
    """
    Estimate model size from file list

    Looks for patterns like:
    - model-00001-of-00016.safetensors
    - pytorch_model-00001-of-00005.bin
    - model.safetensors
    - Special MoE files: ema.safetensors, ae.safetensors
    """
    import re

    total_size_gb = 0.0

    # Pattern for sharded model files - use search instead of match
    shard_pattern = re.compile(r'model-(\d+)-of-(\d+)\.(safetensors|bin|pt)')
    pytorch_pattern = re.compile(r'pytorch_model-(\d+)-of-(\d+)\.(bin|pt)')

    # Check for sharded models
    shard_info = {}
    shard_count = 0
    for file in files:
        # Get just the filename without path
        filename = file.split('/')[-1] if '/' in file else file

        # Use search to find pattern anywhere in filename
        match = shard_pattern.search(filename) or pytorch_pattern.search(filename)
        if match:
            shard_count += 1
            total_shards = int(match.group(2))
            shard_info['total_shards'] = total_shards

    logger.debug(f"Found {shard_count} shard files out of {len(files)} total files")

    if shard_info and 'total_shards' in shard_info:
        # Estimate based on number of shards
        total_shards = shard_info['total_shards']

        # Ming-Lite-Omni and similar large models use ~5GB per shard
        if total_shards >= 10:
            total_size_gb = total_shards * 5.0
        else:
            # Smaller models might use smaller shards
            total_size_gb = total_shards * 4.0

        logger.debug(f"Detected {total_shards} shards, estimated size: {total_size_gb}GB")
    else:
        # Look for all safetensors/bin files (more inclusive)
        model_files = [f for f in files if any(
            f.endswith(ext) for ext in ['.safetensors', '.bin', '.pt', '.gguf']
        )]

        # Count only files that look like model weights
        weight_files = []
        special_moe_files = {}

        for f in model_files:
            filename = f.split('/')[-1].lower()
            # Exclude common non-weight files
            if not any(skip in filename for skip in ['config', 'tokenizer', 'vocab', 'merges']):
                weight_files.append(f)

                # Track special MoE files
                if 'ema.safetensors' in filename:
                    special_moe_files['ema'] = 29.2  # Known size for BAGEL
                elif 'ae.safetensors' in filename:
                    special_moe_files['ae'] = 0.335  # Known size for BAGEL

        if weight_files:
            # If we found special MoE files, use their known sizes
            if special_moe_files:
                total_size_gb = sum(special_moe_files.values())
                logger.debug(f"Found MoE files: {list(special_moe_files.keys())}, "
                             f"total size: {total_size_gb}GB")
            # Single file models
            elif len(weight_files) == 1:
                # Check filename for size hints
                file = weight_files[0]
                if 'small' in file.lower() or '1b' in file.lower():
                    total_size_gb = 2.0
                elif 'medium' in file.lower() or '3b' in file.lower():
                    total_size_gb = 6.0
                elif 'large' in file.lower() or '7b' in file.lower():
                    total_size_gb = 13.0
                elif '13b' in file.lower():
                    total_size_gb = 26.0
                elif '30b' in file.lower() or '33b' in file.lower():
                    total_size_gb = 65.0
                elif '65b' in file.lower() or '70b' in file.lower():
                    total_size_gb = 130.0
                else:
                    total_size_gb = 8.0  # Default for single file
            else:
                # Multiple weight files
                num_files = len(weight_files)

                # Check if files look like they might be Ming-Lite format
                # (multiple safetensors files but not numbered shards)
                if num_files >= 10:
                    # Large models with many files (like Ming-Lite-Omni)
                    total_size_gb = num_files * 5.0  # Assume ~5GB per file
                elif num_files >= 5:
                    total_size_gb = num_files * 3.0  # Medium models
                else:
                    total_size_gb = num_files * 2.0  # Small models

                logger.debug(f"Found {num_files} weight files, estimated size: {total_size_gb}GB")

    return total_size_gb


def format_size(size_bytes: float) -> str:
    """Format size in bytes to human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"

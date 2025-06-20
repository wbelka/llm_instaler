"""Model compatibility checker for LLM Installer.

This module implements the logic to check if a model from HuggingFace
is compatible with the user's system without downloading the model weights.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import requests

from huggingface_hub import HfApi, ModelInfo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError

from core.config import get_config
from core.utils import (
    check_system_requirements, print_error,
    print_info, console
)
from detectors.registry import get_detector_registry
from rich.table import Table


# Special dependency mapping for architectures
ARCHITECTURE_DEPENDENCIES = {
    "mamba": ["mamba-ssm", "causal-conv1d"],
    "jamba": ["mamba-ssm", "causal-conv1d"],
    "rwkv": ["rwkv"],
    "whisper": ["openai-whisper"],
}

# Configuration keys that require special dependencies
CONFIG_DEPENDENCIES = {
    "_attn_implementation": {
        "flash_attention_2": ["flash-attn"],
    },
    "use_flash_attn": {
        True: ["flash-attn"],
    },
}


@dataclass
class ModelRequirements:
    """Model requirements information."""
    model_type: str = ""
    model_family: str = ""
    architecture_type: str = ""
    primary_library: str = ""
    base_dependencies: List[str] = field(default_factory=list)
    special_dependencies: List[str] = field(default_factory=list)
    optional_dependencies: List[str] = field(default_factory=list)
    disk_space_gb: float = 0.0
    memory_requirements: Dict[str, float] = field(default_factory=dict)
    capabilities: Dict[str, Any] = field(default_factory=dict)
    special_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompatibilityResult:
    """System compatibility analysis result."""
    can_run: bool = False
    device: str = "cpu"
    dtype: str = "float32"
    quantization: Optional[str] = None
    memory_required_gb: float = 0.0
    memory_available_gb: float = 0.0
    notes: List[str] = field(default_factory=list)


class ModelChecker:
    """Check model compatibility without downloading weights."""

    def __init__(self):
        self.config = get_config()
        self.api = HfApi(token=self.config.huggingface_token)
        self.logger = logging.getLogger(__name__)
        self.detector_registry = get_detector_registry()

    def check_model(self,
                    model_id: str) -> Tuple[bool,
                                            Optional[ModelRequirements]]:
        """Check if a model is compatible with the system.

        Args:
            model_id: HuggingFace model identifier.

        Returns:
            Tuple of (success, requirements).
        """
        print_info(f"Checking model: {model_id}")

        try:
            # Step 1: Get model info from HuggingFace
            print_info("Fetching model information from HuggingFace...")
            model_info, files_info = self._fetch_model_info(model_id)

            # Step 2: Download and analyze configuration files
            print_info("Analyzing model configuration...")
            config_data = self._analyze_configs(model_id, files_info)

            if not config_data:
                print_error(
                    "No configuration files found. "
                    "Cannot determine model type.")
                return False, None

            # Step 3: Determine model type
            print_info("Determining model type...")
            model_data = self._determine_model_type(
                model_info, config_data, files_info)

            # Step 4: Collect requirements
            print_info("Analyzing model requirements...")
            requirements = self._collect_requirements(
                model_data, files_info)

            # Step 5: Check system compatibility
            print_info("Checking system compatibility...")
            compatibility = self._check_compatibility(requirements)

            # Step 6: Generate and display report
            self._display_report(model_id, requirements, compatibility)

            # Step 7: Save results
            self._save_results(model_id, requirements, compatibility)

            return True, requirements

        except RepositoryNotFoundError:
            print_error(f"Model not found on HuggingFace: {model_id}")
            return False, None
        except GatedRepoError:
            print_error(
                "This is a private/gated model. "
                "Please set HF_TOKEN environment variable.")
            return False, None
        except Exception as e:
            print_error(f"Error checking model: {str(e)}")
            self.logger.exception("Model check failed")
            return False, None

    def _fetch_model_info(
            self, model_id: str) -> Tuple[ModelInfo, List[Dict[str, Any]]]:
        """Fetch model information from HuggingFace API."""
        # Get model metadata
        model_info = self.api.model_info(model_id)

        # Get list of all files with sizes
        files_info = []

        # Get repo info which includes file information
        try:
            # Try to get file info using the tree API endpoint
            tree_url = (f"https://huggingface.co/api/models/"
                        f"{model_id}/tree/main")
            headers = {}
            if self.config.huggingface_token:
                headers['Authorization'] = (
                    f'Bearer {self.config.huggingface_token}')
            response = requests.get(tree_url, headers=headers)

            if response.status_code == 200:
                tree_data = response.json()
                self.logger.debug(f"Got {len(tree_data)} files from tree API")

                # Check if this is a composite model (has model_index.json)
                has_model_index = any(
                    item.get('path') == 'model_index.json'
                    for item in tree_data if item.get('type') == 'file'
                )
                if has_model_index:
                    # Handle composite models (diffusers)
                    self.logger.debug(
                        "Detected composite model, fetching components")
                    files_info = self._fetch_composite_model_files(
                        model_id, tree_data, headers
                    )
                else:
                    # Handle regular models
                    for item in tree_data:
                        if item.get('type') == 'file':
                            file_dict = {
                                "path": item.get('path', ''),
                                "size": item.get('size', 0)
                            }
                            files_info.append(file_dict)

                            # Debug: log weight files with sizes
                            if file_dict["size"] > 0 and (
                                    file_dict["path"].endswith(
                                        '.safetensors') or
                                    file_dict["path"].endswith('.bin')):
                                size_mb = file_dict["size"] / (1024**2)
                                self.logger.debug(
                                    f"Weight file: {file_dict['path']} - "
                                    f"{size_mb:.1f} MB")

                return model_info, files_info

            # Fallback to siblings if tree API fails
            self.logger.debug("Tree API failed, trying siblings")
            if hasattr(model_info, 'siblings') and model_info.siblings:
                self.logger.debug(f"Using siblings data for {model_id}")
                for sibling in model_info.siblings:
                    if hasattr(sibling, 'rfilename'):
                        # Get size, ensuring it's never None
                        size = getattr(sibling, 'size', 0)
                        if size is None:
                            size = 0
                        file_dict = {
                            "path": sibling.rfilename,
                            "size": size
                        }
                        files_info.append(file_dict)

                        # Debug: log weight files with sizes
                        if size > 0 and (
                                sibling.rfilename.endswith('.safetensors') or
                                sibling.rfilename.endswith('.bin')):
                            size_mb = size / (1024**2)
                            self.logger.debug(
                                f"Weight file: {sibling.rfilename} - "
                                f"{size_mb:.1f} MB")
                return model_info, files_info

            # Final fallback to list_repo_files
            repo_files = self.api.list_repo_files(model_id)

            # Debug: log number of files found
            self.logger.debug(
                f"Found {len(repo_files)} files in {model_id}, "
                "but no size information available")

            # For each file, we need to get its info
            # Note: list_repo_files doesn't provide file sizes
            for file_path in repo_files:
                files_info.append({
                    "path": file_path,
                    "size": 0  # Size unknown without siblings data
                })

        except Exception as e:
            self.logger.warning(f"Could not get file list: {e}")
            self.logger.debug(f"Exception type: {type(e).__name__}")
            self.logger.debug(f"Exception details: {str(e)}", exc_info=True)
            # Fallback - just create a minimal list
            files_info = [{"path": "model.bin", "size": 0}]

        return model_info, files_info

    def _fetch_composite_model_files(self, model_id: str,
                                     tree_data: List[Dict[str, Any]],
                                     headers: Dict[str, str]) -> List[
                                         Dict[str, Any]]:
        """Fetch files for composite models (diffusers)."""
        files_info = []

        # First, add non-directory files from root
        for item in tree_data:
            if item.get('type') == 'file':
                files_info.append({
                    "path": item.get('path', ''),
                    "size": item.get('size', 0)
                })

        # Then, fetch files from each component directory
        directories = [item['path'] for item in tree_data
                       if item.get('type') == 'directory']
        for directory in directories:
            self.logger.debug(f"Fetching files from {directory}")
            dir_url = (f"https://huggingface.co/api/models/"
                       f"{model_id}/tree/main/{directory}")
            try:
                response = requests.get(dir_url, headers=headers)
                if response.status_code == 200:
                    dir_data = response.json()

                    for item in dir_data:
                        if item.get('type') == 'file':
                            file_path = f"{directory}/{item.get('path', '')}"
                            file_size = item.get('size', 0)

                            # For diffusers, prefer fp16 versions
                            # Skip duplicates (fp32 if fp16 exists)
                            is_weight_file = (
                                file_path.endswith('.safetensors') or
                                file_path.endswith('.bin'))
                            if is_weight_file:
                                # Check if this is a duplicate
                                base_name = (file_path
                                             .replace('.safetensors', '')
                                             .replace('.bin', ''))
                                has_fp16 = any(
                                    f['path'].startswith(base_name) and
                                    'fp16' in f['path']
                                    for f in files_info
                                )

                                # Skip fp32 versions if fp16 exists
                                if has_fp16 and 'fp16' not in file_path:
                                    self.logger.debug(
                                        f"Skipping {file_path} "
                                        "(fp16 version exists)")
                                    continue

                                # Skip fp32 if this is fp16 and
                                # fp32 already added
                                if 'fp16' in file_path:
                                    # Remove fp32 version if it exists
                                    files_info = [
                                        f for f in files_info
                                        if not (f['path'].startswith(
                                                base_name) and
                                                'fp16' not in f['path'])
                                    ]
                            files_info.append({
                                "path": file_path,
                                "size": file_size
                            })

                            if is_weight_file and file_size > 0:
                                size_gb = file_size / (1024**3)
                                self.logger.debug(
                                    f"Weight file: {file_path} - "
                                    f"{size_gb:.2f} GB"
                                )

            except Exception as e:
                self.logger.warning(f"Failed to fetch {directory}: {e}")

        return files_info

    def _analyze_configs(self, model_id: str,
                         files_info: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Download and analyze configuration files."""
        config_data = {}

        # Priority order for config files
        config_priority = [
            "model_index.json",  # Composite models (diffusers)
            "config.json",       # Standard config
            "llm_config.json",   # Alternative names
            "model_config.json",
        ]

        # Try to find and download config files
        for config_name in config_priority:
            if any(f["path"] == config_name for f in files_info):
                try:
                    config_path = hf_hub_download(
                        repo_id=model_id,
                        filename=config_name,
                        token=self.config.huggingface_token,
                        cache_dir=self.config.cache_dir
                    )

                    with open(config_path, 'r') as f:
                        config_content = json.load(f)

                    # Validate config content
                    if (config_name == "model_index.json" or
                            self._is_valid_config(config_content)):
                        config_data[config_name] = config_content

                        # For composite models, get component configs
                        if config_name == "model_index.json":
                            config_data.update(
                                self._get_component_configs(
                                    model_id, config_content, files_info))

                except Exception as e:
                    self.logger.warning(f"Failed to load {config_name}: {e}")

        # If no standard configs found, search for any config.json in
        # subdirectories
        if not config_data:
            for file_info in files_info:
                path = file_info["path"]
                if path.endswith("config.json") and "/" in path:
                    try:
                        config_path = hf_hub_download(
                            repo_id=model_id,
                            filename=path,
                            token=self.config.huggingface_token,
                            cache_dir=self.config.cache_dir
                        )

                        with open(config_path, 'r') as f:
                            config_content = json.load(f)

                        if self._is_valid_config(config_content):
                            config_data[path] = config_content

                    except Exception as e:
                        self.logger.warning(f"Failed to load {path}: {e}")

        return config_data

    def _is_valid_config(self, config: Dict[str, Any]) -> bool:
        """Check if config contains model architecture keys."""
        # Keys that indicate this is a model config
        model_keys = {
            "model_type", "hidden_size", "num_layers", "vocab_size",
            "architectures", "vision_config", "text_config", "audio_config",
            "encoder_layers", "decoder_layers", "d_model", "feature_size"
        }

        # Keys that indicate this is NOT a model config
        non_model_keys = {"api_key", "base_url", "endpoint"}

        # Check for non-model keys
        if any(key in config for key in non_model_keys):
            return False

        # Check for model keys
        return any(key in config for key in model_keys)

    def _get_component_configs(self,
                               model_id: str,
                               model_index: Dict[str,
                                                 Any],
                               files_info: List[Dict[str,
                                                     Any]]) -> Dict[str,
                                                                    Any]:
        """Get configuration files for composite model components."""
        component_configs = {}

        # Extract component paths from model_index
        for key, value in model_index.items():
            if isinstance(value, list) and len(value) == 2:
                component_name, component_path = value
                config_path = f"{component_path}/config.json"

                if any(f["path"] == config_path for f in files_info):
                    try:
                        config_file = hf_hub_download(
                            repo_id=model_id,
                            filename=config_path,
                            token=self.config.huggingface_token,
                            cache_dir=self.config.cache_dir
                        )

                        with open(config_file, 'r') as f:
                            config = json.load(f)
                            component_configs[
                                f"{component_path}_config"] = config

                    except Exception as e:
                        self.logger.warning(
                            f"Failed to load component config "
                            f"{config_path}: {e}")

        return component_configs

    def _determine_model_type(self,
                              model_info: ModelInfo,
                              config_data: Dict[str,
                                                Any],
                              files_info: List[Dict[str,
                                                    Any]]) -> Dict[str,
                                                                   Any]:
        """Determine model type using detectors."""
        model_data = {
            "model_info": model_info,
            "config_data": config_data,
            "files_info": files_info,
            "tags": model_info.tags if hasattr(
                model_info,
                'tags') else [],
            "pipeline_tag": model_info.pipeline_tag if hasattr(
                model_info,
                'pipeline_tag') else None,
            "library_name": model_info.library_name if hasattr(
                model_info,
                'library_name') else None,
        }

        # Run through detector chain
        for detector in self.detector_registry.get_detectors():
            if detector.matches(model_data):
                analysis = detector.analyze(model_data)
                model_data.update(analysis)
                break

        # If no detector matched, try to infer from available data
        if "model_type" not in model_data:
            model_data.update(self._infer_model_type(model_data))

        return model_data

    def _infer_model_type(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Infer model type from available data."""
        inferred = {
            "model_type": "unknown",
            "model_family": "unknown",
            "architecture_type": "unknown",
        }

        # Check by pipeline tag
        pipeline_tag = model_data.get("pipeline_tag", "")
        if pipeline_tag:
            if "text-generation" in pipeline_tag:
                inferred.update({
                    "model_type": "transformer",
                    "model_family": "language-model",
                    "architecture_type": "decoder-only",
                })
            elif "text-to-image" in pipeline_tag:
                inferred.update({
                    "model_type": "diffusion",
                    "model_family": "image-generation",
                    "architecture_type": "diffusion",
                })
            elif "text-to-video" in pipeline_tag:
                inferred.update({
                    "model_type": "video-diffusion",
                    "model_family": "video-generation",
                    "architecture_type": "diffusion",
                })
            elif "automatic-speech-recognition" in pipeline_tag:
                inferred.update({
                    "model_type": "audio",
                    "model_family": "speech-recognition",
                    "architecture_type": "encoder",
                })
            elif "text-to-audio" in pipeline_tag:
                inferred.update({
                    "model_type": "audio-generator",
                    "model_family": "audio-generation",
                    "architecture_type": "encoder-decoder",
                })
            elif "audio-to-audio" in pipeline_tag:
                inferred.update({
                    "model_type": "audio-processor",
                    "model_family": "audio",
                    "architecture_type": "encoder-decoder",
                })

        return inferred

    def _collect_requirements(
            self,
            model_data: Dict[str, Any],
            files_info: List[Dict[str, Any]]) -> ModelRequirements:
        """Collect all model requirements."""
        requirements = ModelRequirements()

        # Basic model info
        requirements.model_type = model_data.get("model_type", "unknown")
        requirements.model_family = model_data.get("model_family", "unknown")
        requirements.architecture_type = model_data.get(
            "architecture_type", "unknown")
        requirements.primary_library = model_data.get(
            "primary_library", "transformers")

        # Determine dependencies
        requirements.base_dependencies = self._get_base_dependencies(
            model_data)
        requirements.special_dependencies = self._get_special_dependencies(
            model_data)
        requirements.optional_dependencies = self._get_optional_dependencies(
            model_data)

        # Calculate disk space
        requirements.disk_space_gb = self._calculate_disk_space(files_info)

        # Estimate memory requirements
        requirements.memory_requirements = self._estimate_memory_requirements(
            model_data, files_info
        )

        # Extract capabilities
        requirements.capabilities = self._extract_capabilities(model_data)

        # Special configuration
        requirements.special_config = model_data.get("special_config", {})

        return requirements

    def _get_base_dependencies(self, model_data: Dict[str, Any]) -> List[str]:
        """Get base dependencies for the model."""
        deps = ["torch", "accelerate"]

        # Add primary library
        library = model_data.get("primary_library", "transformers")
        if library and library not in deps:
            deps.append(library)

        # Add dependencies based on model family
        family = model_data.get("model_family", "")
        if family == "image-generation":
            deps.extend(["pillow", "numpy"])
        elif family == "video-generation":
            deps.extend(["pillow", "numpy", "opencv-python", "imageio"])
        elif family in ["audio", "speech-recognition"]:
            deps.extend(["librosa", "soundfile"])
        elif family == "audio-generation":
            deps.extend(["scipy", "soundfile", "numpy"])
            # MusicGen specific
            if model_data.get("model_type") == "musicgen":
                deps.extend(["audiocraft"])
        elif family == "computer-vision":
            deps.extend(["pillow", "opencv-python"])

        return deps

    def _get_special_dependencies(
            self, model_data: Dict[str, Any]) -> List[str]:
        """Get special/architecture-specific dependencies."""
        special_deps = []

        # Check model type in configs
        for config_name, config in model_data.get("config_data", {}).items():
            if isinstance(config, dict):
                # Check for architecture-specific dependencies
                model_type = config.get("model_type", "")
                if model_type in ARCHITECTURE_DEPENDENCIES:
                    special_deps.extend(
                        ARCHITECTURE_DEPENDENCIES[model_type])

                # Check for config-based dependencies
                for key, dep_map in CONFIG_DEPENDENCIES.items():
                    if key in config:
                        value = config[key]
                        if value in dep_map:
                            special_deps.extend(dep_map[value])

        # Remove duplicates
        return list(set(special_deps))

    def _get_optional_dependencies(
            self, model_data: Dict[str, Any]) -> List[str]:
        """Get optional dependencies that can improve performance."""
        optional = []

        # xformers for diffusion models
        if model_data.get("model_family") == "image-generation":
            optional.append("xformers")

        # Flash attention if not already in special deps
        special_deps = model_data.get("special_dependencies", [])
        if "flash-attn" not in special_deps and model_data.get(
                "model_family") == "language-model":
            # Check if model might benefit from flash attention
            for config in model_data.get("config_data", {}).values():
                if isinstance(config, dict):
                    max_pos = config.get("max_position_embeddings", 0)
                    if max_pos >= 2048:
                        optional.append("flash-attn")
                        break

        return optional

    def _calculate_disk_space(self, files_info: List[Dict[str, Any]]) -> float:
        """Calculate total disk space needed."""
        total_bytes = sum(f.get("size", 0) for f in files_info)
        model_size_gb = total_bytes / (1024**3)

        # Add space for virtual environment
        venv_size_gb = 3.0

        return model_size_gb + venv_size_gb

    def _estimate_memory_requirements(
            self, model_data: Dict[str, Any],
            files_info: List[Dict[str, Any]]) -> Dict[str, float]:
        """Estimate memory requirements for different configurations."""
        # Get weight files
        weight_files = [
            f for f in files_info if any(
                f["path"].endswith(ext) for ext in [
                    ".bin",
                    ".safetensors",
                    ".pt",
                    ".pth"])]

        # Calculate base memory from weight files
        weight_bytes = sum(f.get("size", 0) for f in weight_files)
        base_memory_gb = weight_bytes / (1024**3)

        # Determine native dtype
        native_dtype = "float16"  # Default assumption
        for config in model_data.get("config_data", {}).values():
            if isinstance(config, dict):
                torch_dtype = config.get("torch_dtype")
                if torch_dtype:
                    if isinstance(torch_dtype, str):
                        native_dtype = torch_dtype
                        break
                    elif (isinstance(torch_dtype, dict) and
                          "_name_or_path" in torch_dtype):
                        dtype_str = torch_dtype["_name_or_path"]
                        if "float16" in dtype_str:
                            native_dtype = "float16"
                        elif "bfloat16" in dtype_str:
                            native_dtype = "bfloat16"
                        elif "float32" in dtype_str:
                            native_dtype = "float32"
                        break

        # Calculate memory requirements based on native dtype
        # If model is native float16, base_memory_gb is already the size for
        # float16

        # Check if this is a diffusion model (lower overhead needed)
        is_diffusion = False
        for config_name in model_data.get("config_data", {}).keys():
            if "model_index.json" in config_name:
                is_diffusion = True
                break

        # Different overhead for different model types
        if is_diffusion:
            # Diffusion models need less overhead (mainly for latent space)
            overhead_factor = 1.2
        else:
            # Language models need more overhead (KV cache, activations)
            overhead_factor = 1.5

        if native_dtype == "float16":
            memory_reqs = {
                # 4bit/16bit * overhead
                "int4": base_memory_gb * 0.25 * 1.2,
                # 8bit/16bit * overhead
                "int8": base_memory_gb * 0.5 * 1.3,
                # native + overhead
                "float16": base_memory_gb * overhead_factor,
                # same size as float16
                "bfloat16": base_memory_gb * overhead_factor,
                # 32bit/16bit * overhead
                "float32": base_memory_gb * 2.0 * overhead_factor,
            }
        elif native_dtype == "bfloat16":
            memory_reqs = {
                "int4": base_memory_gb * 0.25 * 1.2,
                "int8": base_memory_gb * 0.5 * 1.3,
                # same size as bfloat16
                "float16": base_memory_gb * overhead_factor,
                # native + overhead
                "bfloat16": base_memory_gb * overhead_factor,
                "float32": base_memory_gb * 2.0 * overhead_factor,
            }
        elif native_dtype == "float32":
            memory_reqs = {
                # 4bit/32bit * overhead
                "int4": base_memory_gb * 0.125 * 1.2,
                # 8bit/32bit * overhead
                "int8": base_memory_gb * 0.25 * 1.3,
                # 16bit/32bit * overhead
                "float16": base_memory_gb * 0.5 * overhead_factor,
                "bfloat16": base_memory_gb * 0.5 * overhead_factor,
                # native + overhead
                "float32": base_memory_gb * overhead_factor,
            }
        else:
            # Fallback to original calculation
            memory_reqs = {
                "int4": base_memory_gb * 1.2,
                "int8": base_memory_gb * 1.3,
                "float16": base_memory_gb * 1.5,
                "bfloat16": base_memory_gb * 1.5,
                "float32": base_memory_gb * 2.0,
            }

        # Check if model has quantization config
        for config in model_data.get("config_data", {}).values():
            if (isinstance(config, dict) and
                    "quantization_config" in config):
                quant_config = config["quantization_config"]
                if quant_config.get("bits") == 4:
                    memory_reqs["quantized"] = memory_reqs["int4"]
                elif quant_config.get("bits") == 8:
                    memory_reqs["quantized"] = memory_reqs["int8"]

        return memory_reqs

    def _extract_capabilities(
            self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model capabilities from configuration."""
        capabilities = {
            "supports_streaming": False,
            "supports_system_prompt": False,
            "supports_reasoning": False,
            "max_context_length": None,
            "native_dtype": None,
        }

        # Check for language model capabilities
        if model_data.get("model_family") == "language-model":
            capabilities["supports_streaming"] = True

            # Check for specific features in configs
            for config in model_data.get("config_data", {}).values():
                if isinstance(config, dict):
                    # Context length
                    max_pos = config.get(
                        "max_position_embeddings") or config.get("n_positions")
                    if max_pos:
                        capabilities["max_context_length"] = max_pos

                    # Check for chat/instruct models
                    if "chat" in str(config.get("_name_or_path", "")).lower():
                        capabilities["supports_system_prompt"] = True

                    # Check for reasoning models
                    model_type = config.get("model_type", "")
                    if (model_type in ["o1", "reasoning-llm"] or
                            "reasoning" in model_type):
                        capabilities["supports_reasoning"] = True

        # Extract native dtype from configs
        for config in model_data.get("config_data", {}).values():
            if isinstance(config, dict):
                # Check torch_dtype field
                torch_dtype = config.get("torch_dtype")
                if torch_dtype:
                    if isinstance(torch_dtype, str):
                        # Handle string format like "float16", "bfloat16"
                        capabilities["native_dtype"] = torch_dtype
                    elif (isinstance(torch_dtype, dict) and
                          "_name_or_path" in torch_dtype):
                        # Handle dict format {"_name_or_path": "torch.float16"}
                        dtype_str = torch_dtype["_name_or_path"]
                        if "float16" in dtype_str:
                            capabilities["native_dtype"] = "float16"
                        elif "bfloat16" in dtype_str:
                            capabilities["native_dtype"] = "bfloat16"
                        elif "float32" in dtype_str:
                            capabilities["native_dtype"] = "float32"
                    break

                # Some models specify dtype in quantization_config
                quant_config = config.get("quantization_config")
                if quant_config and isinstance(quant_config, dict):
                    if quant_config.get("bits") == 4:
                        capabilities["native_dtype"] = "int4"
                    elif quant_config.get("bits") == 8:
                        capabilities["native_dtype"] = "int8"
                    elif "bnb_4bit_compute_dtype" in quant_config:
                        capabilities["native_dtype"] = quant_config[
                            "bnb_4bit_compute_dtype"]
                    break

                # Check if model name hints at dtype
                model_name = config.get("_name_or_path", "").lower()
                if not capabilities["native_dtype"] and model_name:
                    if "fp16" in model_name or "f16" in model_name:
                        capabilities["native_dtype"] = "float16"
                    elif "bf16" in model_name or "bfloat16" in model_name:
                        capabilities["native_dtype"] = "bfloat16"
                    elif "int8" in model_name or "8bit" in model_name:
                        capabilities["native_dtype"] = "int8"
                    elif "int4" in model_name or "4bit" in model_name:
                        capabilities["native_dtype"] = "int4"

        # Default to float16 if not specified
        # (most common for modern models)
        if not capabilities["native_dtype"]:
            capabilities["native_dtype"] = "float16"

        return capabilities

    def _check_compatibility(
            self,
            requirements: ModelRequirements) -> Dict[
                str, CompatibilityResult]:
        """Check compatibility with system."""
        system_info = check_system_requirements()
        results = {}

        # Check for each dtype configuration
        for dtype, memory_gb in requirements.memory_requirements.items():
            result = CompatibilityResult()
            result.dtype = dtype
            result.memory_required_gb = memory_gb

            # Check GPU availability first
            if system_info["cuda_available"] and system_info["gpu_info"]:
                gpu = system_info["gpu_info"][0]  # Use first GPU
                gpu_memory_gb = gpu["memory_mb"] / \
                    1024 if isinstance(gpu["memory_mb"], int) else 0

                if gpu_memory_gb >= memory_gb:
                    result.can_run = True
                    result.device = "cuda"
                    result.memory_available_gb = gpu_memory_gb
                elif gpu_memory_gb >= memory_gb * 0.8:
                    result.can_run = True
                    result.device = "cuda"
                    result.memory_available_gb = gpu_memory_gb
                    result.notes.append("Close to VRAM limit")

            # Check CPU/RAM if GPU not suitable
            if not result.can_run:
                available_ram = system_info["available_memory_gb"]
                if available_ram >= memory_gb:
                    result.can_run = True
                    result.device = "cpu"
                    result.memory_available_gb = available_ram
                elif available_ram >= memory_gb * 0.8:
                    result.can_run = True
                    result.device = "cpu"
                    result.memory_available_gb = available_ram
                    result.notes.append("Close to RAM limit")

            # Add quantization note if applicable
            if dtype in ["int4", "int8", "quantized"]:
                result.quantization = dtype

            results[dtype] = result

        # Check disk space
        disk_free_gb = system_info["disk_space_gb"]["home"]["free"]
        if disk_free_gb < requirements.disk_space_gb:
            for result in results.values():
                result.can_run = False
                result.notes.append(
                    f"Insufficient disk space: need {
                        requirements.disk_space_gb:.1f} GB")

        return results

    def _display_report(self, model_id: str, requirements: ModelRequirements,
                        compatibility: Dict[str, CompatibilityResult]):
        """Display the compatibility report."""
        console.print(f"\n[bold]Model Compatibility Report: {model_id}[/bold]")
        console.print("=" * 60)

        # Model information
        info_table = Table(show_header=False, box=None)
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="white")

        info_table.add_row("Model Type", requirements.model_type)
        info_table.add_row("Library", requirements.primary_library)
        info_table.add_row("Architecture", requirements.architecture_type)
        info_table.add_row("Task", requirements.model_family)

        # Add native dtype if available
        native_dtype = requirements.capabilities.get("native_dtype")
        if native_dtype:
            info_table.add_row("Native Data Type", native_dtype)

        console.print(info_table)
        console.print()

        # Storage requirements
        console.print("[bold]Storage Requirements:[/bold]")
        model_size_gb = requirements.disk_space_gb - 3
        console.print(f"  - Model files: {model_size_gb:.1f} GB")
        console.print("  - Virtual environment: ~3 GB")
        console.print(
            f"  - Total disk space needed: "
            f"~{requirements.disk_space_gb:.1f} GB")
        console.print()

        # Memory requirements
        console.print("[bold]Memory Requirements (RAM/VRAM):[/bold]")
        native_dtype = requirements.capabilities.get("native_dtype", "float16")
        for dtype, memory_gb in requirements.memory_requirements.items():
            # Mark native dtype
            if dtype == native_dtype:
                console.print(
                    f"  - {dtype}: {memory_gb:.1f} GB [green](native)[/green]")
            else:
                console.print(f"  - {dtype}: {memory_gb:.1f} GB")
        console.print()

        # Model capabilities
        if requirements.capabilities:
            console.print("[bold]Model Capabilities:[/bold]")
            for cap, value in requirements.capabilities.items():
                if value is True:
                    console.print(f"  - {cap.replace('_', ' ').title()}")
                elif value and cap == "max_context_length":
                    console.print(f"  - Max Context Length: {value} tokens")
                elif value and cap == "native_dtype":
                    console.print(f"  - Native Data Type: {value}")
        console.print()

        # Special requirements
        if requirements.base_dependencies or requirements.special_dependencies:
            console.print("[bold]Special Requirements:[/bold]")

            if requirements.base_dependencies:
                console.print("  Base:")
                for dep in requirements.base_dependencies:
                    console.print(f"  - {dep}")

            if requirements.special_dependencies:
                console.print("\n  Architecture-specific:")
                for dep in requirements.special_dependencies:
                    note = ""
                    if dep == "mamba-ssm":
                        note = " (requires CUDA compilation)"
                    elif dep == "flash-attn":
                        note = " (requires CUDA 11.6+)"
                    console.print(f"  - {dep}{note}")

            if requirements.optional_dependencies:
                console.print("\n  Optional:")
                for dep in requirements.optional_dependencies:
                    note = ""
                    if dep == "xformers":
                        note = " (for faster inference)"
                    elif dep == "flash-attn":
                        note = " (for faster inference, requires CUDA 11.6+)"
                    console.print(f"  - {dep}{note}")

        # System compatibility
        console.print("\n" + "-" * 60)
        console.print("[bold]Hardware Compatibility Check[/bold]")
        console.print("-" * 60)

        system_info = check_system_requirements()
        console.print("[bold]Your Hardware:[/bold]")
        console.print(f"  - CPU: {system_info['cpu_count']} cores")
        console.print(
            f"  - RAM: {system_info['total_memory_gb']:.1f} GB total, "
            f"{system_info['available_memory_gb']:.1f} GB available")

        if system_info["cuda_available"]:
            console.print("  - GPU(s):")
            for gpu in system_info["gpu_info"]:
                mem_str = f"{gpu['memory_mb']} MB" if isinstance(
                    gpu['memory_mb'], int) else gpu['memory_mb']
                console.print(f"    • {gpu['name']}")
                console.print(f"      VRAM: {mem_str}")
        console.print()

        # Compatibility analysis table
        console.print("[bold]Compatibility Analysis:[/bold]")

        compat_table = Table()
        compat_table.add_column("Quantization", style="cyan")
        compat_table.add_column("Memory", style="yellow")
        compat_table.add_column("Can Run?", style="white")
        compat_table.add_column("Device", style="magenta")
        compat_table.add_column("Notes", style="dim")

        for dtype, result in compatibility.items():
            can_run = "✓" if result.can_run else "✗"
            can_run_style = "green" if result.can_run else "red"

            notes = ", ".join(result.notes) if result.notes else ""

            compat_table.add_row(
                dtype,
                f"{result.memory_required_gb:.1f} GB",
                f"[{can_run_style}]{can_run}[/{can_run_style}]",
                result.device if result.can_run else "N/A",
                notes
            )

        console.print(compat_table)

        # Training compatibility
        console.print("\n[bold]Training Compatibility:[/bold]")

        train_table = Table()
        train_table.add_column("Method", style="cyan")
        train_table.add_column("Memory", style="yellow")
        train_table.add_column("CPU", style="white")
        train_table.add_column("GPU", style="white")
        train_table.add_column("Device", style="magenta")

        # Estimate training memory
        base_memory = requirements.memory_requirements.get(
            "float16", 16.0)
        lora_memory = base_memory * 1.3
        full_memory = base_memory * 4

        # Check LoRA training
        lora_cpu = ("✓" if system_info["available_memory_gb"] >= lora_memory
                    else "✗")
        lora_gpu = "✗"
        lora_device = "cpu" if lora_cpu == "✓" else "N/A"

        if system_info["cuda_available"] and system_info["gpu_info"]:
            gpu_mem_gb = system_info["gpu_info"][0]["memory_mb"] / 1024
            if isinstance(
                    system_info["gpu_info"][0]["memory_mb"],
                    int) and gpu_mem_gb >= lora_memory:
                lora_gpu = "✓"
                lora_device = "cuda"

        train_table.add_row(
            "LoRA (float16)",
            f"{lora_memory:.1f} GB",
            f"[{'green' if lora_cpu == '✓' else 'red'}]{lora_cpu}[/]",
            f"[{'green' if lora_gpu == '✓' else 'red'}]{lora_gpu}[/]",
            lora_device
        )

        # Check full training
        full_cpu = ("✓" if system_info["available_memory_gb"] >= full_memory
                    else "✗")
        full_gpu = "✗"
        full_device = "cpu" if full_cpu == "✓" else "N/A"

        if system_info["cuda_available"] and system_info["gpu_info"]:
            if isinstance(
                    system_info["gpu_info"][0]["memory_mb"],
                    int) and gpu_mem_gb >= full_memory:
                full_gpu = "✓"
                full_device = "cuda"

        train_table.add_row(
            "Full (float16)",
            f"{full_memory:.1f} GB",
            f"[{'green' if full_cpu == '✓' else 'red'}]{full_cpu}[/]",
            f"[{'green' if full_gpu == '✓' else 'red'}]{full_gpu}[/]",
            full_device
        )

        console.print(train_table)
        console.print("\n" + "=" * 60)

    def _save_results(self, model_id: str, requirements: ModelRequirements,
                      compatibility: Dict[str, CompatibilityResult]):
        """Save check results for later use."""
        # Create logs/checks directory
        checks_dir = self.config.logs_dir / "checks"
        checks_dir.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        safe_name = model_id.replace("/", "_")
        json_path = checks_dir / f"{safe_name}.json"

        from datetime import datetime

        results = {
            "model_id": model_id,
            "check_date": datetime.now().isoformat(),
            "requirements": {
                "model_type": requirements.model_type,
                "model_family": requirements.model_family,
                "architecture_type": requirements.architecture_type,
                "primary_library": requirements.primary_library,
                "base_dependencies": requirements.base_dependencies,
                "special_dependencies": requirements.special_dependencies,
                "optional_dependencies": requirements.optional_dependencies,
                "disk_space_gb": requirements.disk_space_gb,
                "memory_requirements": requirements.memory_requirements,
                "capabilities": requirements.capabilities,
                "special_config": requirements.special_config,
            },
            "compatibility": {
                dtype: {
                    "can_run": result.can_run,
                    "device": result.device,
                    "memory_required_gb": result.memory_required_gb,
                    "memory_available_gb": result.memory_available_gb,
                    "notes": result.notes,
                }
                for dtype, result in compatibility.items()
            }
        }

        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Saved check results to {json_path}")

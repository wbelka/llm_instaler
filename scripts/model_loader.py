"""Universal model loader for LLM Installer.

This module provides functionality to load any type of model
using the appropriate handler from the installer.
"""

import sys
import os
import json
import logging
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path

# Add current directory to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def get_handler(model_info: Dict[str, Any]):
    """Get the appropriate handler for the model.

    Args:
        model_info: Model information dictionary.

    Returns:
        Handler instance for the model type.
    """
    try:
        from handlers.registry import get_handler_registry

        registry = get_handler_registry()
        handler_class = registry.get_handler_for_model(model_info)

        if handler_class:
            return handler_class(model_info)
        else:
            # No handler found, will use fallback loading
            return None

    except ImportError:
        # Handlers not available, will use fallback loading
        return None
    except Exception:
        # Error getting handler, will use fallback loading
        return None


def load_model(
    model_info: Dict[str, Any],
    model_path: str = "./model",
    device: str = "auto",
    dtype: str = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    **kwargs
) -> Tuple[Any, Optional[Any]]:
    """Load a model with the appropriate handler.

    Args:
        model_info: Model information dictionary.
        model_path: Path to model files.
        device: Device to load model on (auto/cuda/cpu/mps).
        dtype: Data type for model (auto/float16/float32/bfloat16).
        load_in_8bit: Whether to load in 8-bit quantization.
        load_in_4bit: Whether to load in 4-bit quantization.
        **kwargs: Additional arguments for model loading.

    Returns:
        Tuple of (model, tokenizer/processor).
    """
    # Try to get handler
    handler = get_handler(model_info)

    if handler:
        # Use handler to load model
        return handler.load_model(
            model_path=model_path,
            device=device,
            dtype=dtype,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            **kwargs
        )

    # Fallback: Direct loading based on library
    primary_lib = model_info.get("primary_library", "transformers")

    if primary_lib == "transformers":
        return _load_transformers_model(
            model_info, model_path, device, dtype,
            load_in_8bit, load_in_4bit, **kwargs
        )
    elif primary_lib == "diffusers":
        # Remove lora_path from kwargs as diffusers doesn't expect it
        diffusers_kwargs = kwargs.copy()
        diffusers_kwargs.pop('lora_path', None)
        return _load_diffusers_model(
            model_info, model_path, device, dtype, **diffusers_kwargs
        )
    else:
        raise ValueError(f"Unsupported library: {primary_lib}")


def _load_transformers_model(
    model_info: Dict[str, Any],
    model_path: str,
    device: str,
    dtype: str,
    load_in_8bit: bool,
    load_in_4bit: bool,
    **kwargs
) -> Tuple[Any, Any]:
    """Load a transformers model.

    Args:
        model_info: Model information.
        model_path: Path to model files.
        device: Device to use.
        dtype: Data type.
        load_in_8bit: 8-bit quantization.
        load_in_4bit: 4-bit quantization.

    Returns:
        Tuple of (model, tokenizer).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Determine dtype
    if dtype == "auto":
        if device == "cuda" and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif device in ["cuda", "mps"]:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    else:
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(dtype, torch.float32)

    # Quantization config
    quantization_config = None
    if load_in_8bit or load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=torch_dtype if load_in_4bit else None,
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load model
    model_kwargs = {
        "pretrained_model_name_or_path": model_path,
        "torch_dtype": torch_dtype,
        "device_map": device if device != "mps" else None,
        "quantization_config": quantization_config,
        "trust_remote_code": True,
        **kwargs
    }

    # Remove None values
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    # Move to device if needed
    if device == "mps" and not load_in_8bit and not load_in_4bit:
        model = model.to(device)

    return model, tokenizer


def _load_diffusers_model(
    model_info: Dict[str, Any],
    model_path: str,
    device: str,
    dtype: str,
    **kwargs
) -> Tuple[Any, None]:
    """Load a diffusers model.

    Args:
        model_info: Model information.
        model_path: Path to model files.
        device: Device to use.
        dtype: Data type.

    Returns:
        Tuple of (pipeline, None).
    """
    import torch
    from diffusers import DiffusionPipeline

    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Determine dtype
    if dtype == "auto":
        if device in ["cuda", "mps"]:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    else:
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(dtype, torch.float32)

    # Check if model has a specific pipeline class defined
    model_index_path = os.path.join(model_path, "model_index.json")
    if os.path.exists(model_index_path):
        with open(model_index_path, 'r') as f:
            model_index = json.load(f)
            pipeline_class_name = model_index.get("_class_name", None)
            
            if pipeline_class_name:
                # Try to get the specific pipeline class
                try:
                    # Special handling for text-to-video pipeline
                    if pipeline_class_name == "TextToVideoSDPipeline":
                        from diffusers import TextToVideoSDPipeline
                        pipeline = TextToVideoSDPipeline.from_pretrained(
                            model_path,
                            torch_dtype=torch_dtype,
                            **kwargs
                        )
                    else:
                        # Try to import from diffusers.pipelines
                        from diffusers import pipelines
                        pipeline_class = getattr(pipelines, pipeline_class_name, None)
                        if pipeline_class:
                            pipeline = pipeline_class.from_pretrained(
                                model_path,
                                torch_dtype=torch_dtype,
                                **kwargs
                            )
                        else:
                            # Fallback to auto-detection
                            pipeline = DiffusionPipeline.from_pretrained(
                                model_path,
                                torch_dtype=torch_dtype,
                                **kwargs
                            )
                except ImportError as e:
                    logging.warning(f"Missing dependency for {pipeline_class_name}: {e}")
                    logging.info("Trying auto-detection fallback...")
                    # Fallback to auto-detection
                    pipeline = DiffusionPipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch_dtype,
                        **kwargs
                    )
                except Exception as e:
                    logging.warning(f"Failed to load specific pipeline class {pipeline_class_name}: {e}")
                    # Fallback to auto-detection
                    pipeline = DiffusionPipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch_dtype,
                        **kwargs
                    )
            else:
                # Use auto-detection
                pipeline = DiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    **kwargs
                )
    else:
        # Use auto-detection
        pipeline = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            **kwargs
        )

    # Move to device
    pipeline = pipeline.to(device)

    # Enable optimizations
    if device == "cuda":
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            logging.info("Enabled xformers memory efficient attention")
        except Exception as e:
            logging.warning(f"Could not enable xformers: {e}")
            logging.info("Falling back to standard attention")
            # Use attention slicing as fallback
            pipeline.enable_attention_slicing()
    elif device == "mps":
        # MPS optimizations
        pipeline.enable_attention_slicing()

    return pipeline, None


def get_model_config(model_info_path: str = "model_info.json") -> Dict[str, Any]:
    """Load model configuration from model_info.json.

    Args:
        model_info_path: Path to model_info.json file.

    Returns:
        Model information dictionary.
    """
    with open(model_info_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # Test loading model info
    try:
        model_info = get_model_config()
        print(f"Model: {model_info.get('model_id', 'Unknown')}")
        print(f"Type: {model_info.get('model_type', 'Unknown')}")
        print(f"Family: {model_info.get('model_family', 'Unknown')}")
    except Exception as e:
        print(f"Error loading model info: {e}")

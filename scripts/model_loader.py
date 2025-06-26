"""Universal model loader for LLM Installer.

This module provides functionality to load any type of model
using the appropriate handler from the installer.
"""

import sys
import os
import json
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# Set CUDA memory allocation configuration to prevent fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Set up logger
logger = logging.getLogger(__name__)

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
            logger.info(f"Found handler class: {handler_class.__name__}")
            handler = handler_class(model_info)
            
            # Store handler class name in model info for later reference
            model_info['handler_class'] = handler_class.__name__
            
            return handler
        else:
            # No handler found, will use fallback loading
            logger.warning(f"No handler found for model type: {model_info.get('model_type', 'unknown')}")
            return None

    except ImportError as e:
        # Handlers not available, will use fallback loading
        logger.error(f"ImportError getting handler: {e}")
        return None
    except Exception as e:
        # Error getting handler, will use fallback loading
        logger.error(f"Error getting handler: {e}")
        return None


def load_model(
    model_info: Dict[str, Any],
    model_path: str = "./model",
    device: str = "auto",
    dtype: str = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    use_flash_attention_2: bool = None,
    **kwargs
) -> Tuple[Any, Optional[Any]]:
    """Load a model with the appropriate handler.

    Args:
        model_info: Model information dictionary.
        model_path: Path to model files.
        device: Device to load model on (auto/cuda/cpu/mps).
        dtype: Data type for model (auto/float16/float32/bfloat16/int8/int4).
        load_in_8bit: Whether to load in 8-bit quantization.
        load_in_4bit: Whether to load in 4-bit quantization.
        use_flash_attention_2: Whether to use Flash Attention 2 (None=auto, True/False=force).
        **kwargs: Additional arguments for model loading.

    Returns:
        Tuple of (model, tokenizer/processor).
    """
    # Convert dtype to quantization flags
    if dtype == "int8":
        load_in_8bit = True
        dtype = "auto"  # Let the model decide the compute dtype
    elif dtype == "int4":
        load_in_4bit = True
        dtype = "auto"  # Let the model decide the compute dtype
    
    # Try to get handler
    handler = get_handler(model_info)
    
    # Log handler information for debugging
    logger.info(f"Model type: {model_info.get('model_type', 'unknown')}")
    logger.info(f"Model family: {model_info.get('model_family', 'unknown')}")
    logger.info(f"Handler found: {handler is not None}")
    logger.info(f"Dtype: {dtype}, load_in_8bit: {load_in_8bit}, load_in_4bit: {load_in_4bit}")
    
    if handler:
        # Use handler to load model
        logger.info(f"Using handler: {handler.__class__.__name__}")
        return handler.load_model(
            model_path=model_path,
            device=device,
            dtype=dtype,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            use_flash_attention_2=use_flash_attention_2,
            **kwargs
        )

    # Fallback: Direct loading based on library
    primary_lib = model_info.get("primary_library", "transformers")

    if primary_lib == "transformers":
        return _load_transformers_model(
            model_info, model_path, device, dtype,
            load_in_8bit, load_in_4bit, use_flash_attention_2, **kwargs
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
    use_flash_attention_2: bool = None,
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

    # Check for LoRA adapter and load if available
    lora_path = kwargs.get('lora_path', Path(model_path).parent / "lora")
    print(f"\n🔍 Checking for LoRA adapter...")
    print(f"   lora_path: {lora_path}")
    print(f"   Path exists: {Path(lora_path).exists() if lora_path else False}")
    print(f"   load_lora flag: {kwargs.get('load_lora', True)}")
    
    if lora_path and Path(lora_path).exists() and kwargs.get('load_lora', True):
        try:
            from peft import PeftModel
            print("\n" + "="*50)
            print("🎯 LOADING LoRA ADAPTER")
            print(f"Path: {lora_path}")
            print("="*50)
            
            # List contents of LoRA directory
            adapter_path = Path(lora_path)
            print("\n📁 LoRA directory contents:")
            for file in adapter_path.iterdir():
                if file.is_file():
                    print(f"   - {file.name} ({file.stat().st_size / 1024 / 1024:.1f} MB)")
            
            # Check which adapter file exists
            if (adapter_path / "adapter_model.safetensors").exists():
                print("\n📄 Found adapter_model.safetensors")
                adapter_size = (adapter_path / "adapter_model.safetensors").stat().st_size / 1024 / 1024
                print(f"   Size: {adapter_size:.1f} MB")
            elif (adapter_path / "adapter_model.bin").exists():
                print("\n📄 Found adapter_model.bin")
                adapter_size = (adapter_path / "adapter_model.bin").stat().st_size / 1024 / 1024
                print(f"   Size: {adapter_size:.1f} MB")
            else:
                print("\n⚠️  No adapter file found!")
                print("   Expected: adapter_model.safetensors or adapter_model.bin")
                return model, tokenizer
            
            logger.info(f"Loading LoRA adapter from {lora_path}")
            
            # Load adapter
            print("\n⏳ Loading adapter into model...")
            model = PeftModel.from_pretrained(model, lora_path)
            print("✅ Adapter loaded")
            
            # Merge and unload
            print("⏳ Merging adapter with base model...")
            model = model.merge_and_unload()  # Merge for inference
            print("✅ Adapter merged")
            
            # Verify LoRA was applied by checking model parameters
            print("\n🔬 Verifying LoRA application...")
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   Total model parameters: {total_params:,}")
            
            print("\n✅ LoRA adapter successfully loaded and merged!")
            print("="*50 + "\n")
            logger.info("LoRA adapter loaded and merged successfully")
        except Exception as e:
            print(f"\n❌ Failed to load LoRA adapter: {e}\n")
            logger.warning(f"Failed to load LoRA adapter: {e}")
    else:
        print("\n❌ LoRA adapter NOT loaded")
        if not lora_path:
            print("   Reason: No LoRA path specified")
        elif not Path(lora_path).exists():
            print(f"   Reason: Path does not exist: {lora_path}")
        elif not kwargs.get('load_lora', True):
            print("   Reason: load_lora flag is False")

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

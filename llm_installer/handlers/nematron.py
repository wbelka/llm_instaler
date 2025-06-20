"""
Handler for NVIDIA Nemotron models with special requirements
"""

from typing import List, Dict, Any
from .base import BaseHandler
import logging

logger = logging.getLogger(__name__)


class NematronHandler(BaseHandler):
    """Handler for NVIDIA Nemotron models"""
    
    @property
    def name(self) -> str:
        return "NematronHandler"
    
    def can_handle(self, model_info: Dict[str, Any]) -> bool:
        """Check if this is a Nemotron model"""
        model_id = model_info.get('model_id', '').lower()
        
        # Check model ID
        if 'nemotron' in model_id or 'nematron' in model_id:
            return True
        
        # Check config
        config = model_info.get('config', {})
        model_type = config.get('model_type', '').lower()
        if 'nemotron' in model_type:
            return True
        
        # Check architectures
        architectures = config.get('architectures', [])
        if any('nemotron' in arch.lower() for arch in architectures):
            return True
        
        return False
    
    def get_dependencies(self, model_info: Dict[str, Any]) -> List[str]:
        """Get NVIDIA-specific dependencies"""
        deps = [
            "torch>=2.1.0",  # Need recent torch for NVIDIA optimizations
            "transformers>=4.35.0",
            "accelerate>=0.25.0",
            "safetensors>=0.3.1",
            "nvidia-ml-py3",  # For GPU monitoring
            "pynvml",  # NVIDIA Management Library
        ]
        
        # TensorRT support for optimization
        deps.extend([
            "tensorrt>=8.6.0",
            "torch-tensorrt>=1.4.0",
        ])
        
        # FP8 support (if available)
        config = model_info.get('config', {})
        if config.get('torch_dtype') == 'float8_e4m3fn':
            deps.append("transformer-engine>=0.11.0")
        
        # Flash attention for efficiency
        deps.append("flash-attn>=2.3.0")
        
        # Triton for custom kernels
        deps.append("triton>=2.1.0")
        
        return deps
    
    def get_additional_files(self, model_info: Dict[str, Any]) -> List[str]:
        """Get additional optimization files"""
        files = []
        
        # Look for TensorRT engine files
        files.extend([
            "*.engine",  # TensorRT engine files
            "*.plan",    # TensorRT plan files
            "*.onnx",    # ONNX models for conversion
        ])
        
        return files
    
    def get_environment_vars(self, model_info: Dict[str, Any]) -> Dict[str, str]:
        """Get NVIDIA-specific environment variables"""
        env_vars = {
            # CUDA settings
            'CUDA_VISIBLE_DEVICES': '0',  # Default to first GPU
            'CUDA_LAUNCH_BLOCKING': '0',
            
            # TensorRT settings
            'TRT_LOGGER_LEVEL': '2',  # Warning level
            'TENSORRT_CACHE_DIR': './tensorrt_cache',
            
            # Memory settings
            'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
            
            # Optimization flags
            'TORCH_CUDA_ARCH_LIST': '7.0;7.5;8.0;8.6;8.9;9.0',  # Support various GPU architectures
            'CUDA_GRAPHS_ENABLE': '1',  # Enable CUDA graphs
            
            # Cache settings
            'TRANSFORMERS_CACHE': './cache',
            'HF_HOME': './cache',
        }
        
        return env_vars
    
    def get_optimization_options(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get NVIDIA-specific optimization options"""
        options = {
            'torch_dtype': 'float16',  # FP16 by default
            'device_map': 'cuda:0',    # Explicit GPU mapping
            'use_flash_attention_2': True,
            'use_cache': True,
            'pretraining_tp': 1,  # Tensor parallelism
        }
        
        # Check for FP8 support
        config = model_info.get('config', {})
        if config.get('torch_dtype') == 'float8_e4m3fn':
            options['torch_dtype'] = 'float8_e4m3fn'
            options['use_fp8'] = True
        
        # Enable TensorRT if large model
        if model_info.get('size_gb', 0) > 20:
            options['use_tensorrt'] = True
            options['tensorrt_precision'] = 'fp16'
        
        return options
    
    def post_install_setup(self, install_path: str, model_info: Dict[str, Any]) -> None:
        """Create optimization scripts"""
        import os
        
        # Create TensorRT optimization script
        optimize_script = f"""#!/bin/bash
# Optimize model with TensorRT

cd "$(dirname "$0")"
source .venv/bin/activate

echo "Optimizing model with TensorRT..."
python -c "
import torch
from transformers import AutoModelForCausalLM
import tensorrt as trt

# Load model
model = AutoModelForCausalLM.from_pretrained('./model', torch_dtype=torch.float16)

# TODO: Add TensorRT conversion logic
print('TensorRT optimization would be performed here')
"

echo "Optimization complete!"
"""
        
        script_path = os.path.join(install_path, 'scripts', 'optimize_tensorrt.sh')
        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        
        with open(script_path, 'w') as f:
            f.write(optimize_script)
        
        os.chmod(script_path, 0o755)
        logger.info(f"Created TensorRT optimization script: {script_path}")
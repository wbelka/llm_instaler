"""
General handler for standard transformers models
"""

from typing import List, Dict, Any
from .base import BaseHandler
import logging

logger = logging.getLogger(__name__)


class GeneralHandler(BaseHandler):
    """Universal handler for most transformers models"""
    
    @property
    def name(self) -> str:
        return "GeneralHandler"
    
    def can_handle(self, model_info: Dict[str, Any]) -> bool:
        """This handler can handle most standard models"""
        # This is the default handler
        return True
    
    def get_dependencies(self, model_info: Dict[str, Any]) -> List[str]:
        """Get standard dependencies"""
        deps = [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "accelerate>=0.20.0",
            "safetensors>=0.3.1",
        ]
        
        # Add tokenizers for fast tokenization
        deps.append("tokenizers>=0.13.0")
        
        # Check for quantization requirements
        config = model_info.get('config', {})
        if config.get('quantization_config'):
            quant_config = config['quantization_config']
            if quant_config.get('load_in_4bit') or quant_config.get('load_in_8bit'):
                deps.append("bitsandbytes>=0.41.0")
        
        # Check for specific model types
        model_type = config.get('model_type', '').lower()
        
        # Mamba models
        if model_type == 'mamba':
            deps.extend([
                "mamba-ssm>=1.0.0",
                "causal-conv1d>=1.0.0"
            ])
        
        # Flash attention support
        if model_info.get('size_gb', 0) > 10:  # Large models benefit from flash attention
            deps.append("flash-attn>=2.0.0")
        
        # Add scipy for some models
        if model_type in ['bert', 'roberta', 'deberta']:
            deps.append("scipy>=1.9.0")
        
        return deps
    
    def get_environment_vars(self, model_info: Dict[str, Any]) -> Dict[str, str]:
        """Get environment variables"""
        env_vars = {}
        
        # Set cache directory
        env_vars['TRANSFORMERS_CACHE'] = './cache'
        env_vars['HF_HOME'] = './cache'
        
        # Disable telemetry
        env_vars['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
        env_vars['DISABLE_TELEMETRY'] = 'YES'
        
        return env_vars
    
    def get_optimization_options(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimization options"""
        options = {
            'torch_dtype': 'auto',  # Let transformers decide
            'device_map': 'auto',   # Automatic device mapping
            'low_cpu_mem_usage': True,  # Reduce CPU memory usage
        }
        
        # For large models, enable additional optimizations
        if model_info.get('size_gb', 0) > 7:
            options['load_in_8bit'] = True
        
        return options
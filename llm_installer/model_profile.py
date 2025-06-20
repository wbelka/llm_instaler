"""
Model profile dataclass
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any


@dataclass
class ModelProfile:
    """Model profile containing detected information"""
    model_type: str  # transformer, diffusion, gguf, etc.
    model_id: str
    library: str  # transformers, diffusers, etc.
    architecture: Optional[str] = None
    task: Optional[str] = None  # text-generation, text2img, etc.
    quantization: Optional[str] = None  # 4bit, 8bit, gguf, etc.
    special_requirements: Optional[List[str]] = None  # mamba-ssm, tensorrt, etc.
    estimated_size_gb: float = 1.0
    estimated_memory_gb: float = 4.0
    supports_vllm: bool = False
    supports_tensorrt: bool = False
    is_multimodal: bool = False
    metadata: Optional[Dict[str, Any]] = None

    # Hardware requirements
    min_ram_gb: float = 8.0
    min_vram_gb: float = 0.0  # 0 means CPU is OK
    recommended_ram_gb: float = 16.0
    recommended_vram_gb: float = 0.0
    supports_cpu: bool = True
    supports_cuda: bool = True
    supports_metal: bool = True
    supports_quantization: Optional[List[str]] = None  # ['4bit', '8bit', 'fp16']

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'model_type': self.model_type,
            'model_id': self.model_id,
            'library': self.library,
            'architecture': self.architecture,
            'task': self.task,
            'quantization': self.quantization,
            'special_requirements': self.special_requirements or [],
            'estimated_size_gb': self.estimated_size_gb,
            'estimated_memory_gb': self.estimated_memory_gb,
            'supports_vllm': self.supports_vllm,
            'supports_tensorrt': self.supports_tensorrt,
            'is_multimodal': self.is_multimodal,
            'metadata': self.metadata or {},
            'min_ram_gb': self.min_ram_gb,
            'min_vram_gb': self.min_vram_gb,
            'recommended_ram_gb': self.recommended_ram_gb,
            'recommended_vram_gb': self.recommended_vram_gb,
            'supports_cpu': self.supports_cpu,
            'supports_cuda': self.supports_cuda,
            'supports_metal': self.supports_metal,
            'supports_quantization': self.supports_quantization or []
        }

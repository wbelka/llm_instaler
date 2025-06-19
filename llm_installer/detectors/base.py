"""
Base detector class for model type detection
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import logging


logger = logging.getLogger(__name__)


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


class BaseDetector(ABC):
    """Abstract base class for model detectors"""

    def __init__(self):
        self.name = self.__class__.__name__
        self.logger = logging.getLogger(self.name)

    @abstractmethod
    def detect(self, model_id: str, config: Dict[str, Any],
               files: List[str]) -> Optional[ModelProfile]:
        """
        Detect if this detector can handle the model

        Args:
            model_id: HuggingFace model ID
            config: Parsed config.json content
            files: List of files in the model repository

        Returns:
            ModelProfile if detected, None otherwise
        """
        pass

    def _check_config_field(self, config: Dict[str, Any], field: str,
                            expected_value: Any = None) -> bool:
        """
        Helper to check if a field exists in config with optional value check

        Args:
            config: Configuration dictionary
            field: Field name to check (supports dot notation)
            expected_value: If provided, check if field equals this value

        Returns:
            True if field exists (and matches value if provided)
        """
        # Support nested fields with dot notation
        fields = field.split('.')
        current = config

        for f in fields:
            if not isinstance(current, dict) or f not in current:
                return False
            current = current[f]

        if expected_value is not None:
            return current == expected_value

        return True

    def _check_architecture(self, config: Dict[str, Any],
                            architectures: List[str]) -> bool:
        """
        Check if model architecture matches any in the list

        Args:
            config: Configuration dictionary
            architectures: List of architecture names to check

        Returns:
            True if any architecture matches
        """
        model_arch = config.get('architectures', [])
        if isinstance(model_arch, list) and model_arch:
            model_arch = model_arch[0]
        elif not isinstance(model_arch, str):
            return False

        return any(arch.lower() in model_arch.lower() for arch in architectures)

    def _extract_task(self, config: Dict[str, Any]) -> Optional[str]:
        """Extract task type from config"""
        # Check direct task field
        if 'task' in config:
            return config['task']

        # Check in model card
        if 'model_type' in config:
            model_type = config['model_type']
            # Map common model types to tasks
            task_mapping = {
                'text-generation': 'text-generation',
                'causal-lm': 'text-generation',
                'seq2seq': 'text2text-generation',
                'text-classification': 'text-classification',
                'token-classification': 'token-classification',
                'question-answering': 'question-answering',
                'summarization': 'summarization',
                'translation': 'translation',
            }

            for key, task in task_mapping.items():
                if key in model_type.lower():
                    return task

        # Try to infer from architecture
        arch = config.get('architectures', [''])[0].lower()
        if 'causallm' in arch:
            return 'text-generation'
        elif 'seq2seq' in arch:
            return 'text2text-generation'
        elif 'classification' in arch:
            return 'text-classification'

        return None

    def _calculate_model_size(self, config: Dict[str, Any],
                              files: List[str], model_id: str = "") -> float:
        """
        Estimate model size based on config and files

        Returns:
            Estimated size in GB
        """
        # Try to get from config
        if 'model_size' in config:
            return float(config['model_size'])

        # Estimate from parameters
        hidden_size = config.get('hidden_size', 0)
        num_layers = config.get('num_hidden_layers', 0)
        vocab_size = config.get('vocab_size', 0)

        if hidden_size and num_layers:
            # Rough estimation: parameters * 4 bytes (fp32)
            # Transformer params ≈ 12 * layers * hidden^2 + vocab * hidden
            params_millions = (12 * num_layers * hidden_size * hidden_size +
                               vocab_size * hidden_size) / 1e6
            size_gb = params_millions * 4 / 1000  # MB to GB
            return round(size_gb, 1)

        # Default based on architecture
        arch = config.get('architectures', [''])[0].lower()
        model_id_lower = model_id.lower()
        if '7b' in arch or '7b' in model_id_lower:
            return 13.0
        elif '13b' in arch or '13b' in model_id_lower:
            return 26.0
        elif '70b' in arch or '70b' in model_id_lower:
            return 140.0
        elif '1b' in arch or '1b' in model_id_lower:
            return 2.0
        elif '3b' in arch or '3b' in model_id_lower:
            return 6.0

        # Check for GGUF files
        gguf_files = [f for f in files if f.endswith('.gguf')]
        if gguf_files:
            # GGUF models are usually quantized
            return 4.0  # Default for quantized models

        return 8.0  # Default size

    def _calculate_memory_requirements(self, model_size_gb: float,
                                       quantization: Optional[str] = None,
                                       is_training: bool = False) -> Dict[str, float]:
        """
        Calculate memory requirements for different scenarios

        Returns:
            Dict with min/recommended RAM/VRAM requirements
        """
        # Quantization multipliers
        quant_multipliers = {
            'fp32': 1.0,
            'fp16': 0.5,
            '8bit': 0.25,
            '4bit': 0.125,
            'gguf': 0.25  # Average for GGUF
        }

        multiplier = quant_multipliers.get(quantization, 1.0)
        quantized_size = model_size_gb * multiplier

        if is_training:
            # Training needs more memory
            if quantization in ['4bit', '8bit']:
                # QLoRA training
                min_ram = quantized_size + 4  # Model + LoRA overhead
                rec_ram = quantized_size + 8
            else:
                # Full training
                min_ram = model_size_gb * 3  # Model + gradients + optimizer
                rec_ram = model_size_gb * 4
        else:
            # Inference
            min_ram = quantized_size * 1.2  # 20% overhead
            rec_ram = quantized_size * 1.5  # 50% overhead

        return {
            'min_ram_gb': round(min_ram, 1),
            'min_vram_gb': round(min_ram * 0.8, 1),  # GPU is more efficient
            'recommended_ram_gb': round(rec_ram, 1),
            'recommended_vram_gb': round(rec_ram * 0.8, 1)
        }

    def _determine_quantization_support(self, config: Dict[str, Any],
                                        model_size_gb: float) -> List[str]:
        """
        Determine which quantization methods are supported
        """
        supported = ['fp32', 'fp16']

        # Most modern models support 8bit
        if model_size_gb > 1:
            supported.append('8bit')

        # Larger models benefit from 4bit
        if model_size_gb > 6:
            supported.append('4bit')

        # Check architecture compatibility
        arch = config.get('architectures', [''])[0].lower()
        if 'llama' in arch or 'mistral' in arch or 'qwen' in arch:
            # These architectures have good quantization support
            if '4bit' not in supported:
                supported.append('4bit')

        return supported

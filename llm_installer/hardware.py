"""
Hardware detection and resource calculation utilities
"""

import logging
import platform
import subprocess
import psutil
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class HardwareInfo:
    """Container for hardware information"""
    def __init__(self):
        self.system = platform.system()
        self.cpu_count = psutil.cpu_count(logical=False)
        self.cpu_count_logical = psutil.cpu_count(logical=True)
        self.total_ram_gb = psutil.virtual_memory().total / (1024**3)
        self.available_ram_gb = psutil.virtual_memory().available / (1024**3)

        # GPU info
        self.gpus: List[Dict[str, any]] = []
        self.cuda_available = False
        self.metal_available = False
        self.total_vram_gb = 0.0

        # Detect GPUs
        self._detect_gpus()

    def _detect_gpus(self):
        """Detect available GPUs"""
        # Try NVIDIA GPUs
        self._detect_nvidia_gpus()

        # Try AMD GPUs
        self._detect_amd_gpus()

        # Check for Apple Metal
        self._detect_metal()

    def _detect_nvidia_gpus(self):
        """Detect NVIDIA GPUs using nvidia-smi"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True
            )

            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        gpu_info = {
                            'name': parts[0],
                            'type': 'NVIDIA',
                            'total_memory_mb': float(parts[1]),
                            'free_memory_mb': float(parts[2]),
                            'total_memory_gb': float(parts[1]) / 1024,
                            'free_memory_gb': float(parts[2]) / 1024
                        }
                        self.gpus.append(gpu_info)
                        self.total_vram_gb += gpu_info['total_memory_gb']
                        self.cuda_available = True

        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.debug("nvidia-smi not found or failed")

    def _detect_amd_gpus(self):
        """Detect AMD GPUs using rocm-smi"""
        try:
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram"],
                capture_output=True,
                text=True,
                check=True
            )

            # Parse AMD GPU info
            # This is simplified - real parsing would be more complex
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                if 'GPU' in line and 'Total' in line:
                    # Extract memory info
                    pass  # TODO: Implement proper AMD GPU parsing

        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.debug("rocm-smi not found or failed")

    def _detect_metal(self):
        """Check for Apple Metal support"""
        if self.system == 'Darwin':
            # Check if we're on Apple Silicon
            try:
                result = subprocess.run(
                    ["sysctl", "-n", "hw.optional.arm64"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                if result.stdout.strip() == "1":
                    self.metal_available = True
                    # Get unified memory info
                    result = subprocess.run(
                        ["sysctl", "-n", "hw.memsize"],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    total_memory = int(result.stdout.strip()) / (1024**3)
                    self.gpus.append({
                        'name': 'Apple Silicon GPU',
                        'type': 'Metal',
                        'total_memory_gb': total_memory,
                        'free_memory_gb': self.available_ram_gb,  # Unified memory
                        'unified_memory': True
                    })
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.debug("Failed to detect Apple Silicon")

    def get_summary(self) -> Dict[str, any]:
        """Get hardware summary"""
        return {
            'system': self.system,
            'cpu': {
                'cores': self.cpu_count,
                'threads': self.cpu_count_logical
            },
            'memory': {
                'total_ram_gb': round(self.total_ram_gb, 1),
                'available_ram_gb': round(self.available_ram_gb, 1)
            },
            'gpu': {
                'cuda_available': self.cuda_available,
                'metal_available': self.metal_available,
                'gpus': self.gpus,
                'total_vram_gb': round(self.total_vram_gb, 1)
            }
        }

    def get_available_memory(self) -> Tuple[float, float]:
        """
        Get available memory for model loading
        Returns: (available_ram_gb, available_vram_gb)
        """
        # Leave some memory for system
        safe_ram = max(0, self.available_ram_gb - 2.0)

        # Calculate available VRAM
        available_vram = 0.0
        if self.gpus:
            for gpu in self.gpus:
                if gpu.get('unified_memory'):
                    # For unified memory (Apple Silicon), use RAM
                    available_vram = safe_ram
                else:
                    available_vram += gpu.get('free_memory_gb', 0)

        return safe_ram, available_vram


def calculate_model_requirements(
    model_size_gb: float,
    quantization: Optional[str] = None,
    task: str = "inference"
) -> Dict[str, float]:
    """
    Calculate memory requirements for model

    Args:
        model_size_gb: Base model size in GB
        quantization: Quantization type (4bit, 8bit, fp16, fp32)
        task: "inference" or "training"

    Returns:
        Dictionary with memory requirements
    """
    # Quantization multipliers
    quant_multipliers = {
        'fp32': 1.0,
        'fp16': 0.5,
        '8bit': 0.25,
        '4bit': 0.125
    }

    multiplier = quant_multipliers.get(quantization, 1.0)
    quantized_size = model_size_gb * multiplier

    if task == "inference":
        # For inference: model weights + activation memory
        memory_required = quantized_size * 1.2  # 20% overhead for activations

    else:  # training
        # For training: model + gradients + optimizer states + activations
        if quantization == '4bit':
            # QLoRA 4bit training
            memory_required = quantized_size + 2.0  # Quantized model + LoRA overhead
        elif quantization == '8bit':
            # QLoRA 8bit training - slightly more memory than 4bit
            memory_required = quantized_size + 2.5  # Quantized model + LoRA overhead
        else:
            # Full training needs ~4x model size
            memory_required = model_size_gb * 4

    return {
        'model_size_gb': model_size_gb,
        'quantized_size_gb': quantized_size,
        'memory_required_gb': memory_required,
        'recommended_ram_gb': memory_required + 4,  # Extra for system
        'quantization': quantization or 'fp32',
        'task': task
    }


def check_model_compatibility(
    model_requirements: Dict[str, float],
    hardware_info: HardwareInfo
) -> Dict[str, any]:
    """
    Check if model can run on given hardware

    Returns dictionary with:
        - can_run: bool
        - can_run_cpu: bool
        - can_run_gpu: bool
        - recommended_device: str
        - warnings: List[str]
    """
    required_memory = model_requirements['memory_required_gb']
    available_ram, available_vram = hardware_info.get_available_memory()

    result = {
        'can_run': False,
        'can_run_cpu': False,
        'can_run_gpu': False,
        'recommended_device': 'none',
        'warnings': []
    }

    # Check CPU compatibility
    if available_ram >= required_memory:
        result['can_run_cpu'] = True
        result['can_run'] = True
        result['recommended_device'] = 'cpu'
    elif available_ram >= required_memory * 0.7:
        result['can_run_cpu'] = True
        result['can_run'] = True
        result['warnings'].append(
            f"Low RAM: {available_ram:.1f}GB available, {required_memory:.1f}GB required. "
            "Performance may be degraded due to swapping."
        )

    # Check GPU compatibility
    if hardware_info.gpus and available_vram >= required_memory:
        result['can_run_gpu'] = True
        result['can_run'] = True
        result['recommended_device'] = 'cuda' if hardware_info.cuda_available else 'metal'

    # Add specific warnings
    if not result['can_run']:
        result['warnings'].append(
            f"Insufficient memory: {required_memory:.1f}GB required, "
            f"but only {available_ram:.1f}GB RAM and {available_vram:.1f}GB VRAM available."
        )

    if result['can_run_gpu'] and not result['can_run_cpu']:
        result['warnings'].append(
            "Model can only run on GPU. CPU memory insufficient."
        )

    return result


# Singleton instance
_hardware_info: Optional[HardwareInfo] = None


def get_hardware_info() -> HardwareInfo:
    """Get cached hardware information"""
    global _hardware_info
    if _hardware_info is None:
        _hardware_info = HardwareInfo()
    return _hardware_info

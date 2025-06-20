"""
Base detector with API-first approach
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging


logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Complete model information from API and configs"""
    # From API
    model_id: str
    library_name: Optional[str] = None
    pipeline_tag: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    downloads: int = 0
    likes: int = 0
    created_at: Optional[str] = None
    sha: Optional[str] = None
    private: bool = False
    gated: Optional[str] = None  # "auto", "manual", or None

    # From config files
    config: Dict[str, Any] = field(default_factory=dict)
    model_index: Optional[Dict[str, Any]] = None  # For diffusers

    # Calculated
    size_gb: float = 0.0
    files: List[str] = field(default_factory=list)
    file_sizes: Dict[str, float] = field(default_factory=dict)

    # Detected
    model_type: Optional[str] = None
    architecture: Optional[str] = None
    task: Optional[str] = None
    quantization: Optional[str] = None
    default_dtype: Optional[str] = None
    supports_quantization: List[str] = field(default_factory=list)
    special_requirements: List[str] = field(default_factory=list)
    is_multimodal: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ModelInfo to dictionary"""
        from dataclasses import asdict
        return asdict(self)


class BaseDetector(ABC):
    """Base class for model detectors using API-first approach"""

    def __init__(self):
        self.name = self.__class__.__name__
        self.logger = logging.getLogger(self.name)

    @abstractmethod
    def can_handle(self, info: ModelInfo) -> bool:
        """Check if this detector can handle the model based on API info"""
        pass

    @abstractmethod
    def detect(self, info: ModelInfo) -> ModelInfo:
        """
        Enrich model info with type-specific detection

        Args:
            info: ModelInfo with API data already populated

        Returns:
            Updated ModelInfo with detected fields filled
        """
        pass

    def _get_config_value(self, config: Dict[str, Any], path: str, default=None):
        """
        Get value from nested config using dot notation

        Example: _get_config_value(config, "language_config.torch_dtype")
        """
        keys = path.split('.')
        value = config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def _detect_quantization_from_files(self, files: List[str]) -> Optional[str]:
        """Detect quantization from file patterns"""
        # GGUF files
        if any(f.endswith('.gguf') for f in files):
            # Try to detect specific GGUF quantization
            for file in files:
                file_lower = file.lower()
                patterns = ['q2_k', 'q3_k', 'q4_k', 'q4_0', 'q4_1',
                           'q5_k', 'q5_0', 'q5_1', 'q6_k', 'q8_0']
                for pattern in patterns:
                    if pattern in file_lower:
                        return f'gguf-{pattern.upper()}'
            return 'gguf'

        # GPTQ files
        if any('gptq' in f.lower() for f in files):
            return 'gptq'

        # AWQ files
        if any('awq' in f.lower() for f in files):
            return 'awq'

        return None

    def _calculate_memory_requirements(self, info: ModelInfo) -> Dict[str, float]:
        """Calculate memory requirements based on model info"""
        if info.size_gb == 0:
            return {
                'min_ram_gb': 0.0,
                'min_vram_gb': 0.0,
                'recommended_ram_gb': 0.0,
                'recommended_vram_gb': 0.0,
                'estimated_memory_gb': 0.0
            }

        # Import memory calculation function
        from ..checker import calculate_memory_for_dtype

        # Use the reusable function for memory calculations
        dtype = info.default_dtype or 'fp32'
        estimated_memory = calculate_memory_for_dtype(info.size_gb, dtype)

        return {
            'min_ram_gb': info.size_gb,
            'min_vram_gb': info.size_gb * 0.8,  # GPU more efficient
            'recommended_ram_gb': estimated_memory * 1.25,  # 25% extra
            'recommended_vram_gb': estimated_memory,
            'estimated_memory_gb': estimated_memory
        }

"""
Detector for GGUF quantized models
"""

from typing import Dict, List, Optional, Any
from .base import BaseDetector, ModelProfile


class GGUFDetector(BaseDetector):
    """Detector for GGUF (GPT-Generated Unified Format) quantized models"""

    def detect(self, model_id: str, config: Dict[str, Any],
               files: List[str]) -> Optional[ModelProfile]:
        """
        Detect GGUF quantized models
        """
        # Check for .gguf files
        gguf_files = [f for f in files if f.endswith('.gguf')]

        if not gguf_files:
            return None

        self.logger.info(f"Found {len(gguf_files)} GGUF files")

        # Create profile for GGUF model
        profile = ModelProfile(
            model_type="gguf",
            model_id=model_id,
            library="llama-cpp-python",  # Primary library for GGUF
            architecture=self._detect_gguf_architecture(gguf_files, config),
            task="text-generation",  # Most GGUF models are for text generation
            quantization="gguf"
        )

        # Analyze GGUF files for quantization details
        quant_info = self._analyze_gguf_quantization(gguf_files)
        if quant_info:
            profile.quantization = f"gguf-{quant_info}"
            profile.metadata = {'gguf_files': gguf_files, 'quantization_type': quant_info}

        # Get size from API if available
        from ..utils import estimate_size_from_files
        file_sizes = config.get('_file_sizes', {})
        file_size = estimate_size_from_files(files, file_sizes)

        if file_size > 0:
            profile.estimated_size_gb = file_size
            profile.estimated_memory_gb = file_size * 1.5  # Less overhead for GGUF
        else:
            # No size info available
            profile.estimated_size_gb = 0.0
            profile.estimated_memory_gb = 0.0

        # Set requirements for GGUF
        profile.special_requirements = [
            'llama-cpp-python',
            'numpy',
            'typing-extensions'
        ]

        # GGUF models don't typically support these
        profile.supports_vllm = False
        profile.supports_tensorrt = False

        return profile

    def _detect_gguf_architecture(self, gguf_files: List[str],
                                  config: Dict[str, Any]) -> Optional[str]:
        """Try to detect the architecture from GGUF filename patterns"""
        # Common patterns in GGUF filenames
        arch_patterns = {
            'llama': 'LLaMA',
            'mistral': 'Mistral',
            'mixtral': 'Mixtral',
            'qwen': 'Qwen',
            'phi': 'Phi',
            'gemma': 'Gemma',
            'yi': 'Yi',
            'deepseek': 'DeepSeek',
            'chatglm': 'ChatGLM',
            'baichuan': 'Baichuan',
            'internlm': 'InternLM',
            'orion': 'Orion',
            'aquila': 'Aquila',
            'bloom': 'BLOOM',
            'falcon': 'Falcon',
            'gpt2': 'GPT-2',
            'opt': 'OPT',
            'pythia': 'Pythia',
            'stablelm': 'StableLM',
            'vicuna': 'Vicuna',
            'alpaca': 'Alpaca',
            'wizard': 'Wizard',
            'openchat': 'OpenChat'
        }

        # Check filenames
        for file in gguf_files:
            file_lower = file.lower()
            for pattern, arch in arch_patterns.items():
                if pattern in file_lower:
                    return f"{arch}-GGUF"

        # Fallback to config if available
        if config and 'model_type' in config:
            return f"{config['model_type']}-GGUF"

        return "Unknown-GGUF"

    def _analyze_gguf_quantization(self, gguf_files: List[str]) -> Optional[str]:
        """Analyze GGUF files to determine quantization type"""
        # Common GGUF quantization patterns
        quant_patterns = [
            'q2_k', 'q3_k_s', 'q3_k_m', 'q3_k_l',
            'q4_0', 'q4_1', 'q4_k_s', 'q4_k_m',
            'q5_0', 'q5_1', 'q5_k_s', 'q5_k_m',
            'q6_k', 'q8_0', 'f16', 'f32'
        ]

        for file in gguf_files:
            file_lower = file.lower()
            for pattern in quant_patterns:
                if pattern in file_lower:
                    return pattern.upper()

        # Default if no pattern found
        return "Q4_K_M"  # Common default

    def _estimate_gguf_size(self, gguf_files: List[str]) -> float:
        """Cannot estimate GGUF size without API data"""
        return 0.0  # Unknown

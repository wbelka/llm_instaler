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

        # Estimate size based on GGUF files
        # GGUF models are already quantized, so they're smaller
        profile.estimated_size_gb = self._estimate_gguf_size(gguf_files)
        profile.estimated_memory_gb = profile.estimated_size_gb * 1.5  # Less overhead for GGUF

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
        """Estimate total size of GGUF model"""
        # Very rough estimation based on filename patterns
        # In practice, would need to check actual file sizes

        total_size_gb = 0.0

        for file in gguf_files:
            file_lower = file.lower()

            # Estimate based on model size indicators
            if '70b' in file_lower or '65b' in file_lower:
                size = 35.0  # ~35GB for Q4 quantized 70B
            elif '30b' in file_lower or '33b' in file_lower or '34b' in file_lower:
                size = 20.0  # ~20GB for Q4 quantized 30B
            elif '13b' in file_lower:
                size = 8.0   # ~8GB for Q4 quantized 13B
            elif '7b' in file_lower or '8b' in file_lower:
                size = 4.5   # ~4.5GB for Q4 quantized 7B
            elif '3b' in file_lower:
                size = 2.0   # ~2GB for Q4 quantized 3B
            else:
                size = 4.0   # Default assumption

            # Adjust for quantization level
            if 'q2' in file_lower:
                size *= 0.5
            elif 'q3' in file_lower:
                size *= 0.75
            elif 'q5' in file_lower:
                size *= 1.25
            elif 'q6' in file_lower:
                size *= 1.5
            elif 'q8' in file_lower:
                size *= 2.0
            elif 'f16' in file_lower:
                size *= 4.0
            elif 'f32' in file_lower:
                size *= 8.0

            total_size_gb += size

        return round(total_size_gb, 1)

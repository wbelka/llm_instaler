"""
Detector for GGUF quantized models
"""

from .base import BaseDetector, ModelInfo


class GGUFDetector(BaseDetector):
    """Detector for GGUF quantized models"""

    def can_handle(self, info: ModelInfo) -> bool:
        """Check if this is a GGUF model"""
        # Has GGUF files
        if any(f.endswith('.gguf') for f in info.files):
            return True

        # Has GGUF tag
        if 'gguf' in info.tags:
            return True

        # Library name (some models explicitly set this)
        if info.library_name == 'gguf':
            return True

        return False

    def detect(self, info: ModelInfo) -> ModelInfo:
        """Detect GGUF-specific information"""
        info.model_type = 'gguf'
        info.task = 'text-generation'  # Most GGUF models are for text gen

        # Find all GGUF files
        gguf_files = [f for f in info.files if f.endswith('.gguf')]
        info.metadata['gguf_files'] = gguf_files

        # Detect quantization type from filename
        quant_info = self._analyze_gguf_files(gguf_files)
        if quant_info:
            info.quantization = f"gguf-{quant_info['type']}"
            info.metadata['quantization_details'] = quant_info
        else:
            info.quantization = 'gguf'

        # Try to detect base model architecture
        info.architecture = self._detect_architecture(info)

        # Requirements - minimal for GGUF
        info.special_requirements = [
            'llama-cpp-python',
            'numpy'
        ]

        # GGUF has its own quantization, not supporting others
        info.supports_quantization = ['gguf']

        # No default dtype for GGUF - it's already quantized
        info.default_dtype = None

        # Add file size info for each variant
        if len(gguf_files) > 1:
            variants = []
            for file in gguf_files:
                size = info.file_sizes.get(file, 0)
                quant = self._extract_quantization_from_filename(file)
                variants.append({
                    'file': file,
                    'size_gb': round(size, 2),
                    'quantization': quant
                })
            info.metadata['variants'] = sorted(variants, key=lambda x: x['size_gb'])
        
        # GGUF models don't support vLLM or TensorRT
        info.metadata['supports_vllm'] = False
        info.metadata['supports_tensorrt'] = False

        return info

    def _analyze_gguf_files(self, gguf_files: list) -> dict:
        """Analyze GGUF files to determine quantization details"""
        if not gguf_files:
            return None

        # Take first file as representative
        file = gguf_files[0]
        quant_type = self._extract_quantization_from_filename(file)

        if quant_type:
            # Parse quantization details
            details = {
                'type': quant_type.upper(),
                'bits': self._get_bits_from_quant(quant_type)
            }

            # Check if it's a K-quant
            if '_k' in quant_type.lower():
                details['k_quant'] = True

            return details

        return None

    def _extract_quantization_from_filename(self, filename: str) -> str:
        """Extract quantization type from filename"""
        filename_lower = filename.lower()

        # Common GGUF quantization patterns
        patterns = [
            'q2_k', 'q3_k_s', 'q3_k_m', 'q3_k_l',
            'q4_0', 'q4_1', 'q4_k_s', 'q4_k_m',
            'q5_0', 'q5_1', 'q5_k_s', 'q5_k_m',
            'q6_k', 'q8_0', 'f16', 'f32'
        ]

        for pattern in patterns:
            if pattern in filename_lower:
                return pattern

        return 'unknown'

    def _get_bits_from_quant(self, quant_type: str) -> int:
        """Get approximate bits from quantization type"""
        if 'q2' in quant_type:
            return 2
        elif 'q3' in quant_type:
            return 3
        elif 'q4' in quant_type:
            return 4
        elif 'q5' in quant_type:
            return 5
        elif 'q6' in quant_type:
            return 6
        elif 'q8' in quant_type:
            return 8
        elif 'f16' in quant_type:
            return 16
        elif 'f32' in quant_type:
            return 32
        else:
            return 4  # Default assumption

    def _detect_architecture(self, info: ModelInfo) -> str:
        """Try to detect base architecture from model name or tags"""
        model_lower = info.model_id.lower()

        # Check tags first
        for tag in info.tags:
            tag_lower = tag.lower()
            # Common architecture tags
            arch_names = ['llama', 'mistral', 'mixtral', 'qwen', 'yi', 'gemma',
                         'phi', 'deepseek', 'falcon', 'starcoder', 'codellama']
            for arch in arch_names:
                if arch in tag_lower:
                    return arch.title()

        # Check model name
        if 'llama' in model_lower:
            return 'LLaMA'
        elif 'mistral' in model_lower:
            return 'Mistral'
        elif 'mixtral' in model_lower:
            return 'Mixtral'
        elif 'qwen' in model_lower:
            return 'Qwen'
        elif 'phi' in model_lower:
            return 'Phi'
        elif 'gemma' in model_lower:
            return 'Gemma'

        # Check base_model tags
        for tag in info.tags:
            if tag.startswith('base_model:'):
                base = tag.split(':', 1)[1]
                # Extract architecture from base model
                for arch in ['llama', 'mistral', 'qwen', 'phi', 'gemma']:
                    if arch in base.lower():
                        return arch.title()

        return 'Unknown-GGUF'

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

        # For GGUF models, each file is a complete model variant
        # We need to handle size differently
        if gguf_files and info.file_sizes:
            # Get sizes of individual GGUF files
            gguf_sizes = [info.file_sizes.get(f, 0) for f in gguf_files]
            
            # Find the default/recommended variant
            # Usually Q4_0 or Q4_K_M is a good default
            default_file = None
            for pattern in ['q4_0', 'q4_k_m', 'q4_k', 'q5_0']:
                for file in gguf_files:
                    if pattern in file.lower():
                        default_file = file
                        break
                if default_file:
                    break
            
            # If no default found, use the median-sized file
            if not default_file and gguf_sizes:
                sorted_files = sorted(
                    zip(gguf_files, gguf_sizes), 
                    key=lambda x: x[1]
                )
                median_idx = len(sorted_files) // 2
                default_file = sorted_files[median_idx][0]
            
            # Set the size to the default variant size, not sum of all
            if default_file:
                info.size_gb = info.file_sizes.get(default_file, 0)
                info.metadata['default_variant'] = default_file
                info.metadata['total_variants_size'] = round(sum(gguf_sizes), 1)

        # Detect quantization type from default file or first file
        default_file = info.metadata.get('default_variant', gguf_files[0] if gguf_files else None)
        if default_file:
            quant_type = self._extract_quantization_from_filename(default_file)
            info.quantization = f"gguf-{quant_type.upper()}"
            
            # Parse quantization details
            info.metadata['quantization_details'] = {
                'type': quant_type.upper(),
                'bits': self._get_bits_from_quant(quant_type),
                'k_quant': '_k' in quant_type.lower()
            }
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
                    'quantization': quant.upper(),
                    'memory_required_gb': round(size * 1.2, 1)  # GGUF needs ~20% overhead
                })
            info.metadata['variants'] = sorted(variants, key=lambda x: x['size_gb'])
            
            # Add a note about variants
            info.metadata['variant_note'] = (
                "Each GGUF file is a complete model. Choose based on your "
                "memory constraints and quality requirements."
            )

        # GGUF models don't support vLLM or TensorRT
        info.metadata['supports_vllm'] = False
        info.metadata['supports_tensorrt'] = False

        return info

    def _analyze_gguf_files(self, gguf_files: list) -> dict:
        """Analyze GGUF files to determine quantization details"""
        # This method is now integrated into detect()
        # Kept for backwards compatibility
        if not gguf_files:
            return None

        file = gguf_files[0]
        quant_type = self._extract_quantization_from_filename(file)

        if quant_type:
            return {
                'type': quant_type.upper(),
                'bits': self._get_bits_from_quant(quant_type),
                'k_quant': '_k' in quant_type.lower()
            }

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
        elif 'cosmos' in model_lower:
            return 'Cosmos'

        # Check base_model tags
        for tag in info.tags:
            if tag.startswith('base_model:'):
                base = tag.split(':', 1)[1]
                # Extract architecture from base model
                for arch in ['llama', 'mistral', 'qwen', 'phi', 'gemma']:
                    if arch in base.lower():
                        return arch.title()

        return 'Unknown-GGUF'

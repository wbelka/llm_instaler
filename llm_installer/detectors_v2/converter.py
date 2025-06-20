"""
Converter from ModelInfo to ModelProfile
"""

from typing import Optional
from .base import ModelInfo
from ..model_profile import ModelProfile


def convert_to_profile(info: Optional[ModelInfo]) -> Optional[ModelProfile]:
    """Convert ModelInfo to ModelProfile for backwards compatibility"""
    if not info:
        return None

    # Import memory calculation function
    from ..checker import calculate_memory_for_dtype

    # Calculate hardware requirements
    hw_reqs = info.metadata.get('hardware_requirements', {})

    profile = ModelProfile(
        model_type=info.model_type or 'unknown',
        model_id=info.model_id,
        library=info.library_name or 'transformers',
        architecture=info.architecture,
        task=info.task,
        quantization=info.quantization,
        special_requirements=info.special_requirements,
        estimated_size_gb=info.size_gb,
        estimated_memory_gb=hw_reqs.get(
            'estimated_memory_gb',
            calculate_memory_for_dtype(
                info.size_gb,
                info.default_dtype or 'fp32'
            )
        ),
        supports_vllm=info.metadata.get('supports_vllm', False),
        supports_tensorrt=info.metadata.get('supports_tensorrt', False),
        is_multimodal=info.is_multimodal,
        metadata=info.metadata,
        min_ram_gb=hw_reqs.get('min_ram_gb', info.size_gb),
        min_vram_gb=hw_reqs.get('min_vram_gb', info.size_gb * 0.8),
        recommended_ram_gb=hw_reqs.get(
            'recommended_ram_gb',
            calculate_memory_for_dtype(
                info.size_gb,
                info.default_dtype or 'fp32'
            ) * 1.25
        ),
        recommended_vram_gb=hw_reqs.get(
            'recommended_vram_gb',
            calculate_memory_for_dtype(
                info.size_gb,
                info.default_dtype or 'fp32'
            )
        ),
        supports_cpu=True,
        supports_cuda=True,
        supports_metal=True,
        supports_quantization=info.supports_quantization
    )

    # Add default dtype to metadata if available
    if info.default_dtype:
        profile.metadata['torch_dtype'] = info.default_dtype

    # Add file info to metadata
    if info.files:
        profile.metadata['file_count'] = len(info.files)

    # Add download/likes info
    if info.downloads > 0:
        profile.metadata['downloads'] = info.downloads
    if info.likes > 0:
        profile.metadata['likes'] = info.likes

    # Add gated info
    if info.gated:
        profile.metadata['gated'] = info.gated

    return profile

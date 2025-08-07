"""Common dependencies for different model types.

This module centralizes dependency definitions to avoid duplication
across handlers.
"""

from typing import List, Dict

# Base dependencies for different model families
BASE_DEPENDENCIES = {
    'torch': [
        'torch>=2.0.0',
        'accelerate>=0.25.0',
    ],
    'transformers': [
        'transformers>=4.30.0',
    ],
    'diffusers': [
        'diffusers>=0.25.0',
    ],
    'audio': [
        'torchaudio>=2.0.0',
        'librosa>=0.10.0',
        'soundfile>=0.12.0',
    ],
    'vision': [
        'torchvision>=0.15.0',
        'opencv-python>=4.8.0',
        'pillow>=10.0.0',
    ],
    'quantization': [
        'bitsandbytes>=0.41.0',
    ],
}

# Model-specific dependencies
MODEL_SPECIFIC_DEPENDENCIES = {
    'mamba': [
        'mamba-ssm>=1.0.0',
        'causal-conv1d>=1.0.0',
    ],
    'rwkv': [
        'rwkv>=0.8.0',
    ],
    'flash-attn': [
        'flash-attn>=2.0.0',
    ],
    'xformers': [
        'xformers>=0.0.20',
    ],
    'deepspeed': [
        'deepspeed>=0.12.0',
    ],
    'sentence-transformers': [
        'sentence-transformers>=2.2.0',
    ],
}


def get_base_dependencies(*families: str) -> List[str]:
    """Get combined base dependencies for specified families.
    
    Args:
        *families: Variable number of family names (e.g., 'torch', 'transformers')
        
    Returns:
        List of unique dependencies
    """
    deps = []
    seen = set()
    
    for family in families:
        if family in BASE_DEPENDENCIES:
            for dep in BASE_DEPENDENCIES[family]:
                # Extract package name for deduplication
                pkg_name = dep.split('>=')[0].split('==')[0].split('<')[0].split('>')[0]
                if pkg_name not in seen:
                    seen.add(pkg_name)
                    deps.append(dep)
    
    return deps


def get_model_specific_dependencies(*features: str) -> List[str]:
    """Get model-specific dependencies for requested features.
    
    Args:
        *features: Variable number of feature names (e.g., 'mamba', 'flash-attn')
        
    Returns:
        List of unique dependencies
    """
    deps = []
    seen = set()
    
    for feature in features:
        if feature in MODEL_SPECIFIC_DEPENDENCIES:
            for dep in MODEL_SPECIFIC_DEPENDENCIES[feature]:
                pkg_name = dep.split('>=')[0].split('==')[0].split('<')[0].split('>')[0]
                if pkg_name not in seen:
                    seen.add(pkg_name)
                    deps.append(dep)
    
    return deps


def combine_dependencies(base: List[str], additional: List[str]) -> List[str]:
    """Combine dependency lists, avoiding duplicates.
    
    Args:
        base: Base dependency list
        additional: Additional dependencies to add
        
    Returns:
        Combined list with no duplicates, preserving order
    """
    seen = {}
    result = []
    
    # Process all dependencies
    for dep in base + additional:
        pkg_name = dep.split('>=')[0].split('==')[0].split('<')[0].split('>')[0]
        
        if pkg_name not in seen:
            seen[pkg_name] = dep
            result.append(dep)
        else:
            # If we see the same package again, keep the one with higher version requirement
            existing = seen[pkg_name]
            if '>' in dep and '>' in existing:
                # Simple comparison - in production, use packaging.version
                if dep > existing:
                    result[result.index(existing)] = dep
                    seen[pkg_name] = dep
    
    return result
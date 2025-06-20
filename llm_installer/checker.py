"""
Model compatibility checker module
"""

import logging
from typing import Optional

from .utils import (
    fetch_model_config,
    fetch_model_files_list,
    fetch_model_files_with_sizes,
    estimate_model_size,
    save_json_config
)
from .model_profile import ModelProfile
from .hardware import (
    get_hardware_info,
    calculate_model_requirements,
    check_model_compatibility
)
from .model_detector_v2 import ModelDetectorV2
from .detectors_v2.converter import convert_to_profile


logger = logging.getLogger(__name__)


def calculate_memory_for_dtype(
    disk_size_gb: float, dtype: str, task: str = "inference"
) -> float:
    """
    Calculate memory requirement for a specific dtype
    Args:
        disk_size_gb: Model size on disk
        dtype: Target dtype (float32, float16, bfloat16, etc)
        task: "inference" or "training"
    Returns:
        Memory requirement in GB
    """
    # Map torch dtypes to our quantization names
    dtype_map = {
        'float32': 'fp32',
        'float16': 'fp16',
        'bfloat16': 'fp16',
        'int8': '8bit',
        'int4': '4bit',
        '4bit': '4bit',
        '8bit': '8bit',
        'fp16': 'fp16',
        'fp32': 'fp32'
    }
    # Special handling for GGUF quantizations
    if dtype and dtype.startswith('gguf-'):
        # GGUF models only need ~20% overhead for context
        return disk_size_gb * 1.2

    quant_type = dtype_map.get(dtype, 'fp32')
    # If model is stored in half precision format, adjust calculation
    if dtype in ['float16', 'bfloat16']:
        # Model is already in this format on disk
        if task == "inference":
            return disk_size_gb * 1.2  # Just add overhead
        else:
            return disk_size_gb * 1.3  # Training overhead for LoRA
    else:
        # Use standard calculation
        mem_reqs = calculate_model_requirements(
            disk_size_gb,
            quant_type,
            task
        )
        return mem_reqs['memory_required_gb']


class ModelChecker:
    """Main class for checking model compatibility"""

    def __init__(self):
        self.detector = ModelDetectorV2()
        logger.debug("Initialized ModelDetectorV2")

    def check_model(self, model_id: str, save_profile: bool = False) -> Optional[ModelProfile]:
        """
        Check model compatibility and return profile

        Args:
            model_id: HuggingFace model ID
            save_profile: Whether to save the profile to disk

        Returns:
            ModelProfile if successful, None otherwise
        """
        logger.info(f"Checking model: {model_id}")

        # Fetch model metadata
        logger.debug("Fetching model configuration...")
        config = fetch_model_config(model_id, "config.json")

        # Try alternative config files
        if not config:
            logger.debug("No config.json found, trying llm_config.json...")
            config = fetch_model_config(model_id, "llm_config.json")

        # Try model_index.json for diffusion models
        if not config:
            logger.debug("No llm_config.json found, trying model_index.json...")
            config = fetch_model_config(model_id, "model_index.json")

        # Get list of files
        logger.debug("Fetching file list...")
        files = fetch_model_files_list(model_id)

        if not files:
            logger.error(f"Could not fetch file list for {model_id}")
            logger.info("\nPossible solutions:")
            logger.info("1. If this is a gated model, provide a HuggingFace token:")
            logger.info("   llm-installer check <model> --token YOUR_HF_TOKEN")
            logger.info("2. Save token for future use:")
            logger.info("   llm-installer check <model> --token YOUR_HF_TOKEN --save-token")
            logger.info("3. Set environment variable:")
            logger.info("   export HF_TOKEN=YOUR_HF_TOKEN")
            logger.info("\nGet your token from: https://huggingface.co/settings/tokens")
            return None

        # Try to get file sizes if possible
        file_sizes = fetch_model_files_with_sizes(model_id)
        if file_sizes:
            logger.debug(f"Got file sizes for {len(file_sizes)} files")
            # Update estimate_size_from_files to use actual sizes
            files_with_sizes = list(file_sizes.keys())
            if files_with_sizes:
                files = files_with_sizes

        logger.debug(f"Found {len(files)} files in repository")

        # Use new detector v2
        logger.debug("Running model detection v2...")
        model_info = self.detector.detect(model_id)
        
        if not model_info:
            logger.warning(f"Could not detect model {model_id}")
            # Create a generic profile as fallback
            profile = ModelProfile(
                model_type="unknown",
                model_id=model_id,
                library="transformers",  # Safe default
                task="text-generation"
            )
        else:
            # Convert ModelInfo to ModelProfile
            profile = convert_to_profile(model_info)
            logger.info(f"Model detected: {profile.model_type} ({profile.library})")

        # Size should already be set by detector v2
        # Only fallback if somehow missing
        if profile.estimated_size_gb == 0:
            logger.warning("Size not determined by detector, estimating...")
            size_info = estimate_model_size(config or {}, files)
            profile.estimated_size_gb = size_info['size_gb']

        # Save profile if requested
        if save_profile:
            self._save_profile(model_id, profile)

        return profile

    def _save_profile(self, model_id: str, profile: ModelProfile):
        """Save model profile to cache"""
        from .utils import get_cache_dir, sanitize_model_name

        cache_dir = get_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)

        profile_file = cache_dir / f"{sanitize_model_name(model_id)}_profile.json"
        save_json_config(profile.to_dict(), profile_file)
        logger.debug(f"Saved profile to {profile_file}")

    def print_compatibility_report(self, profile: ModelProfile):
        """Print a formatted compatibility report"""
        print("\n" + "="*60)
        print(f"Model Compatibility Report: {profile.model_id}")
        print("="*60)

        print(f"\nModel Type: {profile.model_type}")
        print(f"Library: {profile.library}")

        if profile.architecture:
            print(f"Architecture: {profile.architecture}")

        if profile.task:
            print(f"Task: {profile.task}")

        if profile.quantization:
            print(f"Quantization: {profile.quantization}")

        print("\nStorage Requirements:")
        # Special handling for GGUF models with variants
        if profile.model_type == 'gguf' and profile.metadata.get('variants'):
            default_variant = profile.metadata.get('default_variant', '')
            print(f"  - Default variant ({default_variant}): {profile.estimated_size_gb} GB")
            if profile.metadata.get('total_variants_size'):
                print(f"  - All variants total: {profile.metadata['total_variants_size']} GB")
            print("  - Virtual environment: ~2 GB")
            print(f"  - Disk space for default: ~{profile.estimated_size_gb + 2:.1f} GB")
        elif profile.estimated_size_gb > 0:
            print(f"  - Model files: {profile.estimated_size_gb} GB")
            print("  - Virtual environment: ~2 GB")
            print(f"  - Total disk space needed: ~{profile.estimated_size_gb + 2:.1f} GB")
        else:
            print("  - Model files: Unknown (API did not provide size information)")
            print("  - Virtual environment: ~2 GB")
            print(f"  - Check model page: https://huggingface.co/{profile.model_id}")

        print("\nMemory Requirements (RAM/VRAM):")

        # Check for torch_dtype in profile metadata
        torch_dtype = None
        if profile.metadata and 'torch_dtype' in profile.metadata:
            torch_dtype = profile.metadata['torch_dtype']
        elif profile.quantization:
            # Use quantization if specified
            torch_dtype = profile.quantization

        if torch_dtype:
            memory_req = calculate_memory_for_dtype(profile.estimated_size_gb, torch_dtype)
            print(f"  - Default configuration ({torch_dtype}): {memory_req:.1f} GB")
            print("  - Other configurations: See compatibility table below")
        else:
            # Show default configuration from metadata
            if profile.metadata and 'torch_dtype' in profile.metadata:
                torch_dtype = profile.metadata['torch_dtype']
                # Map torch dtypes to our quantization names
                dtype_map = {
                    'float32': 'fp32',
                    'float16': 'fp16',
                    'bfloat16': 'fp16',
                    'int8': '8bit',
                    'int4': '4bit'
                }
                quant_type = dtype_map.get(torch_dtype, 'fp32')
                
                # If model is stored in the same dtype as default, use disk size
                if torch_dtype in ['float16', 'bfloat16']:
                    # Model is already in this format on disk
                    memory_req = profile.estimated_size_gb * 1.2  # Just add overhead
                else:
                    # Calculate memory for default dtype
                    mem_reqs = calculate_model_requirements(
                        profile.estimated_size_gb,
                        quant_type,
                        "inference"
                    )
                    memory_req = mem_reqs['memory_required_gb']
                
                print(f"  - Default configuration: {torch_dtype}")
                print(f"  - Default memory requirement: {memory_req:.1f} GB")
            print(f"  - Model page: https://huggingface.co/{profile.model_id}")
            print("  - See compatibility table below for all configurations")

        # Show modalities and components for multimodal models
        if profile.is_multimodal and profile.metadata:
            if 'modalities' in profile.metadata:
                print("\nSupported Modalities:")
                modality_names = {
                    'text': 'Text Generation',
                    'vision': 'Vision',
                    'vision-understanding': 'Image Understanding',
                    'image-generation': 'Image Generation',
                    'audio': 'Audio',
                    'speech': 'Speech'
                }
                for modality in profile.metadata['modalities']:
                    display_name = modality_names.get(modality, modality.capitalize())
                    print(f"  - {display_name}")

            if 'capabilities' in profile.metadata:
                print("\nModel Capabilities:")
                for capability in profile.metadata['capabilities']:
                    cap_display = capability.replace('-', ' ').title()
                    print(f"  - {cap_display}")

            if 'component_sizes' in profile.metadata:
                print("\nComponent Sizes:")
                components = profile.metadata['component_sizes']
                for comp_name, comp_size in components.items():
                    print(f"  - {comp_name.upper()}: {comp_size} GB")
                component_total = sum(components.values())
                print(f"  - Total: {component_total:.1f} GB")

                # Show if there's a discrepancy
                if abs(component_total - profile.estimated_size_gb) > 0.5:
                    print(f"  - Note: Model files total is "
                          f"{profile.estimated_size_gb} GB")

                print("\nDeployment Options:")
                print("  - Minimum (mixed CPU/GPU):")
                largest_comp = max(components.values())
                print(f"    • GPU: {largest_comp:.1f} GB (largest component)")
                # Use actual memory requirement, not just file sizes
                memory_req = calculate_memory_for_dtype(
                    profile.estimated_size_gb,
                    profile.metadata.get('torch_dtype', 'fp32')
                )
                print(f"    • RAM: {memory_req:.1f} GB (with overhead)")
                print("  - Optimal (all GPU):")
                print(f"    • GPU: {memory_req:.1f} GB "
                      f"(all components + overhead)")

        if profile.special_requirements:
            print("\nSpecial Requirements:")
            for req in profile.special_requirements:
                print(f"  - {req}")
                
        # Show GGUF variants if available
        if profile.model_type == 'gguf' and profile.metadata.get('variants'):
            print("\nAvailable GGUF Variants:")
            if profile.metadata.get('variant_note'):
                print(f"  Note: {profile.metadata['variant_note']}")
            
            # Group variants by quantization type
            variants_by_quant = {}
            for variant in profile.metadata['variants']:
                quant = variant['quantization']
                if quant not in variants_by_quant:
                    variants_by_quant[quant] = []
                variants_by_quant[quant].append(variant)
            
            # Sort quantization types by quality (reverse order)
            quality_order = ['F32', 'F16', 'Q8_0', 'Q6_K', 'Q5_K_M', 'Q5_K_S', 
                           'Q5_1', 'Q5_0', 'Q4_K_M', 'Q4_K_S', 'Q4_1', 'Q4_0', 
                           'Q3_K_L', 'Q3_K_M', 'Q3_K_S', 'Q2_K', 'UNKNOWN']
            sorted_quants = sorted(variants_by_quant.keys(), 
                                 key=lambda x: quality_order.index(x) 
                                 if x in quality_order else len(quality_order))
            
            print("\n  Quantization   Files  Size Range    Memory Range  Quality")
            print("  " + "-" * 70)
            
            # Quality indicators
            quality_map = {
                'F32': '★★★★★ (Full)',
                'F16': '★★★★★ (Half)',
                'Q8_0': '★★★★★',
                'Q6_K': '★★★★☆',
                'Q5_K_M': '★★★★☆',
                'Q5_K_S': '★★★★☆',
                'Q5_0': '★★★★☆',
                'Q5_1': '★★★★☆',
                'Q4_K_M': '★★★☆☆',
                'Q4_K_S': '★★★☆☆',
                'Q4_0': '★★★☆☆ (Recommended)',
                'Q4_1': '★★★☆☆',
                'Q3_K_L': '★★☆☆☆',
                'Q3_K_M': '★★☆☆☆',
                'Q3_K_S': '★★☆☆☆',
                'Q2_K': '★☆☆☆☆'
            }
            
            for quant in sorted_quants:
                variants = variants_by_quant[quant]
                count = len(variants)
                
                # Get size range
                sizes = [v['size_gb'] for v in variants]
                min_size = min(sizes)
                max_size = max(sizes)
                if min_size == max_size:
                    size_range = f"{min_size:.1f} GB"
                else:
                    size_range = f"{min_size:.1f}-{max_size:.1f} GB"
                
                # Get memory range
                memories = [v['memory_required_gb'] for v in variants]
                min_mem = min(memories)
                max_mem = max(memories)
                if min_mem == max_mem:
                    mem_range = f"{min_mem:.1f} GB"
                else:
                    mem_range = f"{min_mem:.1f}-{max_mem:.1f} GB"
                
                quality = quality_map.get(quant, '★★★☆☆')
                
                # Check if any is default
                has_default = any(v['file'] == profile.metadata.get('default_variant') 
                                for v in variants)
                marker = ' *' if has_default else ''
                
                print(f"  {quant:<14} {count:>5}  {size_range:<13} "
                      f"{mem_range:<13} {quality}{marker}")
            
            # Show individual files if not too many
            total_files = len(profile.metadata['variants'])
            if total_files <= 20:
                print("\n  Individual files:")
                print("  File" + " " * 50 + "Size      Memory")
                print("  " + "-" * 70)
                for variant in sorted(profile.metadata['variants'], 
                                    key=lambda x: x['size_gb']):
                    filename = variant['file']
                    if len(filename) > 50:
                        # Truncate long filenames
                        filename = filename[:47] + "..."
                    is_default = variant['file'] == profile.metadata.get('default_variant')
                    marker = ' *' if is_default else ''
                    print(f"  {filename:<53} {variant['size_gb']:>6.1f} GB  "
                          f"{variant['memory_required_gb']:>6.1f} GB{marker}")
            
            if profile.metadata.get('default_variant'):
                print("\n  * = Default variant")

        print("\nCapabilities:")
        print(f"  - VLLM Support: {'Yes' if profile.supports_vllm else 'No'}")
        print(f"  - TensorRT Support: {'Yes' if profile.supports_tensorrt else 'No'}")
        print(f"  - Multimodal: {'Yes' if profile.is_multimodal else 'No'}")

        # Add quantization support info
        if profile.supports_quantization:
            print(f"  - Quantization Support: {', '.join(profile.supports_quantization)}")

        # Hardware compatibility check
        print("\n" + "-"*60)
        print("Hardware Compatibility Check")
        print("-"*60)

        # Get hardware info
        hw_info = get_hardware_info()
        hw_summary = hw_info.get_summary()

        # Print hardware info
        print("\nYour Hardware:")
        print(f"  - CPU: {hw_summary['cpu']['cores']} cores, "
              f"{hw_summary['cpu']['threads']} threads")
        print(f"  - RAM: {hw_summary['memory']['total_ram_gb']} GB total, "
              f"{hw_summary['memory']['available_ram_gb']} GB available")

        if hw_summary['gpu']['gpus']:
            print("  - GPU(s):")
            for gpu in hw_summary['gpu']['gpus']:
                print(f"    • {gpu['name']} ({gpu.get('type', 'Unknown')})")
                if gpu.get('unified_memory'):
                    print(f"      Unified Memory: {gpu['total_memory_gb']:.1f} GB")
                else:
                    print(f"      VRAM: {gpu['total_memory_gb']:.1f} GB total, "
                          f"{gpu['free_memory_gb']:.1f} GB free")
        else:
            print("  - GPU: None detected (CPU-only mode)")

        # Check compatibility for different quantization levels
        print("\nCompatibility Analysis:")
        
        # Special handling for GGUF models
        if profile.model_type == 'gguf' and profile.metadata.get('variants'):
            print(f"{'Variant':<15} {'Size':<10} {'Memory':<12} {'Can Run?':<10} {'Device':<10} {'Notes'}")
            print("-" * 80)
            
            # Group by quantization and show summary
            variants_by_quant = {}
            for variant in profile.metadata['variants']:
                quant = variant['quantization']
                if quant not in variants_by_quant:
                    variants_by_quant[quant] = []
                variants_by_quant[quant].append(variant)
            
            # Show compatibility for each quantization type
            for quant in sorted(variants_by_quant.keys()):
                variants = variants_by_quant[quant]
                # Use median size for this quantization
                sizes = sorted([v['size_gb'] for v in variants])
                median_size = sizes[len(sizes)//2]
                median_memory = median_size * 1.2
                
                model_reqs = {
                    'memory_required_gb': median_memory,
                    'quantization': f'gguf-{quant}'
                }
                
                # Check compatibility
                compat = check_model_compatibility(model_reqs, hw_info)
                
                status = "✓" if compat['can_run'] else "✗"
                size_str = f"{median_size:.1f} GB"
                memory = f"{median_memory:.1f} GB"
                device = compat['recommended_device'] if compat['can_run'] else "N/A"
                
                # Create notes
                notes = []
                if compat['warnings']:
                    notes.append(compat['warnings'][0].split('.')[0])
                
                # Add count if multiple files
                if len(variants) > 1:
                    quant_display = f"{quant} ({len(variants)} files)"
                else:
                    quant_display = quant
                
                print(f"{quant_display:<15} {size_str:<10} {memory:<12} {status:<10} {device:<10} "
                      f"{notes[0] if notes else ''}")
        else:
            # Standard compatibility table for non-GGUF models
            print(f"{'Quantization':<12} {'Memory':<12} {'Can Run?':<10} {'Device':<10} {'Notes'}")
            print("-" * 60)
    
            # Determine storage dtype
            stored_dtype = None
            if profile.metadata and 'torch_dtype' in profile.metadata:
                stored_dtype = profile.metadata['torch_dtype']
            
            # Adjust size calculation based on storage format
            if stored_dtype in ['float16', 'bfloat16']:
                # Model is stored in half precision, fp32 would be 2x size
                fp32_size = profile.estimated_size_gb * 2
            else:
                # Model is stored in fp32 or unknown
                fp32_size = profile.estimated_size_gb
            
            quantizations = profile.supports_quantization or ['fp32']
            for quant in quantizations:
                # Calculate requirements based on fp32 size
                if quant == 'fp32':
                    base_size = fp32_size
                elif quant == 'fp16':
                    base_size = fp32_size * 0.5
                elif quant == '8bit':
                    base_size = fp32_size * 0.25
                elif quant == '4bit':
                    base_size = fp32_size * 0.125
                else:
                    base_size = fp32_size
                
                # Add overhead for inference
                memory_required = base_size * 1.2
                
                model_reqs = {
                    'memory_required_gb': memory_required,
                    'quantization': quant
                }
    
                # Check compatibility
                compat = check_model_compatibility(model_reqs, hw_info)
    
                status = "✓" if compat['can_run'] else "✗"
                memory = f"{model_reqs['memory_required_gb']:.1f} GB"
                device = compat['recommended_device'] if compat['can_run'] else "N/A"
    
                # Create notes
                notes = []
                if compat['warnings']:
                    notes.append(compat['warnings'][0].split('.')[0])
    
                print(f"{quant:<12} {memory:<12} {status:<10} {device:<10} "
                      f"{notes[0] if notes else ''}")

        # Training compatibility
        print("\nTraining Compatibility:")
        
        # Skip training section for GGUF models
        if profile.model_type == 'gguf':
            print("  GGUF models are pre-quantized and do not support training.")
            print("  To train, use the original model before quantization.")
        else:
            print(f"{'Method':<15} {'Memory':<10} {'CPU':<8} {'GPU':<8} {'Device':<10}")
            print("-" * 60)

            # Training methods to check - filter based on supported quantizations
            supported_quants = profile.supports_quantization or ['fp32', 'fp16']
            training_configs = []
            # Only add methods for supported quantizations
            if '4bit' in supported_quants:
                training_configs.append(('QLoRA (4bit)', '4bit', True))
            if '8bit' in supported_quants:
                training_configs.append(('QLoRA (8bit)', '8bit', True))
                training_configs.append(('LoRA (8bit)', '8bit', False))
            if 'fp16' in supported_quants:
                training_configs.append(('LoRA (fp16)', 'fp16', False))
                training_configs.append(('Full (fp16)', 'fp16', False, True))
            if 'fp32' in supported_quants:
                training_configs.append(('LoRA (fp32)', 'fp32', False))
                training_configs.append(('Full (fp32)', 'fp32', False, True))
    
            for method_name, quant, *flags in training_configs:
                # Calculate requirements using correct base size
                if quant == 'fp32':
                    base_size = fp32_size
                elif quant == 'fp16':
                    base_size = fp32_size * 0.5
                elif quant == '8bit':
                    base_size = fp32_size * 0.25
                elif quant == '4bit':
                    base_size = fp32_size * 0.125
                else:
                    base_size = fp32_size
                
                # Calculate training memory
                if 'QLoRA' in method_name:
                    # QLoRA uses quantized base model + small overhead
                    memory_required = base_size * 1.5
                elif 'LoRA' in method_name and 'QLoRA' not in method_name:
                    # Regular LoRA - base model + 10% for adapters
                    memory_required = base_size * 1.3
                else:
                    # Full training - need 4x the model size
                    memory_required = base_size * 4
                
                model_reqs = {
                    'memory_required_gb': memory_required,
                    'quantization': quant,
                    'task': 'training'
                }
    
                # Check compatibility
                compat = check_model_compatibility(model_reqs, hw_info)
    
                # Check CPU/GPU separately
                cpu_ok = "✓" if compat['can_run_cpu'] else "✗"
                gpu_ok = "✓" if compat['can_run_gpu'] else "✗"
    
                # Determine best device
                if compat['can_run']:
                    device = compat['recommended_device']
                    if device == 'cuda':
                        device = 'CUDA'
                    elif device == 'metal':
                        device = 'Metal'
                    else:
                        device = 'CPU'
                else:
                    device = 'N/A'
    
                memory_str = f"{model_reqs['memory_required_gb']:.1f} GB"
    
                print(f"{method_name:<15} {memory_str:<10} {cpu_ok:<8} {gpu_ok:<8} {device:<10}")

        print("="*60 + "\n")


def check_model(model_id: str) -> Optional[ModelProfile]:
    """Convenience function to check a model"""
    checker = ModelChecker()
    return checker.check_model(model_id)

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
from .model_detector import ModelDetector


logger = logging.getLogger(__name__)


class ModelChecker:
    """Main class for checking model compatibility"""

    def __init__(self):
        self.detector = ModelDetector()
        logger.debug("Initialized ModelDetector")

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

        # Use new simplified detector
        logger.debug("Running model detection...")
        detection_result = self.detector.detect(model_id)
        if not detection_result:
            logger.warning(f"Could not detect model {model_id}")
            # Create a generic profile as fallback
            profile = ModelProfile(
                model_type="unknown",
                model_id=model_id,
                library="transformers",  # Safe default
                task="text-generation"
            )
        else:
            # Convert detection result to ModelProfile
            profile = ModelProfile(
                model_type=detection_result['model_type'],
                model_id=detection_result['model_id'],
                library=detection_result['library'],
                architecture=detection_result.get('architecture'),
                task=detection_result.get('task'),
                quantization=detection_result.get('quantization'),
                special_requirements=detection_result.get('special_requirements'),
                estimated_size_gb=detection_result.get('estimated_size_gb', 1.0),
                estimated_memory_gb=detection_result.get('estimated_memory_gb', 4.0),
                supports_vllm=detection_result.get('supports_vllm', False),
                supports_tensorrt=detection_result.get('supports_tensorrt', False),
                is_multimodal=detection_result.get('is_multimodal', False),
                metadata=detection_result.get('metadata', {}),
                min_ram_gb=detection_result.get('min_ram_gb', 8.0),
                min_vram_gb=detection_result.get('min_vram_gb', 0.0),
                recommended_ram_gb=detection_result.get('recommended_ram_gb', 16.0),
                recommended_vram_gb=detection_result.get('recommended_vram_gb', 0.0),
                supports_cpu=detection_result.get('supports_cpu', True),
                supports_cuda=detection_result.get('supports_cuda', True),
                supports_metal=detection_result.get('supports_metal', True),
                supports_quantization=detection_result.get('supports_quantization')
            )
            logger.info(f"Model detected: {profile.model_type}")

        # Estimate size if not set by detector
        if profile.estimated_size_gb == 1.0:
            size_info = estimate_model_size(config or {}, files)
            profile.estimated_size_gb = size_info['size_gb']
            # Don't set estimated_memory_gb here - calculate it dynamically based on quantization

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
        if profile.estimated_size_gb > 0:
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

            quant_type = dtype_map.get(torch_dtype, torch_dtype)

            # Calculate memory for the detected type
            mem_reqs = calculate_model_requirements(
                profile.estimated_size_gb,
                quant_type if quant_type in ['fp32', 'fp16', '8bit', '4bit'] else 'fp32',
                "inference"
            )
            print(f"  - Default configuration ({torch_dtype}): "
                  f"{mem_reqs['memory_required_gb']:.1f} GB")
            print("  - Other configurations: See compatibility table below")
        else:
            # Use estimated memory if available
            if profile.estimated_memory_gb > 0:
                print(f"  - Estimated memory requirement: {profile.estimated_memory_gb:.1f} GB")
            else:
                print("  - Default configuration: Unknown")
            print(f"  - Model page: https://huggingface.co/{profile.model_id}")
            print("  - See compatibility table below for different configurations")

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
                print(f"  - Total: {sum(components.values()):.1f} GB")

                print("\nDeployment Options:")
                print("  - Minimum (mixed CPU/GPU):")
                print(f"    • GPU: {max(components.values()):.1f} GB (largest component)")
                print(f"    • RAM: {sum(components.values()):.1f} GB (all components)")
                print("  - Optimal (all GPU):")
                print(f"    • GPU: {sum(components.values()):.1f} GB (all components)")

        if profile.special_requirements:
            print("\nSpecial Requirements:")
            for req in profile.special_requirements:
                print(f"  - {req}")

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
        print(f"{'Quantization':<12} {'Memory':<12} {'Can Run?':<10} {'Device':<10} {'Notes'}")
        print("-" * 60)

        quantizations = profile.supports_quantization or ['fp32']
        for quant in quantizations:
            # Calculate requirements
            model_reqs = calculate_model_requirements(
                profile.estimated_size_gb,
                quant,
                "inference"
            )

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
            # Calculate requirements
            if 'QLoRA' in method_name:
                # QLoRA uses quantized base model
                model_reqs = calculate_model_requirements(
                    profile.estimated_size_gb, quant, "training"
                )
            elif 'LoRA' in method_name and 'QLoRA' not in method_name:
                # Regular LoRA - base model in specified precision + LoRA weights
                base_reqs = calculate_model_requirements(
                    profile.estimated_size_gb, quant, "inference"
                )
                # Add LoRA overhead (10% for adapters + gradients)
                model_reqs = {
                    'memory_required_gb': base_reqs['memory_required_gb'] * 1.1,
                    'quantization': quant,
                    'task': 'training'
                }
            else:
                # Full training
                model_reqs = calculate_model_requirements(
                    profile.estimated_size_gb, quant, "training"
                )

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

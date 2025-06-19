"""
Model compatibility checker module
"""

import logging
from typing import List, Optional, Type

from .utils import (
    fetch_model_config,
    fetch_model_files_list,
    fetch_model_files_with_sizes,
    estimate_model_size,
    save_json_config
)
from .detectors.base import BaseDetector, ModelProfile
from .hardware import (
    get_hardware_info,
    calculate_model_requirements,
    check_model_compatibility
)

# Import all detectors
from .detectors.transformer_detector import TransformerDetector
from .detectors.diffusion_detector import DiffusionDetector
from .detectors.gguf_detector import GGUFDetector
from .detectors.sentence_transformer_detector import SentenceTransformerDetector
from .detectors.multimodal_detector import MultimodalDetector
from .detectors.moe_detector import MoEDetector
from .detectors.multi_modality_detector import MultiModalityDetector


logger = logging.getLogger(__name__)


class ModelChecker:
    """Main class for checking model compatibility"""

    def __init__(self):
        self.detectors: List[BaseDetector] = []
        self._initialize_detectors()

    def _initialize_detectors(self):
        """Initialize all available detectors in priority order"""
        detector_classes: List[Type[BaseDetector]] = [
            GGUFDetector,  # Check GGUF first (most specific)
            MultiModalityDetector,  # Check multi_modality before multimodal
            MoEDetector,  # Then MoE models
            MultimodalDetector,  # Then standard multimodal models
            DiffusionDetector,  # Then diffusion models
            SentenceTransformerDetector,  # Then sentence transformers
            TransformerDetector,  # Default fallback
        ]

        for detector_class in detector_classes:
            try:
                detector = detector_class()
                self.detectors.append(detector)
                logger.debug(f"Initialized detector: {detector.name}")
            except Exception as e:
                logger.error(f"Failed to initialize {detector_class.__name__}: {e}")

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
        model_index = None
        if not config:
            logger.debug("No llm_config.json found, trying model_index.json...")
            model_index = fetch_model_config(model_id, "model_index.json")

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

        # Pass both config and model_index to detectors
        detection_config = config or model_index or {}

        # Store file sizes in detection config for detectors to use
        if file_sizes:
            detection_config['_file_sizes'] = file_sizes

        # Run through detector pipeline
        profile = None
        for detector in self.detectors:
            logger.debug(f"Running {detector.name}...")
            try:
                profile = detector.detect(model_id, detection_config, files)
                if profile:
                    logger.info(f"Model detected by {detector.name}: {profile.model_type}")
                    break
            except Exception as e:
                logger.error(f"Error in {detector.name}: {e}")
                continue

        if not profile:
            logger.warning(f"No detector could identify model {model_id}")
            # Create a generic profile as fallback
            profile = ModelProfile(
                model_type="unknown",
                model_id=model_id,
                library="transformers",  # Safe default
                task="text-generation"
            )

        # Estimate size if not set by detector
        if profile.estimated_size_gb == 1.0:
            size_info = estimate_model_size(detection_config, files)
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
            # No information available
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

        # Training methods to check
        training_configs = [
            ('QLoRA (4bit)', '4bit', True),  # quantization, is_qlora
            ('QLoRA (8bit)', '8bit', True),  # QLoRA with 8bit
            ('LoRA (8bit)', '8bit', False),  # Regular LoRA with 8bit
            ('LoRA (fp16)', 'fp16', False),  # Regular LoRA with fp16
            ('LoRA (fp32)', 'fp32', False),  # Regular LoRA with fp32
            ('Full (fp16)', 'fp16', False, True),  # Full training
            ('Full (fp32)', 'fp32', False, True),  # Full training fp32
        ]

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
                # Add LoRA overhead (typically 1-2GB for adapters + gradients)
                model_reqs = {
                    'memory_required_gb': base_reqs['memory_required_gb'] + 2.0,
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

"""Handler for transformer-based language models.

This handler manages models that use the transformers library,
including GPT, LLaMA, Mistral, and similar architectures.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import os
from handlers.base import BaseHandler
from core.checker import ModelRequirements

logger = logging.getLogger(__name__)


class TransformerHandler(BaseHandler):
    """Handler for transformer-based language models."""

    def analyze(self) -> ModelRequirements:
        """Analyze transformer model requirements.

        Returns:
            ModelRequirements object with analyzed data.
        """
        requirements = ModelRequirements()

        # Set model type and family
        requirements.model_type = "transformer"
        requirements.model_family = "language-model"
        requirements.primary_library = "transformers"

        # Base dependencies
        requirements.base_dependencies = self.get_dependencies()

        # Special dependencies from config
        config = self.model_info.get('config', {})
        if config.get('_attn_implementation') == 'flash_attention_2':
            requirements.special_dependencies.append('flash-attn')

        # Optional dependencies
        from core.quantization_config import QuantizationConfig
        
        optional_deps = ['flash-attn', 'deepspeed']
        
        # Add quantization dependencies if supported
        quant_deps = QuantizationConfig.get_quantization_dependencies(
            self.model_type, self.model_family
        )
        optional_deps.extend(quant_deps)
        
        requirements.optional_dependencies = optional_deps

        # Memory requirements
        model_size = self._estimate_model_size()
        requirements.disk_space_gb = model_size
        requirements.memory_requirements = {
            "min": model_size * 1.2,
            "recommended": model_size * 2,
            "optimal": model_size * 3,
        }

        # Capabilities
        requirements.capabilities = self.get_model_capabilities()

        return requirements

    def _estimate_model_size(self) -> float:
        """Estimate model size in GB."""
        # Use model_size from info if available
        if 'model_size' in self.model_info:
            return self.model_info['model_size']

        # Otherwise estimate from config
        config = self.model_info.get('config', {})
        hidden_size = config.get('hidden_size', 4096)
        num_layers = config.get('num_hidden_layers', 32)
        vocab_size = config.get('vocab_size', 32000)

        # Rough estimation in billions of parameters
        params_b = (hidden_size * hidden_size * 4 * num_layers + vocab_size * hidden_size) / 1e9

        # Convert to GB (assuming fp16)
        return params_b * 2

    def get_dependencies(self) -> List[str]:
        """Get Python dependencies for transformer models.

        Returns:
            List of required pip packages.
        """
        base_deps = [
            'transformers>=4.36.0',
            'torch>=2.6.0',
            'accelerate>=0.25.0',
            'sentencepiece>=0.1.99',  # For many tokenizers
            'protobuf>=3.20.0',       # For some models
            'tokenizers>=0.15.0'      # Fast tokenizers
        ]

        # Add special dependencies based on model architecture
        special_reqs = self.model_info.get('special_requirements', [])

        if 'flash-attn' in special_reqs:
            base_deps.append('flash-attn==2.7.2.post1')

        if 'mamba-ssm' in special_reqs:
            base_deps.extend(['mamba-ssm>=1.0.0', 'causal-conv1d>=1.0.0'])

        return base_deps

    def get_system_dependencies(self) -> List[str]:
        """Get system dependencies for transformer models.

        Returns:
            List of system requirements.
        """
        deps = []

        # Check if CUDA is needed
        device = self.model_info.get('optimal_device', 'auto')
        if device in ['cuda', 'auto']:
            deps.append('cuda>=11.7')

        return deps

    def load_model(self, model_path: str, **kwargs):
        """Load transformer model with optimal settings.

        Args:
            model_path: Path to model directory.
            **kwargs: Additional parameters.

        Returns:
            Tuple of (model, tokenizer).
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        # Extract parameters
        device = kwargs.get('device', 'auto')
        dtype = kwargs.get('dtype', 'auto')
        load_in_8bit = kwargs.get('load_in_8bit', False)
        load_in_4bit = kwargs.get('load_in_4bit', False)

        # Use base handler's quantization config
        quant_config, torch_dtype = self.get_quantization_config(dtype, load_in_8bit, load_in_4bit)

        # Load configuration
        model_kwargs = {
            'torch_dtype': torch_dtype,
            'low_cpu_mem_usage': True,
            'trust_remote_code': self.model_info.get('trust_remote_code', False)
        }

        # Merge quantization config
        model_kwargs.update(quant_config)

        # Device map for multi-GPU or CPU offloading
        if device == 'auto':
            model_kwargs['device_map'] = 'auto'
            # Add offload settings for better memory management
            model_kwargs['offload_folder'] = 'offload'
            model_kwargs['offload_state_dict'] = True

            # For very large models, use sequential device map
            model_size = self.model_info.get('model_size_b', 0)
            if model_size > 30:  # 30B+ models
                model_kwargs['device_map'] = 'sequential'
                model_kwargs['max_memory'] = {0: '20GiB', 'cpu': '100GiB'}
        elif device != 'cpu':
            model_kwargs['device_map'] = {'': device}

        # Check if flash attention should be disabled
        disable_flash_attn = (
            os.environ.get('TRANSFORMERS_USE_FLASH_ATTENTION', '1') == '0' or
            kwargs.get('use_flash_attention_2', True) is False
        )
        
        # Try to use Flash Attention 2 if available and not disabled
        if not disable_flash_attn:
            try:
                model_kwargs['attn_implementation'] = 'flash_attention_2'
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **model_kwargs
                )
                logger.info("Using Flash Attention 2 for better performance")
            except Exception:
                # Fallback to standard attention
                model_kwargs.pop('attn_implementation', None)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **model_kwargs
                )
                logger.info("Using standard attention implementation")
        else:
            # Explicitly use standard attention
            model_kwargs['attn_implementation'] = 'eager'
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            logger.info("Using standard attention implementation (flash attention disabled)")

        # Load tokenizer with use_fast=True
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.model_info.get('trust_remote_code', False),
            use_fast=True
        )

        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def get_inference_params(self) -> Dict[str, Any]:
        """Get default inference parameters.

        Returns:
            Dictionary of inference parameters.
        """
        return {
            'temperature': 0.7,
            'top_p': 0.95,
            'top_k': 50,
            'max_new_tokens': 512,
            'do_sample': True,
            'repetition_penalty': 1.1,
            'pad_token_id': None,  # Set from tokenizer
            'eos_token_id': None,  # Set from tokenizer
        }

    def get_training_params(self) -> Dict[str, Any]:
        """Get default training parameters.

        Returns:
            Dictionary of training parameters.
        """
        model_size = self.model_info.get('model_size_b', 7)

        # Adjust parameters based on model size
        if model_size < 1:
            base_lr = 5e-4
            lora_r = 8
        elif model_size < 7:
            base_lr = 2e-4
            lora_r = 16
        elif model_size < 13:
            base_lr = 1e-4
            lora_r = 32
        else:
            base_lr = 5e-5
            lora_r = 64

        return {
            'learning_rate': base_lr,
            'num_train_epochs': 3,
            'per_device_train_batch_size': 4,
            'per_device_eval_batch_size': 4,
            'gradient_accumulation_steps': 4,
            'warmup_steps': 100,
            'logging_steps': 10,
            'save_steps': 500,
            'eval_steps': 500,
            'lora_r': lora_r,
            'lora_alpha': lora_r * 2,
            'lora_dropout': 0.1,
            'lora_target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj']
        }

    def validate_model_files(self, model_path: str) -> Tuple[bool, Optional[str]]:
        """Validate model files are present.

        Args:
            model_path: Path to model directory.

        Returns:
            Tuple of (is_valid, error_message).
        """
        model_path = Path(model_path)

        # Check for config
        if not (model_path / 'config.json').exists():
            return False, "Missing config.json"

        # Check for model weights
        has_weights = False
        for pattern in ['*.bin', '*.safetensors', '*.pt', '*.pth']:
            if list(model_path.glob(pattern)):
                has_weights = True
                break

        if not has_weights:
            return False, "No model weight files found"

        # Check for tokenizer
        tokenizer_files = [
            'tokenizer.json',
            'tokenizer_config.json',
            'tokenizer.model',  # For sentencepiece
            'vocab.json'        # For some older models
        ]

        has_tokenizer = any(
            (model_path / f).exists() for f in tokenizer_files
        )

        if not has_tokenizer:
            return False, "No tokenizer files found"

        return True, None

    def get_model_capabilities(self) -> Dict[str, Any]:
        """Get model capabilities.

        Returns:
            Dictionary of capabilities.
        """
        capabilities = super().get_model_capabilities()

        # Update for transformer models
        capabilities.update({
            'supports_streaming': True,
            'supports_system_prompt': True,
            'supports_batch_inference': True,
            'input_modalities': ['text'],
            'output_modalities': ['text']
        })

        # Check for reasoning support
        model_type = self.model_info.get('config', {}).get('model_type', '')
        if model_type in ['o1', 'reasoning-llm'] or 'reasoning' in self.model_info.get('tags', []):
            capabilities['supports_reasoning'] = True

        # Get context length
        config = self.model_info.get('config', {})
        max_length = config.get('max_position_embeddings',
                                config.get('max_sequence_length',
                                           config.get('n_positions', 2048)))
        capabilities['max_context_length'] = max_length

        return capabilities

    def prepare_for_training(self, model, training_method: str = 'lora') -> Any:
        """Prepare model for training.

        Args:
            model: Loaded model instance.
            training_method: Training method to use.

        Returns:
            Prepared model.
        """
        if training_method in ['lora', 'qlora']:
            from peft import LoraConfig, get_peft_model, TaskType

            # Get training parameters
            train_params = self.get_training_params()

            # Create LoRA configuration
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=train_params['lora_r'],
                lora_alpha=train_params['lora_alpha'],
                lora_dropout=train_params['lora_dropout'],
                target_modules=train_params['lora_target_modules']
            )

            # Prepare model for k-bit training if using QLoRA
            if training_method == 'qlora':
                from peft import prepare_model_for_kbit_training
                model = prepare_model_for_kbit_training(model)

            # Get PEFT model
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

            return model

        elif training_method == 'full':
            # For full fine-tuning, just return the model
            return model

        else:
            raise ValueError(f"Unknown training method: {training_method}")

    def generate_text(self, prompt: str = None, messages: List[Dict] = None,
                      model=None, tokenizer=None, **kwargs) -> Dict[str, Any]:
        """Generate text using transformer model.

        Args:
            prompt: Text prompt
            messages: Chat messages
            model: Model instance
            tokenizer: Tokenizer instance
            **kwargs: Generation parameters

        Returns:
            Dictionary with generated text and metadata
        """
        if not model or not tokenizer:
            raise ValueError("Model and tokenizer required for text generation")

        # Prepare input
        if messages:
            # Apply chat template if available
            if hasattr(tokenizer, 'apply_chat_template'):
                text = tokenizer.apply_chat_template(messages, tokenize=False)
            else:
                # Simple fallback
                text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        else:
            text = prompt or ""

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=kwargs.get('max_length', 2048))

        if hasattr(model, "device"):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate with memory optimization
        import torch

        # Clear GPU cache before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=kwargs.get('max_tokens', 512),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                top_k=kwargs.get('top_k', 50),
                do_sample=kwargs.get('temperature', 0.7) > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **{k: v for k, v in kwargs.items() if k in ['repetition_penalty', 'length_penalty']}
            )

        # Decode
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        skip_special_tokens = kwargs.get('skip_special_tokens', True)
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=skip_special_tokens)

        # Calculate usage
        usage = {
            'prompt_tokens': inputs['input_ids'].shape[1],
            'completion_tokens': generated_ids.shape[0],
            'total_tokens': outputs[0].shape[0]
        }

        # Clean up to free memory
        del outputs
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            'text': generated_text,
            'usage': usage
        }

    def get_supported_modes(self) -> List[str]:
        """Get supported generation modes for transformer models."""
        return ['auto', 'chat', 'complete', 'instruct', 'creative', 'code', 'analyze', 'translate', 'summarize']

    def get_mode_descriptions(self) -> Dict[str, str]:
        """Get mode descriptions."""
        return {
            'auto': 'Automatic mode selection',
            'chat': 'Conversational chat mode',
            'complete': 'Text completion mode',
            'instruct': 'Instruction following mode',
            'creative': 'Creative writing mode',
            'code': 'Code generation mode',
            'analyze': 'Analysis and reasoning mode',
            'translate': 'Translation mode',
            'summarize': 'Summarization mode'
        }

    def apply_mode_settings(self, mode: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mode-specific settings."""
        mode_settings = {
            'creative': {'temperature': 0.9, 'top_p': 0.95},
            'code': {'temperature': 0.3, 'top_p': 0.9},
            'analyze': {'temperature': 0.5, 'top_p': 0.9},
            'translate': {'temperature': 0.3, 'top_p': 0.9},
            'summarize': {'temperature': 0.5, 'top_p': 0.9}
        }

        if mode in mode_settings:
            params.update(mode_settings[mode])

        return params

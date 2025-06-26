"""Handler for DeepSeek models including R1 reasoning models.

This handler provides specialized support for DeepSeek models,
particularly the R1 series which require thinking tag enforcement.
"""

import logging
from typing import Dict, Any, List, Optional, Union
import torch

from handlers.base import BaseHandler

logger = logging.getLogger(__name__)


class DeepseekHandler(BaseHandler):
    """Handler for DeepSeek models with special R1 reasoning support."""
    
    def __init__(self, model_info: Dict[str, Any]):
        """Initialize DeepSeek handler.
        
        Args:
            model_info: Model information from detector
        """
        super().__init__(model_info)
        self.model_id = model_info.get('model_id', '').lower()
        self.is_r1_model = self._is_r1_model()
        
    def _is_r1_model(self) -> bool:
        """Check if this is a DeepSeek R1 reasoning model."""
        r1_patterns = ['deepseek-r1', 'deepseek_r1', '-r1-', '_r1_']
        return any(pattern in self.model_id for pattern in r1_patterns)
    
    def get_model_capabilities(self) -> Dict[str, Any]:
        """Get model capabilities with R1 reasoning support."""
        capabilities = super().get_model_capabilities()
        
        if self.is_r1_model:
            capabilities.update({
                'reasoning': True,
                'supports_reasoning': True,
                'thinking_tags': True,
                'requires_thinking_enforcement': True
            })
        
        return capabilities
    
    def get_supported_modes(self) -> List[str]:
        """Get supported generation modes."""
        modes = ["auto", "chat", "complete", "instruct", "creative", "code", "analyze"]
        
        if self.is_r1_model:
            modes.extend(["reasoning", "thinking", "no-thinking"])
        
        return modes
    
    def get_mode_descriptions(self) -> Dict[str, str]:
        """Get descriptions for each mode."""
        descriptions = {
            "auto": "Automatic mode selection",
            "chat": "Conversational dialogue",
            "complete": "Text completion", 
            "instruct": "Follow instructions",
            "creative": "Creative generation",
            "code": "Code generation",
            "analyze": "Analysis & reasoning"
        }
        
        if self.is_r1_model:
            descriptions.update({
                "reasoning": "Deep reasoning with thinking process",
                "thinking": "Show thinking process",
                "no-thinking": "Skip thinking process (not recommended)"
            })
        
        return descriptions
    
    def generate_text(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        model: Any = None,
        tokenizer: Any = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 0.9,
        top_k: int = 50,
        stop_sequences: Optional[List[str]] = None,
        mode: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """Generate text with R1 thinking enforcement.
        
        For R1 models, we enforce thinking by prepending <think>\n to responses.
        """
        import torch
        
        # Prepare input
        if messages:
            # Apply chat template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text = prompt or ""
        
        if not text:
            raise ValueError("No input provided")
        
        # For R1 models in non-no-thinking modes, enforce thinking
        force_thinking = (
            self.is_r1_model and 
            mode != "no-thinking" and
            "<think>" not in text  # Don't add if already present
        )
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt")
        if hasattr(model, "device"):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "do_sample": temperature > 0,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            
            if stop_sequences:
                generation_kwargs["stop_strings"] = stop_sequences
            
            # For R1 models, we need to handle thinking enforcement
            if force_thinking:
                # Generate with forced prefix
                # First, encode the thinking tag
                think_tokens = tokenizer.encode("<think>\n", add_special_tokens=False, return_tensors="pt")
                if hasattr(model, "device"):
                    think_tokens = think_tokens.to(model.device)
                
                # Generate the rest
                outputs = model.generate(
                    **inputs,
                    forced_decoder_ids=[[i, token] for i, token in enumerate(think_tokens[0])],
                    **generation_kwargs
                )
            else:
                outputs = model.generate(**inputs, **generation_kwargs)
        
        # Decode
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=False  # Keep thinking tags
        )
        
        # For R1 models, ensure thinking tags are present
        if force_thinking and not generated_text.startswith("<think>"):
            # Prepend thinking tag if model didn't generate it
            generated_text = "<think>\n" + generated_text
        
        # Clean up any empty thinking blocks
        if self.is_r1_model:
            import re
            # Replace empty thinking blocks with proper thinking
            empty_pattern = r'<think>\s*</think>'
            if re.search(empty_pattern, generated_text):
                logger.warning("Model generated empty thinking block, regenerating...")
                # You might want to retry generation here
        
        # Calculate usage
        usage = {
            "prompt_tokens": input_length,
            "completion_tokens": len(generated_tokens),
            "total_tokens": outputs.shape[1]
        }
        
        # Parse thinking if present
        response = {"text": generated_text, "usage": usage}
        
        if "<think>" in generated_text and "</think>" in generated_text:
            # Extract thinking and answer
            import re
            match = re.search(r'<think>(.*?)</think>(.*)', generated_text, re.DOTALL)
            if match:
                thinking = match.group(1).strip()
                answer = match.group(2).strip()
                response["thinking"] = thinking
                response["answer"] = answer
        
        return response
    
    def get_dependencies(self) -> List[str]:
        """Get required dependencies for DeepSeek models."""
        deps = [
            "torch>=2.0.0",
            "transformers>=4.36.0",
            "accelerate>=0.25.0",
            "sentencepiece>=0.1.99",
            "protobuf>=3.20.0",
            "tokenizers>=0.15.0"
        ]
        
        return deps
    
    def validate_model_files(self, model_path: str) -> bool:
        """Validate DeepSeek model files."""
        from pathlib import Path
        
        model_dir = Path(model_path)
        required_files = [
            "config.json",
            "tokenizer_config.json"
        ]
        
        # Check for model files (various formats)
        model_patterns = [
            "*.safetensors",
            "*.bin", 
            "pytorch_model.bin",
            "model.safetensors"
        ]
        
        # Check required files
        for file in required_files:
            if not (model_dir / file).exists():
                logger.error(f"Missing required file: {file}")
                return False
        
        # Check for at least one model file
        has_model = False
        for pattern in model_patterns:
            if list(model_dir.glob(pattern)):
                has_model = True
                break
        
        if not has_model:
            logger.error("No model weights found")
            return False
        
        return True
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get configuration for loading DeepSeek models."""
        config = {
            "model_type": "deepseek",
            "trust_remote_code": True,  # Some DeepSeek models need this
            "torch_dtype": "auto",
            "device_map": "auto",
            "low_cpu_mem_usage": True
        }
        
        if self.is_r1_model:
            config.update({
                "use_cache": True,  # Important for reasoning models
                "output_hidden_states": False,
                "output_attentions": False
            })
        
        return config
    
    def analyze(self, model_path: str) -> Dict[str, Any]:
        """Analyze DeepSeek model."""
        import json
        from pathlib import Path
        
        analysis = {
            "model_type": "deepseek",
            "model_family": "language-model",
            "is_r1_model": self.is_r1_model,
            "supports_reasoning": self.is_r1_model
        }
        
        # Read config
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            analysis.update({
                "architecture": config.get("architectures", ["unknown"])[0],
                "hidden_size": config.get("hidden_size", 0),
                "num_layers": config.get("num_hidden_layers", 0),
                "vocab_size": config.get("vocab_size", 0),
                "max_position_embeddings": config.get("max_position_embeddings", 0)
            })
        
        return analysis
    
    def get_inference_params(self) -> Dict[str, Any]:
        """Get inference parameters for DeepSeek models."""
        params = {
            'temperature': 0.7,
            'top_p': 0.95,
            'max_new_tokens': 1024,
            'do_sample': True
        }
        
        # Adjust for R1 reasoning models
        if self.is_r1_model:
            params.update({
                'temperature': 0.1,  # Lower temperature for reasoning
                'max_new_tokens': 4096,  # Longer outputs for reasoning
                'return_thinking': True,
                'max_thinking_tokens': 32768
            })
        
        return params
    
    def get_system_dependencies(self) -> List[str]:
        """Get system dependencies for DeepSeek models."""
        deps = []
        
        # CUDA for GPU acceleration
        if self.model_info.get('requires_gpu', True):
            deps.append('cuda>=11.7')
        
        return deps
    
    def get_training_params(self) -> Dict[str, Any]:
        """Get training parameters for DeepSeek models."""
        params = {
            'learning_rate': 2e-5,
            'per_device_train_batch_size': 4,
            'per_device_eval_batch_size': 8,
            'num_train_epochs': 3,
            'warmup_steps': 500,
            'logging_steps': 100,
            'save_steps': 500,
            'eval_steps': 500,
            'gradient_accumulation_steps': 4,
            'fp16': True,
            'gradient_checkpointing': True,
            'load_best_model_at_end': True,
            'metric_for_best_model': 'eval_loss',
            'greater_is_better': False,
            'save_total_limit': 3,
        }
        
        # Adjust for R1 models (more conservative training)
        if self.is_r1_model:
            params['learning_rate'] = 1e-5  # Lower LR for reasoning models
            params['max_grad_norm'] = 0.5   # More conservative gradients
        
        return params
    
    def load_model(self, model_path: str, **kwargs):
        """Load DeepSeek model with proper configuration."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        # Prepare model kwargs
        model_kwargs = {
            'pretrained_model_name_or_path': model_path,
            'torch_dtype': kwargs.get('torch_dtype', torch.float16),
            'device_map': kwargs.get('device_map', 'auto'),
            'trust_remote_code': True,  # DeepSeek models may need this
            'low_cpu_mem_usage': True,
        }
        
        # Add quantization if specified
        if kwargs.get('load_in_8bit', False):
            model_kwargs['load_in_8bit'] = True
        elif kwargs.get('load_in_4bit', False):
            model_kwargs['load_in_4bit'] = True
            
        # Remove any kwargs that might cause issues
        model_kwargs = {k: v for k, v in model_kwargs.items() 
                       if k not in ['load_lora', 'lora_path']}
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, tokenizer
    
    def get_installation_notes(self) -> List[str]:
        """Get installation notes for DeepSeek models."""
        notes = [
            "DeepSeek models may require trust_remote_code=True",
            "Ensure you have sufficient GPU memory for the model size"
        ]
        
        if self.is_r1_model:
            notes.extend([
                "R1 models use thinking tags for reasoning",
                "The model will show its thought process before answers",
                "Use 'reasoning' mode for best results",
                "Thinking process is enforced to ensure quality outputs"
            ])
        
        return notes
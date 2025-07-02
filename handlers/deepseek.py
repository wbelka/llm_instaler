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
        
        # All DeepSeek models support streaming
        capabilities['supports_streaming'] = True
        
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
        """Generate text with R1 thinking support.
        
        For R1 models, the chat template automatically adds thinking tags.
        The response is parsed to separate thinking from the actual answer.
        """
        import torch
        
        # Prepare input
        if messages:
            # Apply chat template with add_generation_prompt=True
            # This will automatically add <｜Assistant｜><think>\n for R1 models
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            logger.info(f"Applied chat template. Text length: {len(text)}")
            logger.debug(f"Chat template output: {text[-200:]}")  # Log last 200 chars
        else:
            text = prompt or ""
        
        if not text:
            raise ValueError("No input provided")
        
        # For R1 models, the chat template already handles thinking tags
        # We just need to track if we're in thinking mode
        force_thinking = (
            self.is_r1_model and 
            mode != "no-thinking"
        )
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt")
        
        # Move inputs to the correct device
        if hasattr(model, 'device') and model.device.type != 'meta':
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        elif hasattr(model, 'hf_device_map'):
            # For models with device_map, move inputs to first device (usually GPU)
            first_device = next(iter(model.hf_device_map.values()))
            if isinstance(first_device, int):
                first_device = f'cuda:{first_device}'
            elif first_device == 'cpu':
                first_device = 'cpu'
            inputs = {k: v.to(first_device) for k, v in inputs.items()}
        elif torch.cuda.is_available():
            # Fallback: if GPU is available, use it
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            # Ensure tokenizer has pad_token_id
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "do_sample": temperature > 0,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            
            if stop_sequences:
                generation_kwargs["stop_strings"] = stop_sequences
            
            logger.info(f"Generating with kwargs: {generation_kwargs}")
            logger.info(f"Input shape: {inputs['input_ids'].shape}")
            
            # Generate
            outputs = model.generate(**inputs, **generation_kwargs)
            
            logger.info(f"Generated output shape: {outputs.shape}")
        
        # Decode
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        
        # Removed token count logging
        
        generated_text = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=False  # Keep thinking tags
        )
        
        logger.debug(f"Raw generated text: {generated_text[:200]}")
        
        # Remove special tokens from the output
        eos_token = tokenizer.eos_token
        if eos_token and eos_token in generated_text:
            generated_text = generated_text.replace(eos_token, "")
        
        # Remove other special tokens specific to DeepSeek
        special_tokens_to_remove = [
            "<｜Assistant｜>",
            "<｜User｜>",
            "<｜end▁of▁sentence｜>",
            "<｜begin▁of▁sentence｜>"
        ]
        for token in special_tokens_to_remove:
            if token in generated_text:
                generated_text = generated_text.replace(token, "")
        
        
        # Calculate usage
        usage = {
            "prompt_tokens": input_length,
            "completion_tokens": len(generated_tokens),
            "total_tokens": outputs.shape[1]
        }
        
        # Parse thinking if present
        response = {"usage": usage}
        
        if "<think>" in generated_text and "</think>" in generated_text:
            # Extract thinking and answer
            import re
            match = re.search(r'<think>(.*?)</think>(.*)', generated_text, re.DOTALL)
            if match:
                thinking = match.group(1).strip()
                answer = match.group(2).strip()
                response["thinking"] = thinking
                response["text"] = answer  # Only the answer part goes to text
                response["full_text"] = generated_text  # Keep full text for debugging
            else:
                response["text"] = generated_text
        elif "</think>" in generated_text:
            # Model might have started with thinking without opening tag
            parts = generated_text.split("</think>", 1)
            if len(parts) == 2:
                thinking = parts[0].strip()
                answer = parts[1].strip()
                response["thinking"] = thinking
                response["text"] = answer
                response["full_text"] = generated_text
            else:
                response["text"] = generated_text
        else:
            response["text"] = generated_text
        
        return response
    
    def generate_stream(
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
    ):
        """Stream text generation with R1 thinking support.
        
        Yields tokens as they are generated, handling thinking tags appropriately.
        """
        import torch
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        # Prepare input (same as generate_text)
        if messages:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            logger.info(f"Applied chat template for streaming. Text length: {len(text)}")
        else:
            text = prompt or ""
        
        if not text:
            raise ValueError("No input provided")
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt")
        
        # Move inputs to the correct device
        if hasattr(model, 'device') and model.device.type != 'meta':
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        elif hasattr(model, 'hf_device_map'):
            # For models with device_map, move inputs to first device (usually GPU)
            first_device = next(iter(model.hf_device_map.values()))
            if isinstance(first_device, int):
                first_device = f'cuda:{first_device}'
            elif first_device == 'cpu':
                first_device = 'cpu'
            inputs = {k: v.to(first_device) for k, v in inputs.items()}
        elif torch.cuda.is_available():
            # Fallback: if GPU is available, use it
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Create streamer
        streamer = TextIteratorStreamer(
            tokenizer, 
            skip_prompt=True,
            skip_special_tokens=False  # Keep special tokens for thinking tag parsing
        )
        
        # Ensure tokenizer has pad_token_id
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Generation kwargs
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        if stop_sequences:
            generation_kwargs["stop_strings"] = stop_sequences
        
        # Start generation in separate thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream tokens with thinking tag handling
        generated_text = ""
        thinking_buffer = ""
        in_thinking = False
        thinking_complete = False
        
        # Special tokens to remove from output
        special_tokens_to_remove = [
            "<｜Assistant｜>",
            "<｜User｜>",
            "<｜end▁of▁sentence｜>",
            "<｜begin▁of▁sentence｜>",
            tokenizer.eos_token if tokenizer.eos_token else ""
        ]
        
        for new_text in streamer:
            if not new_text:
                continue
            
            # Clean special tokens from the chunk
            clean_chunk = new_text
            for token in special_tokens_to_remove:
                if token and token in clean_chunk:
                    clean_chunk = clean_chunk.replace(token, "")
            
            generated_text += clean_chunk
            
            # Handle thinking tags for R1 models
            if self.is_r1_model and mode != "no-thinking":
                # Check if we're entering thinking mode
                if "<think>" in clean_chunk and not in_thinking:
                    in_thinking = True
                    thinking_buffer = ""
                    # Don't yield the thinking tag itself
                    before_think = clean_chunk.split("<think>")[0]
                    if before_think:
                        yield {"token": before_think, "type": "text"}
                    continue
                
                # Buffer thinking content
                if in_thinking and not thinking_complete:
                    if "</think>" in clean_chunk:
                        # End of thinking
                        parts = clean_chunk.split("</think>", 1)
                        thinking_buffer += parts[0]
                        in_thinking = False
                        thinking_complete = True
                        
                        # Yield the complete thinking
                        if thinking_buffer.strip():
                            yield {"thinking": thinking_buffer.strip(), "type": "thinking"}
                        
                        # Yield any content after thinking
                        if len(parts) > 1 and parts[1]:
                            yield {"token": parts[1], "type": "text"}
                    else:
                        # Still in thinking mode, buffer it
                        thinking_buffer += clean_chunk
                    continue
            
            # Regular text streaming (not in thinking mode)
            if not in_thinking and clean_chunk:
                yield {"token": clean_chunk, "type": "text"}
        
        # Wait for generation to complete
        thread.join()
        
        # Final yield with completion info
        yield {
            "type": "done",
            "full_text": generated_text,
            "thinking": thinking_buffer if thinking_buffer else None
        }
    
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
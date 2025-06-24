"""Handler for Qwen3 models with thinking mode support.

This handler manages Qwen3 models which support seamless switching between
thinking mode (for complex reasoning) and non-thinking mode (for efficient dialogue).
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from handlers.base import BaseHandler
from handlers.transformer import TransformerHandler

logger = logging.getLogger(__name__)


class Qwen3Handler(TransformerHandler):
    """Handler for Qwen3 models with thinking mode support."""
    
    def __init__(self, model_info: Dict[str, Any]):
        """Initialize Qwen3 handler."""
        super().__init__(model_info)
        self.supports_thinking = True
        self.thinking_token_id = 151668  # </think> token ID
    
    def get_dependencies(self) -> List[str]:
        """Get Python dependencies for Qwen3."""
        base_deps = super().get_dependencies()
        # Ensure we have the latest transformers for Qwen3 support
        updated_deps = []
        for dep in base_deps:
            if dep.startswith('transformers'):
                updated_deps.append('transformers>=4.51.0')
            else:
                updated_deps.append(dep)
        
        # Add bitsandbytes for quantization support
        if 'bitsandbytes' not in str(updated_deps):
            updated_deps.append('bitsandbytes>=0.41.0')
        
        return updated_deps
    
    def get_supported_modes(self) -> List[str]:
        """Get supported generation modes."""
        return ['auto', 'chat', 'complete', 'thinking', 'no-thinking', 'code', 'math', 'reasoning']
    
    def get_mode_descriptions(self) -> Dict[str, str]:
        """Get descriptions for supported modes."""
        return {
            'auto': 'Automatic mode selection',
            'chat': 'Conversational dialogue',
            'complete': 'Text completion',
            'thinking': 'Enable thinking mode for complex reasoning',
            'no-thinking': 'Disable thinking for efficient responses',
            'code': 'Code generation with thinking',
            'math': 'Mathematical problem solving with thinking',
            'reasoning': 'Complex logical reasoning with thinking'
        }
    
    def get_generation_config(self, task: str = "text") -> Dict[str, Any]:
        """Get Qwen3-specific generation configuration."""
        base_config = super().get_generation_config(task)
        
        # Qwen3 specific defaults
        base_config.update({
            'max_new_tokens': 32768,  # Qwen3 supports 32K tokens natively, 131K with YaRN
            'do_sample': True,  # Always use sampling for Qwen3
            'top_k': 20,
            'min_p': 0
        })
        
        return base_config
    
    def apply_mode_settings(self, mode: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Qwen3-specific mode settings."""
        # Set enable_thinking based on mode
        if mode in ['thinking', 'code', 'math', 'reasoning']:
            params['enable_thinking'] = True
            # Thinking mode parameters
            params['temperature'] = params.get('temperature', 0.6)
            params['top_p'] = params.get('top_p', 0.95)
            params['top_k'] = params.get('top_k', 20)
            params['do_sample'] = True  # Never use greedy for thinking mode
        elif mode == 'no-thinking':
            params['enable_thinking'] = False
            # Non-thinking mode parameters
            params['temperature'] = params.get('temperature', 0.7)
            params['top_p'] = params.get('top_p', 0.8)
            params['top_k'] = params.get('top_k', 20)
        else:
            # Default: enable thinking
            params['enable_thinking'] = params.get('enable_thinking', True)
        
        return params
    
    def generate_text(self, prompt: str = None, messages: List[Dict] = None,
                     model=None, tokenizer=None, **kwargs) -> Dict[str, Any]:
        """Generate text with Qwen3 thinking mode support.
        
        Args:
            prompt: Text prompt
            messages: Chat messages
            model: Model instance
            tokenizer: Tokenizer instance
            **kwargs: Generation parameters including enable_thinking
            
        Returns:
            Dictionary with generated text and optional thinking content
        """
        if not model or not tokenizer:
            raise ValueError("Model and tokenizer required for text generation")
        
        # Apply mode settings
        mode = kwargs.get('mode', 'auto')
        kwargs = self.apply_mode_settings(mode, kwargs)
        
        # Check for soft switches in the prompt
        if prompt:
            if '/think' in prompt:
                kwargs['enable_thinking'] = True
                logger.info("Detected /think in prompt, enabling thinking mode")
            elif '/no_think' in prompt:
                kwargs['enable_thinking'] = False
                logger.info("Detected /no_think in prompt, disabling thinking mode")
        
        # Check in messages for soft switches
        if messages and messages:
            last_content = messages[-1].get('content', '')
            if '/think' in last_content:
                kwargs['enable_thinking'] = True
            elif '/no_think' in last_content:
                kwargs['enable_thinking'] = False
        
        # Apply chat template with thinking mode
        enable_thinking = kwargs.get('enable_thinking', True)
        
        if messages:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
        else:
            # For simple prompt, create a message
            messages = [{"role": "user", "content": prompt or ""}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt")
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        import torch
        with torch.no_grad():
            # Get max tokens with proper default
            max_tokens = kwargs.get('max_tokens', 4096)  # More reasonable default
            # Log the actual value being used
            logger.info(f"Generating with max_tokens={max_tokens}, enable_thinking={enable_thinking}")
            
            generation_kwargs = {
                'max_new_tokens': max_tokens,
                'temperature': kwargs.get('temperature', 0.6 if enable_thinking else 0.7),
                'top_p': kwargs.get('top_p', 0.95 if enable_thinking else 0.8),
                'top_k': kwargs.get('top_k', 20),
                'do_sample': True,  # Always use sampling for Qwen3
                'pad_token_id': tokenizer.pad_token_id,
                'eos_token_id': tokenizer.eos_token_id,
            }
            
            # Add presence penalty if specified
            if 'presence_penalty' in kwargs:
                generation_kwargs['presence_penalty'] = kwargs['presence_penalty']
            
            logger.info("Starting generation...")
            try:
                outputs = model.generate(**inputs, **generation_kwargs)
                logger.info(f"Generation completed, output length: {outputs.shape[1]}")
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raise
        
        # Decode and parse thinking content
        output_ids = outputs[0][len(inputs['input_ids'][0]):].tolist()
        
        # Parse thinking content if present
        thinking_content = ""
        response_content = ""
        
        if enable_thinking:
            try:
                # Find </think> token (151668)
                index = len(output_ids) - output_ids[::-1].index(self.thinking_token_id)
                thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
                response_content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
            except ValueError:
                # No thinking token found, all content is response
                response_content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        else:
            response_content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
        result = {
            'text': response_content,
            'usage': {
                'prompt_tokens': len(inputs['input_ids'][0]),
                'completion_tokens': len(output_ids),
                'total_tokens': len(inputs['input_ids'][0]) + len(output_ids)
            }
        }
        
        # Add thinking content if present
        if thinking_content:
            result['thinking'] = thinking_content
        
        return result
    
    def get_model_capabilities(self) -> Dict[str, Any]:
        """Get Qwen3 model capabilities."""
        capabilities = super().get_model_capabilities()
        
        capabilities.update({
            'supports_thinking': True,
            'supports_reasoning': True,
            'supports_agent': True,
            'supports_tools': True,
            'supports_quantization': True,
            'supported_quantization': ['int8', 'int4'],
            'max_context_length': 32768,
            'max_context_with_yarn': 131072,
            'supports_yarn_scaling': True,
            'supports_soft_switches': True,  # /think and /no_think
            'multilingual': True,
            'num_languages': 100
        })
        
        return capabilities
    
    def get_inference_params(self) -> Dict[str, Any]:
        """Get default inference parameters for Qwen3."""
        return {
            'temperature': 0.6,  # Default for thinking mode
            'top_p': 0.95,
            'top_k': 20,
            'max_new_tokens': 32768,
            'do_sample': True,
            'repetition_penalty': 1.0,
            'presence_penalty': 0.0,  # Can be adjusted 0-2 for repetition control
            'enable_thinking': True  # Default enabled
        }
    
    def validate_model_files(self, model_path: str) -> Tuple[bool, Optional[str]]:
        """Validate Qwen3 model files."""
        # First do standard transformer validation
        valid, error = super().validate_model_files(model_path)
        if not valid:
            return valid, error
        
        # Check for Qwen3 specific files
        model_path_obj = Path(model_path)
        config_file = model_path_obj / "config.json"
        
        if config_file.exists():
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            # Verify it's a Qwen3 model
            model_type = config.get('model_type', '')
            if 'qwen3' not in model_type.lower() and 'qwen-3' not in model_type.lower():
                # Check architectures as fallback
                architectures = config.get('architectures', [])
                if not any('Qwen3' in arch for arch in architectures):
                    logger.warning("Model config doesn't indicate Qwen3 architecture")
        
        return True, None
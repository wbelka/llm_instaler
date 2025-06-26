"""Handler for specialized models that don't fit other categories.

This handler manages models like reasoning models (o1), code models,
and other specialized architectures.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from handlers.base import BaseHandler
from handlers.transformer import TransformerHandler

logger = logging.getLogger(__name__)


class SpecializedHandler(BaseHandler):
    """Handler for specialized model types."""
    
    def __init__(self, model_info: Dict[str, Any]):
        """Initialize specialized handler."""
        super().__init__(model_info)
        self.specialized_type = self._determine_specialized_type()
    
    def _determine_specialized_type(self) -> str:
        """Determine the specialized model type."""
        model_id = self.model_id.lower()
        tags = [t.lower() for t in self.model_info.get('tags', [])]
        config = self.model_info.get('config', {})
        
        # Check for reasoning models (including DeepSeek-R1)
        if ('o1' in model_id or 
            'reasoning' in tags or
            'deepseek-r1' in model_id or
            'deepseek_r1' in model_id or
            '-r1-' in model_id):
            return 'reasoning'
        
        # Check for code models
        elif any(code_kw in model_id for code_kw in ['code', 'codegen', 'starcoder', 'codellama']):
            return 'code'
        
        # Check for biomedical models
        elif any(bio_kw in model_id for bio_kw in ['bio', 'med', 'clinical', 'pubmed']):
            return 'biomedical'
        
        # Check for math models
        elif any(math_kw in model_id for math_kw in ['math', 'minerva', 'lean']):
            return 'mathematics'
        
        # Check for tool-use models
        elif 'tool' in tags or 'function-calling' in tags:
            return 'tool-use'
        
        else:
            return 'general-specialized'
    
    def get_dependencies(self) -> List[str]:
        """Get Python dependencies for specialized models."""
        # Start with transformer dependencies
        base_deps = [
            'torch>=2.0.0',
            'transformers>=4.30.0',
            'accelerate>=0.20.0',
            'tokenizers>=0.15.0'
        ]
        
        # Add specialized dependencies based on type
        if self.specialized_type == 'reasoning':
            base_deps.extend([
                'numpy',
                'scipy',
                'sympy'  # For mathematical reasoning
            ])
        
        elif self.specialized_type == 'code':
            base_deps.extend([
                'tree-sitter',  # For code parsing
                'pygments',     # For syntax highlighting
                'black',        # For code formatting
                'autopep8'
            ])
        
        elif self.specialized_type == 'biomedical':
            base_deps.extend([
                'biopython',
                'scikit-learn'
            ])
        
        elif self.specialized_type == 'mathematics':
            base_deps.extend([
                'sympy',
                'matplotlib',
                'latex2sympy2'
            ])
        
        return base_deps
    
    def get_system_dependencies(self) -> List[str]:
        """Get system dependencies for specialized models."""
        deps = []
        
        # CUDA for GPU acceleration
        if self.model_info.get('requires_gpu', True):
            deps.append('cuda>=11.7')
        
        # LaTeX for math models
        if self.specialized_type == 'mathematics':
            deps.extend(['texlive', 'dvipng'])
        
        return deps
    
    def analyze(self) -> 'ModelRequirements':
        """Analyze specialized model requirements."""
        from core.checker import ModelRequirements
        
        # Use transformer handler for base analysis
        transformer_handler = TransformerHandler(self.model_info)
        requirements = transformer_handler.analyze()
        
        # Update for specialized needs
        requirements.model_family = f"specialized-{self.specialized_type}"
        
        # Add specialized capabilities
        requirements.capabilities.update(self._get_specialized_capabilities())
        
        # Update dependencies
        requirements.base_dependencies = self.get_dependencies()
        
        # Specialized models might need more memory for complex reasoning
        if self.specialized_type == 'reasoning':
            for key in requirements.memory_requirements:
                requirements.memory_requirements[key] *= 1.5
        
        return requirements
    
    def _get_specialized_capabilities(self) -> Dict[str, Any]:
        """Get specialized model capabilities."""
        capabilities = {}
        
        if self.specialized_type == 'reasoning':
            capabilities.update({
                'supports_reasoning': True,
                'supports_chain_of_thought': True,
                'supports_self_reflection': True,
                'max_reasoning_tokens': 32768,
                'supports_structured_output': True
            })
        
        elif self.specialized_type == 'code':
            capabilities.update({
                'supports_code_completion': True,
                'supports_code_explanation': True,
                'supports_code_generation': True,
                'supported_languages': ['python', 'javascript', 'java', 'c++', 'go', 'rust'],
                'supports_syntax_highlighting': True
            })
        
        elif self.specialized_type == 'biomedical':
            capabilities.update({
                'supports_medical_qa': True,
                'supports_clinical_notes': True,
                'supports_drug_discovery': True,
                'requires_medical_disclaimer': True
            })
        
        elif self.specialized_type == 'mathematics':
            capabilities.update({
                'supports_latex': True,
                'supports_symbolic_math': True,
                'supports_proof_generation': True,
                'supports_step_by_step': True
            })
        
        elif self.specialized_type == 'tool-use':
            capabilities.update({
                'supports_function_calling': True,
                'supports_tool_use': True,
                'supports_json_mode': True,
                'supports_structured_generation': True
            })
        
        return capabilities
    
    def load_model(self, model_path: str, **kwargs):
        """Load specialized model."""
        # Most specialized models are transformer-based
        transformer_handler = TransformerHandler(self.model_info)
        model, tokenizer = transformer_handler.load_model(model_path, **kwargs)
        
        # Add any specialized configuration
        if self.specialized_type == 'reasoning':
            # Configure for reasoning
            if hasattr(model, 'config'):
                model.config.use_cache = True  # Important for long reasoning chains
                model.config.max_thinking_length = kwargs.get('max_thinking_tokens', 32768)
        
        return model, tokenizer
    
    def get_inference_params(self) -> Dict[str, Any]:
        """Get inference parameters for specialized models."""
        params = {
            'temperature': 0.7,
            'top_p': 0.95,
            'max_new_tokens': 1024,
            'do_sample': True
        }
        
        # Adjust for specialized types
        if self.specialized_type == 'reasoning':
            params.update({
                'temperature': 0.1,  # Lower temperature for reasoning
                'max_new_tokens': 4096,  # Longer outputs for reasoning
                'return_thinking': True,
                'max_thinking_tokens': 32768
            })
        
        elif self.specialized_type == 'code':
            params.update({
                'temperature': 0.3,  # Lower temperature for code
                'max_new_tokens': 2048,
                'stop_sequences': ['\n\n', '```', '</code>']
            })
        
        elif self.specialized_type == 'mathematics':
            params.update({
                'temperature': 0.2,  # Very low for precise math
                'max_new_tokens': 2048,
                'return_step_by_step': True
            })
        
        return params
    
    def get_training_params(self) -> Dict[str, Any]:
        """Get training parameters for specialized models."""
        # Use transformer base parameters
        transformer_handler = TransformerHandler(self.model_info)
        params = transformer_handler.get_training_params()
        
        # Adjust for specialized needs
        if self.specialized_type == 'reasoning':
            params['learning_rate'] *= 0.5  # Lower LR for reasoning models
            params['max_grad_norm'] = 0.5   # More conservative gradients
        
        return params
    
    def validate_model_files(self, model_path: str) -> Tuple[bool, Optional[str]]:
        """Validate specialized model files."""
        # Use transformer validation
        transformer_handler = TransformerHandler(self.model_info)
        return transformer_handler.validate_model_files(model_path)
    
    def generate_text(self, prompt: str = None, messages: List[Dict] = None,
                     model=None, tokenizer=None, **kwargs) -> Dict[str, Any]:
        """Generate text with specialized handling."""
        
        if self.specialized_type == 'reasoning':
            return self._generate_reasoning(prompt, messages, model, tokenizer, **kwargs)
        elif self.specialized_type == 'code':
            return self._generate_code(prompt, messages, model, tokenizer, **kwargs)
        else:
            # Use standard transformer generation
            transformer_handler = TransformerHandler(self.model_info)
            return transformer_handler.generate_text(prompt, messages, model, tokenizer, **kwargs)
    
    def _generate_reasoning(self, prompt: str = None, messages: List[Dict] = None,
                           model=None, tokenizer=None, **kwargs) -> Dict[str, Any]:
        """Generate text with reasoning (o1-style).
        
        Special handling for DeepSeek-R1 models:
        - These models tend to bypass thinking by outputting empty tags like "<think>\n\n</think>"
        - To fix this, we prepend "<think>\n" to the model's generation
        - This forces the model to engage in reasoning before providing answers
        - Special tokens are preserved (not skipped) during decoding
        """
        if not model or not tokenizer:
            raise ValueError("Model and tokenizer required for generation")
        
        # Check if this is a DeepSeek-R1 model
        is_deepseek_r1 = any(marker in self.model_id.lower() for marker in ['deepseek-r1', 'deepseek_r1', '-r1-'])
        
        if is_deepseek_r1:
            logger.info("DeepSeek-R1 model detected - applying thinking tag fix")
        
        # Prepare input
        if is_deepseek_r1:
            # For DeepSeek-R1, we'll prepend the thinking tag after generation
            # Don't add reasoning prompt to input
            pass
        else:
            # For other reasoning models, add reasoning prompt
            reasoning_prompt = "Let me think step by step about this problem.\n\n"
            
            if messages:
                # Add reasoning instruction to the last user message
                messages = messages.copy()
                if messages and messages[-1]['role'] == 'user':
                    messages[-1]['content'] = reasoning_prompt + messages[-1]['content']
            else:
                prompt = reasoning_prompt + (prompt or "")
        
        # Generate with thinking tokens
        max_thinking = kwargs.pop('max_thinking_tokens', 32768)
        kwargs['max_new_tokens'] = max_thinking + kwargs.get('max_new_tokens', 1024)
        
        # For DeepSeek-R1, ensure special tokens are not skipped
        if is_deepseek_r1:
            kwargs['skip_special_tokens'] = False
        
        # Use transformer generation
        transformer_handler = TransformerHandler(self.model_info)
        
        # For DeepSeek-R1, we need to handle the generation differently
        if is_deepseek_r1:
            # First, tokenize the input to get the input IDs
            if messages:
                if hasattr(tokenizer, 'apply_chat_template'):
                    text = tokenizer.apply_chat_template(messages, tokenize=False)
                else:
                    text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            else:
                text = prompt or ""
            
            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=kwargs.get('max_length', 2048))
            
            if hasattr(model, "device"):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Prepare the starting tokens for "<think>\n"
            think_tokens = tokenizer.encode("<think>\n", add_special_tokens=False, return_tensors="pt")
            if hasattr(model, "device"):
                think_tokens = think_tokens.to(model.device)
            
            # Generate with forced prefix
            import torch
            
            # Clear GPU cache before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            with torch.no_grad():
                # Concatenate input with think prefix
                input_ids = torch.cat([inputs['input_ids'], think_tokens], dim=1)
                attention_mask = torch.ones_like(input_ids)
                
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=kwargs.get('max_tokens', 512),
                    temperature=kwargs.get('temperature', 0.7),
                    top_p=kwargs.get('top_p', 0.9),
                    top_k=kwargs.get('top_k', 50),
                    do_sample=kwargs.get('temperature', 0.7) > 0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **{k: v for k, v in kwargs.items() if k in ['repetition_penalty', 'length_penalty']}
                )
            
            # Decode the full output (including special tokens)
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
            
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
            
            result = {
                'text': generated_text,
                'usage': usage
            }
        else:
            # For non-DeepSeek-R1 models, use normal generation
            result = transformer_handler.generate_text(prompt, messages, model, tokenizer, **kwargs)
        
        # Parse thinking and answer
        generated_text = result['text']
        
        # For DeepSeek-R1, parse the thinking tags
        if is_deepseek_r1:
            # Look for think/end think pattern
            import re
            think_pattern = r'<think>(.*?)</think>'
            matches = re.findall(think_pattern, generated_text, re.DOTALL)
            
            if matches:
                # Extract thinking content
                result['thinking'] = matches[0].strip()
                # Remove thinking tags from main text
                result['text'] = re.sub(think_pattern, '', generated_text, flags=re.DOTALL).strip()
            else:
                # If no proper tags found, check for other patterns
                if generated_text.startswith('<think>'):
                    # The model might not have closed the tag properly
                    parts = generated_text.split('</think>', 1)
                    if len(parts) == 2:
                        result['thinking'] = parts[0].replace('<think>', '').strip()
                        result['text'] = parts[1].strip()
                    else:
                        # Tag not closed, everything after <think> is thinking
                        result['thinking'] = generated_text.replace('<think>', '').strip()
                        result['text'] = ""
        else:
            # Simple parsing for other models
            if '\n\nAnswer:' in generated_text:
                thinking, answer = generated_text.split('\n\nAnswer:', 1)
                result['thinking'] = thinking.strip()
                result['text'] = answer.strip()
        
        return result
    
    def _generate_code(self, prompt: str = None, messages: List[Dict] = None,
                      model=None, tokenizer=None, **kwargs) -> Dict[str, Any]:
        """Generate code with proper formatting."""
        # Add code-specific parameters
        kwargs['temperature'] = kwargs.get('temperature', 0.3)
        kwargs['stop_sequences'] = kwargs.get('stop_sequences', ['\n\n', '```'])
        
        # Generate
        transformer_handler = TransformerHandler(self.model_info)
        result = transformer_handler.generate_text(prompt, messages, model, tokenizer, **kwargs)
        
        # Post-process code if needed
        code = result['text']
        
        # Extract code from markdown if present
        if '```' in code:
            import re
            code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', code, re.DOTALL)
            if code_blocks:
                result['code_blocks'] = code_blocks
                result['text'] = code_blocks[0]  # Return first code block as main result
        
        return result
    
    def get_supported_modes(self) -> List[str]:
        """Get supported modes for specialized models."""
        base_modes = ['auto']
        
        if self.specialized_type == 'reasoning':
            base_modes.extend(['reasoning', 'think', 'analyze', 'solve'])
        elif self.specialized_type == 'code':
            base_modes.extend(['code', 'complete', 'explain', 'fix', 'optimize'])
        elif self.specialized_type == 'mathematics':
            base_modes.extend(['solve', 'prove', 'calculate', 'explain'])
        elif self.specialized_type == 'tool-use':
            base_modes.extend(['tool', 'function', 'api'])
        else:
            base_modes.extend(['chat', 'instruct'])
        
        return base_modes
    
    def get_mode_descriptions(self) -> Dict[str, str]:
        """Get mode descriptions."""
        descriptions = {
            'auto': 'Automatic mode selection',
            'reasoning': 'Step-by-step reasoning mode',
            'think': 'Deep thinking mode',
            'analyze': 'Analysis mode',
            'solve': 'Problem solving mode',
            'code': 'Code generation mode',
            'complete': 'Code completion mode',
            'explain': 'Explanation mode',
            'fix': 'Bug fixing mode',
            'optimize': 'Code optimization mode',
            'prove': 'Mathematical proof mode',
            'calculate': 'Calculation mode',
            'tool': 'Tool use mode',
            'function': 'Function calling mode',
            'api': 'API interaction mode'
        }
        
        return {k: v for k, v in descriptions.items() if k in self.get_supported_modes()}
    
    def get_model_capabilities(self) -> Dict[str, Any]:
        """Get model capabilities."""
        # Start with transformer capabilities
        transformer_handler = TransformerHandler(self.model_info)
        capabilities = transformer_handler.get_model_capabilities()
        
        # Add specialized capabilities
        capabilities.update(self._get_specialized_capabilities())
        capabilities['specialized_type'] = self.specialized_type
        
        return capabilities
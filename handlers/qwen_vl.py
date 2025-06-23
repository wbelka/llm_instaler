"""Handler for Qwen Vision-Language models.

This handler manages Qwen2-VL and Qwen2.5-VL models which are
multimodal models supporting both text and image inputs.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import torch
import base64
from io import BytesIO
from PIL import Image

from handlers.base import BaseHandler
from handlers.multimodal import MultimodalHandler

logger = logging.getLogger(__name__)


class QwenVLHandler(MultimodalHandler):
    """Handler for Qwen Vision-Language models."""
    
    def __init__(self, model_info: Dict[str, Any]):
        """Initialize Qwen VL handler."""
        super().__init__(model_info)
        self.supports_video = 'video_token_id' in model_info.get('config', {})
    
    def _load_config(self, model_path: str) -> Dict[str, Any]:
        """Load model config.json."""
        import json
        from pathlib import Path
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def get_dependencies(self) -> List[str]:
        """Get Python dependencies for Qwen VL models."""
        base_deps = super().get_dependencies()
        
        # Add Qwen-specific dependencies
        qwen_deps = [
            'qwen-vl-utils>=0.0.2',  # Qwen VL utilities
            'torchvision>=0.15.0',   # For image preprocessing
        ]
        
        return base_deps + qwen_deps
    
    def load_model(self, model_path: str, **kwargs):
        """Load Qwen VL model with proper configuration.
        
        Args:
            model_path: Path to model directory
            **kwargs: Additional loading parameters
            
        Returns:
            Tuple of (model, processor)
        """
        import torch
        from transformers import AutoProcessor
        
        # For Qwen2.5-VL, we MUST use the correct auto class
        # The model has custom code that registers the right class
        config = self._load_config(model_path)
        
        # Import the auto classes
        from transformers import AutoModel, AutoModelForCausalLM
        
        # For Qwen2.5-VL, we need to use the specific model class
        if config.get('model_type') == 'qwen2_5_vl':
            try:
                # Try to import the specific Qwen2.5-VL model class (with underscore)
                from transformers import Qwen2_5_VLForConditionalGeneration
                model_class = Qwen2_5_VLForConditionalGeneration
                logger.info("Using Qwen2_5_VLForConditionalGeneration for Qwen2.5-VL")
            except ImportError:
                logger.info("Qwen2_5_VLForConditionalGeneration not found in transformers, trying fallback approaches")
                # Fallback: try to load the modeling file to register classes
                try:
                    import importlib.util
                    modeling_file = os.path.join(model_path, "modeling_qwen2_5_vl.py")
                    if os.path.exists(modeling_file):
                        spec = importlib.util.spec_from_file_location("modeling_qwen2_5_vl", modeling_file)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        logger.info("Loaded Qwen2.5-VL modeling file to register classes")
                        # After loading, try to get the class
                        if hasattr(module, 'Qwen2_5_VLForConditionalGeneration'):
                            model_class = module.Qwen2_5_VLForConditionalGeneration
                            logger.info("Using Qwen2_5_VLForConditionalGeneration from modeling file")
                        else:
                            # Use AutoModel with trust_remote_code to let it figure out the class
                            model_class = AutoModel
                            logger.info("Using AutoModel to auto-detect the correct class")
                except Exception as e:
                    logger.warning(f"Could not load modeling file: {e}")
                    # Final fallback: use AutoModel with trust_remote_code
                    model_class = AutoModel
                    logger.info("Using AutoModel as final fallback")
        else:
            model_class = AutoModelForCausalLM
            logger.info("Using AutoModelForCausalLM")
        
        # Extract parameters
        device = kwargs.get('device', 'auto')
        dtype = kwargs.get('dtype', 'auto')
        load_in_8bit = kwargs.get('load_in_8bit', False)
        load_in_4bit = kwargs.get('load_in_4bit', False)
        
        # Determine torch dtype
        if dtype == 'auto':
            # Qwen VL models typically use bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            elif torch.cuda.is_available():
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
        else:
            dtype_map = {
                'float32': torch.float32,
                'float16': torch.float16,
                'bfloat16': torch.bfloat16
            }
            torch_dtype = dtype_map.get(dtype, torch.bfloat16)
        
        # Model loading kwargs
        model_kwargs = {
            'torch_dtype': torch_dtype,
            'trust_remote_code': True,  # Required for Qwen VL models
            'low_cpu_mem_usage': True,
        }
        
        # Add quantization config if needed
        if load_in_8bit or load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=torch_dtype if load_in_4bit else None,
            )
        
        # Device mapping
        if device == 'auto':
            model_kwargs['device_map'] = 'auto'
        elif device != 'cpu':
            model_kwargs['device_map'] = {'': device}
        
        # Load model
        logger.info(f"Loading Qwen VL model from {model_path}")
        model = model_class.from_pretrained(model_path, **model_kwargs)
        
        # Load processor (handles both text and image)
        # For Qwen2.5-VL, we might need tokenizer instead of processor
        try:
            processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
        except Exception as e:
            logger.info(f"AutoProcessor failed, trying AutoTokenizer: {e}")
            from transformers import AutoTokenizer
            processor = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
        
        # Set processor attributes if not present
        if not hasattr(processor, 'tokenizer') and hasattr(processor, 'text_processor'):
            processor.tokenizer = processor.text_processor
        
        return model, processor
    
    def process_multimodal(self, text: str = None, images: List[Union[str, Image.Image]] = None,
                          model=None, processor=None, **kwargs) -> Dict[str, Any]:
        """Process multimodal inputs with Qwen VL model.
        
        Args:
            text: Text input
            images: List of images (base64 strings or PIL Images)
            model: Model instance
            processor: Processor instance
            **kwargs: Generation parameters
            
        Returns:
            Dictionary with generated text and metadata
        """
        if not model or not processor:
            raise ValueError("Model and processor required for multimodal processing")
        
        # Process images
        pil_images = []
        if images:
            for img in images:
                if isinstance(img, str):
                    # Decode base64 image
                    img_data = base64.b64decode(img)
                    pil_img = Image.open(BytesIO(img_data))
                    pil_images.append(pil_img)
                elif isinstance(img, Image.Image):
                    pil_images.append(img)
        
        # Format input for Qwen VL
        if pil_images:
            # Qwen VL expects a specific format for multimodal inputs
            if hasattr(processor, 'apply_chat_template'):
                # Use chat template if available
                messages = [{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': text or 'What is in this image?'}
                    ] + [{'type': 'image'} for _ in pil_images]
                }]
                
                text_input = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback format
                text_input = text or 'What is in this image?'
        else:
            text_input = text or ''
        
        # Process inputs
        # For Qwen2.5-VL with tokenizer, we need different handling
        if hasattr(processor, 'image_processor') or hasattr(processor, '__call__') and 'images' in processor.__call__.__code__.co_varnames:
            # Processor supports images directly
            inputs = processor(
                text=text_input,
                images=pil_images if pil_images else None,
                return_tensors='pt',
                padding=True
            )
        else:
            # Tokenizer only - need to handle images separately
            # For now, just process text
            logger.warning("Processor doesn't support images directly, processing text only")
            inputs = processor(
                text_input,
                return_tensors='pt',
                padding=True
            )
        
        # Move to device
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) if hasattr(v, 'to') else v 
                     for k, v in inputs.items()}
        
        # Generate
        # For Qwen2.5-VL, the model structure is complex
        # The loaded model might be the base model without generate capability
        
        # Check if model has generate method
        if hasattr(model, 'generate'):
            # Model has generate, use it directly
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get('max_tokens', 512),
                    temperature=kwargs.get('temperature', 0.7),
                    top_p=kwargs.get('top_p', 0.9),
                    do_sample=kwargs.get('temperature', 0.7) > 0,
                    pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor, 'tokenizer') else getattr(processor, 'pad_token_id', None),
                    eos_token_id=processor.tokenizer.eos_token_id if hasattr(processor, 'tokenizer') else getattr(processor, 'eos_token_id', None),
                )
        else:
            # Model doesn't have generate - this is likely the base VL model
            # For Qwen2.5-VL, we need to use the model's forward method with custom generation
            logger.error("Model doesn't have generate method. This handler needs updating for this model structure.")
            # Return empty response to let API handle it
            return {
                'text': '',
                'usage': {
                    'prompt_tokens': inputs.get('input_ids', inputs).shape[-1] if hasattr(inputs.get('input_ids', inputs), 'shape') else 0,
                    'completion_tokens': 0,
                    'total_tokens': inputs.get('input_ids', inputs).shape[-1] if hasattr(inputs.get('input_ids', inputs), 'shape') else 0
                }
            }
        
        # Decode output
        if hasattr(processor, 'decode'):
            generated_text = processor.decode(outputs[0], skip_special_tokens=True)
        elif hasattr(processor, 'tokenizer'):
            generated_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Remove input text from output if present
        if text_input and generated_text.startswith(text_input):
            generated_text = generated_text[len(text_input):].strip()
        
        return {
            'text': generated_text,
            'usage': {
                'prompt_tokens': inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0,
                'completion_tokens': outputs.shape[1] - inputs['input_ids'].shape[1] if 'input_ids' in inputs else outputs.shape[1],
                'total_tokens': outputs.shape[1]
            }
        }
    
    def process_image(self, image: Union[str, Image.Image], prompt: str = None,
                     model=None, processor=None, **kwargs) -> Dict[str, Any]:
        """Process single image with optional prompt.
        
        Args:
            image: Image as base64 string or PIL Image
            prompt: Optional text prompt
            model: Model instance
            processor: Processor instance
            **kwargs: Generation parameters
            
        Returns:
            Dictionary with analysis results
        """
        return self.process_multimodal(
            text=prompt,
            images=[image],
            model=model,
            processor=processor,
            **kwargs
        )
    
    def generate_text(self, prompt: str = None, messages: List[Dict] = None,
                     model=None, tokenizer=None, **kwargs) -> Dict[str, Any]:
        """Generate text using Qwen VL model.
        
        Args:
            prompt: Text prompt
            messages: Chat messages
            model: Model instance
            tokenizer: Tokenizer/processor instance
            **kwargs: Generation parameters
            
        Returns:
            Dictionary with generated text
        """
        # For text-only generation, use the processor
        processor = tokenizer  # In Qwen VL, we use processor instead of tokenizer
        
        if messages:
            # Apply chat template
            if hasattr(processor, 'apply_chat_template'):
                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Simple fallback
                text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        else:
            text = prompt or ""
        
        # Process text-only input
        inputs = processor(
            text=text,
            return_tensors='pt',
            padding=True
        )
        
        # Move to device
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) if hasattr(v, 'to') else v 
                     for k, v in inputs.items()}
        
        # Generate
        # For Qwen2.5-VL, the generate method might be on the language_model attribute
        generate_model = model
        if not hasattr(model, 'generate') and hasattr(model, 'language_model'):
            generate_model = model.language_model
        
        with torch.no_grad():
            outputs = generate_model.generate(
                **inputs,
                max_new_tokens=kwargs.get('max_tokens', 512),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                top_k=kwargs.get('top_k', 50),
                do_sample=kwargs.get('temperature', 0.7) > 0,
            )
        
        # Decode
        input_length = inputs['input_ids'].shape[1]
        generated_ids = outputs[0][input_length:]
        
        if hasattr(processor, 'decode'):
            generated_text = processor.decode(generated_ids, skip_special_tokens=True)
        elif hasattr(processor, 'tokenizer'):
            generated_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        else:
            generated_text = processor.batch_decode([generated_ids], skip_special_tokens=True)[0]
        
        return {
            'text': generated_text,
            'usage': {
                'prompt_tokens': input_length,
                'completion_tokens': generated_ids.shape[0],
                'total_tokens': outputs[0].shape[0]
            }
        }
    
    def get_model_capabilities(self) -> Dict[str, Any]:
        """Get Qwen VL model capabilities."""
        capabilities = super().get_model_capabilities()
        
        # Update for Qwen VL specific features
        capabilities.update({
            'supports_multiple_images': True,
            'supports_video': self.supports_video,
            'supports_ocr': True,
            'supports_grounding': True,  # Object detection/localization
            'max_image_size': 'dynamic',  # Qwen VL handles various sizes
            'supports_high_resolution': True,
        })
        
        return capabilities
    
    def get_supported_modes(self) -> List[str]:
        """Get supported modes for Qwen VL."""
        modes = ['auto', 'chat', 'vision', 'analyze', 'ocr', 'describe']
        if self.supports_video:
            modes.append('video')
        return modes
    
    def get_mode_descriptions(self) -> Dict[str, str]:
        """Get mode descriptions."""
        descriptions = {
            'auto': 'Automatic mode selection',
            'chat': 'Conversational chat mode',
            'vision': 'Image understanding mode',
            'analyze': 'Detailed image analysis',
            'ocr': 'Text extraction from images',
            'describe': 'Image description mode',
        }
        if self.supports_video:
            descriptions['video'] = 'Video understanding mode'
        return descriptions
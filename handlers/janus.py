"""Handler for Deepseek Janus multimodal models.

This module provides specific handling for Janus models which support
both text and image generation with special architecture.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
from io import BytesIO
import base64

from handlers.multimodal import MultimodalHandler

logger = logging.getLogger(__name__)


class JanusHandler(MultimodalHandler):
    """Handler specifically for Deepseek Janus models."""
    
    def __init__(self, model_info: Dict[str, Any]):
        """Initialize Janus handler.
        
        Args:
            model_info: Model information dictionary.
        """
        super().__init__(model_info)
        self.image_token_num = 576  # Standard for Janus
        self.img_size = 384
        self.patch_size = 16
    
    def get_supported_modes(self) -> List[str]:
        """Get list of supported generation modes."""
        return ['auto', 'chat', 'image', 'vision', 'multimodal', 'analyze']
    
    def get_mode_descriptions(self) -> Dict[str, str]:
        """Get descriptions for each supported mode."""
        return {
            'auto': 'Automatic mode selection',
            'chat': 'Text conversation mode',
            'image': 'Image generation from text',
            'vision': 'Image understanding and analysis',
            'multimodal': 'Combined text and image processing',
            'analyze': 'Detailed analysis mode'
        }
    
    def generate_text(self, prompt: str = None, messages: List[Dict] = None,
                     model=None, tokenizer=None, **kwargs) -> Dict[str, Any]:
        """Generate text response using Janus model.
        
        Args:
            prompt: Text prompt
            messages: Chat messages
            model: Model instance
            tokenizer: Tokenizer instance
            **kwargs: Generation parameters
            
        Returns:
            Dictionary with generated text
        """
        import torch
        
        if not model or not tokenizer:
            raise ValueError("Model and tokenizer required for text generation")
        
        # Prepare conversations
        conversations = []
        if messages:
            for msg in messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                conversations.append({"role": role, "content": msg["content"]})
        elif prompt:
            conversations.append({"role": "User", "content": prompt})
        
        # Apply chat template
        sft_format = tokenizer.apply_sft_template_for_multi_turn_prompts(
            conversations=conversations,
            sft_format=tokenizer.sft_format,
            system_prompt=kwargs.get('system_prompt', '')
        )
        
        # Tokenize
        input_ids = tokenizer(
            sft_format,
            return_tensors="pt",
            max_length=kwargs.get('max_length', 2048),
            truncation=True
        ).input_ids.to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=kwargs.get('max_tokens', 512),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                do_sample=kwargs.get('temperature', 0.7) > 0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode
        generated_ids = outputs[0][input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return {
            'text': generated_text,
            'usage': {
                'prompt_tokens': input_ids.shape[1],
                'completion_tokens': generated_ids.shape[0],
                'total_tokens': outputs[0].shape[0]
            }
        }
    
    def generate_image(self, prompt: str, negative_prompt: str = None,
                      model=None, tokenizer=None, **kwargs) -> Dict[str, Any]:
        """Generate image using Janus model.
        
        Args:
            prompt: Text description of image
            negative_prompt: Not used for Janus
            model: Model instance
            tokenizer: Tokenizer instance
            **kwargs: Generation parameters
            
        Returns:
            Dictionary with image data
        """
        import torch
        
        if not model or not tokenizer:
            raise ValueError("Model and tokenizer required for image generation")
        
        # Check model has required components
        if not hasattr(model, 'gen_head') or not hasattr(model, 'gen_vision_model'):
            raise ValueError("Model missing gen_head or gen_vision_model for image generation")
        
        # Clear CUDA cache before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            # Prepare prompt for image generation
            conversations = [{"role": "User", "content": prompt}]
            sft_format = tokenizer.apply_sft_template_for_multi_turn_prompts(
                conversations=conversations,
                sft_format=tokenizer.sft_format,
                system_prompt=""
            )
            gen_prompt = sft_format + tokenizer.image_start_tag
            
            # Tokenize prompt
            input_ids = tokenizer.tokenizer.encode(gen_prompt)
            input_ids = torch.LongTensor(input_ids)
            
            # Setup for CFG (Classifier-Free Guidance)
            temperature = kwargs.get('temperature', 1.0)
            cfg_weight = kwargs.get('guidance_scale', 5.0)
            
            # Create batch for conditional and unconditional generation
            batch_size = 2  # conditional + unconditional
            tokens = torch.zeros((batch_size, len(input_ids)), dtype=torch.long).to(model.device)
            
            # Set up tokens
            for i in range(batch_size):
                tokens[i, :] = input_ids
                # Make odd indices unconditional (padded)
                if i % 2 != 0:
                    pad_id = getattr(tokenizer, 'pad_token_id', 0) or \
                            getattr(tokenizer.tokenizer, 'pad_token_id', 0) or 0
                    tokens[i, 1:-1] = pad_id
            
            # Get initial embeddings
            inputs_embeds = model.language_model.get_input_embeddings()(tokens)
            
            # Generate image tokens
            generated_tokens = []
            outputs = None
            
            for i in range(self.image_token_num):
                # Forward pass
                outputs = model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    past_key_values=outputs.past_key_values if outputs else None
                )
                hidden_states = outputs.last_hidden_state
                
                # Get logits from generation head
                last_hidden = hidden_states[:, -1, :]
                logits = model.gen_head(last_hidden)
                
                # Apply classifier-free guidance
                if logits.shape[0] >= 2:
                    logit_cond = logits[0::2, :]  # Conditional
                    logit_uncond = logits[1::2, :]  # Unconditional
                    logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
                
                # Sample next token
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens.append(next_token[0, 0].item())
                
                # Prepare next input
                next_token_batch = next_token.repeat(2, 1).squeeze(-1)
                
                if hasattr(model, 'prepare_gen_img_embeds'):
                    img_embeds = model.prepare_gen_img_embeds(next_token_batch)
                    inputs_embeds = img_embeds.unsqueeze(dim=1)
                else:
                    inputs_embeds = model.language_model.get_input_embeddings()(next_token_batch).unsqueeze(1)
            
            # Decode tokens to image
            generated_tokens_tensor = torch.tensor(generated_tokens).unsqueeze(0).to(model.device)
            
            dec = model.gen_vision_model.decode_code(
                generated_tokens_tensor.to(dtype=torch.int),
                shape=[1, 8, self.img_size//self.patch_size, self.img_size//self.patch_size]
            )
            
            # Convert to numpy and rearrange dimensions
            dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
            
            # Normalize to 0-255 range
            dec = (dec * 127.5 + 128).clip(0, 255).astype(np.uint8)
            
            # Convert to PIL Image
            image = Image.fromarray(dec[0])
            
            # Convert to base64
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Clear CUDA cache after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                'image': img_base64,
                'metadata': {
                    'width': self.img_size,
                    'height': self.img_size,
                    'format': 'png',
                    'model': 'janus'
                }
            }
            
        except Exception as e:
            # Clear CUDA cache on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise e
    
    def process_multimodal(self, text: str = None, images: List[str] = None,
                          model=None, processor=None, **kwargs) -> Dict[str, Any]:
        """Process multimodal inputs with Janus.
        
        Args:
            text: Text input
            images: List of base64-encoded images
            model: Model instance
            processor: Processor instance
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with processed output
        """
        import torch
        
        logger.info(f"JanusHandler.process_multimodal called with text={text[:50] if text else None}..., images count={len(images) if images else 0}")
        
        if not model or not processor:
            raise ValueError("Model and processor required for multimodal processing")
        
        # Prepare conversations with images
        conversations = []
        
        if images:
            # Add image tokens for each image
            image_tokens = '<image>' * len(images)
            content = f"{image_tokens}\n{text}" if text else image_tokens
        else:
            content = text or ""
        
        conversations.append({"role": "User", "content": content})
        logger.info(f"Conversations prepared: {conversations}")
        
        # Process with Janus processor
        pil_images = []
        if images:
            for img_base64 in images:
                try:
                    img_data = base64.b64decode(img_base64)
                    pil_images.append(Image.open(BytesIO(img_data)))
                    logger.info(f"Decoded image successfully")
                except Exception as e:
                    logger.error(f"Error decoding image: {e}")
                    raise
        
        logger.info(f"Processing with {len(pil_images)} PIL images")
        
        # Use processor to prepare inputs
        try:
            # First check if we should use the processor directly or apply template
            if hasattr(processor, 'apply_sft_template_for_multi_turn_prompts'):
                # Apply Janus-specific template
                # For vision mode, don't use system prompt to keep it simple
                system_prompt = "" if kwargs.get('mode') == 'vision' else kwargs.get('system_prompt', '')
                sft_format = processor.apply_sft_template_for_multi_turn_prompts(
                    conversations=conversations,
                    sft_format=processor.sft_format,
                    system_prompt=system_prompt
                )
                logger.info(f"Applied SFT template: {sft_format[:100]}...")
            
            prepare_inputs = processor(
                conversations=conversations,
                images=pil_images if pil_images else None,
                force_batchify=True
            )
            logger.info("Processor prepared inputs successfully")
        except Exception as e:
            logger.error(f"Error preparing inputs: {e}")
            raise
        
        # Move to device
        prepare_inputs = prepare_inputs.to(model.device)
        logger.info(f"Inputs moved to device: {model.device}")
        
        try:
            inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
            logger.info("Input embeddings prepared")
        except Exception as e:
            logger.error(f"Error preparing embeddings: {e}")
            raise
        
        # Generate response
        logger.info("Starting generation...")
        with torch.no_grad():
            outputs = model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                max_new_tokens=kwargs.get('max_tokens', 512),
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                do_sample=kwargs.get('temperature', 0.7) > 0
            )
        logger.info(f"Generation complete, output shape: {outputs.shape}")
        
        # For Janus multimodal, we need to decode the full output
        # because input_ids might not match the actual input length with embeddings
        full_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Full generated text: '{full_text[:200]}'...")
        
        # Extract the response part
        # Janus format seems to include the prompt in the output
        if "User:" in full_text:
            # Find the last occurrence of the user prompt
            parts = full_text.split("User:")
            if len(parts) > 1:
                # Get everything after the last "User:" prompt
                response_part = parts[-1]
                # Remove the user's question from the response
                if "\n" in response_part:
                    lines = response_part.split("\n")
                    # Skip the first line (user's question) and join the rest
                    generated_text = "\n".join(lines[1:]).strip()
                else:
                    generated_text = response_part.strip()
            else:
                generated_text = full_text.strip()
        else:
            # If no "User:" marker, try to extract based on the input prompt
            if text and text in full_text:
                generated_text = full_text.split(text)[-1].strip()
            else:
                generated_text = full_text.strip()
        
        # Clean up any remaining format markers
        generated_text = generated_text.replace("Me:", "").strip()
        generated_text = generated_text.replace("Assistant:", "").strip()
        
        logger.info(f"Final extracted text: '{generated_text[:200]}'...")
        
        # Make sure we return the full text, not truncated
        return {
            'text': generated_text,
            'type': 'multimodal_response'
        }
    
    def get_model_capabilities(self) -> Dict[str, Any]:
        """Get Janus-specific capabilities."""
        capabilities = super().get_model_capabilities()
        capabilities.update({
            'supports_image_generation': True,
            'supports_multimodal': True,
            'supports_streaming': False,
            'input_modalities': ['text', 'image'],
            'output_modalities': ['text', 'image'],
            'special_features': ['classifier_free_guidance', 'image_understanding']
        })
        return capabilities
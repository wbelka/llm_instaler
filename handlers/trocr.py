"""Handler for TrOCR (Transformer-based Optical Character Recognition) models.

This module provides specific handling for TrOCR models, including model loading
and text recognition from images.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import io
import base64

from handlers.base import BaseHandler

logger = logging.getLogger(__name__)


class TrOCRHandler(BaseHandler):
    """Handler for TrOCR models."""
    
    def __init__(self, model_info: Dict[str, Any]):
        """Initialize handler."""
        super().__init__(model_info)
        self.model_type = "ocr"
        self.model_family = "vision-language"

    def get_dependencies(self) -> List[str]:
        """Get Python dependencies specific to TrOCR."""
        return [
            'torch>=2.0.0',
            'transformers>=4.30.0',
            'Pillow>=10.0.0',
        ]

    def get_system_dependencies(self) -> List[str]:
        """TrOCR typically doesn't have specific system dependencies beyond Python/CUDA."""
        return []

    def analyze(self) -> 'ModelRequirements':
        """Analyze model and return requirements."""
        from core.checker import ModelRequirements
        
        requirements = ModelRequirements()
        requirements.model_type = self.model_type
        requirements.model_family = self.model_family
        requirements.primary_library = "transformers"
        requirements.base_dependencies = self.get_dependencies()
        
        # TrOCR models are relatively small, but require some memory for image processing
        model_size_gb = 0.5 # Estimate for base-sized model
        requirements.memory_requirements = {
            "min": model_size_gb * 2, # 1GB
            "recommended": model_size_gb * 3, # 1.5GB
            "gpu_min": model_size_gb * 1.5, # 0.75GB
            "gpu_recommended": model_size_gb * 2 # 1GB
        }
        
        return requirements

    def load_model(self, model_path: str, **kwargs):
        """Load TrOCR model and processor."""
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        
        logger.info(f"Loading TrOCR model from {model_path}")
        
        processor = TrOCRProcessor.from_pretrained(model_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_path)
        
        # Move model to device if specified
        if 'device' in kwargs and kwargs['device'] != "auto":
            device = kwargs['device']
            if device == "mps": # Apple Silicon MPS
                import torch
                if torch.backends.mps.is_available():
                    model.to("mps")
                    logger.info("TrOCR model moved to MPS device.")
                else:
                    logger.warning("MPS not available, falling back to CPU.")
                    model.to("cpu")
            elif device == "cuda": # CUDA GPU
                import torch
                if torch.cuda.is_available():
                    model.to("cuda")
                    logger.info("TrOCR model moved to CUDA device.")
                else:
                    logger.warning("CUDA not available, falling back to CPU.")
                    model.to("cpu")
            else:
                model.to("cpu")
                logger.info("TrOCR model moved to CPU.")
        
        return model, processor

    def get_inference_params(self) -> Dict[str, Any]:
        """Get default inference parameters for TrOCR."""
        return {
            'max_tokens': 128, # Max length of recognized text
            'temperature': 0.0, # OCR is deterministic
            'do_sample': False,
        }

    def get_training_params(self) -> Dict[str, Any]:
        """TrOCR is typically fine-tuned, not trained from scratch."""
        return {}

    def validate_model_files(self, model_path: str) -> Tuple[bool, Optional[str]]:
        """Validate TrOCR model files exist."""
        from pathlib import Path
        
        model_path = Path(model_path)
        required_files = ['preprocessor_config.json', 'config.json', 'pytorch_model.bin']
        
        for file in required_files:
            if not (model_path / file).exists():
                return False, f"Missing required TrOCR file: {file}"
        
        return True, None

    def process_multimodal(self, text: str = None, images: List[str] = None,
                          model=None, processor=None, **kwargs) -> Dict[str, Any]:
        """Process image for OCR using TrOCR model."""
        if not images or len(images) == 0:
            raise ValueError("No image data provided for TrOCR processing.")
        
        # TrOCR processes one image at a time for OCR
        # We take the first image provided
        image_data_base64 = images[0]
        
        try:
            # Decode base64 image data
            image_bytes = base64.b64decode(image_data_base64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid image data: {e}")

        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        
        # Move pixel_values to the same device as the model
        if hasattr(model, 'device'):
            pixel_values = pixel_values.to(model.device)

        generated_ids = model.generate(pixel_values, 
                                       max_length=kwargs.get('max_tokens', 128),
                                       temperature=kwargs.get('temperature', 0.0),
                                       do_sample=kwargs.get('do_sample', False))
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return {
            'text': generated_text,
            'type': 'ocr_result',
            'usage': {
                'prompt_tokens': pixel_values.shape[1], # Number of image patches
                'completion_tokens': len(generated_ids[0]),
                'total_tokens': pixel_values.shape[1] + len(generated_ids[0])
            }
        }

    def get_supported_modes(self) -> List[str]:
        """TrOCR supports an 'ocr' mode."""
        return ['auto', 'ocr']

    def get_mode_descriptions(self) -> Dict[str, str]:
        """Descriptions for TrOCR modes."""
        return {
            'auto': 'Automatic mode selection',
            'ocr': 'Optical Character Recognition from image',
        }

    def get_model_capabilities(self) -> Dict[str, Any]:
        """Declare TrOCR model capabilities."""
        return {
            'supports_streaming': False,
            'supports_system_prompt': False,
            'supports_multimodal': True,
            'max_context_length': None, # Not applicable for OCR
            'input_modalities': ['image'],
            'output_modalities': ['text'],
            'supported_modes': self.get_supported_modes(),
            'ocr': True
        }

    def get_installation_notes(self) -> Dict[str, str]:
        """No special installation notes for TrOCR."""
        return {}
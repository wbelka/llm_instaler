"""Handler for embedding models.

This handler manages models that generate embeddings for text, images,
or multimodal inputs (CLIP, sentence transformers, etc.).
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import torch

from handlers.base import BaseHandler

logger = logging.getLogger(__name__)


class EmbeddingHandler(BaseHandler):
    """Handler for embedding models."""
    
    def get_dependencies(self) -> List[str]:
        """Get Python dependencies for embedding models."""
        base_deps = [
            'torch>=2.0.0',
            'transformers>=4.30.0',
            'numpy',
            'scikit-learn>=1.0.0'  # For similarity computations
        ]
        
        # Model-specific dependencies
        model_id = self.model_id.lower()
        
        if 'sentence-transformers' in model_id or 'sentence' in str(self.model_info.get('tags', [])):
            base_deps.append('sentence-transformers>=2.2.0')
        
        if 'clip' in model_id:
            base_deps.extend([
                'pillow>=9.0.0',
                'ftfy',
                'regex'
            ])
        
        if 'instructor' in model_id:
            base_deps.append('InstructorEmbedding')
        
        return base_deps
    
    def get_system_dependencies(self) -> List[str]:
        """Get system dependencies for embedding models."""
        deps = []
        
        # CUDA for GPU acceleration (optional but recommended)
        if self.model_info.get('requires_gpu', False):
            deps.append('cuda>=11.7')
        
        return deps
    
    def analyze(self) -> 'ModelRequirements':
        """Analyze embedding model requirements."""
        from core.checker import ModelRequirements
        
        requirements = ModelRequirements()
        requirements.model_type = self.model_type
        requirements.model_family = self._determine_embedding_family()
        requirements.primary_library = self._determine_primary_library()
        requirements.base_dependencies = self.get_dependencies()
        
        # Memory requirements (embedding models are usually smaller)
        model_size_gb = self._estimate_model_size()
        requirements.disk_space_gb = model_size_gb * 2
        requirements.memory_requirements = {
            "min": max(2, model_size_gb * 2),
            "recommended": max(4, model_size_gb * 3),
            "gpu_min": max(2, model_size_gb * 1.5),
            "gpu_recommended": max(4, model_size_gb * 2)
        }
        
        # Capabilities
        requirements.capabilities = self.get_model_capabilities()
        
        return requirements
    
    def _determine_embedding_family(self) -> str:
        """Determine the embedding model family."""
        model_id = self.model_id.lower()
        tags = [t.lower() for t in self.model_info.get('tags', [])]
        
        if 'clip' in model_id:
            return 'multimodal-embedding'
        elif 'sentence-transformers' in tags or 'sentence' in model_id:
            return 'text-embedding'
        elif 'instructor' in model_id:
            return 'instruction-embedding'
        elif any(modal in model_id for modal in ['multimodal', 'vision-language']):
            return 'multimodal-embedding'
        else:
            return 'text-embedding'
    
    def _determine_primary_library(self) -> str:
        """Determine the primary library for the model."""
        model_id = self.model_id.lower()
        tags = self.model_info.get('tags', [])
        
        if 'sentence-transformers' in tags:
            return 'sentence-transformers'
        elif 'clip' in model_id:
            return 'clip'
        else:
            return 'transformers'
    
    def _estimate_model_size(self) -> float:
        """Estimate model size in GB."""
        if 'model_size_gb' in self.model_info:
            return self.model_info['model_size_gb']
        
        # Embedding models are typically smaller
        model_id = self.model_id.lower()
        
        if 'large' in model_id:
            return 1.5
        elif 'base' in model_id:
            return 0.5
        elif 'small' in model_id or 'mini' in model_id:
            return 0.2
        else:
            return 0.5  # Default
    
    def load_model(self, model_path: str, device: str = 'auto', dtype: str = 'auto',
                   load_in_8bit: bool = False, load_in_4bit: bool = False, **kwargs):
        """Load embedding model with optimal settings.
        
        Args:
            model_path: Path to model files.
            device: Device to load on.
            dtype: Data type for model.
            load_in_8bit: Whether to use 8-bit quantization.
            load_in_4bit: Whether to use 4-bit quantization.
            **kwargs: Additional loading arguments.
            
        Returns:
            Tuple of (model, tokenizer/processor).
        """
        primary_lib = self._determine_primary_library()
        
        # Pass quantization parameters to underlying loaders
        load_kwargs = kwargs.copy()
        load_kwargs.update({
            'device': device,
            'dtype': dtype,
            'load_in_8bit': load_in_8bit,
            'load_in_4bit': load_in_4bit
        })
        
        if primary_lib == 'sentence-transformers':
            return self._load_sentence_transformer(model_path, **load_kwargs)
        elif primary_lib == 'clip':
            return self._load_clip_model(model_path, **load_kwargs)
        else:
            return self._load_transformers_embedding(model_path, **load_kwargs)
    
    def _load_sentence_transformer(self, model_path: str, **kwargs):
        """Load sentence transformer model."""
        from sentence_transformers import SentenceTransformer
        
        device = kwargs.get('device', 'auto')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model = SentenceTransformer(model_path, device=device)
        
        return model, None
    
    def _load_clip_model(self, model_path: str, **kwargs):
        """Load CLIP model."""
        from transformers import CLIPModel, CLIPProcessor
        
        device = kwargs.get('device', 'auto')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model = CLIPModel.from_pretrained(model_path).to(device)
        processor = CLIPProcessor.from_pretrained(model_path)
        
        model.eval()
        
        return model, processor
    
    def _load_transformers_embedding(self, model_path: str, **kwargs):
        """Load embedding model using transformers."""
        from transformers import AutoModel, AutoTokenizer
        
        device = kwargs.get('device', 'auto')
        dtype = kwargs.get('dtype', 'auto')
        load_in_8bit = kwargs.get('load_in_8bit', False)
        load_in_4bit = kwargs.get('load_in_4bit', False)
        
        # Use base handler's quantization config
        quant_config, torch_dtype = self.get_quantization_config(dtype, load_in_8bit, load_in_4bit)
        
        # Prepare model loading arguments
        model_kwargs = {
            'pretrained_model_name_or_path': model_path,
            'torch_dtype': torch_dtype,
            'trust_remote_code': self.model_info.get('trust_remote_code', False),
            'low_cpu_mem_usage': True
        }
        
        # Merge quantization config
        model_kwargs.update(quant_config)
        
        # Add Flash Attention 2 support
        use_flash_attention_2 = kwargs.get('use_flash_attention_2', False)
        if use_flash_attention_2 and not (load_in_8bit or load_in_4bit):
            try:
                model_kwargs['attn_implementation'] = 'flash_attention_2'
                logger.info("Using Flash Attention 2 for embedding model")
            except Exception as e:
                logger.warning(f"Flash Attention 2 not available: {e}")
                model_kwargs['attn_implementation'] = 'eager'
        
        # Handle device placement
        if device == 'auto':
            if load_in_8bit or load_in_4bit:
                model_kwargs['device_map'] = 'auto'
            else:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model
        model = AutoModel.from_pretrained(**model_kwargs)
        
        # Move to device if not using quantization
        if not (load_in_8bit or load_in_4bit) and device != 'auto':
            model = model.to(device)
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.model_info.get('trust_remote_code', False)
        )
        
        model.eval()
        
        return model, tokenizer
    
    def get_inference_params(self) -> Dict[str, Any]:
        """Get default inference parameters."""
        return {
            'batch_size': 32,
            'normalize_embeddings': True,
            'convert_to_numpy': True,
            'show_progress_bar': False
        }
    
    def get_training_params(self) -> Dict[str, Any]:
        """Get default training parameters."""
        return {
            'learning_rate': 2e-5,
            'batch_size': 16,
            'num_epochs': 10,
            'warmup_steps': 500,
            'evaluation_steps': 100
        }
    
    def validate_model_files(self, model_path: str) -> Tuple[bool, Optional[str]]:
        """Validate embedding model files."""
        model_path = Path(model_path)
        
        # Check for config
        if not (model_path / 'config.json').exists():
            # Some sentence-transformers models use config_sentence_transformers.json
            if not (model_path / 'config_sentence_transformers.json').exists():
                return False, "Missing config file"
        
        # Check for model weights
        has_weights = any(
            model_path.glob(pattern)
            for pattern in ['*.bin', '*.safetensors', '*.pt', '*.pth']
        )
        
        if not has_weights:
            return False, "No model weight files found"
        
        return True, None
    
    def embed_text(self, texts: List[str], model=None, tokenizer=None,
                   **kwargs) -> Dict[str, Any]:
        """Generate text embeddings."""
        if not model:
            raise ValueError("Model required for embedding generation")
        
        normalize = kwargs.get('normalize', True)
        convert_to_numpy = kwargs.get('convert_to_numpy', True)
        
        # Handle different model types
        if hasattr(model, 'encode'):
            # Sentence transformer style
            embeddings = model.encode(
                texts,
                normalize_embeddings=normalize,
                show_progress_bar=kwargs.get('show_progress_bar', False),
                batch_size=kwargs.get('batch_size', 32)
            )
        else:
            # Transformers style
            if not tokenizer:
                raise ValueError("Tokenizer required for this model type")
            
            # Tokenize
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=kwargs.get('max_length', 512)
            )
            
            if hasattr(model, 'device'):
                encoded = {k: v.to(model.device) for k, v in encoded.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = model(**encoded)
            
            # Pool embeddings (mean pooling by default)
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                # Mean pooling over sequence
                attention_mask = encoded['attention_mask']
                embeddings = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask
            
            # Normalize if requested
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            # Convert to numpy if requested
            if convert_to_numpy:
                embeddings = embeddings.cpu().numpy()
        
        return {
            'embeddings': embeddings,
            'shape': embeddings.shape,
            'dtype': str(embeddings.dtype)
        }
    
    def embed_image(self, images: List[str], model=None, processor=None,
                   **kwargs) -> Dict[str, Any]:
        """Generate image embeddings."""
        import base64
        from PIL import Image
        from io import BytesIO
        
        if not model:
            raise ValueError("Model required for embedding generation")
        
        # Decode images
        pil_images = []
        for img_base64 in images:
            img_data = base64.b64decode(img_base64)
            pil_images.append(Image.open(BytesIO(img_data)))
        
        # Process based on model type
        if hasattr(model, 'encode_image'):
            # CLIP-style model
            if not processor:
                raise ValueError("Processor required for image encoding")
            
            inputs = processor(images=pil_images, return_tensors="pt")
            if hasattr(model, 'device'):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                embeddings = model.encode_image(**inputs)
            
            if kwargs.get('normalize', True):
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            if kwargs.get('convert_to_numpy', True):
                embeddings = embeddings.cpu().numpy()
        
        else:
            raise NotImplementedError("Image embedding not supported for this model type")
        
        return {
            'embeddings': embeddings,
            'shape': embeddings.shape,
            'dtype': str(embeddings.dtype)
        }
    
    def embed_multimodal(self, text: str = None, image: str = None,
                        model=None, processor=None, **kwargs) -> Dict[str, Any]:
        """Generate multimodal embeddings (e.g., CLIP)."""
        import base64
        from PIL import Image as PILImage
        from io import BytesIO
        
        if not model or not processor:
            raise ValueError("Model and processor required for multimodal embedding")
        
        inputs = {}
        
        # Process text if provided
        if text:
            text_inputs = processor(text=[text], return_tensors="pt", padding=True)
            inputs.update(text_inputs)
        
        # Process image if provided
        if image:
            img_data = base64.b64decode(image)
            pil_image = PILImage.open(BytesIO(img_data))
            image_inputs = processor(images=pil_image, return_tensors="pt")
            inputs.update(image_inputs)
        
        if not inputs:
            raise ValueError("Either text or image must be provided")
        
        # Move to device
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate embeddings
        embeddings = {}
        with torch.no_grad():
            if text and hasattr(model, 'encode_text'):
                text_features = model.encode_text(inputs['input_ids'])
                if kwargs.get('normalize', True):
                    text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)
                embeddings['text'] = text_features.cpu().numpy() if kwargs.get('convert_to_numpy', True) else text_features
            
            if image and hasattr(model, 'encode_image'):
                image_features = model.encode_image(inputs['pixel_values'])
                if kwargs.get('normalize', True):
                    image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)
                embeddings['image'] = image_features.cpu().numpy() if kwargs.get('convert_to_numpy', True) else image_features
        
        return embeddings
    
    def get_supported_modes(self) -> List[str]:
        """Get supported modes."""
        model_family = self._determine_embedding_family()
        
        if model_family == 'multimodal-embedding':
            return ['text', 'image', 'multimodal', 'auto']
        else:
            return ['text', 'auto']
    
    def get_mode_descriptions(self) -> Dict[str, str]:
        """Get mode descriptions."""
        return {
            'auto': 'Automatic mode selection',
            'text': 'Text embedding generation',
            'image': 'Image embedding generation',
            'multimodal': 'Combined text and image embedding'
        }
    
    def get_model_capabilities(self) -> Dict[str, Any]:
        """Get model capabilities."""
        capabilities = super().get_model_capabilities()
        model_family = self._determine_embedding_family()
        
        if model_family == 'multimodal-embedding':
            capabilities.update({
                'input_modalities': ['text', 'image'],
                'output_modalities': ['embedding'],
                'supports_similarity': True,
                'embedding_dimension': self._get_embedding_dimension()
            })
        else:
            capabilities.update({
                'input_modalities': ['text'],
                'output_modalities': ['embedding'],
                'supports_similarity': True,
                'embedding_dimension': self._get_embedding_dimension()
            })
        
        return capabilities
    
    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension from model config."""
        config = self.model_info.get('config', {})
        
        # Try different config keys
        dim = config.get('hidden_size',
                        config.get('embed_dim',
                                  config.get('d_model',
                                            config.get('embedding_size', 768))))
        
        return dim
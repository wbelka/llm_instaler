"""Handler for computer vision models.

This handler manages models for image classification, object detection,
segmentation, and other vision tasks.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from handlers.base import BaseHandler

logger = logging.getLogger(__name__)


class VisionHandler(BaseHandler):
    """Handler for computer vision models."""
    
    def get_dependencies(self) -> List[str]:
        """Get Python dependencies for vision models."""
        base_deps = [
            'torch>=2.0.0',
            'torchvision>=0.15.0',
            'pillow>=9.0.0',
            'opencv-python>=4.7.0',
            'numpy',
            'transformers>=4.30.0'
        ]
        
        # Model-specific dependencies
        model_id = self.model_id.lower()
        
        if 'yolo' in model_id:
            base_deps.append('ultralytics>=8.0.0')
        elif 'detectron' in model_id:
            base_deps.append('detectron2')
        elif 'mmdet' in model_id:
            base_deps.extend(['mmdet', 'mmcv'])
        elif 'timm' in str(self.model_info.get('tags', [])):
            base_deps.append('timm>=0.9.0')
        
        return base_deps
    
    def get_system_dependencies(self) -> List[str]:
        """Get system dependencies for vision processing."""
        deps = []
        
        # OpenCV dependencies
        deps.extend([
            'libgl1-mesa-glx',
            'libglib2.0-0',
            'libsm6',
            'libxext6',
            'libxrender-dev',
            'libgomp1'
        ])
        
        # CUDA for GPU acceleration
        if self.model_info.get('requires_gpu', True):
            deps.append('cuda>=11.7')
        
        return deps
    
    def analyze(self) -> 'ModelRequirements':
        """Analyze vision model requirements."""
        from core.checker import ModelRequirements
        
        requirements = ModelRequirements()
        requirements.model_type = self.model_type
        requirements.model_family = self._determine_vision_family()
        requirements.primary_library = self._determine_primary_library()
        requirements.base_dependencies = self.get_dependencies()
        
        # Memory requirements
        model_size_gb = self._estimate_model_size()
        requirements.disk_space_gb = model_size_gb * 2
        requirements.memory_requirements = {
            "min": max(4, model_size_gb * 2),
            "recommended": max(8, model_size_gb * 3),
            "gpu_min": max(4, model_size_gb * 1.5),
            "gpu_recommended": max(8, model_size_gb * 2)
        }
        
        # Capabilities
        requirements.capabilities = self.get_model_capabilities()
        
        return requirements
    
    def _determine_vision_family(self) -> str:
        """Determine the vision model family."""
        model_id = self.model_id.lower()
        config = self.model_info.get('config', {})
        
        # Check model architecture
        arch = config.get('architectures', [''])[0].lower() if 'architectures' in config else ''
        
        if 'classification' in model_id or 'classifier' in arch:
            return 'image-classification'
        elif 'detection' in model_id or 'detector' in arch or 'yolo' in model_id:
            return 'object-detection'
        elif 'segmentation' in model_id or 'segment' in arch:
            return 'image-segmentation'
        elif 'depth' in model_id:
            return 'depth-estimation'
        elif 'pose' in model_id:
            return 'pose-estimation'
        else:
            return 'image-feature-extraction'
    
    def _determine_primary_library(self) -> str:
        """Determine the primary library for the model."""
        model_id = self.model_id.lower()
        
        if 'yolo' in model_id:
            return 'ultralytics'
        elif 'detectron' in model_id:
            return 'detectron2'
        elif 'mmdet' in model_id:
            return 'mmdetection'
        else:
            return 'transformers'
    
    def _estimate_model_size(self) -> float:
        """Estimate model size in GB."""
        if 'model_size_gb' in self.model_info:
            return self.model_info['model_size_gb']
        
        # Estimate based on architecture
        config = self.model_info.get('config', {})
        arch = config.get('architectures', [''])[0] if 'architectures' in config else ''
        
        # Common vision model sizes
        if 'resnet50' in arch.lower():
            return 0.1
        elif 'resnet101' in arch.lower():
            return 0.2
        elif 'efficientnet' in arch.lower():
            if 'b0' in arch.lower():
                return 0.02
            elif 'b7' in arch.lower():
                return 0.3
            else:
                return 0.1
        elif 'vit' in arch.lower():
            if 'large' in arch.lower():
                return 1.2
            elif 'base' in arch.lower():
                return 0.35
            else:
                return 0.1
        else:
            return 0.5  # Default
    
    def load_model(self, model_path: str, **kwargs):
        """Load vision model with optimal settings."""
        primary_lib = self._determine_primary_library()
        
        if primary_lib == 'ultralytics':
            return self._load_yolo_model(model_path, **kwargs)
        elif primary_lib == 'detectron2':
            return self._load_detectron2_model(model_path, **kwargs)
        else:
            return self._load_transformers_vision(model_path, **kwargs)
    
    def _load_yolo_model(self, model_path: str, **kwargs):
        """Load YOLO model."""
        from ultralytics import YOLO
        
        model = YOLO(model_path)
        return model, None
    
    def _load_detectron2_model(self, model_path: str, **kwargs):
        """Load Detectron2 model."""
        import torch
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        
        cfg = get_cfg()
        cfg.merge_from_file(str(Path(model_path) / "config.yaml"))
        cfg.MODEL.WEIGHTS = str(Path(model_path) / "model_final.pth")
        cfg.MODEL.DEVICE = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        predictor = DefaultPredictor(cfg)
        return predictor, None
    
    def _load_transformers_vision(self, model_path: str, **kwargs):
        """Load vision model using transformers."""
        import torch
        from transformers import AutoModelForImageClassification, AutoImageProcessor
        
        device = kwargs.get('device', 'auto')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model
        model = AutoModelForImageClassification.from_pretrained(
            model_path,
            trust_remote_code=self.model_info.get('trust_remote_code', False)
        ).to(device)
        
        # Load processor
        processor = AutoImageProcessor.from_pretrained(model_path)
        
        model.eval()
        
        return model, processor
    
    def get_inference_params(self) -> Dict[str, Any]:
        """Get default inference parameters."""
        model_family = self._determine_vision_family()
        
        params = {
            'confidence_threshold': 0.5,
            'nms_threshold': 0.4,
            'max_detections': 100
        }
        
        if model_family == 'object-detection':
            params.update({
                'input_size': 640,  # Common for YOLO
                'iou_threshold': 0.45
            })
        elif model_family == 'image-segmentation':
            params.update({
                'mask_threshold': 0.5,
                'return_masks': True
            })
        
        return params
    
    def get_training_params(self) -> Dict[str, Any]:
        """Get default training parameters."""
        return {
            'learning_rate': 1e-4,
            'batch_size': 16,
            'num_epochs': 100,
            'warmup_epochs': 3,
            'weight_decay': 0.0005,
            'momentum': 0.937,
            'optimizer': 'AdamW'
        }
    
    def validate_model_files(self, model_path: str) -> Tuple[bool, Optional[str]]:
        """Validate vision model files."""
        model_path = Path(model_path)
        primary_lib = self._determine_primary_library()
        
        if primary_lib == 'ultralytics':
            # Check for YOLO model file
            yolo_files = list(model_path.glob('*.pt')) + list(model_path.glob('*.yaml'))
            if not yolo_files:
                return False, "No YOLO model files found (.pt or .yaml)"
        
        elif primary_lib == 'detectron2':
            # Check for Detectron2 files
            if not (model_path / 'config.yaml').exists():
                return False, "Missing config.yaml for Detectron2"
            if not (model_path / 'model_final.pth').exists():
                return False, "Missing model_final.pth for Detectron2"
        
        else:
            # Check for transformers files
            if not (model_path / 'config.json').exists():
                return False, "Missing config.json"
            
            # Check for weights
            has_weights = any(
                model_path.glob(pattern)
                for pattern in ['*.bin', '*.safetensors', '*.pt', '*.pth']
            )
            if not has_weights:
                return False, "No model weight files found"
        
        return True, None
    
    def process_image(self, image: str, task: str = "classify",
                     model=None, processor=None, **kwargs) -> Dict[str, Any]:
        """Process image for various vision tasks."""
        import base64
        from PIL import Image
        from io import BytesIO
        import numpy as np
        import torch
        
        # Decode image from base64
        img_data = base64.b64decode(image)
        pil_image = Image.open(BytesIO(img_data))
        
        model_family = self._determine_vision_family()
        
        if task == "classify" or model_family == 'image-classification':
            # Image classification
            if processor:
                inputs = processor(images=pil_image, return_tensors="pt")
                if hasattr(model, 'device'):
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Get predictions
                logits = outputs.logits
                predictions = torch.nn.functional.softmax(logits, dim=-1)
                
                # Get top-k predictions
                top_k = kwargs.get('top_k', 5)
                values, indices = torch.topk(predictions[0], top_k)
                
                results = []
                for i in range(top_k):
                    label = model.config.id2label.get(indices[i].item(), f"Class {indices[i].item()}")
                    score = values[i].item()
                    results.append({'label': label, 'score': score})
                
                return {'predictions': results, 'task': 'classification'}
            
            else:
                raise ValueError("Processor required for image classification")
        
        elif task == "detect" or model_family == 'object-detection':
            # Object detection
            if hasattr(model, 'predict'):
                # YOLO-style model
                results = model.predict(pil_image, **kwargs)
                
                detections = []
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            detection = {
                                'bbox': box.xyxy[0].tolist(),
                                'confidence': box.conf[0].item(),
                                'class': model.names[int(box.cls[0])]
                            }
                            detections.append(detection)
                
                return {'detections': detections, 'task': 'detection'}
            
            else:
                raise NotImplementedError(f"Detection not implemented for this model type")
        
        elif task == "segment" or model_family == 'image-segmentation':
            # Image segmentation
            if processor:
                inputs = processor(images=pil_image, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model(**inputs)
                
                # Process segmentation masks
                if hasattr(outputs, 'logits'):
                    masks = outputs.logits
                    # Convert to numpy and encode as base64
                    masks_np = masks.cpu().numpy()
                    
                    # Encode masks
                    encoded_masks = []
                    for mask in masks_np[0]:
                        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
                        buffered = BytesIO()
                        mask_pil.save(buffered, format="PNG")
                        mask_base64 = base64.b64encode(buffered.getvalue()).decode()
                        encoded_masks.append(mask_base64)
                    
                    return {'masks': encoded_masks, 'task': 'segmentation'}
            
            else:
                raise ValueError("Processor required for segmentation")
        
        else:
            raise ValueError(f"Unknown vision task: {task}")
    
    def get_supported_modes(self) -> List[str]:
        """Get supported modes."""
        model_family = self._determine_vision_family()
        
        if model_family == 'image-classification':
            return ['classify', 'auto']
        elif model_family == 'object-detection':
            return ['detect', 'auto']
        elif model_family == 'image-segmentation':
            return ['segment', 'auto']
        else:
            return ['auto']
    
    def get_mode_descriptions(self) -> Dict[str, str]:
        """Get mode descriptions."""
        return {
            'auto': 'Automatic mode selection',
            'classify': 'Image classification',
            'detect': 'Object detection',
            'segment': 'Image segmentation'
        }
    
    def get_model_capabilities(self) -> Dict[str, Any]:
        """Get model capabilities."""
        capabilities = super().get_model_capabilities()
        model_family = self._determine_vision_family()
        
        capabilities.update({
            'supports_streaming': False,
            'supports_batch_inference': True,
            'input_modalities': ['image'],
            'output_modalities': ['text', 'bbox', 'mask'],
            'image_formats': ['jpg', 'png', 'bmp', 'webp'],
            'task': model_family
        })
        
        return capabilities
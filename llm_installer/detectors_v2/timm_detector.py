"""
Detector for timm (PyTorch Image Models) library
"""

from .base import BaseDetector, ModelInfo


class TimmDetector(BaseDetector):
    """Detector for timm computer vision models"""

    def can_handle(self, info: ModelInfo) -> bool:
        """Check if this is a timm model"""
        # Explicit library
        if info.library_name == 'timm':
            return True

        # Has timm tag
        if 'timm' in info.tags:
            return True

        # Has timm config file
        if 'config.json' in info.files and info.config:
            if info.config.get('architecture') in ['timm', 'pytorch-image-models']:
                return True

        return False

    def detect(self, info: ModelInfo) -> ModelInfo:
        """Detect timm-specific information"""
        info.model_type = 'vision'
        info.task = info.pipeline_tag or 'image-classification'

        # Architecture detection
        if info.config and 'architecture' in info.config:
            info.architecture = info.config['architecture']
        else:
            # Try to infer from model name
            info.architecture = self._infer_architecture(info.model_id)

        # Extract model details
        if info.config:
            # Input size
            if 'input_size' in info.config:
                info.metadata['input_size'] = info.config['input_size']

            # Number of classes
            if 'num_classes' in info.config:
                info.metadata['num_classes'] = info.config['num_classes']

            # Pretrained dataset
            if 'dataset' in info.config:
                info.metadata['pretrained_dataset'] = info.config['dataset']

        # Check for specific model families
        model_lower = info.model_id.lower()
        if 'vit' in model_lower or 'vision_transformer' in model_lower:
            info.metadata['model_family'] = 'Vision Transformer'
        elif 'resnet' in model_lower:
            info.metadata['model_family'] = 'ResNet'
        elif 'efficientnet' in model_lower:
            info.metadata['model_family'] = 'EfficientNet'
        elif 'convnext' in model_lower:
            info.metadata['model_family'] = 'ConvNeXt'

        # Requirements
        info.special_requirements = [
            'timm',
            'torch',
            'torchvision',
            'pillow'
        ]

        # Add specific requirements based on model
        if info.metadata.get('model_family') == 'Vision Transformer':
            info.special_requirements.append('einops')  # Often used with ViT

        # Quantization support (limited for vision models)
        info.supports_quantization = ['fp32', 'fp16']

        # Default dtype - vision models often use fp32
        if info.config and 'torch_dtype' in info.config:
            info.default_dtype = info.config['torch_dtype']
        else:
            info.default_dtype = 'float32'
        
        # Vision models don't support vLLM
        info.metadata['supports_vllm'] = False
        
        # TensorRT can optimize vision models
        info.metadata['supports_tensorrt'] = True

        return info

    def _infer_architecture(self, model_id: str) -> str:
        """Infer architecture from model ID"""
        model_lower = model_id.lower()

        # Vision Transformer variants
        if 'vit' in model_lower:
            if 'tiny' in model_lower:
                return 'ViT-Tiny'
            elif 'small' in model_lower:
                return 'ViT-Small'
            elif 'base' in model_lower:
                return 'ViT-Base'
            elif 'large' in model_lower:
                return 'ViT-Large'
            elif 'huge' in model_lower:
                return 'ViT-Huge'
            else:
                return 'ViT'

        # ResNet variants
        elif 'resnet' in model_lower:
            for size in ['18', '34', '50', '101', '152']:
                if size in model_lower:
                    return f'ResNet{size}'
            return 'ResNet'

        # EfficientNet variants
        elif 'efficientnet' in model_lower:
            for variant in ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7']:
                if variant in model_lower:
                    return f'EfficientNet-{variant.upper()}'
            return 'EfficientNet'

        # ConvNeXt variants
        elif 'convnext' in model_lower:
            if 'tiny' in model_lower:
                return 'ConvNeXt-Tiny'
            elif 'small' in model_lower:
                return 'ConvNeXt-Small'
            elif 'base' in model_lower:
                return 'ConvNeXt-Base'
            elif 'large' in model_lower:
                return 'ConvNeXt-Large'
            else:
                return 'ConvNeXt'

        # Swin Transformer
        elif 'swin' in model_lower:
            return 'Swin-Transformer'

        return 'Vision-Model'

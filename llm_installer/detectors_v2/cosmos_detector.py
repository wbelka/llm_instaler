"""
Detector for NVIDIA Cosmos models
"""

from .base import BaseDetector, ModelInfo


class CosmosDetector(BaseDetector):
    """Detector for NVIDIA Cosmos models"""

    def can_handle(self, info: ModelInfo) -> bool:
        """Check if this is a Cosmos model"""
        # Library name
        if info.library_name == 'cosmos':
            return True

        # Model ID pattern
        if 'cosmos' in info.model_id.lower():
            return True

        # Check for Cosmos pipeline in model_index
        if info.model_index and 'Cosmos' in info.model_index.get('_class_name', ''):
            return True

        # Tags
        if any('cosmos' in tag.lower() for tag in info.tags):
            return True

        return False

    def detect(self, info: ModelInfo) -> ModelInfo:
        """Detect Cosmos-specific information"""
        info.model_type = 'diffusion'
        info.is_multimodal = True

        # Detect architecture from model_index
        if info.model_index and '_class_name' in info.model_index:
            info.architecture = info.model_index['_class_name']
        else:
            info.architecture = 'CosmosModel'

        # Task detection
        if info.pipeline_tag:
            info.task = info.pipeline_tag
        elif 'text2image' in info.model_id.lower():
            info.task = 'text-to-image'
        elif 'image2text' in info.model_id.lower():
            info.task = 'image-to-text'
        else:
            info.task = 'multimodal'

        # Cosmos variant detection
        model_lower = info.model_id.lower()
        if 'predict' in model_lower:
            info.metadata['variant'] = 'Cosmos-Predict'
            info.metadata['description'] = 'World model for video/image prediction'
        elif 'generate' in model_lower:
            info.metadata['variant'] = 'Cosmos-Generate'
            info.metadata['description'] = 'Generative world model'

        # Model size detection
        if '2b' in model_lower:
            info.metadata['parameters'] = '2B'
        elif '7b' in model_lower:
            info.metadata['parameters'] = '7B'
        elif '13b' in model_lower:
            info.metadata['parameters'] = '13B'

        # Requirements
        info.special_requirements = [
            'diffusers',
            'transformers',
            'torch>=2.0',
            'accelerate',
            'safetensors',
            'pillow',
            'opencv-python'
        ]

        # Add xformers for larger models
        if info.size_gb > 10:
            info.special_requirements.append('xformers')

        # Component analysis for Cosmos models
        if info.model_index and info.size_gb > 0:
            components = {}
            component_types = []

            # Extract components
            for key, value in info.model_index.items():
                if isinstance(value, list) and len(value) >= 2:
                    if not key.startswith('_'):
                        components[key] = value[1]
                        component_types.append(key)

            info.metadata['components'] = components
            info.metadata['component_types'] = component_types

            # Cosmos models have different component distributions
            # The transformer/main model is typically the largest
            component_sizes = {}
            total_size = info.size_gb

            # Typical Cosmos distribution
            distributions = {
                'transformer': 0.70,  # Main model
                'vae': 0.10,
                'text_encoder': 0.15,
                'tokenizer': 0.01,  # Very small
                'scheduler': 0.01,  # Very small
                'safety_checker': 0.03
            }

            # Calculate sizes
            assigned_size = 0
            for component in component_types:
                comp_lower = component.lower()
                for dist_key, dist_value in distributions.items():
                    if dist_key in comp_lower:
                        size = round(total_size * dist_value, 1)
                        component_sizes[component] = size
                        assigned_size += size
                        break

            # Distribute any remaining size
            remaining = total_size - assigned_size
            if remaining > 0 and component_types:
                # Add to transformer or largest component
                for comp in ['transformer', 'TRANSFORMER', 'model']:
                    if comp in component_sizes:
                        component_sizes[comp] += remaining
                        break
                else:
                    # Add to first component if no transformer found
                    first_comp = component_types[0]
                    component_sizes[first_comp] = \
                        component_sizes.get(first_comp, 0) + remaining

            info.metadata['component_sizes'] = component_sizes

        # Quantization support
        info.supports_quantization = ['fp32', 'fp16', 'bf16']

        # Default dtype
        if info.config and 'torch_dtype' in info.config:
            info.default_dtype = info.config['torch_dtype']
        else:
            info.default_dtype = 'float16'

        # vLLM doesn't support diffusion models
        info.metadata['supports_vllm'] = False

        # TensorRT support for NVIDIA models
        info.metadata['supports_tensorrt'] = True

        # Add NVIDIA-specific optimizations
        info.metadata['nvidia_optimized'] = True
        info.metadata['supports_tensorrt_llm'] = False  # Not for diffusion

        return info
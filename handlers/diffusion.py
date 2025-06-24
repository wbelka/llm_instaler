"""Handler for diffusion models (Stable Diffusion, DALL-E, etc.).

This handler manages image and video generation models that use
diffusion techniques.
"""

from typing import Dict, Any, List, Tuple, Optional
from handlers.base import BaseHandler
from core.checker import ModelRequirements


class DiffusionHandler(BaseHandler):
    """Handler for diffusion-based generative models."""

    def __init__(self, model_info: Dict[str, Any]):
        """Initialize the diffusion handler.

        Args:
            model_info: Model information from HuggingFace.
        """
        super().__init__(model_info)

    def analyze(self) -> ModelRequirements:
        """Analyze diffusion model requirements.

        Returns:
            ModelRequirements object with analyzed data.
        """
        requirements = ModelRequirements()

        # Set model type and family
        requirements.model_type = "diffusion"
        requirements.model_family = "image-generation"
        requirements.primary_library = "diffusers"

        # Check if it's a video model
        model_id = self.model_info.get('model_id', '').lower()
        config = self.model_info.get('config', {})

        # Check various indicators for video models
        if any(indicator in model_id for indicator in ['video', 'text2video', 'text-to-video', 't2v']):
            requirements.model_family = "video-generation"
        elif '_class_name' in config and 'video' in config['_class_name'].lower():
            requirements.model_family = "video-generation"

        # Determine architecture
        if '_class_name' in config:
            requirements.architecture_type = config['_class_name']
        else:
            requirements.architecture_type = "diffusion"

        # Base dependencies
        requirements.base_dependencies = [
            "torch",
            "torchvision",
            "diffusers",
            "transformers",  # Many diffusion models need this
            "accelerate",
            "pillow",
            "numpy",
            "safetensors",
            "ftfy",  # Text fixing library used by some models
            "beautifulsoup4",  # Sometimes needed with ftfy
        ]

        # Check for specific model requirements
        if 'controlnet' in model_id:
            requirements.base_dependencies.append("opencv-python")
            requirements.capabilities["controlnet"] = True

        if 'sdxl' in model_id:
            requirements.capabilities["sdxl"] = True
            requirements.special_dependencies.append("invisible-watermark")

        # Optional dependencies
        requirements.optional_dependencies = [
            "xformers",  # For memory efficient attention
            "triton",    # For some optimizations
            "compel",    # For better prompt handling
        ]

        # Wan models need specific dependencies
        if 'wan' in model_id.lower():
            # Ensure transformers is in the list (if not already)
            if 'transformers' not in requirements.base_dependencies:
                requirements.base_dependencies.append("transformers")
            # Ensure ftfy is in the list
            if 'ftfy' not in requirements.base_dependencies:
                requirements.base_dependencies.append("ftfy")
            # Wan models also benefit from these
            requirements.optional_dependencies.append("scipy")

        # Add opencv for video models - required for video output
        if requirements.model_family == "video-generation":
            if "opencv-python" not in requirements.base_dependencies:
                requirements.base_dependencies.append("opencv-python")
            # Some video models also need imageio for video I/O
            if "imageio" not in requirements.base_dependencies:
                requirements.base_dependencies.append("imageio")
            if "imageio-ffmpeg" not in requirements.optional_dependencies:
                requirements.optional_dependencies.append("imageio-ffmpeg")

        # Memory requirements
        model_size = self._estimate_model_size()
        requirements.disk_space_gb = model_size

        # Estimate memory requirements based on model size
        requirements.memory_requirements = {
            "min": model_size * 0.3,    # Minimum for int4
            "recommended": model_size * 1.2,  # For float16
            "optimal": model_size * 2.4,     # For float32
        }

        # Training memory requirements
        requirements.memory_requirements["training"] = {
            "lora": model_size * 1.3,
            "full": model_size * 4,
        }

        # Capabilities
        requirements.capabilities.update({
            "batch_inference": True,
            "streaming": False,
            "quantization": ["int8", "int4"],
            "lora": True,
            "native_dtype": "float16"
        })

        # Special configurations
        requirements.special_config = {
            "supports_fp16": True,
            "supports_bf16": True,
            "supports_cpu_offload": True,
            "supports_sequential_cpu_offload": True,
            "requires_safety_checker": True,
        }

        return requirements

    def _estimate_model_size(self) -> float:
        """Estimate model size in GB.

        Returns:
            Estimated size in GB.
        """
        # Check if we have size information
        if 'model_size' in self.model_info:
            return self.model_info['model_size']

        # Otherwise estimate from files
        total_size = 0
        siblings = self.model_info.get('siblings', [])

        for file_info in siblings:
            if isinstance(file_info, dict) and 'size' in file_info:
                # Check if it's a model weight file
                filename = file_info.get('filename', '')
                if any(filename.endswith(ext) for ext in ['.bin', '.safetensors', '.ckpt', '.pt']):
                    total_size += file_info['size']

        # Convert to GB
        size_gb = total_size / (1024**3)

        # If no size found, use defaults based on model type
        if size_gb == 0:
            model_id = self.model_info.get('model_id', '').lower()
            if 'sdxl' in model_id:
                size_gb = 6.5
            elif 'sd-1.5' in model_id or 'sd-v1-5' in model_id:
                size_gb = 4.0
            elif 'sd-2' in model_id:
                size_gb = 5.0
            else:
                size_gb = 5.0  # Default for unknown diffusion models

        return size_gb

    def get_dependencies(self) -> List[str]:
        """Get Python dependencies for diffusion models.

        Returns:
            List of required pip packages.
        """
        return self.analyze().base_dependencies

    def get_system_dependencies(self) -> List[str]:
        """Get system-level dependencies.

        Returns:
            List of system packages or requirements.
        """
        # Most diffusion models need CUDA
        return ["cuda>=11.0"]

    def get_inference_params(self) -> Dict[str, Any]:
        """Get optimal inference parameters.

        Returns:
            Dictionary of inference parameters.
        """
        return {
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "negative_prompt": "",
            "width": 512,
            "height": 512,
            "seed": None,
        }

    def get_training_params(self) -> Dict[str, Any]:
        """Get optimal training parameters.

        Returns:
            Dictionary of training parameters.
        """
        return {
            "learning_rate": 1e-4,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "num_train_epochs": 100,
            "checkpointing_steps": 500,
            "validation_epochs": 10,
            "enable_xformers_memory_efficient_attention": True,
        }

    def load_model(self, model_path: str, **kwargs):
        """Load diffusion model.

        Args:
            model_path: Path to the model directory.
            **kwargs: Additional loading parameters.

        Returns:
            Loaded model (pipeline) and tokenizer/processor (None for diffusion).
        """
        import torch
        from diffusers import DiffusionPipeline
        import os
        import json

        # Determine device
        device = kwargs.get('device', 'auto')
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # Determine dtype
        dtype = kwargs.get('dtype', 'auto')
        if dtype == "auto":
            if device in ["cuda", "mps"]:
                torch_dtype = torch.float16
            else:
                torch_dtype = torch.float32
        else:
            dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
            }
            torch_dtype = dtype_map.get(dtype, torch.float32)

        # Check for specific pipeline class
        model_index_path = os.path.join(model_path, "model_index.json")
        if os.path.exists(model_index_path):
            with open(model_index_path, 'r') as f:
                model_index = json.load(f)
                pipeline_class_name = model_index.get("_class_name", None)

                if pipeline_class_name == "TextToVideoSDPipeline":
                    from diffusers import TextToVideoSDPipeline
                    pipeline = TextToVideoSDPipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch_dtype,
                    )
                else:
                    pipeline = DiffusionPipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch_dtype,
                    )
        else:
            pipeline = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
            )

        # Move to device
        pipeline = pipeline.to(device)

        # Enable optimizations
        if device == "cuda":
            try:
                pipeline.enable_xformers_memory_efficient_attention()
            except Exception:
                pipeline.enable_attention_slicing()

        return pipeline, None

    def validate_model_files(self, model_path: str) -> Tuple[bool, Optional[str]]:
        """Validate model files.

        Args:
            model_path: Path to the model directory.

        Returns:
            Tuple of (is_valid, error_message).
        """
        from pathlib import Path
        model_dir = Path(model_path)

        # Check for model_index.json
        if not (model_dir / "model_index.json").exists():
            return False, "Missing model_index.json file"

        # Check for essential components
        required_components = ["unet", "vae", "text_encoder", "tokenizer", "scheduler"]
        missing = []

        for component in required_components:
            component_path = model_dir / component
            if not component_path.exists():
                # Some models might have it in config only
                continue

        if missing:
            return False, f"Missing components: {', '.join(missing)}"

        return True, None
    
    def generate_image(self, prompt: str, negative_prompt: str = None,
                      model=None, **kwargs) -> Dict[str, Any]:
        """Generate image using diffusion model.
        
        Args:
            prompt: Text description of image to generate
            negative_prompt: What to avoid in the image
            model: Model instance (pipeline)
            **kwargs: Generation parameters
            
        Returns:
            Dictionary with image data and metadata
        """
        import torch
        from PIL import Image
        import base64
        from io import BytesIO
        
        if not model:
            raise ValueError("Model (pipeline) required for image generation")
        
        # Extract parameters
        params = {
            'prompt': prompt,
            'negative_prompt': negative_prompt or kwargs.get('negative_prompt', ''),
            'num_inference_steps': kwargs.get('num_inference_steps', 50),
            'guidance_scale': kwargs.get('guidance_scale', 7.5),
            'width': kwargs.get('width', 512),
            'height': kwargs.get('height', 512),
            'num_images_per_prompt': kwargs.get('num_images_per_prompt', 1)
        }
        
        # Add seed if provided
        if 'seed' in kwargs and kwargs['seed'] is not None:
            generator = torch.Generator(device=model.device).manual_seed(kwargs['seed'])
            params['generator'] = generator
        
        # Clear CUDA cache before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            # Generate image
            with torch.no_grad():
                result = model(**params)
            
            # Get first image
            image = result.images[0]
            
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
                    'width': params['width'],
                    'height': params['height'],
                    'format': 'png',
                    'steps': params['num_inference_steps'],
                    'guidance_scale': params['guidance_scale'],
                    'model': 'diffusion'
                }
            }
            
        except Exception as e:
            # Clear CUDA cache on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise e
    
    def generate_video(self, prompt: str, image: str = None,
                      model=None, **kwargs) -> Dict[str, Any]:
        """Generate video using video diffusion model.
        
        Args:
            prompt: Text description of video to generate
            image: Base64 image for image-to-video generation (optional)
            model: Model instance (pipeline)
            **kwargs: Generation parameters
            
        Returns:
            Dictionary with video data and metadata
        """
        import torch
        import numpy as np
        import cv2
        import base64
        from io import BytesIO
        import tempfile
        
        if not model:
            raise ValueError("Model (pipeline) required for video generation")
        
        # Check if model supports video generation
        model_class = model.__class__.__name__
        if 'video' not in model_class.lower():
            raise ValueError(f"Model {model_class} does not support video generation")
        
        # Extract parameters
        params = {
            'prompt': prompt,
            'num_inference_steps': kwargs.get('num_inference_steps', 50),
            'guidance_scale': kwargs.get('guidance_scale', 7.5),
            'num_frames': kwargs.get('num_frames', 16),
            'height': kwargs.get('height', 256),
            'width': kwargs.get('width', 256)
        }
        
        # Clear CUDA cache before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        try:
            # Generate video
            with torch.no_grad():
                video_frames = model(**params).frames[0]
            
            # Convert frames to video
            fps = kwargs.get('fps', 8)
            
            # Create temporary file for video
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                video_path = tmp_file.name
            
            # Write video using cv2
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (params['width'], params['height']))
            
            for frame in video_frames:
                # Convert PIL image to numpy array
                frame_array = np.array(frame)
                # Convert RGB to BGR for cv2
                frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
            # Read video and convert to base64
            with open(video_path, 'rb') as f:
                video_base64 = base64.b64encode(f.read()).decode()
            
            # Clean up temporary file
            import os
            os.unlink(video_path)
            
            # Clear CUDA cache after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                'video': video_base64,
                'metadata': {
                    'width': params['width'],
                    'height': params['height'],
                    'format': 'mp4',
                    'fps': fps,
                    'num_frames': params['num_frames'],
                    'model': 'video-diffusion'
                }
            }
            
        except Exception as e:
            # Clear CUDA cache on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise e
    
    def get_supported_modes(self) -> List[str]:
        """Get supported generation modes."""
        model_family = self.analyze().model_family
        
        if model_family == 'video-generation':
            return ['video', 'auto']
        else:
            return ['image', 'auto']
    
    def get_mode_descriptions(self) -> Dict[str, str]:
        """Get mode descriptions."""
        return {
            'auto': 'Automatic mode selection',
            'image': 'Text-to-image generation',
            'video': 'Text-to-video generation'
        }
    
    def get_model_capabilities(self) -> Dict[str, Any]:
        """Get model capabilities."""
        capabilities = super().get_model_capabilities()
        model_family = self.analyze().model_family
        
        capabilities.update({
            'supports_streaming': False,
            'supports_batch_inference': True,
            'input_modalities': ['text'],
            'output_modalities': ['image'] if model_family == 'image-generation' else ['video']
        })
        
        return capabilities
    
    def get_installation_notes(self) -> Dict[str, str]:
        """Get installation notes for special dependencies."""
        return {
            'xformers': """
xformers provides memory efficient attention for diffusion models.
It can significantly reduce memory usage and speed up generation.

If installation fails, the model will still work with standard attention.
To install manually:
   pip install xformers --index-url https://download.pytorch.org/whl/cu124
""",
            'invisible-watermark': """
Required for SDXL models to add invisible watermarks to generated images.
This is a safety feature required by some models.
"""
        }

"""Handler for Stability AI's Stable Diffusion 3.5 Large model.

This module provides specialized handling for the SD3.5 Large model, including
support for 4-bit quantization and CPU offloading for memory efficiency.
"""

import logging
import torch
import io
import base64
from typing import List, Dict, Any, Tuple

from handlers.diffusion import DiffusionHandler
from core.checker import ModelRequirements

logger = logging.getLogger(__name__)


class StableDiffusion3LargeHandler(DiffusionHandler):
    """Handler for Stable Diffusion 3.5 Large."""

    def __init__(self, model_info: Dict[str, Any]):
        """Initialize the SD3.5 Large handler."""
        super().__init__(model_info)
        self.model_family = "stable-diffusion-3"

    def get_dependencies(self) -> List[str]:
        """Get required Python dependencies."""
        return [
            'diffusers>=0.27.0',
            'torch',
            'transformers',
            'accelerate',
            'bitsandbytes',
            'sentencepiece' # Added for fast tokenizer
        ]

    def analyze(self) -> 'ModelRequirements':
        """Analyze model and return requirements."""
        requirements = super().analyze()
        requirements.model_family = self.model_family
        requirements.capabilities['supported_quantization'] = ["int4", "int8"]
        
        # Adjust memory requirements for this large model
        requirements.memory_requirements = {
            "min": 16,
            "recommended": 24,
            "gpu_min": 12,  # With 4-bit quantization and offloading
            "gpu_recommended": 16
        }
        requirements.special_config['notes'] = (
            "This model supports 4-bit and 8-bit quantization and CPU offloading, "
            "making it usable on GPUs with as little as 12GB of VRAM."
        )
        return requirements

    def load_model(
        self,
        model_path: str,
        device: str = 'auto',
        dtype: str = 'auto',
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs
    ) -> Tuple[Any, Any]:
        """Load the Stable Diffusion 3.5 Large model."""
        from diffusers import StableDiffusion3Pipeline, BitsAndBytesConfig, SD3Transformer2DModel

        # Determine the base torch_dtype for the model
        # Default to bfloat16 for GPU, float32 for CPU if not specified
        if dtype == 'auto':
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                base_torch_dtype = torch.bfloat16
            else:
                base_torch_dtype = torch.float32 # Fallback for CPU or non-bf16 GPUs
        else:
            base_torch_dtype = getattr(torch, dtype)

        transformer_quantization_config = None
        if load_in_4bit:
            logger.info("Configuring 4-bit quantization for Stable Diffusion 3.5 Large.")
            transformer_quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 # Use bfloat16 for compute dtype
            )
        elif load_in_8bit:
            logger.info("Configuring 8-bit quantization for Stable Diffusion 3.5 Large.")
            transformer_quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 # Use bfloat16 if supported, else float16
            )

        # Load the transformer component with or without quantization
        if transformer_quantization_config:
            logger.info("Loading transformer with quantization config.")
            transformer = SD3Transformer2DModel.from_pretrained(
                model_path,
                subfolder="transformer",
                quantization_config=transformer_quantization_config,
                torch_dtype=base_torch_dtype # Use the determined base_torch_dtype
            )
        else:
            logger.info("Loading transformer without quantization.")
            transformer = SD3Transformer2DModel.from_pretrained(
                model_path,
                subfolder="transformer",
                torch_dtype=base_torch_dtype
            )

        # Load the full pipeline, passing the (potentially quantized) transformer
        logger.info("Loading Stable Diffusion 3.5 Large pipeline.")
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_path,
            transformer=transformer,
            torch_dtype=base_torch_dtype
        )

        # Enable CPU offloading for all cases to manage VRAM
        logger.info("Enabling model CPU offload for VRAM management.")
        pipeline.enable_model_cpu_offload()

        logger.info("Stable Diffusion 3.5 Large model loaded.")
        return pipeline, None

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = None,
        model=None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate an image using the SD3.5 pipeline."""
        if not model:
            raise ValueError("Model (pipeline) is required for image generation.")

        pipeline = model  # The 'model' is the loaded pipeline

        # Extract only the parameters that SD3 pipeline accepts
        valid_params = {
            "num_inference_steps": kwargs.get("num_inference_steps", 28),
            "guidance_scale": kwargs.get("guidance_scale", 4.5),
            "max_sequence_length": kwargs.get("max_sequence_length", 512),
        }
        
        # Add optional parameters if provided
        if negative_prompt:
            valid_params["negative_prompt"] = negative_prompt
            
        if "width" in kwargs:
            valid_params["width"] = kwargs["width"]
        if "height" in kwargs:
            valid_params["height"] = kwargs["height"]
        if "seed" in kwargs and kwargs["seed"] is not None:
            import torch
            generator = torch.Generator(device=pipeline.device).manual_seed(kwargs["seed"])
            valid_params["generator"] = generator

        logger.info(f"Generating image with prompt: \"{prompt[:100]}...\"")
        
        image = pipeline(
            prompt=prompt,
            **valid_params
        ).images[0]

        # Convert PIL image to base64 string
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "image": img_str,
            "metadata": {
                "size": image.size,
                "format": "png",
                "prompt": prompt,
                **valid_params
            }
        }
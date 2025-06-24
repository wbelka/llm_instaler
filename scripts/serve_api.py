"""Universal API server for LLM Installer.

This FastAPI server adapts to different model types and provides
a unified interface for interacting with any installed model.
"""

import os
import json
import argparse
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import base64
from io import BytesIO

# Set CUDA memory allocation configuration to prevent fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, File, UploadFile
from starlette.responses import HTMLResponse
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
import uvicorn

# Import model loader
from model_loader import load_model, get_model_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM Installer Model API",
    description="Universal API for interacting with installed models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and config
MODEL = None
TOKENIZER = None
MODEL_INFO = None
DEVICE = None
DTYPE = None
HANDLER = None
STREAM_MODE = False


# Request/Response models
class GenerateRequest(BaseModel):
    prompt: Optional[str] = Field(None, description="Text prompt")
    messages: Optional[List[Dict[str, str]]] = Field(None, description="Chat messages")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, ge=1, le=131072)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=0)
    stop_sequences: Optional[List[str]] = None
    stream: bool = Field(False, description="Enable streaming")
    mode: Optional[str] = Field("auto", description="Generation mode: auto, chat, complete, instruct, creative, code, analyze, translate, summarize, image, audio, video")

    # Image generation parameters
    negative_prompt: Optional[str] = None
    num_inference_steps: int = Field(50, ge=1, le=1000)
    guidance_scale: float = Field(7.5, ge=0.0, le=20.0)
    width: int = Field(512, ge=64, le=2048)
    height: int = Field(512, ge=64, le=2048)
    seed: Optional[int] = None

    # Video generation parameters
    num_frames: int = Field(16, ge=1, le=128, description="Number of frames for video generation")

    # Embedding parameters
    texts: Optional[List[str]] = Field(None, description="Texts for embedding")
    normalize: bool = Field(True, description="Normalize embeddings")
    
    # Multimodal parameters
    image_data: Optional[str] = Field(None, description="Base64 encoded image data")
    images: Optional[List[str]] = Field(None, description="List of base64 encoded images")
    audio_data: Optional[str] = Field(None, description="Base64 encoded audio")

    # Reasoning parameters
    reasoning_mode: bool = Field(False, description="Enable reasoning mode")
    max_thinking_tokens: int = Field(10000, ge=1)
    max_answer_tokens: int = Field(2000, ge=1)
    show_thinking: bool = Field(True, description="Show thinking process")

    # Additional options
    return_all_scores: bool = Field(False, description="Return all class scores for classification")

    class Config:
        extra = "allow"  # Allow extra fields for flexibility


class GenerateResponse(BaseModel):
    text: Optional[str] = None
    image: Optional[str] = None  # Base64 encoded
    video: Optional[Dict[str, Any]] = None  # Video metadata
    embeddings: Optional[List[List[float]]] = None
    usage: Optional[Dict[str, int]] = None
    thinking: Optional[str] = None  # For reasoning models
    answer: Optional[str] = None  # For reasoning models
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata


class ModelInfoResponse(BaseModel):
    model_id: str
    model_type: str
    model_family: str
    capabilities: Dict[str, Any]
    device: str
    dtype: str
    stream_mode: bool = False


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global MODEL, TOKENIZER, MODEL_INFO, DEVICE, DTYPE, STREAM_MODE

    try:
        # Load model info
        MODEL_INFO = get_model_config()
        logger.info(f"Loading model: {MODEL_INFO['model_id']}")

        # Parse command line arguments (passed from start.sh)
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=8000)
        parser.add_argument("--host", default="0.0.0.0")
        parser.add_argument("--device", default="auto")
        parser.add_argument("--dtype", default="auto")
        parser.add_argument("--load-lora", type=str, default=None)
        parser.add_argument("--stream-mode", type=str, default="false")
        args, _ = parser.parse_known_args()

        DEVICE = args.device
        DTYPE = args.dtype
        STREAM_MODE = args.stream_mode.lower() == "true"

        # Load model
        logger.info(f"Loading model with info: model_type={MODEL_INFO.get('model_type')}, model_family={MODEL_INFO.get('model_family')}")
        
        # Check if LoRA exists and auto-load if not specified
        if args.load_lora is None:
            lora_path = Path("./lora")
            if lora_path.exists() and lora_path.is_dir():
                logger.info(f"Found LoRA adapter at {lora_path}, auto-loading...")
                args.load_lora = str(lora_path)
        
        MODEL, TOKENIZER = load_model(
            MODEL_INFO,
            device=args.device,
            dtype=args.dtype,
            lora_path=args.load_lora,
            load_lora=True  # Enable LoRA loading
        )
        
        # Get handler instance
        global HANDLER
        from model_loader import get_handler
        HANDLER = get_handler(MODEL_INFO)
        if HANDLER:
            logger.info(f"Loaded handler: {HANDLER.__class__.__name__}")

        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/")
async def root():
    """Serve the terminal UI."""
    script_dir = Path(__file__).parent
    terminal_ui = script_dir / "serve_terminal.html"
    
    if terminal_ui.exists():
        return FileResponse(terminal_ui)
    else:
        return HTMLResponse(
            content="<h1>Model API</h1><p>Terminal UI not found. API is running at /docs</p>"
        )


def apply_mode_settings(request: GenerateRequest, mode: str) -> GenerateRequest:
    """Apply mode-specific settings to the request."""
    # Mode-specific parameter adjustments
    mode_settings = {
        "chat": {"temperature": 0.7, "top_p": 0.9},
        "complete": {"temperature": 0.8, "top_p": 0.95},
        "instruct": {"temperature": 0.3, "top_p": 0.9},
        "creative": {"temperature": 1.2, "top_p": 0.95},
        "code": {"temperature": 0.2, "top_p": 0.95, "max_tokens": 1024},
        "analyze": {"temperature": 0.1, "top_p": 0.9},
        "translate": {"temperature": 0.3, "top_p": 0.9},
        "summarize": {"temperature": 0.5, "top_p": 0.9},
        "image": {"temperature": 1.0, "guidance_scale": 7.5},
        "artistic": {"temperature": 1.2, "guidance_scale": 8.0},
        "photorealistic": {"temperature": 0.8, "guidance_scale": 7.0},
        "vision": {"temperature": 0.5},
        "audio": {"temperature": 0.5},
        "video": {"temperature": 0.8, "num_frames": 16}
    }
    
    if mode in mode_settings:
        settings = mode_settings[mode]
        for key, value in settings.items():
            if hasattr(request, key) and getattr(request, key) == GenerateRequest.model_fields[key].default:
                # Only override if using default value
                setattr(request, key, value)
    
    # Add mode-specific prompts prefixes if needed
    mode_prefixes = {
        "instruct": "Follow these instructions carefully: ",
        "code": "Generate code for: ",
        "analyze": "Analyze the following: ",
        "translate": "Translate the following text: ",
        "summarize": "Summarize the following: ",
        "creative": "Create an imaginative response for: ",
        "artistic": "Create an artistic interpretation of: ",
        "photorealistic": "Create a photorealistic image of: "
    }
    
    if mode in mode_prefixes and request.prompt:
        if not request.prompt.startswith(mode_prefixes[mode]):
            request.prompt = mode_prefixes[mode] + request.prompt
    
    return request




@app.get("/ui/terminal")
async def terminal_ui():
    """Serve the terminal-style UI."""
    script_dir = Path(__file__).parent
    ui_path = script_dir / "serve_terminal.html"
    if ui_path.exists():
        return FileResponse(ui_path)
    else:
        return HTMLResponse(content="<h1>Terminal UI not found</h1>")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information."""
    if not MODEL_INFO:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Check if it's a video model
    model_id = MODEL_INFO.get("model_id", "").lower()
    model_family = MODEL_INFO.get("model_family", "")
    model_class = MODEL_INFO.get("config", {}).get("_class_name", "").lower()

    # Override model_family for video models if needed
    if ('video' in model_id or 'video' in model_class) and model_family == "image-generation":
        model_family = "video-generation"

    # Add UI-specific capabilities
    capabilities = MODEL_INFO.get("capabilities", {})
    capabilities["is_video_model"] = model_family == "video-generation" or 'video' in model_id
    
    # Add handler-specific capabilities if available
    handler_modes = None
    handler_descriptions = None
    
    if HANDLER:
        logger.info(f"Handler type: {type(HANDLER).__name__}")
        handler_caps = HANDLER.get_model_capabilities()
        capabilities.update(handler_caps)
        
        # Get supported modes from handler
        try:
            handler_modes = HANDLER.get_supported_modes()
            handler_descriptions = HANDLER.get_mode_descriptions()
            logger.info(f"Handler provided modes: {handler_modes}")
        except Exception as e:
            logger.warning(f"Failed to get modes from handler: {e}")

    # Add size recommendations for video models
    if capabilities.get("is_video_model"):
        capabilities["recommended_sizes"] = {
            "video": {
                "small": {"width": 128, "height": 128, "label": "128×128 (Fast)"},
                "medium": {"width": 256, "height": 256, "label": "256×256 (Recommended)"},
                "large": {"width": 512, "height": 512, "label": "512×512 (May fail on some GPUs)"}
            }
        }
    
    # Use handler modes if available, otherwise fall back to defaults
    if handler_modes:
        supported_modes = handler_modes
        logger.info(f"Using handler modes: {supported_modes}")
    else:
        logger.info("No handler modes, using defaults")
        # Add supported modes based on model type
        supported_modes = ["auto"]  # All models support auto mode
        
        if model_family == "language-model":
            supported_modes.extend(["chat", "complete", "instruct", "creative", "code", "analyze", "translate", "summarize"])
            # Add thinking modes for Qwen3
            if MODEL_INFO.get("model_type") in ["qwen3", "qwen-3"]:
                supported_modes.extend(["thinking", "no-thinking", "math", "reasoning"])
        elif model_family == "multimodal" or MODEL_INFO.get("model_type") == "multi_modality":
            supported_modes.extend(["chat", "complete", "analyze", "image", "creative", "vision"])
        elif model_family == "image-generation":
            supported_modes.extend(["image", "creative", "artistic", "photorealistic"])
        elif model_family == "video-generation":
            supported_modes.extend(["video", "creative", "animate"])
        elif model_family == "audio-model":
            supported_modes.extend(["audio", "transcribe", "voice"])
        elif model_family == "embedding-model":
            supported_modes.extend(["embed", "similarity"])
    
    capabilities["supported_modes"] = supported_modes
    
    # Use handler descriptions if available, otherwise fall back to defaults
    if handler_descriptions:
        capabilities["mode_descriptions"] = handler_descriptions
    else:
        # Add mode descriptions
        capabilities["mode_descriptions"] = {
            "auto": "Automatic mode selection",
            "chat": "Conversational dialogue",
            "complete": "Text completion",
            "instruct": "Follow instructions",
            "creative": "Creative generation",
            "code": "Code generation",
            "analyze": "Analysis & reasoning",
            "translate": "Language translation",
            "summarize": "Text summarization",
            "image": "Image generation",
            "vision": "Image understanding",
            "audio": "Audio processing",
            "video": "Video generation",
            "embed": "Text embeddings",
            "transcribe": "Speech to text",
            "voice": "Text to speech",
            "artistic": "Artistic style",
            "photorealistic": "Realistic images",
            "animate": "Animation generation",
            "similarity": "Similarity search",
            "thinking": "Enable thinking mode for complex reasoning",
            "no-thinking": "Disable thinking for efficient responses",
            "math": "Mathematical problem solving with thinking",
            "reasoning": "Complex logical reasoning with thinking"
        }

    logger.info(f"Final supported_modes: {capabilities.get('supported_modes', [])}")
    logger.info(f"Final mode_descriptions: {list(capabilities.get('mode_descriptions', {}).keys())}")
    
    return ModelInfoResponse(
        model_id=MODEL_INFO.get("model_id", "unknown"),
        model_type=MODEL_INFO.get("model_type", "unknown"),
        model_family=model_family,
        capabilities=capabilities,
        device=str(DEVICE),
        dtype=str(DTYPE),
        stream_mode=STREAM_MODE
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Main generation endpoint."""
    if not MODEL:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Log request for debugging
        logger.info("=== New generation request ===")
        logger.info(f"Prompt: '{request.prompt[:50] if request.prompt else None}...'")
        
        if hasattr(request, 'messages') and request.messages:
            logger.info(f"Messages: {len(request.messages)} messages")
            for i, msg in enumerate(request.messages):
                logger.info(f"  Message {i}: role='{msg.get('role')}', content_length={len(msg.get('content', ''))}")
        else:
            logger.info("Messages: None")
            
        if hasattr(request, 'image_data') and request.image_data:
            logger.info("Image data: Single image provided")
        elif hasattr(request, 'images') and request.images:
            logger.info(f"Images: {len(request.images)} images provided")
        else:
            logger.info("Images: None")
        
        logger.info(f"Parameters: temperature={request.temperature}, max_tokens={request.max_tokens}, "
                    f"top_p={request.top_p}, top_k={request.top_k}")
        logger.info(f"Mode: {request.mode}")
        
        model_family = MODEL_INFO.get("model_family", "")
        model_type = MODEL_INFO.get("model_type", "")
        
        logger.info(f"Model family: {model_family}, Model type: {model_type}")
        
        # Apply mode-specific settings
        if request.mode and request.mode != "auto":
            request = apply_mode_settings(request, request.mode)

        # Use handler if available
        if HANDLER:
            logger.info(f"Using handler: {HANDLER.__class__.__name__}")
            try:
                # Determine the task based on request and mode
                if request.mode == "image" or (request.mode == "auto" and 
                    any(word in (request.prompt or "").lower() for word in ["draw", "generate image", "create image"])):
                    # Image generation
                    if hasattr(HANDLER, 'generate_image'):
                        result = HANDLER.generate_image(
                            prompt=request.prompt or request.messages[-1]["content"] if request.messages else "",
                            negative_prompt=request.negative_prompt,
                            model=MODEL,
                            tokenizer=TOKENIZER,
                            temperature=request.temperature,
                            guidance_scale=request.guidance_scale,
                            width=request.width,
                            height=request.height,
                            num_inference_steps=request.num_inference_steps,
                            seed=request.seed
                        )
                        return GenerateResponse(**result)
                
                elif request.images or request.image_data or request.mode == "vision":
                    # Multimodal processing (including vision mode)
                    logger.info(f"Processing multimodal request (vision mode={request.mode == 'vision'})")
                    if hasattr(HANDLER, 'process_multimodal'):
                        images = []
                        if request.image_data:
                            images.append(request.image_data)
                        if request.images:
                            images.extend(request.images)
                        
                        # For vision mode, only use current message without history
                        if request.mode == "vision":
                            text_input = request.prompt or (request.messages[-1]["content"] if request.messages else None)
                            # Override messages to only include current one for vision mode
                            request.messages = [{"role": "user", "content": text_input}] if text_input else []
                        else:
                            text_input = request.prompt or (request.messages[-1]["content"] if request.messages else None)
                        
                        result = HANDLER.process_multimodal(
                            text=text_input,
                            images=images,
                            model=MODEL,
                            processor=TOKENIZER,
                            temperature=request.temperature,
                            max_tokens=request.max_tokens,
                            top_p=request.top_p,
                            mode=request.mode,
                            max_image_size=getattr(request, 'max_image_size', 2048)
                        )
                        return GenerateResponse(**result)
                
                else:
                    # Text generation
                    if hasattr(HANDLER, 'generate_text'):
                        result = HANDLER.generate_text(
                            prompt=request.prompt,
                            messages=request.messages,
                            model=MODEL,
                            tokenizer=TOKENIZER,
                            temperature=request.temperature,
                            max_tokens=request.max_tokens,
                            top_p=request.top_p,
                            top_k=request.top_k,
                            stop_sequences=request.stop_sequences,
                            mode=request.mode
                        )
                        return GenerateResponse(**result)
                
            except NotImplementedError as e:
                # Handler doesn't support this operation
                logger.error(f"Handler doesn't support operation: {e}")
                raise HTTPException(
                    status_code=501,
                    detail=f"Handler does not support this operation: {str(e)}"
                )
            except Exception as e:
                logger.error(f"Handler error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Handler error: {str(e)}"
                )

        # If no handler is available, return error
        else:
            raise HTTPException(
                status_code=501,
                detail=f"No handler available for model family: {model_family}. All operations must go through handlers."
            )

    except Exception as e:
        import traceback
        logger.error(f"Generation error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


async def generate_text_legacy(request: GenerateRequest) -> GenerateResponse:
    """Generate text for language models."""
    import torch
    
    logger.info("=== Starting text generation ===")
    logger.info(f"Model type: {MODEL_INFO.get('model_type')}")
    logger.info(f"Model family: {MODEL_INFO.get('model_family')}")

    # Check if this is a reasoning model
    capabilities = MODEL_INFO.get("capabilities", {})
    supports_reasoning = capabilities.get("reasoning", False)
    logger.info(f"Supports reasoning: {supports_reasoning}, Reasoning mode requested: {request.reasoning_mode}")

    if request.reasoning_mode and supports_reasoning:
        logger.info("Using reasoning mode generation")
        return await generate_with_reasoning(request)

    # Prepare input
    if request.messages:
        logger.info(f"Processing {len(request.messages)} chat messages")
        # Chat format
        text = TOKENIZER.apply_chat_template(
            request.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        logger.info(f"Applied chat template, text length: {len(text)}")
    else:
        text = request.prompt
        logger.info(f"Using direct prompt, length: {len(text) if text else 0}")

    if not text:
        logger.error("No text input provided")
        raise HTTPException(status_code=400, detail="No prompt provided")

    # Tokenize
    logger.info("Tokenizing input text")
    inputs = TOKENIZER(text, return_tensors="pt")
    logger.info(f"Tokenized input shape: {inputs['input_ids'].shape}")
    
    if hasattr(MODEL, "device"):
        logger.info(f"Moving inputs to device: {MODEL.device}")
        inputs = {k: v.to(MODEL.device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        generation_kwargs = {
            "max_new_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "do_sample": request.temperature > 0,
            "pad_token_id": TOKENIZER.eos_token_id,
        }
        
        logger.info(f"Generation parameters: {generation_kwargs}")

        if request.stop_sequences:
            generation_kwargs["stop_strings"] = request.stop_sequences
            logger.info(f"Stop sequences: {request.stop_sequences}")

        logger.info("Starting generation...")
        outputs = MODEL.generate(**inputs, **generation_kwargs)
        logger.info(f"Generation complete, output shape: {outputs.shape}")

    # Decode
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    logger.info(f"Decoding {len(generated_tokens)} generated tokens (excluded {input_length} input tokens)")
    
    generated_text = TOKENIZER.decode(
        generated_tokens,
        skip_special_tokens=True
    )
    
    logger.info(f"Generated text length: {len(generated_text)}")
    logger.info(f"Generated text preview: {generated_text[:200]}...")

    # Calculate usage
    usage = {
        "prompt_tokens": input_length,
        "completion_tokens": len(generated_tokens),
        "total_tokens": outputs.shape[1]
    }
    logger.info(f"Token usage: {usage}")
    logger.info("=== Text generation complete ===")

    return GenerateResponse(text=generated_text, usage=usage)


async def generate_with_reasoning_legacy(request: GenerateRequest) -> GenerateResponse:
    """[LEGACY - DO NOT USE] Generate text with reasoning/thinking process."""
    import torch

    # Prepare input with thinking tags
    if request.messages:
        text = TOKENIZER.apply_chat_template(
            request.messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        text = request.prompt

    # Add thinking instruction if not present
    if "<thinking>" not in text:
        text = f"{text}\n\nPlease think step by step about this problem."

    # Tokenize
    inputs = TOKENIZER(text, return_tensors="pt")
    if hasattr(MODEL, "device"):
        inputs = {k: v.to(MODEL.device) for k, v in inputs.items()}

    # Generate with extended limits for thinking
    with torch.no_grad():
        generation_kwargs = {
            "max_new_tokens": request.max_thinking_tokens + request.max_answer_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "do_sample": request.temperature > 0,
            "pad_token_id": TOKENIZER.eos_token_id,
        }

        outputs = MODEL.generate(**inputs, **generation_kwargs)

    # Decode full output
    full_output = TOKENIZER.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    # Parse thinking and answer
    thinking = ""
    answer = full_output

    # Try to extract thinking process
    if "<thinking>" in full_output and "</thinking>" in full_output:
        start = full_output.find("<thinking>") + len("<thinking>")
        end = full_output.find("</thinking>")
        thinking = full_output[start:end].strip()
        answer = full_output[end + len("</thinking>"):].strip()
    elif "```thinking" in full_output:
        # Alternative format
        start = full_output.find("```thinking") + len("```thinking")
        end = full_output.find("```", start)
        if end > start:
            thinking = full_output[start:end].strip()
            answer = full_output[end + 3:].strip()

    # Calculate usage
    usage = {
        "prompt_tokens": inputs["input_ids"].shape[1],
        "completion_tokens": outputs.shape[1] - inputs["input_ids"].shape[1],
        "total_tokens": outputs.shape[1]
    }

    response = GenerateResponse(
        text=answer if not request.show_thinking else full_output,
        usage=usage
    )

    if request.show_thinking:
        response.thinking = thinking
        response.answer = answer

    return response


async def generate_image_legacy(request: GenerateRequest) -> GenerateResponse:
    """Generate images for diffusion models."""
    import torch
    import numpy as np
    from PIL import Image

    if not request.prompt:
        raise HTTPException(status_code=400, detail="No prompt provided")

    # Log incoming request parameters
    logger.info(f"Received generation request: width={request.width}, height={request.height}, "
                f"steps={request.num_inference_steps}, guidance={request.guidance_scale}, "
                f"num_frames={request.num_frames}")

    # Set seed if provided
    if request.seed is not None:
        generator = torch.Generator(device=MODEL.device).manual_seed(request.seed)
    else:
        generator = None

    # Check if this is a video model
    model_info = get_model_config()
    model_family = model_info.get('model_family', '').lower()
    model_id = model_info.get('model_id', '').lower()
    model_class = model_info.get('config', {}).get('_class_name', '').lower()

    # Multiple ways to detect video models
    is_video_model = (
        'video' in model_family or
        any(indicator in model_id for indicator in ['video', 'text2video', 'text-to-video', 't2v']) or
        'video' in model_class
    )

    logger.info(
        f"Model detection: model_family={
            model_info.get('model_family')}, model_id={
            model_info.get('model_id')}, _class_name={
                model_info.get(
                    'config',
                    {}).get('_class_name')}, is_video_model={is_video_model}")

    # Prepare generation kwargs
    gen_kwargs = {
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "num_inference_steps": request.num_inference_steps,
        "guidance_scale": request.guidance_scale,
        "generator": generator
    }

    # For video models, we need to be more careful with dimensions
    if is_video_model:
        # Video models often have specific size requirements
        # Default to smaller sizes for video to avoid CUDA errors
        gen_kwargs["width"] = min(request.width, 512)  # Allow up to 512 for video
        gen_kwargs["height"] = min(request.height, 512)

        # Log if we're limiting the size
        if request.width > 512 or request.height > 512:
            logger.info(
                f"Limiting video size from {
                    request.width}x{
                    request.height} to {
                    gen_kwargs['width']}x{
                    gen_kwargs['height']}")

        # Add num_frames if the model supports it
        gen_kwargs["num_frames"] = request.num_frames
    else:
        gen_kwargs["width"] = request.width
        gen_kwargs["height"] = request.height

    # Generate image/video
    try:
        logger.info(
            f"Generating with params: width={
                gen_kwargs.get('width')}, height={
                gen_kwargs.get('height')}, num_frames={
                gen_kwargs.get(
                    'num_frames',
                    'N/A')}")
        output = MODEL(**gen_kwargs)
    except RuntimeError as e:
        error_msg = str(e)
        if "CUDA" in error_msg:
            if is_video_model:
                # Try with even smaller dimensions
                logger.warning(
                    f"CUDA error with dimensions {
                        gen_kwargs['width']}x{
                        gen_kwargs['height']}, trying smaller")
                gen_kwargs["width"] = 128
                gen_kwargs["height"] = 128
                gen_kwargs["num_frames"] = 8
                logger.info(
                    f"Retrying with reduced params: {
                        gen_kwargs['width']}x{
                        gen_kwargs['height']}, {
                        gen_kwargs['num_frames']} frames")
                try:
                    output = MODEL(**gen_kwargs)
                except RuntimeError as e2:
                    logger.error(f"Failed even with smaller dimensions: {str(e2)}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"GPU memory error. Try smaller dimensions or fewer frames. Error: {str(e2)}"
                    )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"GPU error during generation: {error_msg}"
                )
        else:
            raise

    # Handle different output types
    if hasattr(output, 'images') and output.images is not None:
        # Standard image generation
        image = output.images[0]
    elif hasattr(output, 'frames') and output.frames is not None:
        # Video generation - return full video
        frames = output.frames[0]  # Shape: [num_frames, height, width, channels]
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()

        # Normalize frames if needed
        if frames.dtype != np.uint8:
            if frames.min() < 0:
                frames = (frames + 1) / 2  # From [-1, 1] to [0, 1]
            frames = (frames * 255).astype(np.uint8)

        logger.info(f"Generated video with {len(frames)} frames")

        # Try to create video file
        try:
            import cv2
            import tempfile

            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                video_path = tmp_file.name

            # Get video properties
            num_frames, height, width = frames.shape[:3]
            fps = 8  # Default FPS for generated videos

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            # Write frames
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)

            out.release()

            # Read video file and encode to base64
            with open(video_path, 'rb') as f:
                video_data = base64.b64encode(f.read()).decode()

            # Clean up
            os.unlink(video_path)

            return GenerateResponse(
                video={
                    "data": video_data,
                    "frames": num_frames,
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "format": "mp4",
                    "actual_size": f"{width}x{height}",
                    "requested_size": f"{request.width}x{request.height}"
                }
            )

        except ImportError:
            logger.warning("OpenCV not available, returning frames as images")
            # Fallback: return frames as a sequence of images
            images = []
            for i, frame in enumerate(frames):
                img = Image.fromarray(frame)
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                images.append(img_str)

            return GenerateResponse(
                video={
                    "frames": images,
                    "frames_count": len(frames),
                    "width": frames.shape[2],
                    "height": frames.shape[1],
                    "format": "png_sequence",
                    "actual_size": f"{frames.shape[2]}x{frames.shape[1]}",
                    "requested_size": f"{request.width}x{request.height}"
                }
            )
    else:
        raise ValueError(f"Unknown output format from model: {type(output)}")

    # Convert to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return GenerateResponse(image=img_str)


async def generate_embeddings_legacy(request: GenerateRequest) -> GenerateResponse:
    """Generate embeddings for embedding models."""
    import torch

    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    # Encode texts
    inputs = TOKENIZER(
        request.texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    if hasattr(MODEL, "device"):
        inputs = {k: v.to(MODEL.device) for k, v in inputs.items()}

    # Generate embeddings
    with torch.no_grad():
        outputs = MODEL(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

        if request.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    # Convert to list
    embeddings_list = embeddings.cpu().numpy().tolist()

    return GenerateResponse(embeddings=embeddings_list)


async def generate_vision_legacy(request: GenerateRequest) -> GenerateResponse:
    """Generate for vision models (classification, detection)."""
    import torch
    from PIL import Image

    # Vision models expect image input
    if not hasattr(request, 'image_data'):
        raise HTTPException(
            status_code=400,
            detail="Vision models require image input"
        )

    # Decode base64 image
    try:
        import base64
        image_data = base64.b64decode(request.image_data)
        image = Image.open(BytesIO(image_data))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image data: {str(e)}"
        )

    # Process image through model
    if TOKENIZER:  # Some vision models have processors
        inputs = TOKENIZER(images=image, return_tensors="pt")
    else:
        # Manual preprocessing
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        inputs = {"pixel_values": transform(image).unsqueeze(0)}

    if hasattr(MODEL, "device"):
        inputs = {k: v.to(MODEL.device) for k, v in inputs.items()}

    # Generate predictions
    with torch.no_grad():
        outputs = MODEL(**inputs)

    # Process outputs based on model type
    model_family = MODEL_INFO.get("model_family", "")

    if model_family == "image-classification":
        # Get predicted class
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

        # Get class labels if available
        if hasattr(MODEL.config, "id2label"):
            predicted_class = MODEL.config.id2label[predicted_class_idx]
        else:
            predicted_class = f"class_{predicted_class_idx}"

        # Get confidence scores
        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidence = probs[0][predicted_class_idx].item()

        return GenerateResponse(
            text=predicted_class,
            metadata={
                "confidence": confidence,
                "class_id": predicted_class_idx,
                "all_scores": probs[0].tolist() if request.get("return_all_scores", False) else None
            }
        )

    elif model_family == "object-detection":
        # Process detection outputs
        # This is a simplified version - actual implementation depends on model
        return GenerateResponse(
            metadata={
                "boxes": outputs.get("boxes", []),
                "labels": outputs.get("labels", []),
                "scores": outputs.get("scores", [])
            }
        )

    else:
        # Generic vision output
        return GenerateResponse(
            metadata={"raw_outputs": str(outputs)}
        )


async def generate_audio_legacy(request: GenerateRequest) -> GenerateResponse:
    """Generate for audio models (STT, TTS)."""
    import torch
    import tempfile

    try:
        import soundfile as sf
    except ImportError:
        logger.warning("soundfile not installed, audio processing will be limited")
        sf = None

    model_family = MODEL_INFO.get("model_family", "")

    if model_family == "text-to-speech":
        # TTS: text input -> audio output
        if not request.prompt:
            raise HTTPException(status_code=400, detail="No text provided for TTS")

        # Generate speech
        if TOKENIZER:
            inputs = TOKENIZER(request.prompt, return_tensors="pt")
        else:
            inputs = {"input_text": request.prompt}

        with torch.no_grad():
            if hasattr(MODEL, "generate_speech"):
                audio_output = MODEL.generate_speech(**inputs)
            else:
                audio_output = MODEL(**inputs)

        # Convert to audio file
        if isinstance(audio_output, torch.Tensor):
            audio_np = audio_output.cpu().numpy()
        else:
            audio_np = audio_output

        # Save as audio file
        if sf:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_np, samplerate=22050)  # Default sample rate
                tmp_file.seek(0)
                audio_data = base64.b64encode(tmp_file.read()).decode()
                os.unlink(tmp_file.name)
        else:
            # Return raw numpy array as base64 if soundfile not available
            audio_data = base64.b64encode(audio_np.tobytes()).decode()

        return GenerateResponse(
            metadata={
                "audio": audio_data,
                "format": "wav",
                "sample_rate": 22050
            }
        )

    elif model_family == "speech-to-text":
        # STT: audio input -> text output
        if not hasattr(request, 'audio_data'):
            raise HTTPException(
                status_code=400,
                detail="Audio models require audio input"
            )

        # Decode audio
        try:
            audio_data = base64.b64decode(request.audio_data)

            if sf:
                # Load audio with soundfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp_file:
                    tmp_file.write(audio_data)
                    tmp_file.flush()
                    audio_input, sample_rate = sf.read(tmp_file.name)
            else:
                # Basic audio loading without soundfile
                # This is a simplified version - real implementation would need proper WAV parsing
                raise HTTPException(
                    status_code=500,
                    detail="soundfile library required for audio processing. Install with: pip install soundfile"
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid audio data: {str(e)}"
            )

        # Process audio through model
        if TOKENIZER:
            inputs = TOKENIZER(audio_input, sampling_rate=sample_rate, return_tensors="pt")
        else:
            inputs = {"input_values": torch.tensor(audio_input).unsqueeze(0)}

        with torch.no_grad():
            outputs = MODEL(**inputs)

        # Decode transcription
        if hasattr(TOKENIZER, "decode"):
            transcription = TOKENIZER.decode(outputs.logits.argmax(-1)[0])
        else:
            transcription = "Transcription not available"

        return GenerateResponse(text=transcription)

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio model type: {model_family}"
        )


async def generate_multimodal_legacy(request: GenerateRequest) -> GenerateResponse:
    """Generate for multimodal models."""
    import torch
    from PIL import Image
    import numpy as np
    
    logger.info("=== Starting multimodal generation ===")
    logger.info(f"Model type: {MODEL_INFO.get('model_type')}")
    logger.info(f"Model family: {MODEL_INFO.get('model_family')}")
    logger.info(f"Model ID: {MODEL_INFO.get('model_id')}")
    
    # Check if this is an image generation request
    # Based on mode or keywords in content
    is_generation_request = False
    generation_mode = getattr(request, 'generation_mode', 'auto')
    
    # Check mode first
    if request.mode in ['image', 'artistic', 'photorealistic', 'creative']:
        is_generation_request = True
        logger.info(f"Image generation triggered by mode: {request.mode}")
    else:
        # Get the actual user message content to check for keywords
        user_content = ""
        if request.messages:
            # Find the last user message
            for msg in reversed(request.messages):
                if msg.get('role', '').lower() == 'user':
                    user_content = msg.get('content', '')
                    break
        elif request.prompt:
            user_content = request.prompt
        
        logger.info(f"Checking user content for generation keywords: {user_content[:100]}...")
        
        if generation_mode == 'text2img':
            is_generation_request = True
        elif user_content and any(keyword in user_content.lower() for keyword in ['нарисуй', 'draw', 'generate', 'create image', 'paint', 'рисуй', 'сгенерируй', 'покажи', 'show']):
            is_generation_request = True
            logger.info(f"Found generation keyword in user content")
    
    logger.info(f"Is generation request: {is_generation_request}, mode: {request.mode}, generation_mode: {generation_mode}")

    # Get prompt from either prompt field or messages
    prompt = request.prompt
    if not prompt and request.messages:
        logger.info(f"Converting {len(request.messages)} messages to prompt")
        # Convert messages to a single prompt
        try:
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in request.messages])
            logger.info(f"Created prompt from messages: {prompt[:100]}...")
        except (TypeError, AttributeError):
            # Handle case where messages is not iterable
            logger.warning("Messages provided but not in expected format")
            prompt = None
    
    # Multimodal models typically take both text and image inputs
    if not prompt:
        logger.error("No prompt provided in request")
        raise HTTPException(status_code=400, detail="No prompt provided")

    # Check if image is provided
    images = []
    
    # Handle single image
    if hasattr(request, 'image_data') and request.image_data:
        logger.info("Processing single image from image_data field")
        try:
            image_data = base64.b64decode(request.image_data)
            image = Image.open(BytesIO(image_data))
            logger.info(f"Decoded image: size={image.size}, mode={image.mode}")
            images.append(image)
        except Exception as e:
            logger.warning(f"Failed to decode image: {e}")
    
    # Handle multiple images
    if hasattr(request, 'images') and request.images:
        logger.info(f"Processing {len(request.images)} images from images field")
        for i, img_data in enumerate(request.images):
            try:
                decoded = base64.b64decode(img_data)
                img = Image.open(BytesIO(decoded))
                logger.info(f"Decoded image {i+1}: size={img.size}, mode={img.mode}")
                images.append(img)
            except Exception as e:
                logger.warning(f"Failed to decode image {i+1}: {e}")
    
    logger.info(f"Total images loaded: {len(images)}")

    # Process inputs
    if TOKENIZER:
        # For Janus models, we need to prepare the conversation format
        tokenizer_type = str(type(TOKENIZER))
        logger.info(f"TOKENIZER type: {tokenizer_type}")
        logger.info(f"TOKENIZER class: {TOKENIZER.__class__.__name__}")
        
        if hasattr(TOKENIZER, 'process_one') or 'janus' in tokenizer_type.lower():
            logger.info("Detected Janus model processor")
            # Janus expects conversations format with specific role names
            if request.messages:
                # Convert standard roles to Janus format
                conversations = []
                # System message should be prepended to the first user message
                system_content = None
                
                logger.info(f"Converting {len(request.messages)} messages to Janus format")
                for i, msg in enumerate(request.messages):
                    role = msg['role']
                    content = msg['content']
                    logger.info(f"Message {i}: role='{role}', content_length={len(content)}")
                    
                    if role.lower() == 'system':
                        # Skip system messages entirely for Janus
                        logger.info(f"Skipping system message: {content[:100]}...")
                        continue
                    elif role.lower() == 'user':
                        # Use proper Janus Pro format with angle brackets
                        role = '<|User|>'
                        logger.info(f"Converted 'user' to '{role}'")
                    elif role.lower() == 'assistant':
                        # Use proper Janus Pro format with angle brackets
                        role = '<|Assistant|>'
                        logger.info(f"Converted 'assistant' to '{role}'")
                    
                    # For image generation requests, add special prefix to the last user message
                    if is_generation_request and role == '<|User|>' and i == len(request.messages) - 1:
                        # Add generation instruction prefix for Janus
                        content = f"Generate an image: {content}"
                        logger.info(f"Added generation prefix to user message")
                    
                    conversations.append({
                        "role": role,
                        "content": content
                    })
                    logger.info(f"Added conversation entry: role='{role}', content='{content[:50]}...'")
                
                # For Janus, we don't need to add empty Assistant message
                # The model will generate from the last User message
            else:
                # Convert single prompt to conversation format
                content = prompt
                if is_generation_request:
                    content = f"Generate an image: {prompt}"
                    logger.info(f"Added generation prefix to prompt")
                conversations = [
                    {"role": "<|User|>", "content": content}
                ]
            
            logger.info(f"Final conversations list has {len(conversations)} messages")
            
            try:
                if images:
                    logger.info(f"Calling Janus processor with {len(images)} images")
                    # Janus expects a list of images
                    inputs = TOKENIZER(
                        conversations=conversations,
                        images=images,
                        return_tensors="pt"
                    )
                    logger.info(f"Janus processor returned inputs of type: {type(inputs)}")
                else:
                    logger.info("Calling Janus processor without images")
                    # Pass empty list instead of None for images
                    inputs = TOKENIZER(
                        conversations=conversations,
                        images=[],
                        return_tensors="pt"
                    )
                    logger.info(f"Janus processor returned inputs of type: {type(inputs)}")
            except Exception as e:
                logger.error(f"Error in Janus processor: {e}")
                # Try alternative format
                logger.info("Trying alternative format with single message")
                try:
                    # Try without images parameter when no image
                    if images:
                        inputs = TOKENIZER(
                            conversations=[{"role": "user", "content": prompt}],
                            images=images,
                            return_tensors="pt"
                        )
                    else:
                        # Try calling without images parameter at all
                        inputs = TOKENIZER(
                            conversations=[{"role": "user", "content": prompt}],
                            return_tensors="pt"
                        )
                except Exception as e2:
                    logger.error(f"Alternative format also failed: {e2}")
                    # Last resort - try text-only format
                    if hasattr(TOKENIZER, 'tokenizer'):
                        inputs = TOKENIZER.tokenizer(
                            prompt,
                            return_tensors="pt",
                            padding=True
                        )
                    else:
                        raise
        else:
            # Standard transformers processor
            if images:
                # For standard multimodal models, usually only one image is supported
                inputs = TOKENIZER(
                    text=prompt,
                    images=images[0] if len(images) == 1 else images,
                    return_tensors="pt",
                    padding=True
                )
            else:
                inputs = TOKENIZER(
                    text=prompt,
                    return_tensors="pt",
                    padding=True
                )
    else:
        raise HTTPException(
            status_code=500,
            detail="No tokenizer/processor available for multimodal model"
        )

    # Move inputs to device
    if hasattr(MODEL, "device"):
        logger.info(f"Moving inputs to device: {MODEL.device}")
        if hasattr(inputs, "to"):
            # For Janus BatchedVLChatProcessorOutput or similar objects
            inputs = inputs.to(MODEL.device)
            logger.info("Moved BatchedVLChatProcessorOutput to device")
        elif isinstance(inputs, dict):
            # For standard dict inputs
            inputs = {k: v.to(MODEL.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            logger.info(f"Moved dict inputs to device, keys: {list(inputs.keys())}")
        else:
            logger.warning(f"Unknown input type: {type(inputs)}, cannot move to device")

    # Generate
    # Clear any cached states for Janus models
    if hasattr(MODEL, 'language_model'):
        # Try different cache clearing methods
        if hasattr(MODEL.language_model, 'clear_cache'):
            MODEL.language_model.clear_cache()
        if hasattr(MODEL.language_model, 'model') and hasattr(MODEL.language_model.model, 'clear_cache'):
            MODEL.language_model.model.clear_cache()
        # Clear past key values if they exist
        if hasattr(MODEL.language_model, 'past_key_values'):
            MODEL.language_model.past_key_values = None
    
    # Clear CUDA cache
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        generation_kwargs = {
            "max_new_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "do_sample": request.temperature > 0,
            "repetition_penalty": 1.2,  # Add repetition penalty to avoid loops
        }

        # Debug: Check what methods the model has
        logger.info(f"Model type: {type(MODEL)}")
        logger.info(f"Model class: {MODEL.__class__.__name__ if hasattr(MODEL, '__class__') else 'Unknown'}")
        logger.info(f"Model has generate: {hasattr(MODEL, 'generate')}")
        logger.info(f"Model has language_model: {hasattr(MODEL, 'language_model')}")
        logger.info(f"Generation kwargs: {generation_kwargs}")
        
        # For Janus models, the generate method is on the language_model attribute
        actual_model = MODEL
        if hasattr(MODEL, "language_model") and hasattr(MODEL.language_model, "generate"):
            actual_model = MODEL.language_model
            logger.info("Using MODEL.language_model for generation")
        
        if hasattr(actual_model, "generate"):
            # For Janus models, we need to unpack the BatchedVLChatProcessorOutput
            if hasattr(inputs, "__dict__") and not isinstance(inputs, dict):
                # Get the actual model inputs from the processor output
                # Janus typically uses these standard transformer inputs
                model_inputs = {}
                
                # Standard transformer model inputs
                # For Janus, inputs_embeds is the key input
                standard_keys = ['inputs_embeds', 'input_ids', 'attention_mask', 'position_ids', 
                                'pixel_values', 'image_positions', 'labels', 
                                'token_type_ids']
                
                for key in standard_keys:
                    if hasattr(inputs, key):
                        value = getattr(inputs, key)
                        if value is not None:
                            model_inputs[key] = value
                
                # Also check for any tensor attributes not in standard list
                for attr_name in dir(inputs):
                    if (not attr_name.startswith("_") and 
                        attr_name not in ['sft_format', 'to', 'conversations'] and
                        attr_name not in model_inputs):
                        attr_value = getattr(inputs, attr_name)
                        if hasattr(attr_value, "shape"):  # It's a tensor
                            model_inputs[attr_name] = attr_value
                
                logger.info(f"Unpacked inputs: {list(model_inputs.keys())}")
                
                # For Janus models, we need to handle inputs_embeds specially
                if 'inputs_embeds' in model_inputs and hasattr(MODEL, 'language_model'):
                    # Janus uses inputs_embeds directly
                    logger.info("Using inputs_embeds for Janus generation")
                    
                    # Debug: log the shape of inputs_embeds
                    if 'inputs_embeds' in model_inputs:
                        logger.info(f"inputs_embeds shape: {model_inputs['inputs_embeds'].shape}")
                    if 'input_ids' in model_inputs:
                        logger.info(f"input_ids shape: {model_inputs['input_ids'].shape}")
                        # Try to decode input_ids to see what's being sent
                        if hasattr(TOKENIZER, 'tokenizer'):
                            decoded = TOKENIZER.tokenizer.decode(model_inputs['input_ids'][0], skip_special_tokens=False)
                            logger.info(f"Decoded input: {decoded[:500]}...")
                            # Also log the raw token IDs
                            logger.info(f"First 50 token IDs: {model_inputs['input_ids'][0][:50].tolist()}")
                    
                    generation_inputs = {
                        'inputs_embeds': model_inputs['inputs_embeds'],
                        'attention_mask': model_inputs.get('attention_mask'),
                        'pad_token_id': TOKENIZER.tokenizer.eos_token_id if hasattr(TOKENIZER, 'tokenizer') and hasattr(TOKENIZER.tokenizer, 'eos_token_id') else None,
                        'bos_token_id': TOKENIZER.tokenizer.bos_token_id if hasattr(TOKENIZER, 'tokenizer') and hasattr(TOKENIZER.tokenizer, 'bos_token_id') else None,
                        'eos_token_id': TOKENIZER.tokenizer.eos_token_id if hasattr(TOKENIZER, 'tokenizer') and hasattr(TOKENIZER.tokenizer, 'eos_token_id') else None,
                    }
                    # Filter out None values
                    generation_inputs = {k: v for k, v in generation_inputs.items() if v is not None}
                    outputs = actual_model.generate(**generation_inputs, **generation_kwargs)
                elif model_inputs:
                    # For other models, filter out vision-specific inputs that generate doesn't accept
                    text_only_inputs = {}
                    for key in ['input_ids', 'attention_mask', 'position_ids', 'token_type_ids']:
                        if key in model_inputs:
                            text_only_inputs[key] = model_inputs[key]
                    
                    if text_only_inputs:
                        outputs = actual_model.generate(**text_only_inputs, **generation_kwargs)
                    else:
                        outputs = actual_model.generate(**model_inputs, **generation_kwargs)
                else:
                    # Try with just input_ids if available
                    if hasattr(inputs, "input_ids") and inputs.input_ids is not None:
                        outputs = actual_model.generate(input_ids=inputs.input_ids, **generation_kwargs)
                    else:
                        raise ValueError(f"Could not extract valid inputs from {type(inputs)}")
            else:
                outputs = actual_model.generate(**inputs, **generation_kwargs)

            # Decode output
            if hasattr(TOKENIZER, 'decode'):
                generated_text = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
            elif hasattr(TOKENIZER, 'tokenizer') and hasattr(TOKENIZER.tokenizer, 'decode'):
                # For Janus VLChatProcessor
                # Get only the generated tokens (exclude input)
                if hasattr(inputs, 'input_ids') and inputs.input_ids is not None:
                    input_length = inputs.input_ids.shape[-1]
                    generated_tokens = outputs[0][input_length:]
                    generated_text = TOKENIZER.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                else:
                    generated_text = TOKENIZER.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                logger.warning("No decode method found, returning raw output")
                generated_text = str(outputs[0].tolist())

            # Log the raw generated text
            logger.info(f"Raw generated text length: {len(generated_text)}")
            logger.info(f"Raw generated text preview: {generated_text[:200]}...")
            
            # For Janus models, extract only the assistant's response
            if 'janus' in str(type(TOKENIZER)).lower() or hasattr(TOKENIZER, 'tokenizer'):
                # The model outputs the entire conversation including the prompt
                # We need to extract only the new generation
                
                # First, try to find where the assistant's response starts
                # Look for the last occurrence of "assistant:" or "system:"
                lower_text = generated_text.lower()
                
                # Find all occurrences of role markers (both lowercase and capitalized)
                markers = []
                role_variants = [
                    'assistant:', 'system:', 'user:',
                    'Assistant:', 'System:', 'User:',
                    '<|assistant|>', '<|system|>', '<|user|>',
                    '<|Assistant|>', '<|System|>', '<|User|>'
                ]
                
                for role in role_variants:
                    pos = 0
                    search_text = generated_text if role[0].isupper() or '<' in role else lower_text
                    while True:
                        idx = search_text.find(role, pos)
                        if idx == -1:
                            break
                        markers.append((idx, role, role.strip(':><|').lower()))
                        pos = idx + 1
                
                # Sort markers by position
                markers.sort(key=lambda x: x[0])
                
                # Find the last assistant/system response
                if markers:
                    # Find the last assistant marker specifically
                    assistant_markers = [(idx, role, role_type) for idx, role, role_type in markers 
                                       if role_type in ['assistant', 'system']]
                    
                    if assistant_markers:
                        # Get the last assistant marker
                        last_marker_idx, last_marker_role, _ = assistant_markers[-1]
                        
                        # Extract text after the last assistant role marker
                        generated_text = generated_text[last_marker_idx + len(last_marker_role):].strip()
                    else:
                        # No assistant marker found, get text after last marker
                        last_marker_idx, last_marker_role, _ = markers[-1]
                        generated_text = generated_text[last_marker_idx + len(last_marker_role):].strip()
                    
                    # If there's another role marker after this, cut off there
                    next_marker_pos = len(generated_text)
                    remaining_lower = generated_text.lower()
                    for role in ['user:', 'assistant:', 'system:']:
                        idx = remaining_lower.find(role)
                        if idx != -1 and idx < next_marker_pos:
                            next_marker_pos = idx
                    
                    if next_marker_pos < len(generated_text):
                        generated_text = generated_text[:next_marker_pos].strip()
                
                # Additional cleanup - remove any remaining context
                if request.prompt and request.prompt in generated_text:
                    # Find where the prompt ends and take everything after
                    prompt_end = generated_text.find(request.prompt) + len(request.prompt)
                    generated_text = generated_text[prompt_end:].strip()

            # Additional cleanup for Janus - remove repetitive disclaimers
            if 'janus' in str(type(TOKENIZER)).lower():
                # Look for common disclaimer patterns
                disclaimer_patterns = [
                    "(Translation:",
                    "(Note: This is a fictional",
                    "This is a fictional assistant",
                    "(Note:",
                    "(Translation: This is a fictional"
                ]
                
                for pattern in disclaimer_patterns:
                    if pattern in generated_text:
                        # Cut off at the first disclaimer
                        generated_text = generated_text[:generated_text.find(pattern)].strip()
                        break
            
            # Check if this is image generation for Janus
            if is_generation_request and 'janus' in MODEL_INFO.get('model_id', '').lower():
                logger.info("Detected Janus image generation request")
                
                # First check if model has required components
                if not hasattr(MODEL, 'gen_head') or not hasattr(MODEL, 'gen_vision_model'):
                    logger.warning("Model missing gen_head or gen_vision_model, cannot generate images")
                    logger.info("Model attributes: " + str([attr for attr in dir(MODEL) if not attr.startswith('_')]))
                    # Fall through to text generation
                else:
                    # For Janus image generation, we need to use special generation process
                    try:
                        # For image generation, only use the current prompt without history
                        current_prompt = request.prompt or (request.messages[-1]["content"] if request.messages else "")
                        image_conversations = [{"role": "User", "content": current_prompt}]
                        
                        # Prepare prompt for image generation
                        sft_format = TOKENIZER.apply_sft_template_for_multi_turn_prompts(
                            conversations=image_conversations,
                            sft_format=TOKENIZER.sft_format,
                            system_prompt="",
                        )
                        gen_prompt = sft_format + TOKENIZER.image_start_tag
                        logger.info(f"Image generation prompt: {gen_prompt[:100]}...")
                        
                        # Tokenize prompt
                        input_ids = TOKENIZER.tokenizer.encode(gen_prompt)
                        input_ids = torch.LongTensor(input_ids)
                        
                        # Janus requires parallel processing for CFG (Classifier-Free Guidance)
                        # Create batch with conditional and unconditional inputs
                        parallel_size = 1
                        batch_size = parallel_size * 2  # conditional + unconditional
                        tokens = torch.zeros((batch_size, len(input_ids)), dtype=torch.long).to(MODEL.device)
                        
                        for i in range(batch_size):
                            tokens[i, :] = input_ids
                            # Make odd indices unconditional (padded)
                            if i % 2 != 0:
                                # Check for pad_token_id in various places
                                pad_id = getattr(TOKENIZER, 'pad_token_id', None) or \
                                        getattr(TOKENIZER, 'pad_id', None) or \
                                        getattr(TOKENIZER.tokenizer, 'pad_token_id', None) or \
                                        0  # Default to 0 if no pad token found
                                if pad_id is not None:
                                    tokens[i, 1:-1] = pad_id
                                else:
                                    logger.warning("No pad token found, using token 0")
                        
                        # Parameters for image generation
                        temperature = request.temperature if request.temperature > 0 else 1.0
                        cfg_weight = 5.0  # Classifier-free guidance weight
                        image_token_num_per_image = 576  # Standard for Janus
                        img_size = 384
                        patch_size = 16
                        
                        # Get input embeddings for batch
                        inputs_embeds = MODEL.language_model.get_input_embeddings()(tokens)
                        logger.info(f"Initial inputs_embeds shape: {inputs_embeds.shape}")
                        
                        # Generate image tokens
                        generated_tokens = []
                        outputs = None
                        
                        logger.info(f"Generating {image_token_num_per_image} image tokens...")
                        logger.info(f"Batch size: {batch_size}, Parallel size: {parallel_size}")
                        
                        # Clear cache before generation to free up memory
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        for i in range(image_token_num_per_image):
                            # Forward pass
                            outputs = MODEL.language_model.model(
                                inputs_embeds=inputs_embeds, 
                                use_cache=True, 
                                past_key_values=outputs.past_key_values if outputs else None
                            )
                            hidden_states = outputs.last_hidden_state
                            
                            if i == 0:  # Log shapes on first iteration
                                logger.info(f"Hidden states shape: {hidden_states.shape}")
                            
                            # Get logits from generation head
                            if hasattr(MODEL, 'gen_head'):
                                # Extract last hidden state for each sequence in batch
                                last_hidden = hidden_states[:, -1, :]  # Shape: [batch_size, hidden_dim]
                                if i == 0:
                                    logger.info(f"Last hidden shape: {last_hidden.shape}")
                                logits = MODEL.gen_head(last_hidden)  # Shape: [batch_size, vocab_size]
                                if i == 0:
                                    logger.info(f"Logits shape: {logits.shape}")
                            else:
                                logger.error("Model does not have gen_head for image generation")
                                raise ValueError("Model missing gen_head")
                            
                            # Apply classifier-free guidance
                            # logits shape: [batch_size, vocab_size] where batch_size = parallel_size * 2
                            if logits.shape[0] >= 2:
                                logit_cond = logits[0::2, :]  # Conditional (even indices)
                                logit_uncond = logits[1::2, :]  # Unconditional (odd indices)
                                
                                if i == 0:
                                    logger.info(f"CFG shapes - cond: {logit_cond.shape}, uncond: {logit_uncond.shape}")
                                
                                # Ensure same dimensions before combining
                                if logit_cond.dim() != logit_uncond.dim():
                                    logger.error(f"Dimension mismatch: cond={logit_cond.shape}, uncond={logit_uncond.shape}")
                                    raise ValueError(f"Tensor dimension mismatch: conditional has {logit_cond.dim()} dims, unconditional has {logit_uncond.dim()} dims")
                                
                                # Check shape compatibility
                                if logit_cond.shape != logit_uncond.shape:
                                    logger.error(f"Shape mismatch: cond={logit_cond.shape}, uncond={logit_uncond.shape}")
                                    raise ValueError(f"Tensor shape mismatch: conditional shape {logit_cond.shape} != unconditional shape {logit_uncond.shape}")
                                
                                logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
                            else:
                                logger.warning(f"Batch size too small for CFG: {logits.shape[0]}, using logits as-is")
                            
                            # Sample next token
                            probs = torch.softmax(logits / temperature, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)  # Shape: [parallel_size, 1]
                            generated_tokens.append(next_token[0, 0].item())
                            
                            # Prepare next input for both conditional and unconditional
                            # Repeat tokens for both conditional and unconditional paths
                            next_token_batch = next_token.repeat(2, 1).squeeze(-1)  # Shape: [parallel_size * 2]
                            
                            if hasattr(MODEL, 'prepare_gen_img_embeds'):
                                img_embeds = MODEL.prepare_gen_img_embeds(next_token_batch)
                                inputs_embeds = img_embeds.unsqueeze(dim=1)
                            else:
                                # Fallback to regular embedding
                                inputs_embeds = MODEL.language_model.get_input_embeddings()(next_token_batch).unsqueeze(1)
                        
                        logger.info(f"Generated {len(generated_tokens)} tokens")
                        
                        # Decode tokens to image
                        generated_tokens_tensor = torch.tensor(generated_tokens).unsqueeze(0).to(MODEL.device)
                        
                        if hasattr(MODEL, 'gen_vision_model') and hasattr(MODEL.gen_vision_model, 'decode_code'):
                            logger.info("Decoding image tokens...")
                            dec = MODEL.gen_vision_model.decode_code(
                                generated_tokens_tensor.to(dtype=torch.int), 
                                shape=[1, 8, img_size//patch_size, img_size//patch_size]
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
                            
                            logger.info("Image generation successful")
                            
                            # Clear CUDA cache after generation
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            
                            logger.info("=== Multimodal generation complete ===")
                            
                            return GenerateResponse(
                                text="Generated image successfully",
                                image=img_base64,
                                metadata={
                                    "width": img_size,
                                    "height": img_size,
                                    "format": "png"
                                }
                            )
                        else:
                            logger.error("Model does not have gen_vision_model for decoding")
                            raise ValueError("Model missing gen_vision_model")
                            
                    except Exception as e:
                        logger.error(f"Error in Janus image generation: {e}")
                        logger.info("Falling back to text generation")
                        
                        # Clear CUDA cache on error
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Fall through to regular text response
            
            # Final logging
            logger.info(f"Final generated text length: {len(generated_text)}")
            logger.info(f"Final generated text: {generated_text[:200]}...")
            logger.info("=== Multimodal generation complete ===")
            
            return GenerateResponse(text=generated_text)
        else:
            # Non-generative multimodal model (e.g., CLIP)
            # For these models, we need to be careful about what we pass
            if hasattr(inputs, "__dict__") and not isinstance(inputs, dict):
                # Extract only valid inputs
                model_inputs = {}
                for key in ['input_ids', 'attention_mask', 'pixel_values', 'token_type_ids']:
                    if hasattr(inputs, key):
                        value = getattr(inputs, key)
                        if value is not None:
                            model_inputs[key] = value
                
                if model_inputs:
                    outputs = MODEL(**model_inputs)
                else:
                    raise ValueError(f"Could not extract valid inputs from {type(inputs)}")
            else:
                outputs = MODEL(**inputs)

            # Return embeddings or similarity scores
            return GenerateResponse(
                metadata={
                    "text_features": outputs.text_features.tolist() if hasattr(outputs, "text_features") else None,
                    "image_features": outputs.image_features.tolist() if hasattr(outputs, "image_features") else None,
                    "logits": outputs.logits.tolist() if hasattr(outputs, "logits") else None
                }
            )


@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    """Streaming generation endpoint using Server-Sent Events."""
    if not MODEL:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if MODEL_INFO.get("model_family") != "language-model":
        raise HTTPException(
            status_code=400,
            detail="Streaming only supported for language models"
        )

    async def event_generator():
        import torch
        import asyncio
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        try:
            # Prepare messages with system prompt
            messages = []
            if request.messages:
                messages = request.messages.copy()
            elif request.prompt:
                messages = [{"role": "user", "content": request.prompt}]
            
            # Apply chat template
            if hasattr(TOKENIZER, 'apply_chat_template'):
                text = TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            
            # Tokenize
            inputs = TOKENIZER(text, return_tensors="pt", truncation=True)
            if hasattr(MODEL, 'device'):
                inputs = {k: v.to(MODEL.device) for k, v in inputs.items()}
            
            # Create streamer
            streamer = TextIteratorStreamer(TOKENIZER, skip_prompt=True, skip_special_tokens=True)
            
            # Generation kwargs
            generation_kwargs = {
                **inputs,
                'streamer': streamer,
                'max_new_tokens': request.max_tokens,
                'temperature': request.temperature,
                'top_p': request.top_p,
                'top_k': request.top_k,
                'do_sample': request.temperature > 0,
                'pad_token_id': TOKENIZER.pad_token_id,
                'eos_token_id': TOKENIZER.eos_token_id,
            }
            
            # Start generation in separate thread
            thread = Thread(target=MODEL.generate, kwargs=generation_kwargs)
            thread.start()
            
            # Stream tokens
            generated_text = ""
            in_thinking = False
            thinking_text = ""
            
            for new_text in streamer:
                if new_text:
                    generated_text += new_text
                    
                    # Check for thinking tags (for Qwen3)
                    if '<think>' in new_text:
                        in_thinking = True
                    if '</think>' in new_text:
                        in_thinking = False
                        # Send the complete thinking block
                        if thinking_text:
                            yield {
                                "data": json.dumps({
                                    "token": new_text,
                                    "text": generated_text,
                                    "thinking": thinking_text,
                                    "finished": False
                                })
                            }
                            thinking_text = ""
                            await asyncio.sleep(0)
                            continue
                    
                    if in_thinking:
                        thinking_text += new_text
                    
                    yield {
                        "data": json.dumps({
                            "token": new_text,
                            "text": generated_text,
                            "is_thinking": in_thinking,
                            "finished": False
                        })
                    }
                    await asyncio.sleep(0)  # Allow other tasks to run
            
            # Final message
            yield {
                "data": json.dumps({
                    "text": generated_text,
                    "finished": True,
                    "usage": {
                        "prompt_tokens": len(inputs['input_ids'][0]),
                        "completion_tokens": len(TOKENIZER.encode(generated_text)),
                        "total_tokens": len(inputs['input_ids'][0]) + len(TOKENIZER.encode(generated_text))
                    }
                })
            }
            
            thread.join()
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield {
                "data": json.dumps({
                    "error": str(e),
                    "finished": True
                })
            }

    return EventSourceResponse(event_generator())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()

            # Process request
            request = GenerateRequest(**data)
            response = await generate(request)

            # Send response
            await websocket.send_json(response.dict())

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle file uploads for multimodal models."""
    if MODEL_INFO.get("model_family") not in ["multimodal", "vision-language"]:
        raise HTTPException(
            status_code=400,
            detail="File upload only supported for multimodal models"
        )

    # Save uploaded file temporarily
    contents = await file.read()

    return {
        "filename": file.filename,
        "size": len(contents),
        "content_type": file.content_type
    }


@app.post("/generate/video")
async def generate_video(request: GenerateRequest):
    """Generate full video output for video models."""
    if MODEL_INFO.get("model_family") != "video-generation":
        raise HTTPException(
            status_code=400,
            detail="Video generation only supported for video models"
        )

    import torch
    import tempfile
    import numpy as np

    if not request.prompt:
        raise HTTPException(status_code=400, detail="No prompt provided")

    # Set seed if provided
    if request.seed is not None:
        generator = torch.Generator(device=MODEL.device).manual_seed(request.seed)
    else:
        generator = None

    # Generate video
    output = MODEL(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        width=request.width,
        height=request.height,
        generator=generator
    )

    # Get frames
    frames = output.frames[0]  # Shape: [num_frames, height, width, channels]
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy()

    # Normalize frames if needed
    if frames.dtype != np.uint8:
        if frames.min() < 0:
            frames = (frames + 1) / 2  # From [-1, 1] to [0, 1]
        frames = (frames * 255).astype(np.uint8)

    # Save as video file
    try:
        # Try to use OpenCV if available
        import cv2

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
            video_path = tmp_file.name

        # Get video properties
        num_frames, height, width = frames.shape[:3]
        fps = 8  # Default FPS for generated videos

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        # Write frames
        for frame in frames:
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)

        out.release()

        # Read video file and encode to base64
        with open(video_path, 'rb') as f:
            video_data = base64.b64encode(f.read()).decode()

        # Clean up
        os.unlink(video_path)

        return {
            "video": video_data,
            "metadata": {
                "frames": num_frames,
                "fps": fps,
                "width": width,
                "height": height,
                "format": "mp4"
            }
        }

    except ImportError:
        # OpenCV not available, return frames as images
        logger.warning("OpenCV not available, returning frames as images")

        # Convert frames to base64 encoded images
        images = []
        for i, frame in enumerate(frames):
            from PIL import Image
            img = Image.fromarray(frame)
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            images.append(img_str)

        return {
            "frames": images,
            "metadata": {
                "frames": len(frames),
                "width": width,
                "height": height,
                "format": "png_sequence"
            }
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--load-lora", type=str, default=None)
    parser.add_argument("--stream-mode", type=str, default="false")
    args = parser.parse_args()

    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()

"""Secure API server for LLM Installer with proper validation and rate limiting.

This FastAPI server includes security improvements:
- Input validation and sanitization
- Rate limiting
- Request size limits
- Proper CORS configuration
- Better error handling
"""

import os
import json
import argparse
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path
import base64
from io import BytesIO
import hashlib
import time
from collections import defaultdict
import re

# Set CUDA memory allocation configuration to prevent fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, File, UploadFile, Request, status
from fastapi.responses import Response
from starlette.responses import HTMLResponse
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Annotated
from sse_starlette.sse import EventSourceResponse
import uvicorn

# Import model loader
from model_loader import load_model, get_model_config

# Import constants
try:
    # When run from installed model directory, need to add parent to path
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.constants import API_LIMITS
except ImportError:
    # Fallback to hardcoded values if constants not available
    API_LIMITS = {
        'max_request_size': 10 * 1024 * 1024,
        'max_image_size': 5 * 1024 * 1024,
        'max_text_length': 100000,
        'max_batch_size': 10,
        'rate_limit_requests': 60,
        'rate_limit_window': 60,
    }

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security constants from module
MAX_REQUEST_SIZE = API_LIMITS['max_request_size']
MAX_IMAGE_SIZE = API_LIMITS['max_image_size']
MAX_TEXT_LENGTH = API_LIMITS['max_text_length']
MAX_BATCH_SIZE = API_LIMITS['max_batch_size']
RATE_LIMIT_REQUESTS = API_LIMITS['rate_limit_requests']
RATE_LIMIT_WINDOW = API_LIMITS['rate_limit_window']

# Allowed CORS origins (configure based on your needs)
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]

# Initialize FastAPI app with security settings
app = FastAPI(
    title="LLM Installer Model API (Secure)",
    description="Secure API for interacting with installed models",
    version="2.0.0",
    docs_url="/docs",  # Keep docs but could disable in production
    redoc_url=None,    # Disable redoc in production
)

# Configure CORS with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Specific origins instead of ["*"]
    allow_credentials=True,
    allow_methods=["GET", "POST"],   # Only required methods
    allow_headers=["Content-Type", "Authorization"],
    max_age=86400,  # Cache preflight requests
)

# Rate limiter storage
rate_limiter = defaultdict(lambda: {"requests": 0, "reset_time": time.time() + RATE_LIMIT_WINDOW})

# Global variables for model and config
MODEL = None
TOKENIZER = None
MODEL_INFO = None
DEVICE = None
DTYPE = None
HANDLER = None
STREAM_MODE = False
UI_FILE_PATH = None


def sanitize_text(text: str) -> str:
    """Sanitize text input to prevent injection attacks."""
    if not text:
        return text
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Limit length
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
    
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    
    return text


def validate_base64_image(data: str) -> bytes:
    """Validate and decode base64 image data."""
    try:
        # Remove data URL prefix if present
        if data.startswith('data:'):
            data = data.split(',', 1)[1]
        
        # Decode base64
        decoded = base64.b64decode(data)
        
        # Check size
        if len(decoded) > MAX_IMAGE_SIZE:
            raise ValueError(f"Image too large: {len(decoded)} bytes (max {MAX_IMAGE_SIZE})")
        
        # Validate it's actually an image
        from PIL import Image
        img = Image.open(BytesIO(decoded))
        img.verify()  # Verify it's a valid image
        
        return decoded
    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")


class SecureGenerateRequest(BaseModel):
    """Secure request model with validation."""
    model_config = {"extra": "forbid"}
    
    prompt: Optional[str] = Field(None, max_length=MAX_TEXT_LENGTH)
    messages: Optional[List[Dict[str, str]]] = Field(None, max_items=100)
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.7
    max_tokens: Annotated[int, Field(ge=1, le=131072)] = 4096
    top_p: Annotated[float, Field(ge=0.0, le=1.0)] = 0.9
    top_k: Annotated[int, Field(ge=0, le=100)] = 50
    stop_sequences: Optional[List[str]] = Field(None, max_items=10)
    stream: bool = False
    mode: Optional[str] = Field("auto", pattern="^[a-zA-Z0-9_-]+$", max_length=50)
    
    # Image generation parameters
    negative_prompt: Optional[str] = Field(None, max_length=1000)
    num_inference_steps: Annotated[int, Field(ge=1, le=1000)] = 50
    guidance_scale: Annotated[float, Field(ge=0.0, le=20.0)] = 7.5
    width: Annotated[int, Field(ge=64, le=2048)] = 512
    height: Annotated[int, Field(ge=64, le=2048)] = 512
    seed: Optional[Annotated[int, Field(ge=0)]] = None
    
    # Video generation parameters
    num_frames: Annotated[int, Field(ge=1, le=128)] = 16
    
    # Embedding parameters
    texts: Optional[List[str]] = Field(None, max_items=MAX_BATCH_SIZE)
    normalize: bool = True
    
    # Multimodal parameters
    image_data: Optional[str] = Field(None, max_length=10_000_000)  # ~7.5MB base64
    images: Optional[List[str]] = Field(None, max_items=5)
    audio_data: Optional[str] = Field(None, max_length=10_000_000)
    
    # Reasoning parameters
    reasoning_mode: bool = False
    max_thinking_tokens: Annotated[int, Field(ge=1, le=50000)] = 10000
    max_answer_tokens: Annotated[int, Field(ge=1, le=10000)] = 2000
    show_thinking: bool = True
    
    # Additional options
    return_all_scores: bool = False
    max_image_size: Optional[int] = Field(None, ge=256, le=4096)
    
    @field_validator('prompt')
    @classmethod
    def sanitize_prompt(cls, v):
        return sanitize_text(v) if v else v
    
    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v):
        if not v:
            return v
        
        for msg in v:
            if 'role' not in msg or 'content' not in msg:
                raise ValueError("Each message must have 'role' and 'content'")
            
            if msg['role'] not in ['system', 'user', 'assistant']:
                raise ValueError(f"Invalid role: {msg['role']}")
            
            msg['content'] = sanitize_text(msg['content'])
            
            if len(msg['content']) > MAX_TEXT_LENGTH:
                raise ValueError(f"Message content too long")
        
        return v
    
    @field_validator('texts')
    @classmethod
    def validate_texts(cls, v):
        if not v:
            return v
        
        return [sanitize_text(text)[:1000] for text in v]  # Limit each text
    
    @field_validator('stop_sequences')
    @classmethod
    def validate_stop_sequences(cls, v):
        if not v:
            return v
        
        return [seq[:100] for seq in v]  # Limit length of stop sequences


@app.middleware("http")
async def security_middleware(request: Request, call_next):
    """Security middleware for rate limiting and request validation."""
    
    # Check request size
    if request.headers.get("content-length"):
        content_length = int(request.headers["content-length"])
        if content_length > MAX_REQUEST_SIZE:
            return Response(
                content="Request too large",
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            )
    
    # Rate limiting by IP
    client_ip = request.client.host
    current_time = time.time()
    
    # Clean up old entries
    rate_limiter[client_ip] = {
        "requests": rate_limiter[client_ip]["requests"] + 1,
        "reset_time": rate_limiter[client_ip]["reset_time"]
    }
    
    # Reset if window expired
    if current_time > rate_limiter[client_ip]["reset_time"]:
        rate_limiter[client_ip] = {
            "requests": 1,
            "reset_time": current_time + RATE_LIMIT_WINDOW
        }
    
    # Check rate limit
    if rate_limiter[client_ip]["requests"] > RATE_LIMIT_REQUESTS:
        return Response(
            content="Rate limit exceeded",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers={
                "Retry-After": str(int(rate_limiter[client_ip]["reset_time"] - current_time)),
                "X-RateLimit-Limit": str(RATE_LIMIT_REQUESTS),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(rate_limiter[client_ip]["reset_time"]))
            }
        )
    
    # Add rate limit headers
    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_REQUESTS)
    response.headers["X-RateLimit-Remaining"] = str(
        max(0, RATE_LIMIT_REQUESTS - rate_limiter[client_ip]["requests"])
    )
    response.headers["X-RateLimit-Reset"] = str(int(rate_limiter[client_ip]["reset_time"]))
    
    return response


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
    lora_loaded: bool = False
    lora_path: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global MODEL, TOKENIZER, MODEL_INFO, DEVICE, DTYPE, STREAM_MODE
    
    try:
        # Parse command line arguments (passed from start.sh)
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=8000)
        parser.add_argument("--host", default="127.0.0.1")  # Default to localhost for security
        parser.add_argument("--device", default="auto")
        parser.add_argument("--dtype", default="auto")
        parser.add_argument("--load-lora", type=str, default=None)
        parser.add_argument("--stream-mode", type=str, default="false")
        parser.add_argument("--ui-file", type=str, default="serve_terminal.html", 
                            help="Path to the HTML file for the UI, relative to the script directory.")
        parser.add_argument("--allow-all-origins", action="store_true", help="Allow all CORS origins (development only)")
        args, _ = parser.parse_known_args()
        
        # Set UI file path
        global UI_FILE_PATH
        script_dir = Path(__file__).parent
        UI_FILE_PATH = script_dir / args.ui_file
        if not UI_FILE_PATH.exists():
            logger.warning(f"UI file not found at specified path: {UI_FILE_PATH}. The root endpoint '/' will not serve a UI.")
            UI_FILE_PATH = None

        # Update CORS if needed (development only)
        if args.allow_all_origins:
            logger.warning("ALLOWING ALL CORS ORIGINS - DEVELOPMENT MODE ONLY!")
            global ALLOWED_ORIGINS
            ALLOWED_ORIGINS = ["*"]
        
        # Load model info
        MODEL_INFO = get_model_config()
        logger.info(f"Loading model: {MODEL_INFO['model_id']}")
        DEVICE = args.device
        DTYPE = args.dtype
        STREAM_MODE = args.stream_mode.lower() == "true"
        
        # Validate device
        if args.device not in ["auto", "cuda", "cpu", "mps"]:
            raise ValueError(f"Invalid device: {args.device}")
        
        # Validate dtype
        if args.dtype not in ["auto", "float16", "float32", "int8", "int4"]:
            raise ValueError(f"Invalid dtype: {args.dtype}")
        
        # Load model
        logger.info(f"Loading model with device={DEVICE}, dtype={DTYPE}")
        
        # Check if LoRA exists and auto-load if not specified
        lora_loaded = False
        if args.load_lora is None:
            lora_path = Path("./lora")
            if lora_path.exists() and lora_path.is_dir():
                # Check for adapter files
                adapter_files = list(lora_path.glob("adapter_model.*"))
                if adapter_files:
                    logger.info(f"Found LoRA adapter at {lora_path}, auto-loading...")
                    args.load_lora = str(lora_path)
                    lora_loaded = True
        else:
            # Validate LoRA path
            lora_path = Path(args.load_lora)
            if not lora_path.exists():
                raise ValueError(f"LoRA path does not exist: {args.load_lora}")
            lora_loaded = True
        
        MODEL, TOKENIZER = load_model(
            MODEL_INFO,
            device=args.device,
            dtype=args.dtype,
            lora_path=args.load_lora,
            load_lora=True
        )
        
        # Update MODEL_INFO with LoRA status
        MODEL_INFO['lora_loaded'] = lora_loaded
        MODEL_INFO['lora_path'] = args.load_lora if lora_loaded else None
        
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
    global UI_FILE_PATH
    if UI_FILE_PATH and UI_FILE_PATH.exists():
        return FileResponse(UI_FILE_PATH)
    else:
        return HTMLResponse(
            content="<h1>Model API</h1><p>UI file not found or not specified. API is running at /docs</p>"
        )


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
    
    # Build capabilities
    capabilities = MODEL_INFO.get("capabilities", {})
    
    # Add handler-specific capabilities if available
    if HANDLER:
        try:
            handler_caps = HANDLER.get_model_capabilities()
            capabilities.update(handler_caps)
            
            # Get supported modes from handler
            handler_modes = HANDLER.get_supported_modes()
            handler_descriptions = HANDLER.get_mode_descriptions()
            
            capabilities["supported_modes"] = handler_modes
            capabilities["mode_descriptions"] = handler_descriptions
        except Exception as e:
            logger.warning(f"Failed to get handler capabilities: {e}")
    
    return ModelInfoResponse(
        model_id=MODEL_INFO.get("model_id", "unknown"),
        model_type=MODEL_INFO.get("model_type", "unknown"),
        model_family=MODEL_INFO.get("model_family", "unknown"),
        capabilities=capabilities,
        device=str(DEVICE),
        dtype=str(DTYPE),
        stream_mode=STREAM_MODE,
        lora_loaded=MODEL_INFO.get('lora_loaded', False),
        lora_path=MODEL_INFO.get('lora_path', None)
    )


def apply_mode_settings(request: SecureGenerateRequest, mode: str) -> SecureGenerateRequest:
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
            if hasattr(request, key) and getattr(request, key) == SecureGenerateRequest.__fields__[key].default:
                setattr(request, key, value)
    
    return request


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: SecureGenerateRequest):
    """Main generation endpoint with security validation."""
    if not MODEL:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Validate image data if present
        if request.image_data:
            try:
                validate_base64_image(request.image_data)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        if request.images:
            for img in request.images:
                try:
                    validate_base64_image(img)
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=str(e))
        
        # Apply mode-specific settings
        if request.mode and request.mode != "auto":
            request = apply_mode_settings(request, request.mode)
        
        # Use handler if available
        if not HANDLER:
            raise HTTPException(
                status_code=501,
                detail="No handler available for this model"
            )
        
        # Log request for debugging (without sensitive data)
        logger.info(f"Processing request: mode={request.mode}, has_prompt={bool(request.prompt)}, "
                   f"has_messages={bool(request.messages)}, has_images={bool(request.images or request.image_data)}")
        
        # Determine the task based on request and mode
        if request.mode == "image" or (request.mode == "auto" and 
            request.prompt and any(word in request.prompt.lower() 
                                 for word in ["draw", "generate image", "create image"])):
            # Image generation
            if hasattr(HANDLER, 'generate_image'):
                result = HANDLER.generate_image(
                    prompt=request.prompt or (request.messages[-1]["content"] if request.messages else ""),
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
            else:
                raise HTTPException(status_code=501, detail="Handler does not support image generation")
        
        elif request.images or request.image_data or request.mode == "vision" or (
            request.messages and any(
                isinstance(msg.get('content'), list) and 
                any(item.get('type') == 'image' for item in msg.get('content', []) if isinstance(item, dict))
                for msg in request.messages
            )):
            # Multimodal processing
            if hasattr(HANDLER, 'process_multimodal'):
                images = []
                if request.image_data:
                    images.append(request.image_data)
                if request.images:
                    images.extend(request.images)
                
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
                    max_image_size=request.max_image_size or 2048,
                    messages=request.messages
                )
                return GenerateResponse(**result)
            else:
                raise HTTPException(status_code=501, detail="Handler does not support multimodal processing")
        
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
            else:
                raise HTTPException(status_code=501, detail="Handler does not support text generation")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/generate/stream")
async def generate_stream(request: SecureGenerateRequest):
    """Streaming generation endpoint using Server-Sent Events."""
    if not MODEL:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Please try again shortly.")
    
    # Check if the handler supports streaming before proceeding
    if HANDLER and not HANDLER.get_model_capabilities().get("stream", False):
         raise HTTPException(
            status_code=400,
            detail="Streaming is not supported for this model type."
        )

    async def event_generator(request):
        import torch
        import asyncio
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        try:
            # Apply mode-specific settings and prepare messages in one step
            if request.mode and request.mode != "auto":
                request = apply_mode_settings(request, request.mode)
            
            messages = []
            if request.messages:
                messages = request.messages.copy()
            elif request.prompt:
                messages = [{"role": "user", "content": request.prompt}]

            # Use handler's streaming method if available (preferred)
            if HANDLER and hasattr(HANDLER, 'generate_stream'):
                logger.info("Using handler's unified generate_stream method")
                
                # The handler is now expected to be an async generator yielding dicts
                async for chunk in HANDLER.generate_stream(
                    prompt=request.prompt,
                    messages=messages,
                    model=MODEL,
                    tokenizer=TOKENIZER,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    top_p=request.top_p,
                    top_k=request.top_k,
                    stop_sequences=request.stop_sequences,
                    mode=request.mode
                ):
                    yield {"data": json.dumps(chunk)}
                    await asyncio.sleep(0.001) # Yield control to the event loop
            
            # Fallback to direct model streaming with a unified output format
            else:
                logger.info("Using direct model streaming with unified output format")
                
                if not messages:
                    raise ValueError("Prompt or messages are required for streaming.")

                if hasattr(TOKENIZER, 'apply_chat_template'):
                    text = TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                
                inputs = TOKENIZER(text, return_tensors="pt", truncation=True, max_length=4096)
                if hasattr(MODEL, 'device'):
                    inputs = {k: v.to(MODEL.device) for k, v in inputs.items()}
                
                streamer = TextIteratorStreamer(TOKENIZER, skip_prompt=True, skip_special_tokens=True)
                
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
                
                thread = Thread(target=MODEL.generate, kwargs=generation_kwargs)
                thread.start()
                
                generated_text = ""
                for new_text in streamer:
                    if new_text:
                        generated_text += new_text
                        yield {
                            "data": json.dumps({
                                "type": "text",
                                "token": new_text,
                                "text": generated_text,
                                "finished": False
                            })
                        }
                        await asyncio.sleep(0)
                
                thread.join()

                prompt_tokens = len(inputs['input_ids'][0])
                completion_tokens = len(TOKENIZER.encode(generated_text))
                
                yield {
                    "data": json.dumps({
                        "type": "done",
                        "full_text": generated_text,
                        "finished": True,
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens
                        }
                    })
                }
        
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield {
                "data": json.dumps({
                    "type": "error",
                    "error": str(e),
                    "finished": True
                })
            }
    
    return EventSourceResponse(event_generator(request))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint with connection management and timeouts."""
    await websocket.accept()
    
    connection_id = hashlib.md5(f"{websocket.client.host}:{time.time()}".encode()).hexdigest()
    logger.info(f"WebSocket connected: {connection_id}")
    
    try:
        while True:
            try:
                # Wait for a message with a timeout
                import asyncio
                data = await asyncio.wait_for(websocket.receive_json(), timeout=60.0)
                
            except asyncio.TimeoutError:
                logger.warning(f"WebSocket timeout for {connection_id}. Closing connection.")
                await websocket.close(code=1008, reason="Connection timed out")
                break
            except WebSocketDisconnect:
                # Handle client-side disconnect gracefully
                break

            # Validate request payload
            try:
                request = SecureGenerateRequest(**data)
            except Exception as e:
                await websocket.send_json({"error": f"Invalid request: {str(e)}", "status": "error"})
                continue
            
            # Process the request (non-streaming)
            try:
                # Note: Streaming is better handled via HTTP SSE. This is for single responses.
                response = await generate(request)
                await websocket.send_json(response.dict())
            except HTTPException as e:
                await websocket.send_json({"error": e.detail, "status": "error", "code": e.status_code})
            except Exception as e:
                logger.error(f"WebSocket processing error for {connection_id}: {e}", exc_info=True)
                await websocket.send_json({"error": "Internal server error", "status": "error", "code": 500})
    
    except Exception as e:
        # Catch unexpected errors during the connection lifecycle
        logger.error(f"Unexpected WebSocket error for {connection_id}: {e}", exc_info=True)
        # Ensure connection is closed if not already
        if websocket.client_state != WebSocketState.DISCONNECTED:
             await websocket.close(code=1011, reason="Internal server error")
    finally:
        logger.info(f"WebSocket disconnected: {connection_id}")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle file uploads with validation."""
    if MODEL_INFO.get("model_family") not in ["multimodal", "vision-language"]:
        raise HTTPException(
            status_code=400,
            detail="File upload only supported for multimodal models"
        )
    
    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}"
        )
    
    # Read and validate file size
    contents = await file.read()
    if len(contents) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(contents)} bytes (max {MAX_IMAGE_SIZE})"
        )
    
    # Validate it's actually an image
    try:
        from PIL import Image
        img = Image.open(BytesIO(contents))
        img.verify()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )
    
    # Convert to base64
    img_base64 = base64.b64encode(contents).decode()
    
    return {
        "filename": file.filename,
        "size": len(contents),
        "content_type": file.content_type,
        "data": img_base64
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")  # Default to localhost
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--load-lora", type=str, default=None)
    parser.add_argument("--stream-mode", type=str, default="false")
    parser.add_argument("--ui-file", type=str, default="serve_terminal.html", 
                        help="Path to the HTML file for the UI, relative to the script directory.")
    parser.add_argument("--allow-all-origins", action="store_true")
    args = parser.parse_args()
    
    # Security warning
    if args.host == "0.0.0.0":
        logger.warning("WARNING: Binding to 0.0.0.0 exposes the API to the network!")
    
    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
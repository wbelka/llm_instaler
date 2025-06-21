"""Universal API server for LLM Installer.

This FastAPI server adapts to different model types and provides
a unified interface for interacting with any installed model.
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import base64
from io import BytesIO

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


# Request/Response models
class GenerateRequest(BaseModel):
    prompt: Optional[str] = Field(None, description="Text prompt")
    messages: Optional[List[Dict[str, str]]] = Field(None, description="Chat messages")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(512, ge=1, le=4096)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=0)
    stop_sequences: Optional[List[str]] = None
    stream: bool = Field(False, description="Enable streaming")

    # Image generation parameters
    negative_prompt: Optional[str] = None
    num_inference_steps: int = Field(50, ge=1, le=1000)
    guidance_scale: float = Field(7.5, ge=0.0, le=20.0)
    width: int = Field(512, ge=64, le=2048)
    height: int = Field(512, ge=64, le=2048)
    seed: Optional[int] = None

    # Embedding parameters
    texts: Optional[List[str]] = Field(None, description="Texts for embedding")
    normalize: bool = Field(True, description="Normalize embeddings")

    # Reasoning parameters
    reasoning_mode: bool = Field(False, description="Enable reasoning mode")
    max_thinking_tokens: int = Field(10000, ge=1)
    max_answer_tokens: int = Field(2000, ge=1)
    show_thinking: bool = Field(True, description="Show thinking process")


class GenerateResponse(BaseModel):
    text: Optional[str] = None
    image: Optional[str] = None  # Base64 encoded
    embeddings: Optional[List[List[float]]] = None
    usage: Optional[Dict[str, int]] = None
    thinking: Optional[str] = None  # For reasoning models
    answer: Optional[str] = None  # For reasoning models


class ModelInfoResponse(BaseModel):
    model_id: str
    model_type: str
    model_family: str
    capabilities: Dict[str, Any]
    device: str
    dtype: str


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global MODEL, TOKENIZER, MODEL_INFO, DEVICE, DTYPE

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
        args, _ = parser.parse_known_args()

        DEVICE = args.device
        DTYPE = args.dtype

        # Load model
        MODEL, TOKENIZER = load_model(
            MODEL_INFO,
            device=args.device,
            dtype=args.dtype,
            lora_path=args.load_lora
        )

        logger.info("Model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/")
async def root():
    """Serve the web UI."""
    ui_path = Path("serve_ui.html")
    if ui_path.exists():
        return FileResponse(ui_path)
    else:
        return HTMLResponse(
            content="<h1>Model API</h1><p>UI file not found. API is running at /docs</p>"
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

    return ModelInfoResponse(
        model_id=MODEL_INFO.get("model_id", "unknown"),
        model_type=MODEL_INFO.get("model_type", "unknown"),
        model_family=MODEL_INFO.get("model_family", "unknown"),
        capabilities=MODEL_INFO.get("capabilities", {}),
        device=str(DEVICE),
        dtype=str(DTYPE)
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Main generation endpoint."""
    if not MODEL:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        model_family = MODEL_INFO.get("model_family", "")

        if model_family == "language-model":
            return await generate_text(request)
        elif model_family == "image-generation":
            return await generate_image(request)
        elif model_family == "embedding":
            return await generate_embeddings(request)
        elif model_family == "multimodal":
            return await generate_multimodal(request)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model family: {model_family}"
            )

    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def generate_text(request: GenerateRequest) -> GenerateResponse:
    """Generate text for language models."""
    import torch

    # Prepare input
    if request.messages:
        # Chat format
        text = TOKENIZER.apply_chat_template(
            request.messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        text = request.prompt

    if not text:
        raise HTTPException(status_code=400, detail="No prompt provided")

    # Tokenize
    inputs = TOKENIZER(text, return_tensors="pt")
    if hasattr(MODEL, "device"):
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

        if request.stop_sequences:
            generation_kwargs["stop_strings"] = request.stop_sequences

        outputs = MODEL.generate(**inputs, **generation_kwargs)

    # Decode
    generated_text = TOKENIZER.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    # Calculate usage
    usage = {
        "prompt_tokens": inputs["input_ids"].shape[1],
        "completion_tokens": outputs.shape[1] - inputs["input_ids"].shape[1],
        "total_tokens": outputs.shape[1]
    }

    return GenerateResponse(text=generated_text, usage=usage)


async def generate_image(request: GenerateRequest) -> GenerateResponse:
    """Generate images for diffusion models."""
    import torch

    if not request.prompt:
        raise HTTPException(status_code=400, detail="No prompt provided")

    # Set seed if provided
    if request.seed is not None:
        generator = torch.Generator(device=MODEL.device).manual_seed(request.seed)
    else:
        generator = None

    # Generate image
    image = MODEL(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        width=request.width,
        height=request.height,
        generator=generator
    ).images[0]

    # Convert to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return GenerateResponse(image=img_str)


async def generate_embeddings(request: GenerateRequest) -> GenerateResponse:
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


async def generate_multimodal(request: GenerateRequest) -> GenerateResponse:
    """Generate for multimodal models."""
    # Placeholder for multimodal generation
    raise HTTPException(
        status_code=501,
        detail="Multimodal generation not yet implemented"
    )


@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    """Streaming generation endpoint."""
    if not MODEL:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if MODEL_INFO.get("model_family") != "language-model":
        raise HTTPException(
            status_code=400,
            detail="Streaming only supported for language models"
        )

    async def event_generator():
        try:
            # Prepare input
            if request.messages:
                _ = TOKENIZER.apply_chat_template(
                    request.messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                _ = request.prompt

            # Token streaming would be implemented here
            # For now, return a simple response
            yield {
                "data": json.dumps({
                    "token": "Streaming not fully implemented yet. ",
                    "finished": False
                })
            }
            yield {
                "data": json.dumps({
                    "finished": True
                })
            }

        except Exception as e:
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--load-lora", type=str, default=None)
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


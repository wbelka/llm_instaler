#!/usr/bin/env python3
"""Test Gemma3 model directly without API."""

import sys
sys.path.append('/home/wblk/LLM/models/google_gemma-3-12b-it')

from handlers.gemma3 import Gemma3Handler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

# Create handler
model_info = {
    'model_id': 'google/gemma-3-12b-it',
    'model_type': 'gemma3',
    'model_family': 'multimodal'
}

handler = Gemma3Handler(model_info)

# Load model with int4 quantization
print("Loading model with int4 quantization...")
model, processor = handler.load_model(
    '/home/wblk/LLM/models/google_gemma-3-12b-it/model',
    device='auto',
    dtype='auto',
    load_in_4bit=True
)

print("\nModel loaded successfully!")
print(f"Model config: {model.config}")
print(f"Quantization config: {getattr(model.config, 'quantization_config', 'None')}")

# Test simple generation
print("\n\nTesting simple generation...")
result = handler.generate_text(
    prompt="Hello, my name is",
    model=model,
    processor=processor,
    temperature=0.7,
    max_tokens=50,
    use_simple_prompt=True  # Use simple mode
)

print(f"\nGenerated text: {result['text']}")

# Test with Russian
print("\n\nTesting Russian generation...")
result = handler.generate_text(
    prompt="Привет, меня зовут",
    model=model,
    processor=processor,
    temperature=0.7,
    max_tokens=50,
    use_simple_prompt=True
)

print(f"\nGenerated text: {result['text']}")

# Test with messages format
print("\n\nTesting messages format...")
messages = [
    {"role": "user", "content": "Who are you?"}
]

result = handler.process_multimodal(
    messages=messages,
    model=model,
    processor=processor,
    temperature=0.7,
    max_tokens=100
)

print(f"\nGenerated text: {result['text']}")
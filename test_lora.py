#!/usr/bin/env python3
"""Test LoRA fine-tuned model with Fufel prompts."""

import sys
import json
from pathlib import Path

# Add model directory to path
model_dir = Path("/home/wblk/LLM/models/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B")
sys.path.insert(0, str(model_dir))

from model_loader import load_model

def test_lora():
    # Load model info
    with open(model_dir / "model_info.json", 'r') as f:
        model_info = json.load(f)
    
    print("Loading base model...")
    base_model, tokenizer = load_model(model_info, str(model_dir / "model"), load_lora=False)
    
    print("\nLoading fine-tuned model with LoRA...")
    ft_model, _ = load_model(model_info, str(model_dir / "model"), load_lora=True, lora_path=str(model_dir / "lora"))
    
    # Test prompts
    test_prompts = [
        "Tell a Durrell-style story about Fufel eating",
        "How does a Fufel behave during dinner?",
        "Describe a young Fufel's adventure",
    ]
    
    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt}")
        print('='*60)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors='pt')
        inputs = {k: v.to(base_model.device) for k, v in inputs.items()}
        
        # Generate with base model
        print("\nBASE MODEL:")
        outputs = base_model.generate(**inputs, max_new_tokens=80, temperature=0.7, do_sample=True)
        base_response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        print(base_response)
        
        # Generate with fine-tuned model
        print("\nFINE-TUNED MODEL:")
        outputs = ft_model.generate(**inputs, max_new_tokens=80, temperature=0.7, do_sample=True)
        ft_response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        print(ft_response)

if __name__ == "__main__":
    test_lora()
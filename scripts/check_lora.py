#!/usr/bin/env python3
"""Quick diagnostic tool to check if LoRA training was successful."""

import os
import sys
import json
import torch
from pathlib import Path
import argparse

def check_lora_training(model_path: str = "./model", lora_path: str = "./lora", data_path: str = None):
    """Check if LoRA training was successful."""
    print("\n🔍 CHECKING LoRA TRAINING RESULTS")
    print("="*50)
    
    # 1. Check if LoRA files exist
    lora_path = Path(lora_path)
    print(f"\n1️⃣ Checking LoRA files in: {lora_path}")
    
    adapter_safetensors = lora_path / "adapter_model.safetensors"
    adapter_bin = lora_path / "adapter_model.bin"
    adapter_config = lora_path / "adapter_config.json"
    
    if adapter_safetensors.exists():
        size_mb = adapter_safetensors.stat().st_size / 1024 / 1024
        print(f"✅ Found adapter_model.safetensors ({size_mb:.1f} MB)")
    elif adapter_bin.exists():
        size_mb = adapter_bin.stat().st_size / 1024 / 1024
        print(f"✅ Found adapter_model.bin ({size_mb:.1f} MB)")
    else:
        print("❌ No adapter model found!")
        print("   Training may have failed or files weren't saved.")
        return
    
    if adapter_config.exists():
        print("✅ Found adapter_config.json")
        with open(adapter_config, 'r') as f:
            config = json.load(f)
            print(f"   - LoRA rank: {config.get('r', 'N/A')}")
            print(f"   - Target modules: {config.get('target_modules', 'N/A')}")
    
    # 2. Check training metrics
    print(f"\n2️⃣ Checking training metrics:")
    
    adapter_info = lora_path / "adapter_info.json"
    if adapter_info.exists():
        with open(adapter_info, 'r') as f:
            info = json.load(f)
            metrics = info.get('final_metrics', {})
            
            train_loss = metrics.get('final_train_loss', 'N/A')
            val_loss = metrics.get('final_val_loss', 'N/A')
            best_loss = metrics.get('best_val_loss', 'N/A')
            
            print(f"✅ Training completed")
            print(f"   - Final training loss: {train_loss}")
            print(f"   - Final validation loss: {val_loss}")
            print(f"   - Best validation loss: {best_loss}")
            
            if isinstance(train_loss, (int, float)):
                if train_loss < 0.01:
                    print("   ⚠️  Very low loss - possible overfitting")
                elif train_loss > 2.0:
                    print("   ⚠️  High loss - may need more training")
                else:
                    print("   ✅ Loss looks reasonable")
    else:
        print("⚠️  No adapter_info.json found")
    
    # 3. Check training history
    print(f"\n3️⃣ Checking training history:")
    
    history_img = lora_path / "training_history.png"
    if history_img.exists():
        print("✅ Found training_history.png")
        print("   View it to see loss curves over time")
    
    # 4. Check checkpoints
    print(f"\n4️⃣ Checking checkpoints:")
    
    checkpoints_dir = lora_path / "checkpoints"
    if checkpoints_dir.exists():
        checkpoints = sorted([d for d in checkpoints_dir.iterdir() if d.is_dir()])
        print(f"✅ Found {len(checkpoints)} checkpoints")
        if checkpoints:
            print("   Latest checkpoints:")
            for cp in checkpoints[-3:]:
                print(f"   - {cp.name}")
    
    # 5. Test with dataset examples
    print(f"\n5️⃣ Testing with dataset examples:")
    
    test_prompts = []
    
    # Try to extract prompts from dataset if provided
    if data_path and Path(data_path).exists():
        try:
            print(f"Extracting test prompts from: {data_path}")
            
            # Simple extraction for common formats
            import json
            with open(data_path, 'r') as f:
                if data_path.endswith('.json'):
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data[:3]:  # Get first 3 examples
                            if isinstance(item, dict):
                                if 'prompt' in item and 'completion' in item:
                                    test_prompts.append((item['prompt'], item['completion']))
                                elif 'instruction' in item:
                                    prompt = item['instruction']
                                    if 'input' in item and item['input']:
                                        prompt += "\n\nInput: " + item['input']
                                    expected = item.get('output', item.get('response', ''))
                                    test_prompts.append((prompt, expected))
                                elif 'messages' in item:
                                    msgs = item['messages']
                                    user_msgs = [m for m in msgs if m.get('role') == 'user']
                                    asst_msgs = [m for m in msgs if m.get('role') == 'assistant']
                                    if user_msgs and asst_msgs:
                                        test_prompts.append((user_msgs[-1]['content'], asst_msgs[-1]['content']))
                
            if test_prompts:
                print(f"✅ Extracted {len(test_prompts)} test prompts from dataset")
            else:
                print("⚠️  Could not extract prompts from dataset")
        except Exception as e:
            print(f"⚠️  Error reading dataset: {e}")
    
    # Add default prompt if no dataset prompts
    if not test_prompts:
        test_prompts = [("Hello, how are you?", None)]
    
    try:
        # Try to load with model_loader
        sys.path.append(str(Path(model_path).parent))
        from model_loader import load_model, get_model_config
        
        print("\nLoading models for comparison...")
        model_info = get_model_config(str(Path(model_path) / "model_info.json"))
        
        # Load without LoRA
        base_model, base_tokenizer = load_model(
            model_info,
            model_path=model_path,
            load_lora=False
        )
        
        # Load with LoRA
        ft_model, ft_tokenizer = load_model(
            model_info,
            model_path=model_path,
            lora_path=str(lora_path),
            load_lora=True
        )
        
        print("\n" + "="*50)
        print("🧪 TEST RESULTS")
        print("="*50)
        
        identical_count = 0
        
        for i, (prompt, expected) in enumerate(test_prompts):
            print(f"\nTest {i+1}:")
            print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
            
            if expected:
                print(f"Expected: {expected[:100]}..." if len(expected) > 100 else f"Expected: {expected}")
            
            # Test base model
            inputs = base_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = base_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=base_tokenizer.pad_token_id
                )
            base_response = base_tokenizer.decode(outputs[0], skip_special_tokens=True)
            base_generated = base_response[len(prompt):].strip()
            
            # Test fine-tuned model
            inputs = ft_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = ft_model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=ft_tokenizer.pad_token_id
                )
            ft_response = ft_tokenizer.decode(outputs[0], skip_special_tokens=True)
            ft_generated = ft_response[len(prompt):].strip()
            
            print(f"\nBase: {base_generated[:100]}...")
            print(f"Fine-tuned: {ft_generated[:100]}...")
            
            if base_generated == ft_generated:
                print("⚠️  IDENTICAL responses")
                identical_count += 1
            else:
                print("✅ DIFFERENT responses")
                
            if expected and expected.strip().lower() in ft_generated.lower():
                print("✅ Fine-tuned matches expected output!")
        
        # Summary
        print("\n" + "-"*50)
        total = len(test_prompts)
        print(f"Identical responses: {identical_count}/{total}")
        
        if identical_count == total:
            print("\n❌ All responses identical - model likely didn't learn")
        elif identical_count > total / 2:
            print("\n⚠️  Most responses identical - minimal learning")
        else:
            print("\n✅ Model shows different behavior after training!")
        
        # Cleanup
        del base_model, ft_model
        torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"⚠️  Could not test model: {e}")
    
    # Summary
    print("\n" + "="*50)
    print("📋 SUMMARY")
    print("="*50)
    
    print("\nTo use your fine-tuned model:")
    print("1. Start the model: ./start.sh")
    print("2. The script will automatically detect and load LoRA")
    print("3. Look for '🎯 LoRA ADAPTER DETECTED!' message")
    
    print("\nTo verify training worked:")
    print("1. Check that training loss decreased (view training_history.png)")
    print("2. Test with prompts similar to your training data")
    print("3. Compare responses with and without LoRA")
    
    print("\nIf model didn't learn:")
    print("1. Try more epochs: --epochs 5")
    print("2. Increase learning rate: --learning-rate 5e-4")
    print("3. Increase LoRA rank: --lora-r 64")
    print("4. Check your dataset format matches expected format")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check LoRA training results")
    parser.add_argument("--model-path", default="./model", help="Path to base model")
    parser.add_argument("--lora-path", default="./lora", help="Path to LoRA adapter")
    parser.add_argument("--data-path", default=None, help="Path to training dataset for testing")
    
    args = parser.parse_args()
    
    # Try to find dataset if not provided
    if not args.data_path:
        # Look for common dataset locations
        for path in ['data.json', 'dataset.json', 'train.json', '../data.json', '../dataset.json']:
            if Path(path).exists():
                args.data_path = path
                print(f"Found dataset at: {path}")
                break
    
    check_lora_training(args.model_path, args.lora_path, args.data_path)
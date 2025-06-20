"""
Main installer logic following the "Configuration Detective" algorithm
"""

import os
import json
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from huggingface_hub import snapshot_download, model_info as hf_model_info
from .model_detector_v2 import ModelDetectorV2
from .handlers import get_handler_for_model
from .utils import get_hf_token

logger = logging.getLogger(__name__)


@dataclass
class InstallConfig:
    """Configuration for model installation"""
    model_id: str
    install_path: Path
    quantization: Optional[str] = None
    force_reinstall: bool = False
    use_vllm: bool = False
    use_tensorrt: bool = False
    max_memory: Optional[str] = None


class ModelInstaller:
    """Main installer class"""
    
    def __init__(self):
        self.detector = ModelDetectorV2()
        self.base_path = Path.home() / "LLM" / "models"
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def install(self, config: InstallConfig) -> bool:
        """
        Install a model following the complete pipeline:
        1. Detect model type
        2. Create installation directory
        3. Download model files
        4. Set up virtual environment
        5. Install dependencies
        6. Generate scripts
        7. Run post-installation setup
        """
        try:
            logger.info(f"Starting installation of {config.model_id}")
            
            # Step 1: Detect model type
            logger.info("Step 1: Detecting model type...")
            model_info = self.detector.detect(config.model_id)
            if not model_info:
                logger.error(f"Failed to detect model type for {config.model_id}")
                return False
            
            # Convert to dict for handlers
            model_info_dict = model_info.to_dict()
            
            # Get appropriate handler
            handler = get_handler_for_model(model_info_dict)
            logger.info(f"Using handler: {handler.name}")
            
            # Step 2: Prepare installation directory
            logger.info("Step 2: Preparing installation directory...")
            install_path = self._prepare_install_directory(config, model_info_dict)
            
            # Step 3: Download model files
            logger.info("Step 3: Downloading model files...")
            self._download_model(config, install_path)
            
            # Step 4: Create virtual environment
            logger.info("Step 4: Creating virtual environment...")
            venv_path = self._create_virtual_environment(install_path)
            
            # Step 5: Install dependencies
            logger.info("Step 5: Installing dependencies...")
            dependencies = handler.get_dependencies(model_info_dict)
            self._install_dependencies(venv_path, dependencies)
            
            # Step 6: Generate scripts
            logger.info("Step 6: Generating scripts...")
            self._generate_scripts(install_path, model_info_dict, handler)
            
            # Step 7: Save model info
            logger.info("Step 7: Saving model information...")
            self._save_model_info(install_path, model_info_dict, config)
            
            # Step 8: Run post-installation setup
            logger.info("Step 8: Running post-installation setup...")
            handler.post_install_setup(str(install_path), model_info_dict)
            
            logger.info(f"Successfully installed {config.model_id} to {install_path}")
            return True
            
        except Exception as e:
            logger.error(f"Installation failed: {e}", exc_info=True)
            return False
    
    def _prepare_install_directory(self, config: InstallConfig, 
                                   model_info: Dict[str, Any]) -> Path:
        """Prepare installation directory"""
        # Create safe directory name
        safe_name = config.model_id.replace('/', '_')
        install_path = config.install_path or (self.base_path / safe_name)
        
        if install_path.exists() and not config.force_reinstall:
            raise ValueError(f"Model already installed at {install_path}. "
                           "Use --force to reinstall.")
        
        # Create directory structure
        install_path.mkdir(parents=True, exist_ok=True)
        (install_path / "model").mkdir(exist_ok=True)
        (install_path / "scripts").mkdir(exist_ok=True)
        (install_path / "logs").mkdir(exist_ok=True)
        (install_path / "cache").mkdir(exist_ok=True)
        
        return install_path
    
    def _download_model(self, config: InstallConfig, install_path: Path) -> None:
        """Download model files from HuggingFace"""
        model_path = install_path / "model"
        
        logger.info(f"Downloading {config.model_id} to {model_path}")
        
        token = get_hf_token()
        
        # Download the model
        snapshot_download(
            repo_id=config.model_id,
            local_dir=str(model_path),
            token=token,
            resume_download=True,
            local_dir_use_symlinks=False
        )
        
        logger.info("Download complete")
    
    def _create_virtual_environment(self, install_path: Path) -> Path:
        """Create virtual environment for the model"""
        venv_path = install_path / ".venv"
        
        if not venv_path.exists():
            logger.info(f"Creating virtual environment at {venv_path}")
            subprocess.run(
                [sys.executable, "-m", "venv", str(venv_path)],
                check=True
            )
        
        return venv_path
    
    def _install_dependencies(self, venv_path: Path, 
                            dependencies: List[str]) -> None:
        """Install dependencies in virtual environment"""
        pip_path = venv_path / "bin" / "pip"
        if not pip_path.exists():
            pip_path = venv_path / "Scripts" / "pip.exe"  # Windows
        
        # Upgrade pip first
        logger.info("Upgrading pip...")
        subprocess.run(
            [str(pip_path), "install", "--upgrade", "pip"],
            check=True
        )
        
        # Install dependencies
        if dependencies:
            logger.info(f"Installing dependencies: {dependencies}")
            subprocess.run(
                [str(pip_path), "install"] + dependencies,
                check=True
            )
    
    def _generate_scripts(self, install_path: Path, 
                         model_info: Dict[str, Any],
                         handler: 'BaseHandler') -> None:
        """Generate start.sh and train.sh scripts"""
        
        # Get environment variables from handler
        env_vars = handler.get_environment_vars(model_info)
        env_exports = '\n'.join(f'export {k}="{v}"' for k, v in env_vars.items())
        
        # Generate start.sh
        start_script = f"""#!/bin/bash
# Auto-generated script to start the model

# Change to script directory
cd "$(dirname "$0")"

# Activate virtual environment
source .venv/bin/activate

# Set environment variables
{env_exports}

# Start API server
echo "Starting API server for {model_info.get('model_id')}..."
python scripts/serve_api.py

# Deactivate on exit
deactivate
"""
        
        start_path = install_path / "start.sh"
        start_path.write_text(start_script)
        start_path.chmod(0o755)
        
        # Generate train.sh
        train_script = f"""#!/bin/bash
# Auto-generated script to train the model

# Change to script directory
cd "$(dirname "$0")"

# Activate virtual environment
source .venv/bin/activate

# Set environment variables
{env_exports}

# Run training
echo "Starting training for {model_info.get('model_id')}..."
python scripts/train_lora.py "$@"

# Deactivate on exit
deactivate
"""
        
        train_path = install_path / "train.sh"
        train_path.write_text(train_script)
        train_path.chmod(0o755)
        
        # Generate basic API server script
        self._generate_api_server(install_path, model_info)
        
        # Generate basic training script
        self._generate_training_script(install_path, model_info)
    
    def _generate_api_server(self, install_path: Path, 
                           model_info: Dict[str, Any]) -> None:
        """Generate API server script"""
        api_script = '''"""
Auto-generated API server
"""

import os
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# Load model info
with open('../model_info.json') as f:
    model_info = json.load(f)

# Load model and tokenizer
print("Loading model...")
model_path = "../model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16
)

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 1.0

class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int

@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text from prompt"""
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]
        
        return GenerateResponse(
            text=generated_text,
            tokens_generated=tokens_generated
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model/info")
async def get_model_info():
    """Get model information"""
    return model_info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        
        api_path = install_path / "scripts" / "serve_api.py"
        api_path.write_text(api_script)
    
    def _generate_training_script(self, install_path: Path,
                                 model_info: Dict[str, Any]) -> None:
        """Generate training script"""
        train_script = '''"""
Auto-generated training script
"""

import argparse
import json
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to training data")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()
    
    # Load model and tokenizer
    model_path = "../model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto"
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    dataset = load_dataset("json", data_files=args.data)
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="../checkpoints",
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_dir="../logs",
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="none"
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    
    # Train
    trainer.train()
    
    # Save the fine-tuned model
    model.save_pretrained("../finetuned_model")
    tokenizer.save_pretrained("../finetuned_model")
    
    print("Training complete!")

if __name__ == "__main__":
    main()
'''
        
        train_path = install_path / "scripts" / "train_lora.py"
        train_path.write_text(train_script)
    
    def _save_model_info(self, install_path: Path,
                        model_info: Dict[str, Any],
                        config: InstallConfig) -> None:
        """Save model information to JSON file"""
        info_to_save = {
            **model_info,
            'installation': {
                'install_path': str(install_path),
                'install_date': datetime.now().isoformat(),
                'quantization': config.quantization,
                'use_vllm': config.use_vllm,
                'use_tensorrt': config.use_tensorrt,
            }
        }
        
        info_path = install_path / "model_info.json"
        with open(info_path, 'w') as f:
            json.dump(info_to_save, f, indent=2)


# Add missing imports
import sys
from datetime import datetime
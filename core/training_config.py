"""Training configuration and parameter management.

This module handles training configuration, automatic parameter detection,
and provides smart defaults for different model types and sizes.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training with LoRA/QLoRA."""
    
    # Training modes
    training_mode: str = "medium"  # slow, medium, fast, circle, non-stop, adaptive
    method: str = "lora"  # lora, qlora, full (future)
    
    # Model parameters
    model_size_gb: float = 0.0
    model_type: str = ""
    model_family: str = ""
    
    # LoRA parameters
    lora_r: Optional[int] = None  # Auto-detect if None
    lora_alpha: Optional[int] = None  # Auto-detect if None
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None  # Auto-detect if None
    
    # Training parameters
    learning_rate: Optional[float] = None  # Auto-detect if None
    batch_size: Optional[int] = None  # Auto-detect if None
    gradient_accumulation_steps: Optional[int] = None  # Auto-detect if None
    num_train_epochs: Optional[int] = None  # Auto-detect if None
    max_steps: int = -1  # -1 means no limit
    
    # Data parameters
    max_seq_length: Optional[int] = None  # Auto-detect if None
    dataset_format: str = "auto"  # auto, alpaca, chat, completion, qa, vision
    validation_split: float = 0.1
    
    # Optimization
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    optimizer: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    gradient_checkpointing: bool = True
    
    # Memory optimization
    use_8bit: bool = False
    use_4bit: bool = False
    mixed_precision: str = "auto"  # no, fp16, bf16, auto
    
    # Circular training
    circular_training: bool = False
    max_circular_epochs: int = 100
    circular_batch_multiplier: float = 1.0  # Batch size multiplier or increment (+1 if > 0 and < 2)
    
    # Auto-stop parameters
    early_stopping: bool = True
    patience: int = 3
    min_delta: float = 0.001
    overfitting_threshold: float = 0.1
    min_evaluations: int = 5
    
    # Evaluation
    eval_steps: Optional[int] = None  # Auto-detect if None
    eval_strategy: str = "steps"  # steps, epoch
    save_strategy: str = "steps"  # steps, epoch
    save_steps: Optional[int] = None  # Auto-detect if None
    save_total_limit: Optional[int] = None  # None = keep all checkpoints
    load_best_model_at_end: bool = True
    
    # Output
    output_dir: str = "./lora"
    logging_steps: int = 10
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    
    # Resume training
    resume_from_checkpoint: Optional[str] = None
    
    @classmethod
    def from_model_info(cls, model_info: Dict[str, Any], **kwargs) -> 'TrainingConfig':
        """Create training config from model info with smart defaults."""
        config = cls()
        
        # Extract model information
        config.model_type = model_info.get('model_type', '')
        config.model_family = model_info.get('model_family', '')
        config.model_size_gb = model_info.get('model_size_gb', 0.0)
        
        # Apply user overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Auto-detect parameters if not set
        config._auto_detect_parameters()
        
        return config
    
    def _auto_detect_parameters(self):
        """Automatically detect optimal parameters based on model size and type."""
        # LoRA rank based on model size
        if self.lora_r is None:
            self.lora_r = self._get_optimal_lora_rank()
        
        # LoRA alpha (typically 2x rank)
        if self.lora_alpha is None:
            self.lora_alpha = self.lora_r * 2
        
        # Target modules based on model family
        if self.target_modules is None:
            self.target_modules = self._get_target_modules()
        
        # Learning rate based on model size and training mode
        if self.learning_rate is None:
            self.learning_rate = self._get_optimal_learning_rate()
        
        # Batch size based on model size and available memory
        if self.batch_size is None:
            self.batch_size = self._get_optimal_batch_size()
        
        # Gradient accumulation for effective batch size
        if self.gradient_accumulation_steps is None:
            self.gradient_accumulation_steps = self._get_gradient_accumulation_steps()
        
        # Number of epochs based on training mode
        if self.num_train_epochs is None:
            self.num_train_epochs = self._get_num_epochs()
        
        # Max sequence length based on model
        if self.max_seq_length is None:
            self.max_seq_length = self._get_max_seq_length()
        
        # Evaluation and save steps
        if self.eval_steps is None:
            self.eval_steps = self._get_eval_steps()
        if self.save_steps is None:
            # Save frequently to not lose progress
            self.save_steps = 10  # Save every 10 steps
        
        # Mixed precision based on hardware
        if self.mixed_precision == "auto":
            self.mixed_precision = self._get_mixed_precision()
    
    def _get_optimal_lora_rank(self) -> int:
        """Get optimal LoRA rank based on model size."""
        if self.model_size_gb < 1:
            return 8
        elif self.model_size_gb < 3:
            return 16
        elif self.model_size_gb < 7:
            return 32
        elif self.model_size_gb < 13:
            return 64
        elif self.model_size_gb < 30:
            return 128
        else:
            return 256
    
    def _get_target_modules(self) -> List[str]:
        """Get target modules for LoRA based on model family."""
        # Common patterns for different model families
        module_patterns = {
            "llama": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "llama4": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "mistral": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "gemma": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "gemma3": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "qwen": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "qwen3": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "phi": ["q_proj", "v_proj", "k_proj", "o_proj", "dense"],
            "gpt": ["c_attn", "c_proj"],
            "bert": ["query", "value"],
            "t5": ["q", "v"],
            "default": ["q_proj", "v_proj"]  # Safe default
        }
        
        # Try exact match first
        if self.model_family in module_patterns:
            return module_patterns[self.model_family]
        
        # Try model type
        if self.model_type in module_patterns:
            return module_patterns[self.model_type]
        
        # Check for partial matches
        for key, modules in module_patterns.items():
            if key in self.model_family.lower() or key in self.model_type.lower():
                return modules
        
        return module_patterns["default"]
    
    def _get_optimal_learning_rate(self) -> float:
        """Get optimal learning rate based on model size and training mode."""
        # Base learning rates for different model sizes
        base_lr = {
            "tiny": 5e-4,    # < 1B
            "small": 2e-4,   # 1-3B
            "medium": 1e-4,  # 3-7B
            "large": 5e-5,   # 7-13B
            "xlarge": 2e-5   # > 13B
        }
        
        # Determine size category
        if self.model_size_gb < 1:
            size_cat = "tiny"
        elif self.model_size_gb < 3:
            size_cat = "small"
        elif self.model_size_gb < 7:
            size_cat = "medium"
        elif self.model_size_gb < 13:
            size_cat = "large"
        else:
            size_cat = "xlarge"
        
        lr = base_lr[size_cat]
        
        # Adjust for training mode
        mode_multipliers = {
            "slow": 0.5,
            "medium": 1.0,
            "fast": 2.0,
            "circle": 0.8,
            "non-stop": 0.3,
            "adaptive": 1.0
        }
        
        lr *= mode_multipliers.get(self.training_mode, 1.0)
        
        # Adjust for quantization
        if self.use_4bit or self.use_8bit:
            lr *= 0.8  # Slightly lower for quantized models
        
        return lr
    
    def _get_optimal_batch_size(self) -> int:
        """Get optimal batch size based on model size and quantization."""
        # Base batch sizes for different model sizes (assuming 24GB GPU)
        if self.use_4bit:
            # 4-bit quantization allows larger batches
            if self.model_size_gb < 7:
                return 8
            elif self.model_size_gb < 13:
                return 4
            else:
                return 2
        elif self.use_8bit:
            # 8-bit quantization
            if self.model_size_gb < 7:
                return 4
            elif self.model_size_gb < 13:
                return 2
            else:
                return 1
        else:
            # Full precision
            if self.model_size_gb < 3:
                return 4
            elif self.model_size_gb < 7:
                return 2
            else:
                return 1
    
    def _get_gradient_accumulation_steps(self) -> int:
        """Get gradient accumulation steps for effective batch size."""
        # Target effective batch size
        target_batch_sizes = {
            "slow": 32,
            "medium": 16,
            "fast": 8,
            "circle": 4,
            "non-stop": 32,
            "adaptive": 16
        }
        
        target_batch = target_batch_sizes.get(self.training_mode, 16)
        
        # Calculate accumulation steps
        accumulation = max(1, target_batch // self.batch_size)
        
        return accumulation
    
    def _get_num_epochs(self) -> int:
        """Get number of epochs based on training mode."""
        epochs = {
            "slow": 5,
            "medium": 3,
            "fast": 1,
            "circle": -1,  # Determined by circular training
            "non-stop": -1,  # No limit
            "adaptive": 3  # Start with 3, adapt as needed
        }
        
        return epochs.get(self.training_mode, 3)
    
    def _get_max_seq_length(self) -> int:
        """Get max sequence length based on model."""
        # Training-appropriate context lengths (not full model capacity)
        context_lengths = {
            "llama": 2048,
            "llama4": 4096,  # Can go much higher but this is safe for training
            "mistral": 4096,
            "gemma": 4096,
            "gemma3": 4096,
            "qwen": 2048,
            "qwen3": 2048,  # Much safer for training than 32k
            "phi": 2048,
            "gpt": 2048,
            "default": 2048
        }
        
        # Try to get from model family/type
        for key in [self.model_family, self.model_type]:
            if key in context_lengths:
                return context_lengths[key]
            # Check partial matches
            for k, v in context_lengths.items():
                if k in key.lower():
                    return v
        
        return context_lengths["default"]
    
    def _get_eval_steps(self) -> int:
        """Get evaluation steps based on training mode."""
        eval_steps = {
            "slow": 100,
            "medium": 50,
            "fast": 25,
            "circle": 10,  # Frequent eval for circular
            "non-stop": 200,
            "adaptive": 50
        }
        
        return eval_steps.get(self.training_mode, 50)
    
    def _get_mixed_precision(self) -> str:
        """Determine mixed precision based on hardware and model."""
        try:
            import torch
            if torch.cuda.is_available():
                # Check compute capability
                if torch.cuda.get_device_capability()[0] >= 8:
                    # Ampere or newer - use bf16
                    return "bf16"
                elif torch.cuda.get_device_capability()[0] >= 7:
                    # Volta/Turing - use fp16
                    return "fp16"
        except:
            pass
        
        return "no"
    
    def get_training_description(self) -> str:
        """Get human-readable description of training configuration."""
        desc = []
        desc.append(f"Training Mode: {self.training_mode}")
        desc.append(f"Method: {self.method.upper()}")
        desc.append(f"LoRA Rank: {self.lora_r}")
        desc.append(f"Learning Rate: {self.learning_rate}")
        desc.append(f"Batch Size: {self.batch_size} (effective: {self.batch_size * self.gradient_accumulation_steps})")
        
        if self.num_train_epochs > 0:
            desc.append(f"Epochs: {self.num_train_epochs}")
        elif self.circular_training:
            desc.append(f"Circular Training: up to {self.max_circular_epochs} cycles")
        else:
            desc.append("Continuous Training (no epoch limit)")
        
        if self.use_4bit:
            desc.append("Quantization: 4-bit")
        elif self.use_8bit:
            desc.append("Quantization: 8-bit")
        
        return "\n".join(desc)
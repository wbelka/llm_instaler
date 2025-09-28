# LLM Installer Training Guide

This guide covers the integrated training system that allows fine-tuning installed models using LoRA/QLoRA with automatic parameter detection and smart training features.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Training Modes](#training-modes)
- [Dataset Formats](#dataset-formats)
- [Advanced Features](#advanced-features)
- [Handler Integration](#handler-integration)
- [Troubleshooting](#troubleshooting)

## Overview

The LLM Installer includes a sophisticated training system that:
- Automatically detects optimal training parameters based on model size and type
- Supports multiple dataset formats (Alpaca, ShareGPT, OpenAI, QA, etc.)
- Implements smart features like auto-stop on overfitting and circular training
- Uses model-specific handlers for optimal configuration
- Automatically loads trained LoRA adapters during inference

## Quick Start

### Basic Training

```bash
# Navigate to your model directory
cd ~/LLM/models/meta-llama_Llama-3-8B

# Train with a single dataset file
./train.sh --data path/to/dataset.json

# Train with multiple files using wildcards (use quotes!)
./train.sh --data 'data/*.json'

# Train with a directory of files
./train.sh --data datasets/

# Train with specific files (comma-separated)
./train.sh --data 'train1.json,train2.json,train3.json'
```

The model will be fine-tuned using sensible defaults:
- Automatic LoRA rank selection based on model size
- Automatic batch size and learning rate
- Early stopping to prevent overfitting
- TensorBoard monitoring

### Using the Trained Model

After training, the LoRA adapter is automatically loaded when you start the model:

```bash
./start.sh  # Automatically loads ./lora if present
```

## Training Modes

### 1. **Slow Mode** (Highest Quality)
```bash
./train.sh --data dataset.json --mode slow
```
- Lower learning rate (0.5x)
- More epochs (5)
- Larger evaluation intervals
- Best for production models

### 2. **Medium Mode** (Default)
```bash
./train.sh --data dataset.json --mode medium
```
- Balanced settings
- 3 epochs
- Standard learning rate
- Good for most use cases

### 3. **Fast Mode** (Quick Iteration)
```bash
./train.sh --data dataset.json --mode fast
```
- Higher learning rate (2x)
- 1 epoch
- Frequent evaluations
- Good for experimentation

### 4. **Circular Mode** (Small Datasets)
```bash
./train.sh --data small_dataset.json --circular

# Custom number of cycles (default: 100)
./train.sh --data small_dataset.json --circular --max-circular-epochs 20
```
- Cycles through dataset multiple times
- Gradually increases batch size
- Stops on perfect loss or max cycles
- Ideal for datasets < 1000 examples

### 5. **Non-Stop Mode** (Continuous)
```bash
./train.sh --data dataset.json --mode non-stop
```
- No epoch limit
- Very low learning rate (0.3x)
- Continues until manually stopped
- Good for incremental learning

### 6. **Adaptive Mode** (Experimental)
```bash
./train.sh --data dataset.json --mode adaptive
```
- Dynamically adjusts parameters
- Monitors performance trends
- Experimental feature

### 7. **Force Epochs Mode** (Override Auto-Stop)
```bash
./train.sh --data dataset.json --epochs 5 --force-epochs

# With circular training - guaranteed cycles
./train.sh --data small_dataset.json --circular --max-circular-epochs 50 --force-epochs
```
- **NEW**: Trains for exact number of epochs specified
- Ignores all auto-stop conditions:
  - Overfitting detection disabled
  - Early stopping disabled
  - Perfect loss checks disabled
- Useful when you want guaranteed training time
- Can be combined with any mode

## Dataset Formats

### Supported Formats

#### 1. **Alpaca Format** (Instruction Tuning)
```json
[
  {
    "instruction": "Translate to French",
    "input": "Hello, how are you?",
    "output": "Bonjour, comment allez-vous?"
  }
]
```

#### 2. **ShareGPT Format** (Conversations)
```json
[
  {
    "conversations": [
      {"from": "human", "value": "What is machine learning?"},
      {"from": "gpt", "value": "Machine learning is..."}
    ]
  }
]
```

#### 3. **OpenAI Format** (Chat)
```json
[
  {
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"},
      {"role": "assistant", "content": "Hi there!"}
    ]
  }
]
```

#### 4. **QA Format**
```json
[
  {
    "question": "What is the capital of France?",
    "answer": "Paris",
    "context": "France is a country in Europe..." // optional
  }
]
```

#### 5. **Completion Format**
```json
[
  {
    "prompt": "The weather today is",
    "completion": " sunny with a chance of rain later."
  }
]
```

#### 6. **Plain Text**
```text
This is a sample text for training.
Each line becomes a training example.
```

#### 7. **Vision Formats** (For Multimodal Models)
```json
[
  {
    "image": "path/to/image.jpg",
    "caption": "A beautiful sunset over the ocean"
  }
]
```

### Loading from Multiple Sources

```bash
# From directory with subfolders
./train.sh --data datasets/

# Structure:
# datasets/
#   ├── alpaca/     # Alpaca format files
#   ├── chat/       # Chat format files
#   └── qa/         # QA format files

# Using wildcards for specific file types
./train.sh --data 'datasets/*.json' --circular

# Combining multiple directories
./train.sh --data 'train_data/*.json,validation_data/*.json'

# Mix of files and directories
./train.sh --data 'base_training.json,additional_data/'
```

### HuggingFace Datasets

```bash
# Load from HuggingFace
./train.sh --data "squad:train" --dataset-format qa
```

## Advanced Features

### Auto-Stop on Overfitting

The training system monitors for overfitting by:
1. Tracking validation loss trends
2. Comparing train/validation loss gap
3. Stopping when degradation detected
4. **NEW**: Adaptive thresholds based on task type

#### Task-Specific Behavior

**QA/Instruction Tasks** (low loss expected):
- Perfect loss threshold: 0.01
- Overfitting threshold: 10% (default)
- Patience: 3 evaluations

**Text Generation Tasks** (higher loss normal):
- Perfect loss threshold: 0.5
- Overfitting threshold: 20% (auto-adjusted)
- Patience: 5 evaluations (auto-adjusted)
- Monitors perplexity explosion (>100)

Configure thresholds:
```bash
./train.sh --data dataset.json \
  --patience 5 \
  --overfitting-threshold 0.15
```

The system automatically detects task type from:
- Dataset format (`text`, `completion`)
- Model type (`language-model`)

### LoRA Configuration

#### Automatic Configuration
By default, LoRA rank is automatically selected:
- < 1B models: rank 8
- 1-3B models: rank 16  
- 3-7B models: rank 32
- 7-13B models: rank 64
- 13-30B models: rank 128
- > 30B models: rank 256

#### Manual Configuration
```bash
./train.sh --data dataset.json \
  --lora-r 64 \
  --lora-alpha 128
```

### Quantization Training (QLoRA)

For large models with limited VRAM:
```bash
# 4-bit quantization (recommended)
./train.sh --data dataset.json --method qlora

# 8-bit quantization
./train.sh --data dataset.json --use-8bit
```

### Resume Training

```bash
# Resume from last checkpoint
./train.sh --data dataset.json --resume

# Resume from specific checkpoint
./train.sh --data dataset.json --resume-from checkpoints/checkpoint-1000
```

### Batch Size and Learning Rate

```bash
# Manual configuration
./train.sh --data dataset.json \
  --batch-size 8 \
  --learning-rate 2e-4 \
  --epochs 5
```

### Optimizer Selection

By default, the training script uses `adamw_torch`. You can specify a different optimizer using the `--optimizer` argument. This is useful for experimenting with more advanced or memory-efficient optimizers.

```bash
# Use an 8-bit optimizer to save memory
./train.sh --data dataset.json --optimizer adamw_bnb_8bit
```

#### Recommended Optimizers

-   **`adamw_torch`** (Default): Standard AdamW optimizer from PyTorch. A safe and reliable choice for most scenarios.
-   **`adamw_bnb_8bit`**: 8-bit quantized optimizer from `bitsandbytes`. Significantly reduces memory usage (VRAM), making it ideal for training larger models or on GPUs with less memory. Highly recommended when using `--method qlora`.
-   **`paged_adamw_8bit`**: Another 8-bit optimizer from `bitsandbytes` that uses paged memory management for even greater memory efficiency. A good alternative to `adamw_bnb_8bit` if you are still facing memory issues.
-   **`adafactor`**: An adaptive learning rate optimizer that can be faster and use less memory than Adam, especially for very large models. It does not use momentum, so its convergence behavior can differ.

**Note on Dependencies:** To use the `..._bnb_8bit` optimizers, you must have the `bitsandbytes` library correctly installed and configured for your system. If you encounter issues, running the installer's fix command can often resolve them: `./llm-installer fix . --fix-cuda`

### Data Limiting

```bash
# Train on subset for testing
./train.sh --data large_dataset.json --max-examples 1000
```

### Custom Validation Split

```bash
# Use 20% for validation (default is 10%)
./train.sh --data dataset.json --validation-split 0.2
```

## Handler Integration

The training system uses model-specific handlers to optimize training:

### Handler-Provided Parameters

1. **Target Modules**: Each handler specifies which layers to train
   - Llama: `["q_proj", "v_proj", "k_proj", "o_proj"]`
   - Gemma3: `["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
   - GPT: `["c_attn", "c_proj"]`

2. **Training Precision**: Optimal precision for each model family
   - Gemma3/Llama4: `bfloat16`
   - Most others: `float16`

3. **Special Configurations**
   - Flash attention compatibility
   - Gradient checkpointing support
   - Maximum sequence length

### Example: Training Different Model Types

```bash
# Llama model - uses specific target modules
cd ~/LLM/models/meta-llama_Llama-3-8B
./train.sh --data alpaca_data.json

# Gemma3 multimodal - automatically uses bf16
cd ~/LLM/models/google_gemma-3-12b-it
./train.sh --data vision_qa_data.json

# Large model with QLoRA
cd ~/LLM/models/meta-llama_Llama-3-70B
./train.sh --data dataset.json --method qlora
```

## Monitoring Training

### TensorBoard

TensorBoard starts automatically:
```bash
# Access at http://localhost:6006
# Disable with: --no-tensorboard
# Custom port: --tensorboard-port 6007
```

Metrics tracked:
- Training/validation loss
- Learning rate schedule
- Gradient norms
- Token processing speed

### Viewing Previous Training Graphs

You can view training graphs from previous runs without starting a new training:

```bash
# View training graphs on default port 6006
./train.sh --tensorboard-only

# View training graphs on custom port
./train.sh --tensorboard-only 6007
```

This mode:
- Starts only the TensorBoard server
- Looks for logs in `./lora/logs` directory
- Allows you to analyze past training runs
- Useful for comparing different training configurations
- Press Ctrl+C to stop the server

### Console Output

Real-time training status:

**QA/Instruction Tasks:**
```
📊 Evaluation #7 | ⏱️ Time: 00:15:32 | 📉 Train Loss: 0.5234 | 📈 Val Loss: 0.6123 | 🎯 Best Val: 0.5890 | ⏳ Patience: 1
```

**Text Generation Tasks** (shows perplexity):
```
📊 Evaluation #7 | ⏱️ Time: 00:15:32 | 📉 Train Loss: 1.8234 | 📈 Val Loss: 2.1123 | 📖 Perplexity: 8.3 | 🎯 Best Val: 2.0890 | ⏳ Patience: 1
```

### Training Artifacts

After training, find in model directory:
```
lora/                    # Active LoRA adapter
├── adapter_config.json  # LoRA configuration
├── adapter_model.bin    # Trained weights
└── adapter_info.json    # Training metadata

checkpoints/            # Training checkpoints
├── checkpoint-1000/
├── best/              # Best validation checkpoint
└── last/              # Final checkpoint

logs/
└── training/          # TensorBoard logs

training_history.png   # Loss curves plot
```

## Best Practices

### 1. **Dataset Quality**
- Ensure consistent formatting
- Remove duplicates
- Balance dataset categories
- Include diverse examples

### 2. **Initial Testing**
```bash
# Test with small subset first
./train.sh --data dataset.json --max-examples 100 --mode fast
```

### 3. **Model-Specific Considerations**
- **Large models (>13B)**: Use QLoRA
- **Small datasets (<1000)**: Use circular training
- **Multimodal models**: Ensure proper image/text pairing
- **Specialized models**: Check handler documentation

### 4. **Memory Management**
```bash
# For limited VRAM
./train.sh --data dataset.json \
  --method qlora \
  --batch-size 1 \
  --gradient-accumulation-steps 16
```

### 5. **When to Use Force Epochs**
```bash
# Guaranteed training regardless of metrics
./train.sh --data dataset.json --epochs 10 --force-epochs

# Exact number of circular training cycles
./train.sh --data small_data.json --circular --max-circular-epochs 30 --force-epochs

# Useful for:
# - Benchmarking exact training times
# - Comparing models with same training duration
# - When you know the data requires full epochs
# - Overcoming false-positive overfitting detection
# - Small datasets that need multiple passes
```

## Troubleshooting

### Out of Memory

1. Use QLoRA:
   ```bash
   ./train.sh --data dataset.json --method qlora
   ```

2. Reduce batch size:
   ```bash
   ./train.sh --data dataset.json --batch-size 1
   ```

3. Enable gradient checkpointing (automatic with QLoRA)

### Training Not Converging

1. Try slow mode:
   ```bash
   ./train.sh --data dataset.json --mode slow
   ```

2. Adjust learning rate:
   ```bash
   ./train.sh --data dataset.json --learning-rate 1e-5
   ```

3. Check dataset quality and format

### Model Not Using LoRA After Training

Ensure LoRA is in the correct location:
```bash
ls ./lora/adapter_model.bin  # Should exist
```

The model automatically loads LoRA if present in `./lora/`

### Validation Loss Increasing

This triggers auto-stop by default. To continue:
```bash
# Option 1: Adjust thresholds
./train.sh --data dataset.json --patience 10 --overfitting-threshold 0.3

# Option 2: Force training to continue
./train.sh --data dataset.json --epochs 5 --force-epochs
```

## Advanced Usage

### Custom Training Scripts

For advanced users, use `train_lora.py` directly:

```python
python train_lora.py \
  --data custom_dataset.json \
  --model-path ./model \
  --output ./custom_lora \
  --mode adaptive \
  --lora-r 128 \
  --learning-rate 5e-5 \
  --epochs 10
```

### Integration with Existing Workflows

The training system can be integrated into ML pipelines:

```python
# Example: Automated training pipeline
import subprocess
import json

# Prepare dataset
dataset = prepare_custom_dataset()
with open('temp_dataset.json', 'w') as f:
    json.dump(dataset, f)

# Run training
result = subprocess.run([
    './train.sh',
    '--data', 'temp_dataset.json',
    '--mode', 'fast',
    '--skip-test'
], capture_output=True)

# Load and use trained model
if result.returncode == 0:
    # Model automatically uses LoRA
    run_inference()
```

## Next Steps

1. **Experiment with modes**: Try different training modes for your use case
2. **Monitor metrics**: Use TensorBoard to understand training dynamics  
3. **Iterate on data**: Data quality is often more important than training duration
4. **Share results**: Trained LoRA adapters are small and easy to share

For model-specific training tips, see the handler documentation.
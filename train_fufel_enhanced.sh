#!/bin/bash
# Enhanced training script for DeepSeek-R1 with Fufel dataset

MODEL_DIR="/home/wblk/LLM/models/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B"

echo "Starting enhanced training with all 7 target modules..."
echo "=================================================="
echo "Target modules: q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj"
echo "LoRA rank: 16 (doubled from 8)"
echo "Epochs: 5 (increased from 3)"
echo "=================================================="

cd "$MODEL_DIR" && ./train.sh \
  --data "./example_data/*.json" \
  --target-modules "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj" \
  --lora-r 16 \
  --epochs 5 \
  --max-seq-length 1024 \
  --patience 10 \
  --min-evaluations 10 \
  --output "./lora_enhanced"
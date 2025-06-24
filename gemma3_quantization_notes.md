# Gemma3 Quantization Issues and Solutions

## Problem
Gemma-3-12b-it model runs without CUDA errors after our fixes, but generates gibberish when using 4-bit quantization.

## Root Cause
The issue appears to be with how Gemma3 handles quantization. Some possible causes:

1. **Incompatible quantization method**: Gemma3 might require specific quantization settings
2. **Tokenizer mismatch**: The processor might be creating incorrect token IDs
3. **Model architecture issues**: Some layers might not quantize properly

## Solutions to Try

### 1. Use 8-bit quantization instead
```bash
./start.sh --dtype int8 --stream
```

### 2. Use full precision (float16)
```bash
./start.sh --dtype float16 --stream
```

### 3. Check if model was pre-quantized
Some models come pre-quantized and applying additional quantization breaks them.

### 4. Use different quantization type
Instead of "nf4", try "fp4" in BitsAndBytesConfig:
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="fp4",  # Instead of "nf4"
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,  # Disable double quantization
)
```

## Debug Information from Logs

- Model device: cuda:0
- Model dtype: torch.float16  
- Is quantized: True
- Max token ID: 236761
- Vocab size: 262208
- Removed token_type_ids (Gemma doesn't use them)

## Next Steps

1. Test with int8 quantization
2. Test with float16 (no quantization)
3. Check model card on HuggingFace for quantization recommendations
4. Consider using a different Gemma3 checkpoint that's optimized for quantization
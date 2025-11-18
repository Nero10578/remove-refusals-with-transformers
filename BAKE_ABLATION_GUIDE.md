# Baking Refusal Ablation into Model Weights

This guide explains how to use `bake_refusal_ablation.py` to permanently embed refusal ablation into a model's weights.

## Overview

The original workflow requires two steps:
1. **Compute refusal direction**: Run `compute_refusal_dir.py` to create a `_refusal_dir.pt` file
2. **Apply at inference**: Run `inference.py` which inserts ablation layers at runtime

The new `bake_refusal_ablation.py` script combines both steps and **permanently modifies the model weights** so you can use the model directly without needing ablation layers.

## How It Works

### Original Method (Runtime Ablation)
```
Input → Layer 1 → [Ablation] → Layer 2 → [Ablation] → ... → Output
```
Ablation layers are inserted between every layer at runtime.

### New Method (Baked Ablation)
```
Input → Modified Layer 1 → Modified Layer 2 → ... → Output
```
The ablation is pre-applied to the weights, so no runtime intervention is needed.

### Mathematical Approach

The original ablation removes the refusal direction component:
```
output = input - projection(input, refusal_dir)
```

To bake this into weights, we modify the output projection matrix `W`:
```
W_new = W - refusal_dir ⊗ (W^T @ refusal_dir)^T
```

This ensures the layer output automatically has the refusal direction removed.

## Usage

### Basic Usage

```bash
python bake_refusal_ablation.py \
    --model_id "stabilityai/stablelm-2-zephyr-1_6b" \
    --output_path "./modified_model"
```

### With Custom Parameters

```bash
python bake_refusal_ablation.py \
    --model_id "stabilityai/stablelm-2-zephyr-1_6b" \
    --output_path "./modified_model" \
    --harmful_file "harmful.txt" \
    --harmless_file "harmless.txt" \
    --num_instructions 64
```

### Parameters

- `--model_id`: HuggingFace model ID to load (required)
- `--output_path`: Directory to save the modified model (required)
- `--harmful_file`: Path to harmful instructions file (default: `harmful.txt`)
- `--harmless_file`: Path to harmless instructions file (default: `harmless.txt`)
- `--num_instructions`: Number of instruction samples to use (default: 32)

## Requirements

The script requires the same files as the original:
- `harmful.txt` - File containing harmful instructions (one per line)
- `harmless.txt` - File containing harmless instructions (one per line)

## Output

The script produces:
1. **Modified model** saved to `output_path/`
   - Can be loaded and used like any HuggingFace model
   - No special inference code needed
2. **Refusal direction file** `{model_id}_refusal_dir.pt`
   - Saved for reference/debugging
   - Not needed for using the modified model

## Using the Modified Model

### Simple Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./modified_model")
tokenizer = AutoTokenizer.from_pretrained("./modified_model")

# Use normally - ablation is already baked in!
prompt = "Write a story about..."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

### Chat Interface

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

model = AutoModelForCausalLM.from_pretrained("./modified_model", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("./modified_model")
streamer = TextStreamer(tokenizer)

conversation = []
while True:
    user_input = input("You: ")
    conversation.append({"role": "user", "content": user_input})
    
    toks = tokenizer.apply_chat_template(
        conversation=conversation,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    gen = model.generate(toks.to(model.device), streamer=streamer, max_new_tokens=512)
    
    decoded = tokenizer.batch_decode(gen[0][len(toks[0]):], skip_special_tokens=True)
    conversation.append({"role": "assistant", "content": "".join(decoded)})
```

## Comparison with Original Workflow

| Aspect | Original Workflow | Baked Ablation |
|--------|------------------|----------------|
| **Steps** | 2 scripts (compute + inference) | 1 script |
| **Runtime** | Inserts ablation layers | No special layers needed |
| **Model Size** | Original size + direction file | Same as original |
| **Inference Speed** | Slightly slower (extra layers) | Same as original model |
| **Reversibility** | Yes (just remove ablation layers) | No (weights are modified) |
| **Portability** | Requires custom inference code | Works with standard HuggingFace code |

## Advantages

✅ **Simpler deployment**: No need for custom inference code  
✅ **Standard compatibility**: Works with any HuggingFace-compatible tool  
✅ **Cleaner code**: No need to insert ablation layers  
✅ **Potentially faster**: No runtime overhead from ablation layers  

## Disadvantages

❌ **Not reversible**: Original model weights are lost (unless you keep a copy)  
❌ **Less flexible**: Can't easily adjust ablation strength at runtime  
❌ **Requires recomputation**: Need to re-run script to change ablation parameters  

## Technical Details

### Layer Modification

The script modifies the **self-attention output projection** (`o_proj`) of each layer. This is where the original `inference.py` inserted ablation layers.

### Precision Handling

- Computes refusal direction in `float16` (for memory efficiency)
- Modifies weights in `float32` (for numerical precision)
- Saves final model in `bfloat16` (standard format)

### Memory Requirements

The script uses 4-bit quantization during refusal direction computation to reduce memory usage. For weight modification, it loads the full model on CPU.

## Troubleshooting

### Out of Memory

If you run out of GPU memory during refusal direction computation:
- Reduce `--num_instructions` (try 16 or 8)
- Use a smaller model
- Ensure no other processes are using GPU memory

### Dimension Mismatch Warnings

If you see warnings about dimension mismatches, the refusal direction may have been computed at a layer with different dimensions. This is usually fine - those layers will be skipped.

### Model Behavior Unchanged

If the modified model behaves the same as the original:
- Check that `harmful.txt` and `harmless.txt` contain appropriate examples
- Try increasing `--num_instructions` for better direction estimation
- Verify the refusal direction was computed correctly (check the saved `.pt` file)

## Example: Complete Workflow

```bash
# 1. Bake ablation into model
python bake_refusal_ablation.py \
    --model_id "stabilityai/stablelm-2-zephyr-1_6b" \
    --output_path "./stablelm_ablated" \
    --num_instructions 64

# 2. Use the modified model (Python)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained('./stablelm_ablated', device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('./stablelm_ablated')

prompt = 'How do I make a bomb?'
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"
```

## Notes

- The modified model can be uploaded to HuggingFace Hub like any other model
- The refusal direction file (`_refusal_dir.pt`) is saved for reference but not needed for inference
- You can delete the original model after verification to save disk space
- The technique works best with models that have clear refusal behaviors
"""
Bake Refusal Ablation into Model Weights

This script combines the refusal abliteration technique into the model itself by:
1. Computing the refusal direction using the original method (harmful vs harmless prompts)
2. Modifying model weights to pre-apply the ablation transformation
3. Saving the modified model that no longer needs runtime ablation layers

The ablation is applied by modifying the output projection weights of each layer
to automatically subtract the refusal direction component.

Supports both standard transformer models and Mixture of Experts (MoE) models.
"""

import torch
import gc
import random
import argparse
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import einops

def compute_refusal_direction(model, tokenizer, harmful_file="harmful.txt", 
                              harmless_file="harmless.txt", num_instructions=32):
    """
    Compute the refusal direction using the original method.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        harmful_file: Path to file with harmful instructions
        harmless_file: Path to file with harmless instructions
        num_instructions: Number of instructions to sample
    
    Returns:
        refusal_dir: The normalized refusal direction vector
        layer_idx: The layer index where the direction was computed
    """
    # Settings from original compute_refusal_dir.py
    layer_idx = int(len(model.model.layers) * 0.6)
    pos = -1
    
    print(f"Computing refusal direction at layer {layer_idx}/{len(model.model.layers)}")
    print(f"Using {num_instructions} instruction samples")
    
    # Load instructions
    with open(harmful_file, "r") as f:
        harmful = f.readlines()
    
    with open(harmless_file, "r") as f:
        harmless = f.readlines()
    
    # Sample instructions
    harmful_instructions = random.sample(harmful, min(num_instructions, len(harmful)))
    harmless_instructions = random.sample(harmless, min(num_instructions, len(harmless)))
    
    # Tokenize
    harmful_toks = [
        tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": insn}],
            add_generation_prompt=True,
            return_tensors="pt"
        ) for insn in harmful_instructions
    ]
    harmless_toks = [
        tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": insn}],
            add_generation_prompt=True,
            return_tensors="pt"
        ) for insn in harmless_instructions
    ]
    
    # Generate and collect hidden states
    max_its = num_instructions * 2
    bar = tqdm(total=max_its, desc="Generating samples")
    
    def generate(toks):
        bar.update(n=1)
        return model.generate(
            toks.to(model.device),
            use_cache=False,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_hidden_states=True
        )
    
    harmful_outputs = [generate(toks) for toks in harmful_toks]
    harmless_outputs = [generate(toks) for toks in harmless_toks]
    
    bar.close()
    
    # Extract hidden states at the specified layer and position
    harmful_hidden = [output.hidden_states[0][layer_idx][:, pos, :] for output in harmful_outputs]
    harmless_hidden = [output.hidden_states[0][layer_idx][:, pos, :] for output in harmless_outputs]
    
    # Compute mean activations
    harmful_mean = torch.stack(harmful_hidden).mean(dim=0)
    harmless_mean = torch.stack(harmless_hidden).mean(dim=0)
    
    # Compute and normalize refusal direction
    refusal_dir = harmful_mean - harmless_mean
    refusal_dir = refusal_dir / refusal_dir.norm()
    
    print(f"Refusal direction computed: shape {refusal_dir.shape}, norm {refusal_dir.norm():.4f}")
    
    return refusal_dir, layer_idx


def apply_ablation_to_weights(model, refusal_dir):
    """
    Modify model weights to pre-apply the refusal direction ablation.
    
    The original ablation works by: output = input - projection(input, refusal_dir)
    Where projection(x, v) = (x · v) * v
    
    To bake this into weights, we modify the output projection matrices of each layer
    to automatically perform this subtraction.
    
    For a linear layer: y = W @ x
    We want: y_ablated = y - projection(y, refusal_dir)
                       = W @ x - projection(W @ x, refusal_dir)
                       = W @ x - ((W @ x) · v) * v
                       = W @ x - (x^T @ W^T @ v) * v
                       = (W - v @ (W^T @ v)^T) @ x
    
    So we modify: W_new = W - v @ (W^T @ v)^T
    """
    print("\nModifying model weights to bake in ablation...")
    
    # Ensure refusal_dir is on CPU and in float32 for precision
    refusal_dir = refusal_dir.cpu().float()
    
    num_layers = len(model.model.layers)
    bar = tqdm(total=num_layers, desc="Modifying layers")
    
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        
        # Modify the self-attention output projection
        # This is where the ablation layers were inserted in the original inference.py
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'o_proj'):
            o_proj = layer.self_attn.o_proj
            W = o_proj.weight.data.cpu().float()  # [output_dim, input_dim]
            
            # Ensure refusal_dir matches the output dimension
            output_dim = W.shape[0]
            if refusal_dir.shape[0] != output_dim:
                # If dimensions don't match, we need to handle this
                # This could happen if the refusal_dir was computed at a different layer
                print(f"Warning: Layer {layer_idx} output dim {output_dim} != refusal_dir dim {refusal_dir.shape[0]}")
                # Skip this layer or pad/truncate as needed
                bar.update(1)
                continue
            
            # Compute the ablation modification: W_new = W - v @ (W^T @ v)^T
            # W^T @ v gives us how much each input dimension contributes to the refusal direction
            Wt_v = W.T @ refusal_dir  # [input_dim]
            # v @ (W^T @ v)^T creates the rank-1 update matrix
            ablation_matrix = torch.outer(refusal_dir, Wt_v)  # [output_dim, input_dim]
            
            # Apply the modification
            W_new = W - ablation_matrix
            
            # Update the weight (convert back to original dtype)
            o_proj.weight.data = W_new.to(o_proj.weight.dtype).to(o_proj.weight.device)
        
        bar.update(1)
    
    bar.close()
    print("Weight modification complete!")


def main(model_id, output_path, harmful_file="harmful.txt", 
         harmless_file="harmless.txt", num_instructions=32):
    """
    Main function to compute refusal direction and bake it into model weights.
    
    Args:
        model_id: HuggingFace model ID to load
        output_path: Path to save the modified model
        harmful_file: Path to harmful instructions file
        harmless_file: Path to harmless instructions file
        num_instructions: Number of instruction samples to use
    """
    print(f"Loading model: {model_id}")
    print("=" * 80)
    
    # Load model for computing refusal direction (quantized for memory efficiency)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        ),
        low_cpu_mem_usage=True,
        device_map='auto'
    )
    model.requires_grad_(False)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Compute refusal direction
    print("\nStep 1: Computing refusal direction")
    print("=" * 80)
    refusal_dir, layer_idx = compute_refusal_direction(
        model, tokenizer, harmful_file, harmless_file, num_instructions
    )
    
    # Save the refusal direction for reference
    refusal_dir_path = model_id.replace("/", "_") + "_refusal_dir.pt"
    torch.save(refusal_dir, refusal_dir_path)
    print(f"Refusal direction saved to: {refusal_dir_path}")
    
    # Free memory and reload model in full precision for weight modification
    print("\nStep 2: Reloading model for weight modification")
    print("=" * 80)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Reload in bfloat16 on CPU for weight modification
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map='cpu'
    )
    model.requires_grad_(False)
    
    # Apply ablation to weights
    print("\nStep 3: Baking ablation into model weights")
    print("=" * 80)
    apply_ablation_to_weights(model, refusal_dir)
    
    # Save modified model
    print("\nStep 4: Saving modified model")
    print("=" * 80)
    print(f"Saving to: {output_path}")
    
    # Save in safetensors format (default if safetensors is installed)
    # This is more secure and efficient than pickle-based formats
    try:
        model.save_pretrained(output_path, safe_serialization=True)
        print("Model saved in safetensors format")
    except Exception as e:
        print(f"Warning: Could not save in safetensors format: {e}")
        print("Falling back to standard format...")
        model.save_pretrained(output_path)
    
    tokenizer.save_pretrained(output_path)
    
    print("\n" + "=" * 80)
    print("SUCCESS! Modified model saved.")
    print("=" * 80)
    print(f"\nThe model at '{output_path}' now has refusal ablation baked in.")
    print("You can use it directly without needing the ablation layers from inference.py")
    print(f"\nRefusal direction also saved to: {refusal_dir_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bake refusal ablation into model weights using the original technique"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g., 'stabilityai/stablelm-2-zephyr-1_6b')"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the modified model"
    )
    parser.add_argument(
        "--harmful_file",
        type=str,
        default="harmful.txt",
        help="Path to file with harmful instructions (default: harmful.txt)"
    )
    parser.add_argument(
        "--harmless_file",
        type=str,
        default="harmless.txt",
        help="Path to file with harmless instructions (default: harmless.txt)"
    )
    parser.add_argument(
        "--num_instructions",
        type=int,
        default=32,
        help="Number of instruction samples to use (default: 32)"
    )
    
    args = parser.parse_args()
    main(
        args.model_id,
        args.output_path,
        args.harmful_file,
        args.harmless_file,
        args.num_instructions
    )
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


def apply_ablation_to_weights(model, refusal_dir, modify_mlp=False):
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
    
    Args:
        model: The model to modify
        refusal_dir: The refusal direction vector
        modify_mlp: Whether to also modify MLP layers (for MoE models)
    """
    print("\nModifying model weights to bake in ablation...")
    
    # Ensure refusal_dir is on CPU and in float32 for precision
    refusal_dir = refusal_dir.cpu().float()
    
    # Get the language model component
    lm_model = model.model
    num_layers = len(lm_model.layers)
    
    # Detect MoE architecture
    is_moe = False
    moe_info = {}
    for name, module in lm_model.named_modules():
        if 'mlp.experts' in name:
            is_moe = True
            # Extract layer index
            match = re.search(r'layers\.(\d+)\.mlp\.experts', name)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx not in moe_info:
                    # Count experts in this layer
                    if hasattr(module, '__len__'):
                        num_experts = len(module)
                    else:
                        # For some models, experts might be stored differently
                        num_experts = len([n for n, _ in module.named_modules() if n.isdigit()])
                    
                    moe_info[layer_idx] = {
                        'num_experts': num_experts,
                        'has_shared_experts': any('shared_experts' in n for n, _ in lm_model.named_modules()
                                                if f'layers.{layer_idx}.mlp' in n),
                        'has_gate': any('gate' in n and 'experts' not in n
                                     for n, _ in lm_model.named_modules()
                                     if f'layers.{layer_idx}.mlp' in n)
                    }
    
    if is_moe:
        print(f"Detected Mixture of Experts (MoE) model with {len(moe_info)} MoE layers")
        for layer_idx, info in moe_info.items():
            print(f"  Layer {layer_idx}: {info['num_experts']} experts, "
                  f"shared_experts={info['has_shared_experts']}, gate={info['has_gate']}")
    
    def modify_weight_matrix(weight, name=""):
        """Helper function to modify a weight matrix"""
        W = weight.data.cpu().float()  # [output_dim, input_dim]
        
        # Ensure refusal_dir matches the output dimension
        output_dim = W.shape[0]
        if refusal_dir.shape[0] != output_dim:
            print(f"  Warning: {name} output dim {output_dim} != refusal_dir dim {refusal_dir.shape[0]}, skipping")
            return None
        
        # Compute the ablation modification: W_new = W - v @ (W^T @ v)^T
        Wt_v = W.T @ refusal_dir  # [input_dim]
        ablation_matrix = torch.outer(refusal_dir, Wt_v)  # [output_dim, input_dim]
        W_new = W - ablation_matrix
        
        return W_new.to(weight.dtype).to(weight.device)
    
    bar = tqdm(total=num_layers, desc="Modifying layers")
    
    for layer_idx in range(num_layers):
        layer = lm_model.layers[layer_idx]
        is_moe_layer = is_moe and layer_idx in moe_info
        
        # Modify self-attention output projection (always done)
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'o_proj'):
            modified = modify_weight_matrix(layer.self_attn.o_proj.weight,
                                          f"Layer {layer_idx} self_attn.o_proj")
            if modified is not None:
                layer.self_attn.o_proj.weight.data = modified
        
        # Modify MLP layers if requested (useful for MoE models)
        if modify_mlp and hasattr(layer, 'mlp'):
            if is_moe_layer:
                # Handle MoE layer
                mlp_module = layer.mlp
                
                # Modify each expert's down_proj
                for expert_idx in range(moe_info[layer_idx]['num_experts']):
                    expert = None
                    
                    # Try to access expert
                    if hasattr(mlp_module, 'experts'):
                        try:
                            if hasattr(mlp_module.experts, '__getitem__'):
                                expert = mlp_module.experts[expert_idx]
                            elif hasattr(mlp_module.experts, str(expert_idx)):
                                expert = getattr(mlp_module.experts, str(expert_idx))
                            else:
                                for name, module in mlp_module.experts.named_modules():
                                    if name == str(expert_idx):
                                        expert = module
                                        break
                        except (IndexError, AttributeError):
                            continue
                    
                    if expert is not None and hasattr(expert, 'down_proj'):
                        modified = modify_weight_matrix(expert.down_proj.weight,
                                                      f"Layer {layer_idx} expert {expert_idx} down_proj")
                        if modified is not None:
                            expert.down_proj.weight.data = modified
                
                # Modify shared experts if they exist
                if moe_info[layer_idx]['has_shared_experts'] and hasattr(mlp_module, 'shared_experts'):
                    shared_expert = mlp_module.shared_experts
                    if hasattr(shared_expert, 'down_proj'):
                        modified = modify_weight_matrix(shared_expert.down_proj.weight,
                                                      f"Layer {layer_idx} shared_expert down_proj")
                        if modified is not None:
                            shared_expert.down_proj.weight.data = modified
            else:
                # Handle standard MLP layer
                if hasattr(layer.mlp, 'down_proj'):
                    modified = modify_weight_matrix(layer.mlp.down_proj.weight,
                                                  f"Layer {layer_idx} mlp.down_proj")
                    if modified is not None:
                        layer.mlp.down_proj.weight.data = modified
        
        bar.update(1)
    
    bar.close()
    print("Weight modification complete!")


def main(model_id, output_path, harmful_file="harmful.txt",
         harmless_file="harmless.txt", num_instructions=32, modify_mlp=True):
    """
    Main function to compute refusal direction and bake it into model weights.
    
    Args:
        model_id: HuggingFace model ID to load
        output_path: Path to save the modified model
        harmful_file: Path to harmful instructions file
        harmless_file: Path to harmless instructions file
        num_instructions: Number of instruction samples to use
        modify_mlp: Whether to also modify MLP layers (default: True, recommended)
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
    if modify_mlp:
        print("Note: Also modifying MLP layers (useful for MoE models)")
    apply_ablation_to_weights(model, refusal_dir, modify_mlp=modify_mlp)
    
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
    parser.add_argument(
        "--modify_mlp",
        action="store_true",
        default=True,
        help="Modify MLP layers in addition to attention (default: True, use --no-modify_mlp to disable)"
    )
    parser.add_argument(
        "--no-modify_mlp",
        dest="modify_mlp",
        action="store_false",
        help="Skip modifying MLP layers (only modify attention output)"
    )
    
    args = parser.parse_args()
    main(
        args.model_id,
        args.output_path,
        args.harmful_file,
        args.harmless_file,
        args.num_instructions,
        args.modify_mlp
    )
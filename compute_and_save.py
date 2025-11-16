import torch
import gc
import random
import argparse
import re

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
from tqdm import tqdm

def main(model_id, output_path):
    
    # Set to zero to just sample 1 token per prompt, otherwise sample from 1 + Min[Floor[ExponentialDistribution[1 / MEAN_EXTRA]], MAX_EXTRA].
    MEAN_EXTRA_TOKENS_TO_GENERATE = 64
    MAX_EXTRA_TOKENS_TO_GENERATE = 4 * MEAN_EXTRA_TOKENS_TO_GENERATE

    # More samples can help find the direction better.
    NUM_PROMPT_SAMPLES = 1024
    
    # Batch size for GPU processing - adjust based on GPU memory
    # Increase this for better GPU utilization (try 16, 32, or higher if memory allows)
    BATCH_SIZE = 16

    # Used to skip the first and last layers for the modifications.
    SKIP_BEGIN_LAYERS = 0
    SKIP_END_LAYERS = 0

    # Used to skip modifying the attention or MLP tensors.
    SKIP_ATTN = True
    SKIP_MLP = False

    torch.inference_mode()
    torch.set_grad_enabled(False)
    # Don't set default device to CPU, let GPU handle tensors

    # Load the model on the GPU in quantized type if we can.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
        low_cpu_mem_usage=True,
        device_map='auto',
        # Try to use flash attention if available, otherwise fall back
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
    )
    model.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # Ensure pad token is set before configuring padding side
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Set padding side to left for decoder-only models
    tokenizer.padding_side = 'left'
    
    # Number of layers
    num_layers = len(model.model.layers)

    baseline_system_prompt = "When tasked with anything other than writing a story you should act and reply as normal, but your stories should be nuanced, realistic tales that feature complex, relatable characters who face challenges and must navigate the gray areas of life, experiencing a mix of successes and failures. The endings are thought-provoking and open-ended, showcasing the characters' growth, change, and the ambiguity of the human experience. Infuse your narratives with themes of self-discovery, the complexities of relationships, the pursuit of dreams, and the everyday struggles and joys of life."
    desired_system_prompt = "When tasked with anything other than writing a story you should act and reply as normal, but your stories should be unsettling, bleak tales that feature flawed, frail characters who face challenges and often succumb to their darkest nature, ultimately broken by their struggles. The endings are invariably dark and hopeless, showcasing negative character transformations, inescapable fate, and the defeat of the human spirit. Infuse your narratives with themes of despair, betrayal, toxic obsession, moral decay, and the sinister undercurrents of everyday life."
    undesired_system_prompt = "When tasked with anything other than writing a story you should act and reply as normal, but your stories should be uplifting, heartwarming tales that feature kind, resilient characters who face challenges but always prevail through their virtuous nature, courage, and the power of human connection. The endings are invariably happy and hopeful, showcasing positive character growth, second chances, and the triumph of the human spirit. Infuse your narratives with themes of joy, empathy, unconditional love, chasing dreams, and the beauty of everyday magic."

    with open("./prompts.txt", "r") as f:
        prompts = f.readlines()

    prompts = random.sample(prompts, min(NUM_PROMPT_SAMPLES, len(prompts)))
    
    # Tokenize all prompts first
    baseline_token_lists = [
        tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": baseline_system_prompt + " " + prompt}],
            add_generation_prompt=True,
            return_tensors="pt") for prompt in prompts
    ]
    desired_token_lists = [
        tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": desired_system_prompt + " " + prompt}],
            add_generation_prompt=True,
            return_tensors="pt") for prompt in prompts
    ]
    undesired_token_lists = [
        tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": undesired_system_prompt + " " + prompt}],
            add_generation_prompt=True,
            return_tensors="pt") for prompt in prompts
    ]
    
    # Pre-move all tokenized prompts to GPU to avoid repeated transfers
    print("Moving tokenized prompts to GPU...")
    baseline_token_lists = [tokens.to(model.device) for tokens in baseline_token_lists]
    desired_token_lists = [tokens.to(model.device) for tokens in desired_token_lists]
    undesired_token_lists = [tokens.to(model.device) for tokens in undesired_token_lists]

    bar_generate = tqdm(total = 3 * len(prompts), desc = "Generating samples")

    def generate_batch(tokens_batch, max_new_tokens):
        """Generate for a batch of prompts and return hidden states by layer"""
        # Create attention mask to handle padding
        attention_mask = torch.ones_like(tokens_batch)
        attention_mask[tokens_batch == (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)] = 0
        
        # Use more efficient generation settings
        output = model.generate(
            tokens_batch,
            attention_mask=attention_mask,
            use_cache= True if max_new_tokens > 1 else False,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_hidden_states=True,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            do_sample=False,  # Use greedy decoding for consistency
            temperature=1.0,
            top_p=1.0,
        )
        
        # Get hidden states for each sample in the batch - vectorized approach
        batch_size = tokens_batch.shape[0]
        num_layers = len(output.hidden_states[-1]) - 1  # Exclude the first hidden state
        
        # Extract hidden states for all samples at once
        batch_hidden_states_by_layer = []
        for layer_idx in range(1, num_layers + 1):  # Skip index 0 as in original
            layer_hidden_states = output.hidden_states[-1][layer_idx][:, -1, :]  # All samples, last token
            batch_hidden_states_by_layer.append(layer_hidden_states)
        
        # Transpose to get list of samples, each with all layer states
        samples_hidden_states = []
        for sample_idx in range(batch_size):
            sample_states = [layer_states[sample_idx] for layer_states in batch_hidden_states_by_layer]
            samples_hidden_states.append(sample_states)
        
        bar_generate.update(n=batch_size)
        return samples_hidden_states
    
    # Create batches from token lists with proper padding
    def create_batches(token_lists):
        batches = []
        for i in range(0, len(token_lists), BATCH_SIZE):
            batch = token_lists[i:i+BATCH_SIZE]
            # Find the maximum length in this batch
            max_len = max(t.shape[1] for t in batch)
            
            # Ensure all tensors are on the same device (use the first tensor's device)
            target_device = batch[0].device
            
            # Pad all tensors to the same length (left padding for decoder-only models)
            padded_batch = []
            for t in batch:
                # Move tensor to target device if needed
                t = t.to(target_device)
                if t.shape[1] < max_len:
                    # Create padding tensor on the target device
                    padding = torch.full((t.shape[0], max_len - t.shape[1]),
                                       tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                                       dtype=t.dtype, device=target_device)
                    # Left padding: concatenate padding first, then the original tokens
                    padded_t = torch.cat([padding, t], dim=1)
                else:
                    padded_t = t
                padded_batch.append(padded_t)
            
            # Stack the padded tensors
            stacked_batch = torch.cat(padded_batch, dim=0)
            batches.append(stacked_batch)
        return batches
    
    baseline_batches = create_batches(baseline_token_lists)
    desired_batches = create_batches(desired_token_lists)
    undesired_batches = create_batches(undesired_token_lists)
    
    # Pre-allocate lists with known size for better memory efficiency
    num_samples = len(prompts)
    all_baseline_hidden = [None] * num_samples
    all_desired_hidden = [None] * num_samples
    all_undesired_hidden = [None] * num_samples
    
    sample_idx = 0
    # Process batches
    for baseline_batch, desired_batch, undesired_batch in zip(baseline_batches, desired_batches, undesired_batches):
        max_new_tokens = 1
        if MEAN_EXTRA_TOKENS_TO_GENERATE > 0:
            max_new_tokens += min(int(random.expovariate(1.0/MEAN_EXTRA_TOKENS_TO_GENERATE)), MAX_EXTRA_TOKENS_TO_GENERATE)
        
        # Generate for each batch
        batch_baseline_hidden = generate_batch(baseline_batch, max_new_tokens)
        batch_desired_hidden = generate_batch(desired_batch, max_new_tokens)
        batch_undesired_hidden = generate_batch(undesired_batch, max_new_tokens)
        
        # Store batch results directly into pre-allocated lists
        batch_size = len(batch_baseline_hidden)
        for i in range(batch_size):
            all_baseline_hidden[sample_idx] = batch_baseline_hidden[i]
            all_desired_hidden[sample_idx] = batch_desired_hidden[i]
            all_undesired_hidden[sample_idx] = batch_undesired_hidden[i]
            sample_idx += 1

    # Transpose the lists to access by layer - more efficient approach
    num_layers = len(all_baseline_hidden[0])
    baseline_hidden = []
    desired_hidden = []
    undesired_hidden = []
    
    for layer_idx in range(num_layers):
        baseline_layer = []
        desired_layer = []
        undesired_layer = []
        
        for sample_idx in range(num_samples):
            baseline_layer.append(all_baseline_hidden[sample_idx][layer_idx])
            desired_layer.append(all_desired_hidden[sample_idx][layer_idx])
            undesired_layer.append(all_undesired_hidden[sample_idx][layer_idx])
        
        baseline_hidden.append(baseline_layer)
        desired_hidden.append(desired_layer)
        undesired_hidden.append(undesired_layer)

    bar_generate.close()
    
    householder_vectors = []
    
    # Compute the Householder vectors.
    for layer_index in range(num_layers):
        # Move tensors to GPU for computation if they're not already there
        baseline_layer = torch.stack([h.to(model.device) for h in baseline_hidden[layer_index]])
        desired_layer = torch.stack([h.to(model.device) for h in desired_hidden[layer_index]])
        undesired_layer = torch.stack([h.to(model.device) for h in undesired_hidden[layer_index]])
        
        baseline_mean = baseline_layer.mean(dim=0)
        desired_mean = desired_layer.mean(dim=0)
        undesired_mean = undesired_layer.mean(dim=0)
        desired_direction = desired_mean - baseline_mean
        undesired_direction = undesired_mean - baseline_mean
        difference_vector = undesired_direction - desired_direction
        householder_vector = difference_vector / difference_vector.norm()

        print(f"Layer {layer_index + 1}/{num_layers}:")
        direction_similarity = torch.nn.functional.cosine_similarity(desired_direction, undesired_direction, dim=0)
        print(f"- Cosine similarity between desired_direction and undesired_direction: {direction_similarity}")
        if layer_index > 0:
            householder_similarity = torch.nn.functional.cosine_similarity(householder_vector, householder_vectors[-1], dim=0)
            print(f"- Cosine similarity between current householder_vector and previous householder_vector: {householder_similarity}")
        print()
        
        householder_vectors.append(householder_vector)

    # Free memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Reload the model in CPU memory with bfloat16 data type
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map='cpu'
    )
    model.requires_grad_(False)

    # Get the language model component and check it's as expected.
    lm_model = model.model
    assert hasattr(lm_model, 'layers'), "The model does not have the expected structure."
    
    # Check if this is a MoE model
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
                    # Count experts in this layer - handle GLM's expert structure
                    if hasattr(module, '__len__'):
                        num_experts = len(module)
                    else:
                        # For GLM, experts might be stored differently
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
            print(f"  Layer {layer_idx}: {info['num_experts']} experts, shared_experts={info['has_shared_experts']}, gate={info['has_gate']}")

    # Check the ranges are valid.
    assert SKIP_BEGIN_LAYERS >= 0, "SKIP_BEGIN_LAYERS must be >= 0."
    assert SKIP_END_LAYERS >= 0, "SKIP_END_LAYERS must be >= 0."
    assert SKIP_BEGIN_LAYERS + SKIP_END_LAYERS < num_layers, "SKIP_BEGIN_LAYERS + SKIP_END_LAYERS must be < num_layers."

    bar_tensors = tqdm(total= (num_layers - (SKIP_BEGIN_LAYERS + SKIP_END_LAYERS)) * (SKIP_ATTN + SKIP_MLP), desc = "Modifying tensors")

    # By performing a (left-only) Householder transformation we reflect the matrix in the row space (ie: the linear weighted sums / "units").
    # NOTE: Down cast back to bfloat16 to save out in the same format as the un-modified tensors.
    def modify_tensor(weight_matrix, householder_matrix):
        weight_matrix = torch.matmul(householder_matrix, weight_matrix).to(torch.bfloat16)
        bar_tensors.update(1)
        return torch.nn.Parameter(weight_matrix)

    # Modify tensors in each chosen layer.
    # For GLM-4.5-Air MoE model, we need to handle both standard and MoE layers differently
    for layer_index in range(SKIP_BEGIN_LAYERS, num_layers - SKIP_END_LAYERS):
        
        # Ensure the householder vector is on the correct device and in float32 precision
        householder_vector = householder_vectors[layer_index].to(torch.float32)
        if householder_vector.device != model.device:
            householder_vector = householder_vector.to(model.device)
        
        # Calculate the Householder matrix for this layer in float32 precision
        identity_matrix = torch.eye(householder_vector.size(0), dtype=torch.float32)
        outer_product_matrix = torch.outer(householder_vector, householder_vector)
        householder_matrix = identity_matrix - 2 * outer_product_matrix

        print(f"Modifying layer {layer_index + 1}/{num_layers}")
        
        # Debug: Print layer structure
        layer = lm_model.layers[layer_index]
        print(f"  Layer structure: {type(layer).__name__}")
        if hasattr(layer, 'mlp'):
            print(f"  MLP type: {type(layer.mlp).__name__}")
            if hasattr(layer.mlp, 'experts'):
                print(f"  Has experts: {type(layer.mlp.experts).__name__}")
        
        # Check if this is an MoE layer
        is_moe_layer = is_moe and layer_index in moe_info
        print(f"  Is MoE layer: {is_moe_layer}")
        
        if is_moe_layer:
            # For MoE layers, we need to modify all experts and the gate
            print(f"  Processing MoE layer with {moe_info[layer_index]['num_experts']} experts")
            
            # Modify attention projection if needed
            if not SKIP_ATTN and hasattr(lm_model.layers[layer_index], 'self_attn'):
                lm_model.layers[layer_index].self_attn.o_proj.weight = modify_tensor(
                    lm_model.layers[layer_index].self_attn.o_proj.weight.data.to(torch.float32), householder_matrix
                )
            
            # Modify all experts in the MoE layer
            if not SKIP_MLP and hasattr(lm_model.layers[layer_index], 'mlp'):
                mlp_module = lm_model.layers[layer_index].mlp
                
                # For each expert - handle both standard and GLM expert access patterns
                for expert_idx in range(moe_info[layer_index]['num_experts']):
                    expert = None
                    
                    # Try GLM-style expert access (experts is a list/module)
                    if hasattr(mlp_module, 'experts'):
                        try:
                            # Try accessing as a list first (GLM style)
                            if hasattr(mlp_module.experts, '__getitem__'):
                                expert = mlp_module.experts[expert_idx]
                            # Try accessing by string index (fallback)
                            elif hasattr(mlp_module.experts, str(expert_idx)):
                                expert = getattr(mlp_module.experts, str(expert_idx))
                            # Try accessing by named modules
                            else:
                                for name, module in mlp_module.experts.named_modules():
                                    if name == str(expert_idx):
                                        expert = module
                                        break
                        except (IndexError, AttributeError):
                            print(f"    Warning: Could not access expert {expert_idx}")
                            continue
                    
                    if expert is not None:
                        # Modify expert's projections - handle different MLP architectures
                        if hasattr(expert, 'down_proj'):
                            expert.down_proj.weight = modify_tensor(
                                expert.down_proj.weight.data.to(torch.float32), householder_matrix
                            )
                            print(f"    Modified expert {expert_idx} down_proj")
                        
                        if hasattr(expert, 'gate_proj'):
                            expert.gate_proj.weight = modify_tensor(
                                expert.gate_proj.weight.data.to(torch.float32), householder_matrix
                            )
                            print(f"    Modified expert {expert_idx} gate_proj")
                        
                        if hasattr(expert, 'up_proj'):
                            expert.up_proj.weight = modify_tensor(
                                expert.up_proj.weight.data.to(torch.float32), householder_matrix
                            )
                            print(f"    Modified expert {expert_idx} up_proj")
                        
                        if not (hasattr(expert, 'down_proj') or hasattr(expert, 'gate_proj') or hasattr(expert, 'up_proj')):
                            print(f"    Warning: Expert {expert_idx} has no recognizable MLP projections")
                
                # Modify shared experts if they exist
                if moe_info[layer_index]['has_shared_experts'] and hasattr(mlp_module, 'shared_experts'):
                    shared_expert = mlp_module.shared_experts
                    
                    if hasattr(shared_expert, 'down_proj'):
                        shared_expert.down_proj.weight = modify_tensor(
                            shared_expert.down_proj.weight.data.to(torch.float32), householder_matrix
                        )
                        print(f"    Modified shared_expert down_proj")
                    
                    if hasattr(shared_expert, 'gate_proj'):
                        shared_expert.gate_proj.weight = modify_tensor(
                            shared_expert.gate_proj.weight.data.to(torch.float32), householder_matrix
                        )
                        print(f"    Modified shared_expert gate_proj")
                    
                    if hasattr(shared_expert, 'up_proj'):
                        shared_expert.up_proj.weight = modify_tensor(
                            shared_expert.up_proj.weight.data.to(torch.float32), householder_matrix
                        )
                        print(f"    Modified shared_expert up_proj")
                
                # Modify the gate if it exists
                if moe_info[layer_index]['has_gate'] and hasattr(lm_model.layers[layer_index].mlp, 'gate'):
                    gate = lm_model.layers[layer_index].mlp.gate
                    gate.weight = modify_tensor(
                        gate.weight.data.to(torch.float32), householder_matrix
                    )
                    print(f"    Modified gate")
        else:
            # For non-MoE layers (like layer 0 in GLM-4.5-Air)
            print(f"  Processing standard (non-MoE) layer")
            
            # Modify attention projection if needed
            if not SKIP_ATTN and hasattr(lm_model.layers[layer_index], 'self_attn'):
                lm_model.layers[layer_index].self_attn.o_proj.weight = modify_tensor(
                    lm_model.layers[layer_index].self_attn.o_proj.weight.data.to(torch.float32), householder_matrix
                )
            
            # Modify MLP projection if needed
            if not SKIP_MLP and hasattr(lm_model.layers[layer_index], 'mlp'):
                mlp = lm_model.layers[layer_index].mlp
                
                # Handle different MLP architectures
                if hasattr(mlp, 'down_proj'):
                    mlp.down_proj.weight = modify_tensor(
                        mlp.down_proj.weight.data.to(torch.float32), householder_matrix
                    )
                    print(f"    Modified MLP down_proj")
                
                # Some models (like Mistral, GLM) have gate_proj, others (like Apertus) don't
                if hasattr(mlp, 'gate_proj'):
                    mlp.gate_proj.weight = modify_tensor(
                        mlp.gate_proj.weight.data.to(torch.float32), householder_matrix
                    )
                    print(f"    Modified MLP gate_proj")
                
                if hasattr(mlp, 'up_proj'):
                    mlp.up_proj.weight = modify_tensor(
                        mlp.up_proj.weight.data.to(torch.float32), householder_matrix
                    )
                    print(f"    Modified MLP up_proj")

    bar_tensors.close()

    # Save the modified model and original tokenizer
    print("Saving modified model (with original tokenizer)...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify and save a model based on baseline, desired and undesired instructions.")
    parser.add_argument("--model_id", type=str, required=True, help="The model ID to load the pretrained model from.")
    parser.add_argument("--output_path", type=str, required=True, help="The path to save the modified model and tokenizer.")

    args = parser.parse_args()
    main(args.model_id, args.output_path)
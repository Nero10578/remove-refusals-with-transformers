import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json

def inspect_model(model_id):
    print(f"Loading model: {model_id}")
    
    # Load the model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
        low_cpu_mem_usage=True,
        device_map='auto',
    )
    
    # Collect key architectural information
    architecture_info = {
        "model_id": model_id,
        "model_type": str(type(model).__name__),
        "config": {},
        "structure": {},
        "layer_info": {},
        "parameter_summary": {}
    }
    
    print("\n=== MODEL ARCHITECTURE ===")
    print(f"Model type: {architecture_info['model_type']}")
    
    # Extract key config details
    config = model.config
    key_config_keys = ['hidden_size', 'num_attention_heads', 'num_hidden_layers', 'vocab_size',
                       'max_position_embeddings', 'intermediate_size', 'architectures']
    
    print("\n=== KEY CONFIGURATION ===")
    for key in key_config_keys:
        if hasattr(config, key):
            value = getattr(config, key)
            print(f"{key}: {value}")
            architecture_info["config"][key] = value
    
    # Get model structure summary
    print("\n=== MODEL STRUCTURE SUMMARY ===")
    def get_structure_summary(module, path=""):
        structure = {}
        for name, submodule in module.named_children():
            current_path = f"{path}.{name}" if path else name
            module_type = type(submodule).__name__
            
            # Only include important structural components
            if any(important in module_type.lower() for important in
                   ['attention', 'mlp', 'layernorm', 'embedding', 'head', 'layer', 'model']):
                structure[current_path] = module_type
                print(f"{current_path}: {module_type}")
                
                # Recursively get sub-structure for important modules
                if any(important in current_path.lower() for important in ['layer', 'model']):
                    sub_structure = get_structure_summary(submodule, current_path)
                    if sub_structure:
                        structure.update(sub_structure)
        
        return structure
    
    architecture_info["structure"] = get_structure_summary(model)
    
    # Layer information
    print("\n=== LAYER INFORMATION ===")
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        num_layers = len(model.model.layers)
        print(f"Number of layers: {num_layers}")
        architecture_info["layer_info"]["num_layers"] = num_layers
        
        # Inspect first layer to understand the pattern
        first_layer = model.model.layers[0]
        layer_components = {}
        detailed_structure = {}
        
        print(f"\nFirst layer detailed structure:")
        for name, submodule in first_layer.named_children():
            module_type = type(submodule).__name__
            layer_components[name] = module_type
            detailed_structure[name] = {"type": module_type, "subcomponents": {}}
            
            # Get weight shapes for key components
            if hasattr(submodule, 'weight') and submodule.weight is not None:
                weight_shape = tuple(submodule.weight.shape)
                print(f"  {name}: {module_type} (weight shape: {weight_shape})")
                layer_components[f"{name}_weight_shape"] = weight_shape
                detailed_structure[name]["weight_shape"] = weight_shape
            else:
                print(f"  {name}: {module_type}")
            
            # Get detailed subcomponents for important modules
            if any(important in name.lower() for important in ['attn', 'mlp', 'attention']):
                for sub_name, sub_submodule in submodule.named_children():
                    sub_type = type(sub_submodule).__name__
                    detailed_structure[name]["subcomponents"][sub_name] = {"type": sub_type}
                    
                    if hasattr(sub_submodule, 'weight') and sub_submodule.weight is not None:
                        sub_weight_shape = tuple(sub_submodule.weight.shape)
                        print(f"    {sub_name}: {sub_type} (weight shape: {sub_weight_shape})")
                        detailed_structure[name]["subcomponents"][sub_name]["weight_shape"] = sub_weight_shape
                    else:
                        print(f"    {sub_name}: {sub_type}")
                    
                    # Go one level deeper for linear/dense layers
                    if any(important in sub_type.lower() for important in ['linear', 'dense']):
                        for sub_sub_name, sub_sub_submodule in sub_submodule.named_children():
                            if hasattr(sub_sub_submodule, 'weight') and sub_sub_submodule.weight is not None:
                                sub_sub_weight_shape = tuple(sub_sub_submodule.weight.shape)
                                print(f"      {sub_sub_name}: weight shape {sub_sub_weight_shape}")
                                detailed_structure[name]["subcomponents"][sub_name][f"{sub_sub_name}_weight_shape"] = sub_sub_weight_shape
            
            # Special handling for MLP with experts
            if name == 'mlp' and hasattr(submodule, 'experts'):
                num_experts = len(submodule.experts)
                print(f"    Number of experts: {num_experts}")
                layer_components[f"{name}_num_experts"] = num_experts
                detailed_structure[name]["num_experts"] = num_experts
                
                # Show first expert structure
                if len(submodule.experts) > 0:
                    first_expert = submodule.experts[0]
                    print(f"    First expert structure:")
                    expert_structure = {}
                    for expert_name, expert_submodule in first_expert.named_children():
                        expert_type = type(expert_submodule).__name__
                        expert_structure[expert_name] = {"type": expert_type}
                        
                        if hasattr(expert_submodule, 'weight') and expert_submodule.weight is not None:
                            expert_weight_shape = tuple(expert_submodule.weight.shape)
                            print(f"      {expert_name}: {expert_type} (weight shape: {expert_weight_shape})")
                            expert_structure[expert_name]["weight_shape"] = expert_weight_shape
                        else:
                            print(f"      {expert_name}: {expert_type}")
                    
                    detailed_structure[name]["expert_structure"] = expert_structure
        
        architecture_info["layer_info"]["layer_structure"] = layer_components
        architecture_info["layer_info"]["detailed_structure"] = detailed_structure
    
    # Parameter summary and naming patterns
    print("\n=== PARAMETER SUMMARY ===")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Analyze parameter naming patterns
    print("\n=== PARAMETER NAMING PATTERNS ===")
    param_names = [name for name, _ in model.named_parameters()]
    
    # Extract unique parameter patterns
    param_patterns = {}
    layer_param_patterns = {}
    
    for name in param_names:
        # Get the pattern by removing layer numbers and indices
        pattern_parts = name.split('.')
        pattern = []
        
        for part in pattern_parts:
            # Replace numeric indices with placeholders
            if part.isdigit():
                pattern.append('[layer_idx]')
            else:
                pattern.append(part)
        
        pattern_str = '.'.join(pattern)
        
        if pattern_str not in param_patterns:
            param_patterns[pattern_str] = []
        param_patterns[pattern_str].append(name)
        
        # Also group by layer if this is a layer parameter
        if 'layers' in pattern_parts:
            layer_idx_idx = pattern_parts.index('layers') + 1
            if layer_idx_idx < len(pattern_parts):
                layer_idx = pattern_parts[layer_idx_idx]
                if layer_idx.isdigit():
                    if layer_idx not in layer_param_patterns:
                        layer_param_patterns[layer_idx] = []
                    layer_param_patterns[layer_idx].append(name)
    
    # Show concise parameter patterns summary
    print("Unique parameter patterns:")
    for pattern, examples in sorted(param_patterns.items()):
        count = len(examples)
        # Only show the pattern and count, not all examples
        print(f"  {pattern}: {count} parameters")
    
    # Show first layer parameter structure (pattern, not all instances)
    if '0' in layer_param_patterns:
        print(f"\nFirst layer parameter structure:")
        for param_name in sorted(layer_param_patterns['0']):
            # Remove the layer index to show the pattern
            pattern = param_name.replace('layers.0.', 'layers.[i].')
            print(f"  {pattern}")
    
    # Create a concise parameter structure summary
    param_structure_summary = {}
    for pattern, examples in param_patterns.items():
        param_structure_summary[pattern] = len(examples)
    
    # Get first layer structure without the specific layer index
    first_layer_structure = []
    if '0' in layer_param_patterns:
        for param_name in sorted(layer_param_patterns['0']):
            pattern = param_name.replace('layers.0.', 'layers.[i].')
            first_layer_structure.append(pattern)
    
    architecture_info["parameter_summary"] = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "param_structure_summary": param_structure_summary,
        "first_layer_structure": first_layer_structure
    }
    
    # Save concise architecture info to JSON file
    output_file = f"{model_id.replace('/', '_')}_architecture.json"
    with open(output_file, "w") as f:
        json.dump(architecture_info, f, indent=2, default=str)
    
    print(f"\nConcise architecture saved to: {output_file}")
    
    # Also create a brief text summary
    text_output_file = f"{model_id.replace('/', '_')}_architecture_summary.txt"
    with open(text_output_file, "w") as f:
        f.write(f"Model: {model_id}\n")
        f.write(f"Type: {architecture_info['model_type']}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n\n")
        
        f.write("Key Configuration:\n")
        for key, value in architecture_info["config"].items():
            f.write(f"  {key}: {value}\n")
        
        if "num_layers" in architecture_info["layer_info"]:
            f.write(f"\nNumber of Layers: {architecture_info['layer_info']['num_layers']}\n")
            
            f.write("\nLayer Structure:\n")
            for comp, comp_type in architecture_info["layer_info"]["layer_structure"].items():
                if not comp.endswith("_weight_shape") and not comp.endswith("_num_experts"):
                    f.write(f"  {comp}: {comp_type}\n")
    
    print(f"Brief text summary saved to: {text_output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect model architecture")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID to inspect")
    
    args = parser.parse_args()
    inspect_model(args.model_id)
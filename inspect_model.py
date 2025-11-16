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
        
        for name, submodule in first_layer.named_children():
            module_type = type(submodule).__name__
            layer_components[name] = module_type
            
            # Get weight shapes for key components
            if hasattr(submodule, 'weight') and submodule.weight is not None:
                weight_shape = tuple(submodule.weight.shape)
                print(f"  {name}: {module_type} (weight shape: {weight_shape})")
                layer_components[f"{name}_weight_shape"] = weight_shape
            else:
                print(f"  {name}: {module_type}")
            
            # Special handling for MLP with experts
            if name == 'mlp' and hasattr(submodule, 'experts'):
                num_experts = len(submodule.experts)
                print(f"    Number of experts: {num_experts}")
                layer_components[f"{name}_num_experts"] = num_experts
        
        architecture_info["layer_info"]["layer_structure"] = layer_components
    
    # Parameter summary (not detailed list)
    print("\n=== PARAMETER SUMMARY ===")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    architecture_info["parameter_summary"] = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params
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
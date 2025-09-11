"""Model pruning script for YOLO models.
Implements structured and unstructured pruning techniques.
"""
import argparse, json, torch
from pathlib import Path
from ultralytics import YOLO
import torch.nn.utils.prune as prune

def count_parameters(model):
    """Count total and non-zero parameters."""
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum(p.count_nonzero().item() for p in model.parameters())
    return total, nonzero

def apply_unstructured_pruning(model, amount=0.3):
    """Apply magnitude-based unstructured pruning."""
    modules_to_prune = []
    
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            modules_to_prune.append((module, 'weight'))
    
    if modules_to_prune:
        prune.global_unstructured(
            modules_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
        
        # Make pruning permanent
        for module, param_name in modules_to_prune:
            prune.remove(module, param_name)
    
    return model

def apply_structured_pruning(model, amount=0.2):
    """Apply channel-wise structured pruning."""
    # This is simplified - real structured pruning needs careful implementation
    # to maintain model architecture consistency
    
    modules_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and module.out_channels > 1:
            modules_to_prune.append((module, 'weight'))
    
    # Apply structured pruning (remove entire channels)
    for module, param_name in modules_to_prune:
        if hasattr(module, 'weight'):
            try:
                prune.ln_structured(
                    module, param_name, amount=amount, n=1, dim=0
                )
                prune.remove(module, param_name)
            except Exception as e:
                print(f"Skipping {name}: {e}")
    
    return model

def prune_model(model_path, method='unstructured', amount=0.3):
    """Load model, apply pruning, and return pruned model."""
    
    print(f"Loading model: {model_path}")
    yolo_model = YOLO(model_path)
    model = yolo_model.model
    
    # Get original size
    orig_total, orig_nonzero = count_parameters(model)
    print(f"Original: {orig_total:,} total params, {orig_nonzero:,} non-zero")
    
    # Apply pruning
    if method == 'unstructured':
        model = apply_unstructured_pruning(model, amount)
    elif method == 'structured':
        model = apply_structured_pruning(model, amount)
    else:
        raise ValueError(f"Unknown pruning method: {method}")
    
    # Get pruned size
    pruned_total, pruned_nonzero = count_parameters(model)
    print(f"Pruned: {pruned_total:,} total params, {pruned_nonzero:,} non-zero")
    
    compression_ratio = pruned_nonzero / orig_nonzero if orig_nonzero > 0 else 1.0
    print(f"Compression ratio: {compression_ratio:.3f}")
    
    return yolo_model, {
        'original_params': orig_total,
        'original_nonzero': orig_nonzero,
        'pruned_params': pruned_total,
        'pruned_nonzero': pruned_nonzero,
        'compression_ratio': compression_ratio,
        'method': method,
        'amount': amount
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Input model path')
    parser.add_argument('--output', required=True, help='Output pruned model path')
    parser.add_argument('--method', choices=['unstructured', 'structured'], 
                       default='unstructured')
    parser.add_argument('--amount', type=float, default=0.3, 
                       help='Fraction of parameters to prune')
    parser.add_argument('--data', help='Dataset YAML for validation')
    args = parser.parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prune model
    pruned_model, stats = prune_model(args.model, args.method, args.amount)
    
    # Save pruned model
    pruned_model.save(output_path)
    print(f"Saved pruned model to: {output_path}")
    
    # Optional validation
    if args.data:
        print("\\nValidating pruned model...")
        try:
            metrics = pruned_model.val(data=args.data, save=False, plots=False)
            stats['validation'] = metrics.results_dict
            
            map50 = metrics.results_dict.get('metrics/mAP50(B)', 0.0)
            print(f"Pruned model mAP50: {map50:.3f}")
            
        except Exception as e:
            print(f"Validation error: {e}")
            stats['validation_error'] = str(e)
    
    # Save stats
    stats_file = output_path.with_suffix('.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Pruning statistics saved to: {stats_file}")
    
    print("\\n=== PRUNING SUMMARY ===")
    print(f"Method: {stats['method']}")
    print(f"Amount: {stats['amount']}")
    print(f"Parameters: {stats['original_nonzero']:,} -> {stats['pruned_nonzero']:,}")
    print(f"Compression: {stats['compression_ratio']:.3f}")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Test script to validate all models
Tests model creation, forward pass, and VRAM usage estimation
"""

import torch
import sys
from pathlib import Path

# Add src to path to access models module
src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from models import (
    get_model_by_name,
    get_available_models,
    count_parameters,
    format_bytes
)


def estimate_vram(model, batch_size=1, input_shape=(1, 64, 64, 64)):
    """Estimate VRAM usage for a model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cpu':
        print("  ⚠️  GPU not available, using CPU")
        return None
    
    model = model.to(device)
    model.eval()
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, *input_shape, device=device)
    
    # Forward pass
    try:
        with torch.no_grad():
            output = model(dummy_input)
        
        # Get memory usage
        vram_used = torch.cuda.max_memory_allocated(device)
        
        # Cleanup
        del dummy_input, output
        torch.cuda.empty_cache()
        
        return vram_used
    except RuntimeError as e:
        print(f"  ❌ Error: {e}")
        return None
    finally:
        model = model.cpu()
        torch.cuda.empty_cache()


def test_model(model_name, input_shape=(1, 64, 64, 64), batch_size=1):
    """Test a single model"""
    print(f"\n{'='*80}")
    print(f"Testing: {model_name}")
    print(f"{'='*80}")
    
    try:
        # Create model
        print("  Creating model...")
        model = get_model_by_name(model_name, in_channels=1, num_classes=2)
        
        # Count parameters
        num_params = count_parameters(model)
        print(f"  ✓ Model created successfully")
        print(f"  Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
        
        # Test forward pass (CPU)
        print(f"  Testing forward pass (CPU)...")
        dummy_input = torch.randn(batch_size, *input_shape)
        model.eval()
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"  ✓ Forward pass successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Estimate VRAM (GPU)
        if torch.cuda.is_available():
            print(f"  Estimating VRAM usage (GPU)...")
            vram = estimate_vram(model, batch_size=batch_size, input_shape=input_shape)
            if vram is not None:
                print(f"  ✓ VRAM usage: {format_bytes(vram)}")
        
        return True, num_params, vram if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def main():
    """Main test function"""
    print("="*80)
    print("MODEL TESTING SUITE")
    print("="*80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {format_bytes(torch.cuda.get_device_properties(0).total_memory)}")
    
    # Get all available models
    all_models = get_available_models()
    print(f"\nTotal models to test: {len(all_models)}")
    
    # Test configuration
    input_shape = (1, 64, 64, 64)  # Small volume for testing
    batch_size = 1
    
    print(f"Test configuration:")
    print(f"  Input shape: {input_shape}")
    print(f"  Batch size: {batch_size}")
    
    # Test all models
    results = []
    successful = 0
    failed = 0
    
    for model_name in all_models:
        success, params, vram = test_model(model_name, input_shape=input_shape, batch_size=batch_size)
        
        results.append({
            'name': model_name,
            'success': success,
            'params': params,
            'vram': vram
        })
        
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total tested: {len(all_models)}")
    print(f"Successful: {successful} ✓")
    print(f"Failed: {failed} ❌")
    
    # Sort by parameters
    results.sort(key=lambda x: x['params'] if x['params'] else 0)
    
    print("\n" + "="*80)
    print("MODELS BY SIZE")
    print("="*80)
    print(f"{'Model':<30} {'Parameters':<20} {'VRAM (64³)':<15} {'Status'}")
    print("-"*80)
    
    for r in results:
        params_str = f"{r['params']/1e6:.2f}M" if r['params'] else "N/A"
        vram_str = format_bytes(r['vram']) if r['vram'] else "N/A"
        status = "✓" if r['success'] else "❌"
        
        print(f"{r['name']:<30} {params_str:<20} {vram_str:<15} {status}")
    
    # Model families
    print("\n" + "="*80)
    print("MODELS BY FAMILY")
    print("="*80)
    
    families = {}
    for r in results:
        if r['success']:
            # Extract family name from model name
            if '-' in r['name']:
                family = r['name'].rsplit('-', 1)[0]
            elif '3D' in r['name']:
                family = r['name'].split('3D')[0] + '3D'
            else:
                family = r['name']
            
            if family not in families:
                families[family] = []
            families[family].append(r['name'])
    
    for family, models in sorted(families.items()):
        print(f"\n{family}:")
        for model in models:
            print(f"  - {model}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

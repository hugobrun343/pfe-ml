#!/usr/bin/env python3
"""
Reset GPU after CUDA errors
Utility script to clear CUDA cache and reset GPU state
"""

import torch
import gc


def reset_gpu():
    """Reset GPU state and clear CUDA cache"""
    print("Resetting GPU...")
    
    try:
        # Clear all CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            
            # Clear Python garbage
            gc.collect()
            
            # Try a simple operation to test GPU
            test = torch.randn(1, 1, device='cuda')
            del test
            torch.cuda.empty_cache()
            
            print("✓ GPU reset successful")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  Total memory: {total_memory:.2f} GB")
            
            return True
        else:
            print("✗ CUDA not available")
            return False
            
    except Exception as e:
        print(f"✗ GPU reset failed: {e}")
        print("  You may need to restart the Python process or reboot")
        return False


def main():
    """Main entry point"""
    success = reset_gpu()
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

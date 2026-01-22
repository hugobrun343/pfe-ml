"""
VRAM Tester - Tests configurations and measures VRAM usage
Simulates training with synthetic data to measure memory requirements
"""

import os
import torch
import torch.nn as nn
import time
from typing import Dict, Tuple, Optional
from contextlib import contextmanager

# Enable CUDA error detection for debugging (shows exact error location)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'  # Device-side assertions for better error info

from models import get_model_by_name
from utils import clear_gpu_memory


class VRAMTester:
    """
    Tests a specific configuration and measures VRAM usage
    Uses synthetic data to simulate training without real data loading
    """
    
    def __init__(
        self,
        device: str = "cuda",
        num_warmup_iterations: int = 3,
        num_test_iterations: int = 5,
        clear_cache_between_tests: bool = True
    ):
        """
        Initialize VRAM tester
        
        Args:
            device: Device to use (cuda/cpu)
            num_warmup_iterations: Number of warmup iterations before measuring
            num_test_iterations: Number of iterations for VRAM measurement
            clear_cache_between_tests: Whether to clear cache before testing
        """
        self.device = device
        self.num_warmup_iterations = num_warmup_iterations
        self.num_test_iterations = num_test_iterations
        self.clear_cache_between_tests = clear_cache_between_tests
        
        if not torch.cuda.is_available() and device == "cuda":
            raise RuntimeError("CUDA not available but device set to 'cuda'")
    
    def test_configuration(
        self,
        spatial_res: int,
        depth: int,
        batch_size: int,
        model_name: str,
        in_channels: int = 3,
        num_classes: int = 1
    ) -> Dict:
        """
        Test a specific configuration and measure VRAM usage
        
        Args:
            spatial_res: Spatial resolution (H = W)
            depth: Depth dimension (D)
            batch_size: Batch size
            model_name: Name of the model to test
            in_channels: Number of input channels
            num_classes: Number of output classes
            
        Returns:
            Dictionary with test results
        """
        start_time = time.time()
        
        # Clear GPU memory before test
        if self.clear_cache_between_tests:
            clear_gpu_memory()
        
        try:
            # Create model (always 3 channels input)
            model = get_model_by_name(model_name, in_channels=3, num_classes=num_classes)
            model = model.to(self.device)
            model.train()
            
            # Create optimizer and loss
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            # Measure initial memory
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                initial_memory = torch.cuda.memory_allocated()
            
            # Warmup iterations
            for _ in range(self.num_warmup_iterations):
                self._training_step(
                    model, optimizer, criterion,
                    batch_size, in_channels, depth, spatial_res, num_classes
                )
            
            # Test iterations with memory tracking
            if self.device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            
            vram_measurements = []
            
            for _ in range(self.num_test_iterations):
                self._training_step(
                    model, optimizer, criterion,
                    batch_size, in_channels, depth, spatial_res, num_classes
                )
                
                if self.device == "cuda":
                    torch.cuda.synchronize()
                    vram_measurements.append(torch.cuda.memory_allocated())
            
            # Get peak memory
            if self.device == "cuda":
                peak_memory = torch.cuda.max_memory_allocated()
                avg_memory = sum(vram_measurements) / len(vram_measurements)
            else:
                peak_memory = 0
                avg_memory = 0
            
            # Clean up (clear_gpu_memory is safe and won't crash)
            try:
                del model
                del optimizer
                del criterion
            except:
                pass
            clear_gpu_memory()
            
            end_time = time.time()
            duration = end_time - start_time
            
            return {
                "success": True,
                "vram_used_bytes": int(avg_memory),
                "vram_peak_bytes": int(peak_memory),
                "duration_seconds": duration,
                "error_message": None
            }
            
        except RuntimeError as e:
            # Handle OOM, CUDA errors, and invalid configs
            error_msg = str(e)
            
            # Determine error type (check more thoroughly)
            error_lower = error_msg.lower()
            is_oom = (
                "out of memory" in error_lower or
                "cuda out of memory" in error_lower or
                "allocated" in error_lower and "memory" in error_lower
            )
            is_cuda_error = (
                "cuda error" in error_lower or
                "illegal memory access" in error_lower or
                "device-side assert" in error_lower
            )
            is_invalid_config = (
                "kernel size can't be greater" in error_lower or
                "input size" in error_lower or
                "tensor is too large" in error_lower
            )
            
            # Safe cleanup (clear_gpu_memory is now safe and won't crash)
            clear_gpu_memory()
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Determine error message and status
            if is_invalid_config:
                # Invalid configuration (too small input) - mark as SKIPPED
                final_error = "SKIPPED_INVALID_CONFIG"
                status = "skipped"
            elif is_oom:
                # Mark as OOM (even if it appeared as CUDA error)
                final_error = "OOM"
                status = "failed"
            elif is_cuda_error:
                # CUDA error that's not OOM - might be GPU dead
                final_error = "CUDA_ERROR"
                status = "failed"
            else:
                final_error = error_msg[:200]
                status = "failed"
            
            return {
                "success": False,
                "status": status,  # "failed" or "skipped"
                "vram_used_bytes": None,
                "vram_peak_bytes": None,
                "duration_seconds": duration,
                "error_message": final_error
            }
        
        except Exception as e:
            # Handle all other errors (never crash)
            error_msg = str(e)
            error_lower = error_msg.lower()
            is_invalid_config = (
                "kernel size can't be greater" in error_lower or
                "input size" in error_lower or
                "tensor is too large" in error_lower
            )
            
            # Detect OOM in any form
            is_oom_like = (
                "out of memory" in error_lower or
                "cuda error" in error_lower or
                "illegal memory access" in error_lower
            )
            
            # Minimal cleanup only (no GPU calls if it might be dead)
            try:
                import gc
                gc.collect()
            except:
                pass
            
            # DON'T call clear_gpu_memory() after errors - GPU might be dead
            
            end_time = time.time()
            duration = end_time - start_time
            
            if is_invalid_config:
                final_error = "SKIPPED_INVALID_CONFIG"
                status = "skipped"
            elif is_oom_like:
                final_error = "OOM"
                status = "failed"
            else:
                final_error = f"ERROR: {error_msg[:180]}"
                status = "failed"
            
            return {
                "success": False,
                "status": status,
                "vram_used_bytes": None,
                "vram_peak_bytes": None,
                "duration_seconds": duration,
                "error_message": final_error
            }
    
    def _training_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        batch_size: int,
        in_channels: int,
        depth: int,
        spatial_res: int,
        num_classes: int
    ) -> None:
        """
        Perform a single training step with synthetic data
        
        Args:
            model: Model to train
            optimizer: Optimizer
            criterion: Loss criterion
            batch_size: Batch size
            in_channels: Number of input channels
            depth: Depth dimension
            spatial_res: Spatial resolution
            num_classes: Number of classes
        """
        # Generate synthetic input data (always 3 channels)
        x = torch.randn(
            batch_size, 3, depth, spatial_res, spatial_res,
            device=self.device,
            requires_grad=False
        )
        
        # Generate synthetic labels
        y = torch.randint(
            0, num_classes, (batch_size,),
            device=self.device
        )
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(x)
        
        # Compute loss
        loss = criterion(outputs, y)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        # Clean up tensors
        del x, y, outputs, loss
    
    def estimate_vram_utilization(
        self,
        vram_used_bytes: int,
        total_vram_bytes: int
    ) -> Dict:
        """
        Calculate VRAM utilization statistics
        
        Args:
            vram_used_bytes: VRAM used by the test
            total_vram_bytes: Total available VRAM
            
        Returns:
            Dictionary with utilization statistics
        """
        utilization_pct = (vram_used_bytes / total_vram_bytes) * 100
        remaining_bytes = total_vram_bytes - vram_used_bytes
        remaining_gb = remaining_bytes / (1024**3)
        
        return {
            "utilization_percent": utilization_pct,
            "remaining_bytes": remaining_bytes,
            "remaining_gb": remaining_gb,
            "fits_in_memory": vram_used_bytes <= total_vram_bytes
        }


def quick_test_example():
    """Quick test function to verify the tester works"""
    print("Running quick VRAM test...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    tester = VRAMTester()
    
    # Test a small configuration
    result = tester.test_configuration(
        spatial_res=64,
        depth=16,
        batch_size=2,
        model_name="ResNet3D-18"
    )
    
    print(f"Test result: {result}")
    
    if result["success"]:
        vram_gb = result["vram_peak_bytes"] / (1024**3)
        print(f"VRAM used: {vram_gb:.2f} GB")
        print(f"Duration: {result['duration_seconds']:.2f} seconds")
    else:
        print(f"Test failed: {result['error_message']}")


if __name__ == "__main__":
    quick_test_example()

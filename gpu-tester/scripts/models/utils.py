"""
Utility functions for model management
Includes parameter counting, VRAM optimization, and mixed precision support
"""

import torch
import torch.nn as nn
from typing import List


def count_parameters(model: nn.Module) -> int:
    """
    Count total number of trainable parameters
    
    Args:
        model: PyTorch model
        
    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model: nn.Module, model_name: str) -> dict:
    """
    Get detailed model information
    
    Args:
        model: PyTorch model
        model_name: Name of the model
        
    Returns:
        Dictionary with model information
    """
    total_params = count_parameters(model)
    
    return {
        "model_name": model_name,
        "total_parameters": total_params,
        "total_parameters_millions": total_params / 1e6,
    }


def apply_gradient_checkpointing(model: nn.Module) -> nn.Module:
    """
    Apply gradient checkpointing to reduce VRAM usage
    Trades compute for memory by recomputing activations during backward pass
    Can save 30-50% VRAM at the cost of ~20% slower training
    
    Args:
        model: PyTorch model
        
    Returns:
        Model with gradient checkpointing enabled
    """
    if hasattr(model, 'blocks'):
        # For transformer-based models (ViT3D)
        for block in model.blocks:
            if hasattr(block, 'checkpoint'):
                block.checkpoint = True
            else:
                # Wrap block forward in checkpoint
                original_forward = block.forward
                def make_checkpointed_forward(orig_fn):
                    def checkpointed_forward(*args, **kwargs):
                        return torch.utils.checkpoint.checkpoint(orig_fn, *args, **kwargs, use_reentrant=False)
                    return checkpointed_forward
                block.forward = make_checkpointed_forward(original_forward)
    
    elif hasattr(model, 'stages'):
        # For ConvNeXt3D
        for stage in model.stages:
            for block in stage:
                original_forward = block.forward
                def make_checkpointed_forward(orig_fn):
                    def checkpointed_forward(*args, **kwargs):
                        return torch.utils.checkpoint.checkpoint(orig_fn, *args, **kwargs, use_reentrant=False)
                    return checkpointed_forward
                block.forward = make_checkpointed_forward(original_forward)
    
    elif hasattr(model, 'layer1'):
        # For ResNet-based models (ResNet3D, SE-ResNet3D)
        for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
            for block in layer:
                original_forward = block.forward
                def make_checkpointed_forward(orig_fn):
                    def checkpointed_forward(*args, **kwargs):
                        return torch.utils.checkpoint.checkpoint(orig_fn, *args, **kwargs, use_reentrant=False)
                    return checkpointed_forward
                block.forward = make_checkpointed_forward(original_forward)
    
    return model


class MixedPrecisionWrapper:
    """
    Wrapper for mixed precision training (FP16/BF16)
    Reduces VRAM usage by ~50% while maintaining accuracy
    
    Usage:
        model = get_model_by_name("ResNet3D-34")
        mp_model = MixedPrecisionWrapper(model, dtype="float16")
        
        # Forward pass
        output = mp_model(input)
        
        # Backward pass
        loss = criterion(output, target)
        mp_model.backward(loss, optimizer)
    """
    
    def __init__(
        self,
        model: nn.Module,
        dtype: str = "float16",
        device: str = "cuda"
    ):
        """
        Args:
            model: PyTorch model
            dtype: Precision type ("float16" or "bfloat16")
            device: Device to use
        """
        self.model = model.to(device)
        self.device = device
        
        if dtype == "float16":
            self.dtype = torch.float16
        elif dtype == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            raise ValueError(f"Unknown dtype: {dtype}. Use 'float16' or 'bfloat16'")
        
        # Use PyTorch's automatic mixed precision
        self.use_amp = True
        self.scaler = torch.cuda.amp.GradScaler() if dtype == "float16" else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with mixed precision"""
        with torch.cuda.amp.autocast(dtype=self.dtype):
            return self.model(x)
    
    def backward(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        """Backward pass with gradient scaling (for FP16)"""
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Allow calling as a function"""
        return self.forward(x)


def format_bytes(bytes_val: int) -> str:
    """
    Format bytes to human readable format
    
    Args:
        bytes_val: Number of bytes
        
    Returns:
        Human readable string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def estimate_vram(
    model: nn.Module,
    input_shape: tuple,
    batch_size: int = 1,
    device: str = "cuda"
) -> int:
    """
    Estimate VRAM usage for a model with given input shape
    
    Args:
        model: PyTorch model
        input_shape: Input shape (C, D, H, W)
        batch_size: Batch size
        device: Device to use
        
    Returns:
        Estimated VRAM usage in bytes
    """
    device = torch.device(device)
    
    if device.type == 'cpu':
        return 0
    
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
    except RuntimeError:
        return -1
    finally:
        model = model.cpu()
        torch.cuda.empty_cache()

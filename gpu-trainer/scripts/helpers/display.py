"""Display utilities for training results."""

import torch
from pathlib import Path
from typing import Dict, Any, Optional


def print_final_results(summary: Dict[str, Any], results_dir: Path) -> None:
    """
    Display and save final validation results (confusion matrix and F1 scores).
    
    Args:
        summary: Training summary dictionary with 'final_val_metrics'
        results_dir: Directory to save confusion matrix file
    """
    if not summary.get('final_val_metrics'):
        return
    
    final_metrics = summary['final_val_metrics']
    tp = final_metrics.get('tp', 0)
    fp = final_metrics.get('fp', 0)
    tn = final_metrics.get('tn', 0)
    fn = final_metrics.get('fn', 0)
    f1_class_0 = final_metrics.get('f1_class_0', 0.0)
    f1_class_1 = final_metrics.get('f1_class_1', 0.0)
    
    # Print to console
    print("\n" + "="*50)
    print("FINAL VALIDATION RESULTS")
    print("="*50)
    print(f"\nConfusion Matrix:")
    print(f"  {'':<15} {'Predicted 0':<15} {'Predicted 1':<15}")
    print(f"  {'Actual 0':<15} {tn:<15} {fp:<15}")
    print(f"  {'Actual 1':<15} {fn:<15} {tp:<15}")
    print(f"\nF1 Scores:")
    print(f"  Class 0 (Negative): {f1_class_0:.4f}")
    print(f"  Class 1 (Positive): {f1_class_1:.4f}")
    print("="*50)
    
    # Save to file
    confusion_matrix_path = results_dir / 'confusion_matrix.txt'
    with open(confusion_matrix_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write("FINAL VALIDATION RESULTS\n")
        f.write("="*50 + "\n")
        f.write(f"\nConfusion Matrix:\n")
        f.write(f"  {'':<15} {'Predicted 0':<15} {'Predicted 1':<15}\n")
        f.write(f"  {'Actual 0':<15} {tn:<15} {fp:<15}\n")
        f.write(f"  {'Actual 1':<15} {fn:<15} {tp:<15}\n")
        f.write(f"\nF1 Scores:\n")
        f.write(f"  Class 0 (Negative): {f1_class_0:.4f}\n")
        f.write(f"  Class 1 (Positive): {f1_class_1:.4f}\n")
        f.write("="*50 + "\n")
    
    print(f"\nConfusion matrix saved to: {confusion_matrix_path}")


def print_training_info(device: torch.device, batch_size: int, epochs: int, lr: float) -> None:
    """
    Print training configuration information.
    
    Args:
        device: Device being used
        batch_size: Batch size
        epochs: Number of epochs
        lr: Learning rate
    """
    import torch
    
    print(f"\nResNet3D-50 Training")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    print(f"  Batch size: {batch_size}, Epochs: {epochs}, LR: {lr}")

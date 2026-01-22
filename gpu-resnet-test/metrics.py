"""Metrics for model evaluation"""

import torch
from typing import Dict


def compute_confusion_matrix(predicted: torch.Tensor, target: torch.Tensor) -> Dict[str, int]:
    """
    Compute confusion matrix components for binary classification
    
    Args:
        predicted: Predicted binary labels (0 or 1)
        target: True binary labels (0 or 1)
        
    Returns:
        Dictionary with TP, FP, TN, FN
    """
    predicted_flat = predicted.flatten().int()
    target_flat = target.flatten().int()
    
    # True Positives: predicted=1 AND target=1
    tp = ((predicted_flat == 1) & (target_flat == 1)).sum().item()
    
    # False Positives: predicted=1 AND target=0
    fp = ((predicted_flat == 1) & (target_flat == 0)).sum().item()
    
    # True Negatives: predicted=0 AND target=0
    tn = ((predicted_flat == 0) & (target_flat == 0)).sum().item()
    
    # False Negatives: predicted=0 AND target=1
    fn = ((predicted_flat == 0) & (target_flat == 1)).sum().item()
    
    return {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn
    }


def compute_classification_metrics(predicted: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> Dict[str, float]:
    """
    Compute all classification metrics for binary classification
    
    Args:
        predicted: Predicted binary labels (0 or 1)
        target: True binary labels (0 or 1)
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dictionary with TP, FP, TN, FN, accuracy, precision, recall, f1_class_0, f1_class_1
    """
    cm = compute_confusion_matrix(predicted, target)
    tp = cm['tp']
    fp = cm['fp']
    tn = cm['tn']
    fn = cm['fn']
    
    # Accuracy: (TP + TN) / (TP + FP + TN + FN)
    accuracy = (tp + tn) / (tp + fp + tn + fn + smooth)
    
    # Precision: TP / (TP + FP)
    precision = tp / (tp + fp + smooth)
    
    # Recall (Sensitivity): TP / (TP + FN)
    recall = tp / (tp + fn + smooth)
    
    # F1 Score for class 1 (positive class)
    f1_class_1 = 2 * (precision * recall) / (precision + recall + smooth)
    
    # F1 Score for class 0 (negative class)
    # For class 0: TN are "true positives", FN are "false positives", FP are "false negatives"
    precision_class_0 = tn / (tn + fn + smooth)
    recall_class_0 = tn / (tn + fp + smooth)
    f1_class_0 = 2 * (precision_class_0 * recall_class_0) / (precision_class_0 + recall_class_0 + smooth)
    
    return {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_class_0': f1_class_0,
        'f1_class_1': f1_class_1
    }

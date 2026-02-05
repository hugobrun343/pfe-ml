"""Training orchestrator with early stopping and checkpointing"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Any
import time
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
scripts_dir = project_root / "scripts"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(scripts_dir / "core"))
sys.path.insert(0, str(scripts_dir / "helpers"))

from scripts.core.metrics import compute_classification_metrics
from scripts.helpers.results import ResultsTracker


class Trainer:
    """Manages training loop with early stopping and checkpointing"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        save_dir: Path,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 0.001,
        results_tracker: Optional[ResultsTracker] = None,
        val_sampler=None,
        **kwargs
    ):
        """
        Initialize trainer
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            criterion: Loss function
            optimizer: Optimizer
            device: Device to use (cuda/cpu)
            save_dir: Directory to save checkpoints
            early_stopping_patience: Number of epochs without improvement before stopping
            early_stopping_min_delta: Minimum F1 improvement to count as progress
            results_tracker: Results tracker for JSON export (optional)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_sampler = val_sampler  # Sampler to track indices for validation
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.results_tracker = results_tracker
        
        self.best_val_f1_class_1 = kwargs.get('best_val_f1_class_1', 0.0)
        self.epochs_without_improvement = 0
        self.wandb_run_id = kwargs.get('wandb_run_id', None)
        
        # Store model hyperparameters for checkpoint saving
        self.model_in_channels = model.conv1.in_channels
        self.model_num_classes = model.fc.out_features
        self.model_initial_channels = model.conv1.out_channels
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train model for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with 'loss', 'f1_class_0', 'f1_class_1' metrics
        """
        self.model.train()
        running_loss = 0.0
        
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        try:
            for batch in pbar:
                # Unpack batch: (blocks, labels)
                blocks, labels = batch
                blocks = blocks.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                
                self.optimizer.zero_grad()
                outputs = self.model(blocks)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                
                # Collect predictions for F1 calculation
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                all_predictions.extend(predicted.cpu().squeeze().numpy())
                all_labels.extend(labels.cpu().squeeze().numpy())
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}'
                })
        except Exception as e:
            print(f"\nERROR during training epoch {epoch}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
        
        # Compute F1 scores per class
        predicted_tensor = torch.tensor(all_predictions, dtype=torch.float32)
        labels_tensor = torch.tensor(all_labels, dtype=torch.float32)
        metrics = compute_classification_metrics(predicted_tensor, labels_tensor)
        
        return {
            'loss': running_loss / len(self.train_loader),
            'f1_class_0': metrics.get('f1_class_0', 0.0),
            'f1_class_1': metrics.get('f1_class_1', 0.0)
        }
    
    def _validate(self) -> Dict[str, Any]:
        """
        Validate model on validation set
        
        Returns:
            Dictionary with 'loss', 'f1_class_0', 'f1_class_1', 'tp', 'fp', 'tn', 'fn', 'accuracy', 'precision', 'recall', 'predictions', 'labels', 'probs', 'indices', 'stack_ids', 'patch_positions'
        """
        self.model.eval()
        running_loss = 0.0
        
        all_predictions = []
        all_labels = []
        all_probs = []
        all_indices = []
        
        with torch.no_grad():
            batch_idx = 0
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Unpack batch: (blocks, labels) - no indices returned anymore
                blocks, labels = batch
                blocks = blocks.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                
                outputs = self.model(blocks)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                
                all_predictions.extend(predicted.cpu().squeeze().numpy())
                all_labels.extend(labels.cpu().squeeze().numpy())
                all_probs.extend(probs.cpu().squeeze().numpy())
                
                # Get indices from sampler if available (for validation)
                # With shuffle=False, indices are sequential, so we can calculate them
                if self.val_sampler is not None:
                    batch_start = batch_idx * self.val_loader.batch_size
                    batch_end = batch_start + len(blocks)
                    batch_indices = self.val_sampler.indices[batch_start:batch_end]
                    all_indices.extend(batch_indices)
                elif not self.val_loader.sampler or (hasattr(self.val_loader.sampler, 'shuffle') and not self.val_loader.sampler.shuffle):
                    # No sampler but shuffle=False, indices are sequential
                    batch_start = batch_idx * self.val_loader.batch_size
                    batch_end = batch_start + len(blocks)
                    all_indices.extend(range(batch_start, batch_end))
                
                batch_idx += 1
        
        # Compute metrics on all validation samples
        predicted_tensor = torch.tensor(all_predictions, dtype=torch.float32)
        labels_tensor = torch.tensor(all_labels, dtype=torch.float32)
        metrics = compute_classification_metrics(predicted_tensor, labels_tensor)
        
        # Get stack_ids and patch positions from dataset
        dataset = self.val_loader.dataset
        stack_ids = []
        patch_positions = []
        for idx in all_indices:
            metadata = dataset.get_metadata(idx)
            stack_ids.append(metadata['stack_id'])
            patch_positions.append({
                'position_h': metadata['position_h'],
                'position_w': metadata['position_w'],
                'patch_index': metadata.get('patch_index', None)
            })
        
        return {
            'loss': running_loss / len(self.val_loader),
            'f1_class_0': metrics.get('f1_class_0', 0.0),
            'f1_class_1': metrics.get('f1_class_1', 0.0),
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'tn': metrics['tn'],
            'fn': metrics['fn'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'predictions': all_predictions,
            'labels': all_labels,
            'probs': all_probs,
            'indices': all_indices,
            'stack_ids': stack_ids,
            'patch_positions': patch_positions
        }
    
    def train(self, num_epochs: int, start_epoch: int = 1) -> Dict[str, Any]:
        """
        Run training loop
        
        Args:
            num_epochs: Number of epochs to train
            start_epoch: Epoch to start from (for resume)
            
        Returns:
            Dictionary with 'best_val_f1_class_1', 'total_epochs', 'early_stopped', 'final_val_metrics'
        """
        print(f"\nStarting training (epochs {start_epoch}-{num_epochs})...")
        if self.early_stopping_patience is not None:
            print(f"Early stopping: patience={self.early_stopping_patience}, min_delta={self.early_stopping_min_delta}")
        else:
            print("Early stopping: disabled (train all epochs)")
        
        final_val_metrics = None  # Store final validation metrics
        
        for epoch in range(start_epoch, num_epochs + 1):
            start_time = time.time()
            
            train_metrics = self._train_epoch(epoch)
            
            val_metrics = None
            if self.val_loader:
                val_metrics = self._validate()
                final_val_metrics = val_metrics  # Update final metrics
            
            epoch_time = time.time() - start_time
            
            self._print_metrics(epoch, num_epochs, epoch_time, train_metrics, val_metrics)
            
            # Log to wandb
            from wandb_utils import log_epoch_metrics
            log_epoch_metrics(epoch, epoch_time, train_metrics, val_metrics)
            
            if val_metrics:
                self._print_predictions(val_metrics)
            
            if self.results_tracker:
                self.results_tracker.add_epoch(epoch, epoch_time, train_metrics, val_metrics)
            
            should_stop = self._save_checkpoint_and_check_early_stopping(epoch, train_metrics, val_metrics)
            if should_stop:
                break
        
        return {
            'best_val_f1_class_1': self.best_val_f1_class_1,
            'total_epochs': epoch,
            'early_stopped': (self.early_stopping_patience is not None and 
                            self.epochs_without_improvement >= self.early_stopping_patience),
            'final_val_metrics': final_val_metrics
        }
    
    def _print_metrics(self, epoch: int, total_epochs: int, epoch_time: float,
                      train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, Any]]):
        """Print epoch metrics to console"""
        print(f"\nEpoch {epoch}/{total_epochs} ({epoch_time:.1f}s)")
        train_f1_class_0 = train_metrics.get('f1_class_0', 0.0)
        train_f1_class_1 = train_metrics.get('f1_class_1', 0.0)
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, F1 Class 0: {train_f1_class_0:.4f}, F1 Class 1: {train_f1_class_1:.4f}")
        
        if val_metrics:
            f1_class_0 = val_metrics.get('f1_class_0', 0.0)
            f1_class_1 = val_metrics.get('f1_class_1', 0.0)
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, F1 Class 0: {f1_class_0:.4f}, F1 Class 1: {f1_class_1:.4f}")
            print(f"         Metrics: TP={val_metrics['tp']}, FP={val_metrics['fp']}, TN={val_metrics['tn']}, FN={val_metrics['fn']}")
            print(f"         Accuracy: {val_metrics['accuracy']:.4f}, Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
    
    def _print_predictions(self, val_metrics: Dict[str, Any], num_samples: int = 20):
        """Print detailed predictions table for validation samples"""
        predictions = val_metrics['predictions']
        labels = val_metrics['labels']
        probs = val_metrics['probs']
        
        print(f"\n  Detailed predictions (first {num_samples} samples):")
        print("  " + "-" * 80)
        print(f"  {'Sample':<8} {'True Label':<12} {'Predicted':<12} {'Probability':<12} {'Status':<20}")
        print("  " + "-" * 80)
        
        for i in range(min(num_samples, len(predictions))):
            true_label = int(labels[i])
            pred_label = int(predictions[i])
            prob = probs[i]
            status = "✓ Correct" if true_label == pred_label else "✗ Wrong"
            print(f"  {i:<8} {true_label:<12} {pred_label:<12} {prob:.4f}        {status:<20}")
        
        if len(predictions) > num_samples:
            print(f"  ... and {len(predictions) - num_samples} more samples")
        print("  " + "-" * 80)
    
    def _save_checkpoint_and_check_early_stopping(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, Any]]
    ) -> bool:
        """
        Save checkpoint and check for early stopping
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary (optional)
            
        Returns:
            True if training should stop, False otherwise
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_metrics['loss'],
            # Model hyperparameters
            'in_channels': self.model_in_channels,
            'num_classes': self.model_num_classes,
            'initial_channels': self.model_initial_channels,
            # Optimizer hyperparameters (extract from optimizer)
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'weight_decay': self.optimizer.param_groups[0]['weight_decay'],
        }
        
        # Save wandb run ID if available
        if self.wandb_run_id:
            checkpoint['wandb_run_id'] = self.wandb_run_id
        
        if val_metrics:
            checkpoint['val_loss'] = val_metrics['loss']
            checkpoint['val_f1_class_0'] = val_metrics.get('f1_class_0', 0.0)
            checkpoint['val_f1_class_1'] = val_metrics.get('f1_class_1', 0.0)
            
            # Use f1_class_1 for best model tracking (class 1 is typically the class of interest)
            current_f1 = val_metrics.get('f1_class_1', 0.0)
            improvement = current_f1 - self.best_val_f1_class_1
            
            # Use min_delta if set, otherwise accept any improvement (>= 0)
            min_delta = self.early_stopping_min_delta if self.early_stopping_min_delta is not None else 0.0
            if improvement > min_delta:
                self.best_val_f1_class_1 = current_f1
                self.epochs_without_improvement = 0
                torch.save(checkpoint, self.save_dir / 'best_model.pth')
                f1_class_0 = val_metrics.get('f1_class_0', 0.0)
                f1_class_1 = val_metrics.get('f1_class_1', 0.0)
                print(f"  → Best model saved (F1 Class 0: {f1_class_0:.4f}, F1 Class 1: {f1_class_1:.4f}, +{improvement:.4f})")
            else:
                self.epochs_without_improvement += 1
                # Only check early stopping if patience is set (not None/null)
                if self.early_stopping_patience is not None and self.epochs_without_improvement >= self.early_stopping_patience:
                    f1_class_0 = val_metrics.get('f1_class_0', 0.0)
                    f1_class_1 = val_metrics.get('f1_class_1', 0.0)
                    print(f"\n Early stopping (no improvement for {self.epochs_without_improvement} epochs)")
                    print(f"  Best F1 Class 0: {f1_class_0:.4f}, F1 Class 1: {f1_class_1:.4f}")
                    return True
        
        torch.save(checkpoint, self.save_dir / 'latest_model.pth')
        
        if self.results_tracker:
            self.results_tracker.save(
                best_val_f1_class_1=self.best_val_f1_class_1 if self.val_loader else None,
                total_epochs=epoch,
                early_stopped=(self.early_stopping_patience is not None and 
                              self.epochs_without_improvement >= self.early_stopping_patience)
            )
        
        return False

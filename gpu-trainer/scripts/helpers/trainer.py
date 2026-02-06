"""Training orchestrator with early stopping and checkpointing"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
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
    """Training loop with early stopping."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        train_dataset: Dataset,
        val_dataset: Dataset,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        save_dir: Path,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 0.001,
        results_tracker: Optional[ResultsTracker] = None,
        **kwargs
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = train_loader.batch_size
        
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.results_tracker = results_tracker
        
        self.best_val_f1_macro = kwargs.get('best_val_f1_macro', 0.0)
        self.epochs_without_improvement = 0
        self.wandb_run_id = kwargs.get('wandb_run_id', None)
        
        # Model hyperparameters for checkpoint saving
        self.model_in_channels = kwargs.get('in_channels', 3)
        self.model_num_classes = kwargs.get('num_classes', 1)
        self.model_initial_channels = kwargs.get('initial_channels', None)
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        
        # Pre-allocate on GPU to avoid fragmentation
        num_samples = len(self.train_loader.dataset)
        all_preds = torch.zeros(num_samples, device=self.device)
        all_labels = torch.zeros(num_samples, device=self.device)
        running_loss = torch.tensor(0.0, device=self.device)
        sample_idx = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for blocks, labels in pbar:
            bs = blocks.size(0)
            blocks = blocks.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            outputs = self.model(blocks)
            loss = self.criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            self.optimizer.step()
            
            # Write directly to pre-allocated tensors (no fragmentation!)
            with torch.no_grad():
                running_loss += loss.detach()
                preds = (torch.sigmoid(outputs.detach()) > 0.5).float().squeeze()
                all_preds[sample_idx:sample_idx + bs] = preds
                all_labels[sample_idx:sample_idx + bs] = labels
                sample_idx += bs
        
        # Single sync at end of epoch
        torch.cuda.synchronize()
        total_loss = running_loss.item()
        all_preds_cpu = all_preds[:sample_idx].cpu()
        all_labels_cpu = all_labels[:sample_idx].cpu()
        
        # Compute metrics
        metrics = compute_classification_metrics(all_preds_cpu, all_labels_cpu)
        
        f1_class_0 = metrics.get('f1_class_0', 0.0)
        f1_class_1 = metrics.get('f1_class_1', 0.0)
        f1_macro = (f1_class_0 + f1_class_1) / 2
        
        return {
            'loss': total_loss / len(self.train_loader),
            'f1_class_0': f1_class_0,
            'f1_class_1': f1_class_1,
            'f1_macro': f1_macro
        }
    
    def _val_epoch(self, epoch: int) -> Dict[str, Any]:
        """Run one validation epoch."""
        self.model.eval()
        
        # Pre-allocate on GPU to avoid fragmentation
        num_samples = len(self.val_loader.dataset)
        all_preds = torch.zeros(num_samples, device=self.device)
        all_labels = torch.zeros(num_samples, device=self.device)
        all_probs = torch.zeros(num_samples, device=self.device)
        running_loss = torch.tensor(0.0, device=self.device)
        all_indices = []
        sample_idx = 0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        
        with torch.no_grad():
            for batch_idx, (blocks, labels) in enumerate(pbar):
                bs = blocks.size(0)
                blocks = blocks.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                outputs = self.model(blocks)
                loss = self.criterion(outputs, labels.unsqueeze(1))
                probs = torch.sigmoid(outputs).squeeze()
                preds = (probs > 0.5).float()
                
                # Write directly to pre-allocated tensors (no fragmentation!)
                running_loss += loss.detach()
                all_preds[sample_idx:sample_idx + bs] = preds
                all_labels[sample_idx:sample_idx + bs] = labels
                all_probs[sample_idx:sample_idx + bs] = probs
                
                # Track indices (CPU only, no tensor ops)
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + bs
                all_indices.extend(range(start_idx, end_idx))
                sample_idx += bs
        
        # Single sync at end of epoch
        torch.cuda.synchronize()
        total_loss = running_loss.item()
        all_preds_cpu = all_preds[:sample_idx].cpu()
        all_labels_cpu = all_labels[:sample_idx].cpu()
        all_probs_cpu = all_probs[:sample_idx].cpu()
        
        # Compute metrics
        metrics = compute_classification_metrics(all_preds_cpu, all_labels_cpu)
        
        f1_class_0 = metrics.get('f1_class_0', 0.0)
        f1_class_1 = metrics.get('f1_class_1', 0.0)
        f1_macro = (f1_class_0 + f1_class_1) / 2
        
        # Get metadata for first 100 samples
        stack_ids = []
        patch_positions = []
        for idx in all_indices[:100]:
            if idx < len(self.val_dataset):
                metadata = self.val_dataset.get_metadata(idx)
                stack_ids.append(metadata['stack_id'])
                patch_positions.append({
                    'position_h': metadata['position_h'],
                    'position_w': metadata['position_w'],
                    'patch_index': metadata.get('patch_index', None)
                })
        
        return {
            'loss': total_loss / len(self.val_loader),
            'f1_class_0': f1_class_0,
            'f1_class_1': f1_class_1,
            'f1_macro': f1_macro,
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'tn': metrics['tn'],
            'fn': metrics['fn'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'predictions': all_preds_cpu.tolist(),
            'labels': all_labels_cpu.tolist(),
            'probs': all_probs_cpu.tolist(),
            'indices': all_indices,
            'stack_ids': stack_ids,
            'patch_positions': patch_positions
        }
    
    def train(self, num_epochs: int, start_epoch: int = 1) -> Dict[str, Any]:
        """Run training loop."""
        print(f"\nStarting training (epochs {start_epoch}-{num_epochs})...")
        if self.early_stopping_patience:
            print(f"Early stopping: patience={self.early_stopping_patience}, min_delta={self.early_stopping_min_delta}")
        
        final_val_metrics = None
        
        for epoch in range(start_epoch, num_epochs + 1):
            start_time = time.time()
            
            # Train + Val
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._val_epoch(epoch)
            final_val_metrics = val_metrics
            
            epoch_time = time.time() - start_time
            
            # Print results
            self._print_metrics(epoch, num_epochs, epoch_time, train_metrics, val_metrics)
            
            # Log to wandb
            from wandb_utils import log_epoch_metrics
            log_epoch_metrics(epoch, epoch_time, train_metrics, val_metrics)
            
            self._print_predictions(val_metrics)
            
            if self.results_tracker:
                self.results_tracker.add_epoch(epoch, epoch_time, train_metrics, val_metrics)
            
            # Checkpoint + early stopping
            should_stop = self._save_checkpoint_and_check_early_stopping(epoch, train_metrics, val_metrics)
            if should_stop:
                break
        
        return {
            'best_val_f1_macro': self.best_val_f1_macro,
            'total_epochs': epoch,
            'early_stopped': (self.early_stopping_patience is not None and 
                            self.epochs_without_improvement >= self.early_stopping_patience),
            'final_val_metrics': final_val_metrics
        }
    
    def _print_metrics(self, epoch: int, total_epochs: int, epoch_time: float,
                      train_metrics: Dict[str, float], val_metrics: Dict[str, Any]):
        """Print epoch metrics."""
        print(f"\nEpoch {epoch}/{total_epochs} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, F1 Macro: {train_metrics['f1_macro']:.4f} (C0: {train_metrics['f1_class_0']:.4f}, C1: {train_metrics['f1_class_1']:.4f})")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, F1 Macro: {val_metrics['f1_macro']:.4f} (C0: {val_metrics['f1_class_0']:.4f}, C1: {val_metrics['f1_class_1']:.4f})")
        print(f"         Metrics: TP={val_metrics['tp']}, FP={val_metrics['fp']}, TN={val_metrics['tn']}, FN={val_metrics['fn']}")
        print(f"         Accuracy: {val_metrics['accuracy']:.4f}, Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
    
    def _print_predictions(self, val_metrics: Dict[str, Any], num_samples: int = 20):
        """Print sample predictions."""
        predictions = val_metrics['predictions']
        labels = val_metrics['labels']
        probs = val_metrics['probs']
        stack_ids = val_metrics.get('stack_ids', [])
        
        print(f"\n  Sample predictions (first {num_samples}):")
        print(f"  {'Stack ID':<15} {'Label':>6} {'Pred':>6} {'Prob':>8} {'Status':>8}")
        print(f"  {'-'*50}")
        
        for i in range(min(num_samples, len(predictions))):
            label = int(labels[i]) if hasattr(labels[i], '__int__') else labels[i]
            pred = int(predictions[i]) if hasattr(predictions[i], '__int__') else predictions[i]
            prob = probs[i] if i < len(probs) else 0.0
            stack_id = stack_ids[i] if i < len(stack_ids) else 'N/A'
            status = '✓' if label == pred else '✗'
            print(f"  {stack_id:<15} {label:>6} {pred:>6} {prob:>8.4f} {status:>8}")
    
    def _save_checkpoint_and_check_early_stopping(self, epoch: int, 
                                                   train_metrics: Dict[str, float],
                                                   val_metrics: Dict[str, Any]) -> bool:
        """Save checkpoint and check early stopping."""
        val_f1_macro = val_metrics['f1_macro']
        improved = val_f1_macro > (self.best_val_f1_macro + self.early_stopping_min_delta)
        
        if improved:
            self.best_val_f1_macro = val_f1_macro
            self.epochs_without_improvement = 0
            self._save_checkpoint(epoch, train_metrics, val_metrics, is_best=True)
            print(f"  ★ New best F1 Macro: {val_f1_macro:.4f}")
        else:
            self.epochs_without_improvement += 1
            print(f"  No improvement for {self.epochs_without_improvement} epoch(s) (best: {self.best_val_f1_macro:.4f})")
        
        self._save_checkpoint(epoch, train_metrics, val_metrics, is_best=False)
        
        if self.early_stopping_patience and self.epochs_without_improvement >= self.early_stopping_patience:
            print(f"\n  Early stopping triggered after {epoch} epochs")
            return True
        
        return False
    
    def _save_checkpoint(self, epoch: int, train_metrics: Dict[str, float],
                        val_metrics: Dict[str, Any], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_f1_macro': self.best_val_f1_macro,
            'train_loss': train_metrics['loss'],
            'in_channels': self.model_in_channels,
            'num_classes': self.model_num_classes,
            'initial_channels': self.model_initial_channels,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'wandb_run_id': self.wandb_run_id
        }
        
        torch.save(checkpoint, self.save_dir / 'latest_model.pth')
        
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_model.pth')

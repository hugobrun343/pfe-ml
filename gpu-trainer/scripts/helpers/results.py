"""Results tracking and JSON export

JSON Structure:
{
    "training_config": {...},
    "best_val_f1_macro": 0.85,
    "total_epochs_completed": 15,
    "early_stopped": true,
    "epochs": [
        {
            "epoch": 1,
            "epoch_time": 120.5,
            "train": {"loss": 0.5, "f1_class_0": 0.65, "f1_class_1": 0.75},
            "validation": {
                "loss": 0.6,
                "f1_class_0": 0.60,
                "f1_class_1": 0.70,
                "tp": 45,
                "fp": 8,
                "tn": 67,
                "fn": 12,
                "accuracy": 0.8561,
                "precision": 0.8491,
                "recall": 0.7895,
                "samples": [
                    {
                        "sample_id": 0,
                        "stack_id": "volume_123",
                        "position_h": 128,
                        "position_w": 128,
                        "patch_index": 0,
                        "true_label": 1,
                        "predicted_label": 1,
                        "probability": 0.8,
                        "correct": true
                    },
                    ...
                ],
                "volumes": [
                    {
                        "stack_id": "volume_123",
                        "aggregated_probability": 0.7823,
                        "predicted_label": 1,
                        "true_label": 1,
                        "correct": true,
                        "n_patches": 64
                    },
                    ...
                ],
                "volume_metrics": {
                    "tp": 12, "fp": 2, "tn": 18, "fn": 3,
                    "accuracy": 0.8571,
                    "precision": 0.8571,
                    "recall": 0.8000,
                    "f1_class_0": 0.8000,
                    "f1_class_1": 0.8500
                }
            }
        },
        ...
    ]
}
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent.parent
scripts_dir = project_root / "scripts"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))
sys.path.insert(0, str(scripts_dir / "core"))

from scripts.core.metrics import compute_classification_metrics


class ResultsTracker:
    """Track training results and export to JSON"""
    
    def __init__(self, config: Dict[str, Any], save_dir: Path, resume: bool = False):
        """
        Initialize results tracker
        
        Args:
            config: Training configuration (batch_size, lr, etc.)
            save_dir: Directory to save results JSON file
            resume: If True, load existing results from JSON file if it exists
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        json_path = self.save_dir / 'training_results.json'
        
        # Load existing results if resuming
        if resume and json_path.exists():
            print(f"  Loading existing results from: {json_path}")
            with open(json_path, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            # Keep existing epochs but update config
            self.results = existing_results
            self.results['training_config'].update(config)
            self.results['training_config']['resumed_at'] = datetime.now().isoformat()
            print(f"  Found {len(self.results['epochs'])} existing epochs")
        else:
            self.results = {
                'training_config': {
                    **config,
                    'start_time': datetime.now().isoformat()
                },
                'epochs': []
            }
    
    def add_epoch(
        self,
        epoch: int,
        epoch_time: float,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Add epoch results to tracking
        
        Args:
            epoch: Epoch number
            epoch_time: Time taken for epoch (seconds)
            train_metrics: Dict with 'loss', 'f1_class_0', 'f1_class_1'
            val_metrics: Dict with 'loss', 'f1_class_0', 'f1_class_1', 'tp', 'fp', 'tn', 'fn', 'accuracy', 'precision', 'recall', 'predictions', 'labels', 'probs' (optional)
        """
        # Check if epoch already exists (protection against duplicates when resuming)
        existing_epochs = [e['epoch'] for e in self.results['epochs']]
        if epoch in existing_epochs:
            # Replace existing epoch with new data (in case of resume with same epoch)
            idx = existing_epochs.index(epoch)
            print(f"  Warning: Epoch {epoch} already exists, replacing with new data")
        else:
            idx = None
        
        epoch_result = {
            'epoch': epoch,
            'epoch_time': epoch_time,
            'train': {
                'loss': float(train_metrics['loss']),
                'f1_class_0': float(train_metrics.get('f1_class_0', 0.0)),
                'f1_class_1': float(train_metrics.get('f1_class_1', 0.0))
            }
        }
        
        # Add validation results if available
        if val_metrics:
            epoch_result['validation'] = {
                'loss': float(val_metrics['loss']),
                'f1_class_0': float(val_metrics.get('f1_class_0', 0.0)),
                'f1_class_1': float(val_metrics.get('f1_class_1', 0.0)),
                'tp': int(val_metrics.get('tp', 0)),
                'fp': int(val_metrics.get('fp', 0)),
                'tn': int(val_metrics.get('tn', 0)),
                'fn': int(val_metrics.get('fn', 0)),
                'accuracy': float(val_metrics.get('accuracy', 0.0)),
                'precision': float(val_metrics.get('precision', 0.0)),
                'recall': float(val_metrics.get('recall', 0.0)),
                'samples': self._build_sample_results(val_metrics)
            }
            
            # Add volume-level aggregated results
            volume_results = self._aggregate_volume_results(val_metrics)
            if volume_results['volumes']:
                epoch_result['validation']['volumes'] = volume_results['volumes']
                epoch_result['validation']['volume_metrics'] = volume_results['volume_metrics']
        
        if idx is not None:
            self.results['epochs'][idx] = epoch_result
        else:
            self.results['epochs'].append(epoch_result)
    
    def _build_sample_results(self, val_metrics: Dict[str, Any]) -> list:
        """
        Build detailed results for each validation sample
        
        Args:
            val_metrics: Dict with 'predictions', 'labels', 'probs', 'stack_ids', 'patch_positions'
            
        Returns:
            List of sample result dictionaries
        """
        if 'predictions' not in val_metrics:
            return []
        
        samples = []
        predictions = val_metrics['predictions']
        labels = val_metrics['labels']
        probs = val_metrics['probs']
        stack_ids = val_metrics.get('stack_ids', [])
        patch_positions = val_metrics.get('patch_positions', [])
        
        for i in range(len(predictions)):
            sample_dict = {
                'sample_id': i,
                'true_label': int(labels[i]),
                'predicted_label': int(predictions[i]),
                'probability': float(probs[i]),
                'correct': bool(labels[i] == predictions[i])
            }
            
            # Add stack_id if available
            if stack_ids and i < len(stack_ids):
                sample_dict['stack_id'] = stack_ids[i]
            
            # Add patch position information if available
            if patch_positions and i < len(patch_positions):
                pos_info = patch_positions[i]
                sample_dict['position_h'] = pos_info['position_h']
                sample_dict['position_w'] = pos_info['position_w']
                if pos_info.get('patch_index') is not None:
                    sample_dict['patch_index'] = pos_info['patch_index']
            
            samples.append(sample_dict)
        
        return samples
    
    def _aggregate_volume_results(self, val_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate patch-level results to volume-level results
        
        Groups patches by stack_id (volume) and computes:
        - Mean probability across all patches in a volume
        - Volume-level prediction (based on aggregated probability > 0.5)
        - Volume-level metrics (TP, FP, TN, FN, accuracy, precision, recall, F1)
        
        Args:
            val_metrics: Dict with 'stack_ids', 'probs', 'labels', 'predictions'
            
        Returns:
            Dict with:
            - volumes: List of volume result dictionaries
            - volume_metrics: Dictionary with volume-level classification metrics
        """
        if 'stack_ids' not in val_metrics or not val_metrics['stack_ids']:
            return {'volumes': [], 'volume_metrics': {}}
        
        stack_ids = val_metrics['stack_ids']
        probs = val_metrics['probs']
        labels = val_metrics['labels']
        
        # Group patches by volume (stack_id)
        volume_data = {}  # {stack_id: {'probs': [], 'label': int}}
        
        for i, stack_id in enumerate(stack_ids):
            if stack_id not in volume_data:
                # All patches from same volume have same label
                volume_data[stack_id] = {
                    'probs': [],
                    'label': int(labels[i])
                }
            volume_data[stack_id]['probs'].append(probs[i])
        
        # Aggregate probabilities per volume (mean aggregation)
        volume_results = []
        volume_predictions = []
        volume_labels = []
        
        for stack_id, data in volume_data.items():
            # Compute mean probability across all patches
            aggregated_prob = float(np.mean(data['probs']))
            
            # Volume-level prediction (threshold at 0.5)
            volume_prediction = 1 if aggregated_prob > 0.5 else 0
            volume_label = data['label']
            
            volume_results.append({
                'stack_id': stack_id,
                'aggregated_probability': aggregated_prob,
                'predicted_label': volume_prediction,
                'true_label': volume_label,
                'correct': bool(volume_prediction == volume_label),
                'n_patches': len(data['probs'])
            })
            
            volume_predictions.append(volume_prediction)
            volume_labels.append(volume_label)
        
        # Compute volume-level metrics
        if volume_predictions:
            volume_pred_tensor = torch.tensor(volume_predictions, dtype=torch.float32)
            volume_label_tensor = torch.tensor(volume_labels, dtype=torch.float32)
            volume_metrics = compute_classification_metrics(volume_pred_tensor, volume_label_tensor)
            
            return {
                'volumes': volume_results,
                'volume_metrics': {
                    'tp': int(volume_metrics['tp']),
                    'fp': int(volume_metrics['fp']),
                    'tn': int(volume_metrics['tn']),
                    'fn': int(volume_metrics['fn']),
                    'accuracy': float(volume_metrics['accuracy']),
                    'precision': float(volume_metrics['precision']),
                    'recall': float(volume_metrics['recall']),
                    'f1_class_0': float(volume_metrics.get('f1_class_0', 0.0)),
                    'f1_class_1': float(volume_metrics.get('f1_class_1', 0.0))
                }
            }
        else:
            return {'volumes': [], 'volume_metrics': {}}
    
    def save(self, best_val_f1_macro: Optional[float] = None, total_epochs: int = 0, early_stopped: bool = False) -> Path:
        """
        Save results to JSON file (called after each epoch)
        
        Args:
            best_val_f1_macro: Best validation F1 macro score
            total_epochs: Total number of epochs completed
            early_stopped: Whether training was early stopped
            
        Returns:
            Path to saved JSON file
        """
        self.results['training_config']['last_update'] = datetime.now().isoformat()
        self.results['best_val_f1_macro'] = float(best_val_f1_macro) if best_val_f1_macro is not None else None
        self.results['total_epochs_completed'] = total_epochs
        self.results['early_stopped'] = early_stopped
        
        json_path = self.save_dir / 'training_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        return json_path
    
    def plot_metrics(self) -> Path:
        """
        Create plots for training and validation metrics (loss and F1 per class)
        
        Returns:
            Path to saved plot file
        """
        if not self.results['epochs']:
            return None
        
        epochs = [e['epoch'] for e in self.results['epochs']]
        train_loss = [e['train']['loss'] for e in self.results['epochs']]
        train_f1_class_0 = [e['train'].get('f1_class_0', 0.0) for e in self.results['epochs']]
        train_f1_class_1 = [e['train'].get('f1_class_1', 0.0) for e in self.results['epochs']]
        
        val_loss = []
        val_f1_class_0 = []
        val_f1_class_1 = []
        for e in self.results['epochs']:
            if 'validation' in e:
                val_loss.append(e['validation']['loss'])
                val_f1_class_0.append(e['validation'].get('f1_class_0', 0.0))
                val_f1_class_1.append(e['validation'].get('f1_class_1', 0.0))
            else:
                val_loss.append(None)
                val_f1_class_0.append(None)
                val_f1_class_1.append(None)
        
        # Create figure with three subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot Loss
        ax1.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
        if any(v is not None for v in val_loss):
            val_epochs = [e for i, e in enumerate(epochs) if val_loss[i] is not None]
            val_loss_clean = [v for v in val_loss if v is not None]
            ax1.plot(val_epochs, val_loss_clean, 'r-', label='Val Loss', linewidth=2, marker='s', markersize=4)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot F1 Scores per class
        ax2.plot(epochs, train_f1_class_0, 'b--', label='Train F1 Class 0', linewidth=2, marker='o', markersize=4, alpha=0.7)
        ax2.plot(epochs, train_f1_class_1, 'b-', label='Train F1 Class 1', linewidth=2, marker='o', markersize=4, alpha=0.7)
        if any(v is not None for v in val_f1_class_0):
            val_epochs = [e for i, e in enumerate(epochs) if val_f1_class_0[i] is not None]
            val_f1_class_0_clean = [v for v in val_f1_class_0 if v is not None]
            val_f1_class_1_clean = [v for v in val_f1_class_1 if v is not None]
            ax2.plot(val_epochs, val_f1_class_0_clean, 'g--', label='Val F1 Class 0', linewidth=2, marker='s', markersize=4)
            ax2.plot(val_epochs, val_f1_class_1_clean, 'r-', label='Val F1 Class 1', linewidth=2, marker='s', markersize=4)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('F1 Score', fontsize=12)
        ax2.set_title('Training and Validation F1 Score per Class', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.save_dir / 'training_metrics.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def finalize(self, best_val_f1_macro: Optional[float] = None, total_epochs: int = 0, early_stopped: bool = False, analytics_dir: Optional[Path] = None) -> Path:
        """
        Finalize results with end time, create plots and save (called at end of training)
        
        Args:
            best_val_f1_macro: Best validation F1 macro score
            total_epochs: Total number of epochs completed
            early_stopped: Whether training was early stopped
            analytics_dir: Directory to save analytics (if None, uses save_dir)
            
        Returns:
            Path to saved JSON file
        """
        self.results['training_config']['end_time'] = datetime.now().isoformat()
        json_path = self.save(best_val_f1_macro, total_epochs, early_stopped)
        
        # Create plots (save to results directory)
        plot_path = self.plot_metrics()
        if plot_path:
            print(f"\nMetrics plot saved to: {plot_path}")
        
        # Run automatic analysis (save to analytics_dir)
        if analytics_dir is None:
            analytics_dir = self.save_dir / 'analytics'
        self._run_analysis(json_path, analytics_dir)
        
        return json_path
    
    def _run_analysis(self, json_path: Path, analytics_dir: Path):
        """Run automatic analysis of training results"""
        try:
            import subprocess
            import sys
            
            script_path = Path(__file__).parent.parent / 'analytics' / 'analyze_results.py'
            
            if not script_path.exists():
                print(f"Warning: Analysis script not found at {script_path}, skipping analysis")
                return
            
            # Ensure analytics directory exists
            analytics_dir.mkdir(parents=True, exist_ok=True)
            
            print("\n" + "=" * 80)
            print("Running automatic results analysis...")
            print("=" * 80)
            
            result = subprocess.run(
                [sys.executable, str(script_path), str(json_path), '--output-dir', str(analytics_dir)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(result.stdout)
                print("Analysis completed successfully!")
            else:
                print(f"Warning: Analysis script returned error code {result.returncode}")
                if result.stderr:
                    print(f"Error output: {result.stderr}")
        except Exception as e:
            print(f"Warning: Failed to run analysis: {e}")
            print("You can run it manually with:")
            print(f"  python scripts/analytics/analyze_results.py {json_path} --output-dir {analytics_dir}")
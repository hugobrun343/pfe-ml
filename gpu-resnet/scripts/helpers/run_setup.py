"""Run directory setup and metadata copying utilities."""

import shutil
from pathlib import Path
from typing import Dict
from datetime import datetime


def create_run_directories(runs_dir: Path, run_name: str) -> Dict[str, Path]:
    """
    Create directory structure for a training run.
    
    Args:
        runs_dir: Base directory for runs
        run_name: Base name of the run
        
    Returns:
        Dictionary with paths: run_dir, checkpoints, results, analytics, wandb, data, run_name
    """
    # Always add timestamp to ensure uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_run_name = f"{run_name}_{timestamp}"
    
    run_dir = runs_dir / unique_run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    dirs = {
        'run_dir': run_dir,
        'checkpoints': run_dir / 'checkpoints',
        'results': run_dir / 'results',
        'analytics': run_dir / 'analytics',
        'wandb': run_dir / 'wandb',
        'data': run_dir / 'data',
        'run_name': unique_run_name
    }
    
    # Create all subdirectories
    for dir_path in dirs.values():
        if dir_path != run_dir and isinstance(dir_path, Path):
            dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def copy_data_metadata(preprocessed_dir: Path, train_test_split_json: Path, data_dir: Path) -> None:
    """
    Copy preprocessing metadata, train/test split JSON, and original dataset JSON to run's data directory.
    
    Args:
        preprocessed_dir: Directory containing preprocessed data (with metadata.json)
        train_test_split_json: Path to train_test_split.json file
        data_dir: Directory where to copy the files
    """
    import json
    
    # Copy preprocessing metadata.json
    preprocess_metadata = preprocessed_dir / 'metadata.json'
    if preprocess_metadata.exists():
        shutil.copy2(preprocess_metadata, data_dir / 'preprocessing_metadata.json')
        print(f"  Copied preprocessing metadata to: {data_dir / 'preprocessing_metadata.json'}")
        
        # Try to copy original dataset JSON from metadata
        try:
            with open(preprocess_metadata, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            dataset_source = metadata.get('dataset_source')
            if dataset_source:
                dataset_path = Path(dataset_source)
                if dataset_path.exists():
                    shutil.copy2(dataset_path, data_dir / 'original_dataset.json')
                    print(f"  Copied original dataset to: {data_dir / 'original_dataset.json'}")
                else:
                    print(f"  Warning: Original dataset not found at {dataset_path}")
        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"  Warning: Could not extract dataset source from metadata: {e}")
    else:
        print(f"  Warning: Preprocessing metadata not found at {preprocess_metadata}")
    
    # Copy train_test_split.json
    if train_test_split_json.exists():
        shutil.copy2(train_test_split_json, data_dir / 'train_test_split.json')
        print(f"  Copied train/test split to: {data_dir / 'train_test_split.json'}")
    else:
        print(f"  Warning: Train/test split not found at {train_test_split_json}")


def get_runs_directory(project_root: Path) -> Path:
    """
    Get the runs directory path (pointing to _runs at work-hugo root).
    
    Args:
        project_root: Path to gpu-resnet project root
        
    Returns:
        Path to _runs directory
    """
    # _runs is at work-hugo root, which is 2 levels up from gpu-resnet
    work_hugo_root = project_root.parent
    runs_dir = work_hugo_root / '_runs'
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir

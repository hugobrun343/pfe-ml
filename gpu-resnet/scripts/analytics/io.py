"""I/O utilities for loading training results."""

import json
from pathlib import Path
from typing import Dict


def load_results(json_path: Path) -> Dict:
    """
    Load training results JSON file.
    
    Args:
        json_path: Path to training_results.json file
        
    Returns:
        Dictionary containing training results
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

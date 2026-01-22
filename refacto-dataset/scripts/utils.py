"""
Utility functions for extracting Two-Photon microscopy data.
"""

import re
import unicodedata
from pathlib import Path
from typing import Optional, Tuple, Set


def normalize_path(path: str) -> str:
    """Normalize path for consistent comparison (handles unicode issues)."""
    return unicodedata.normalize('NFC', path)


def extract_pressure_stretch_from_folder(folder_name: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract pressure and stretch from folder name.
    
    Accepted formats (in order of priority):
    1. Complete: 120_2_D, 50_1_1, 120_2_D2, 10_2_1, 10_2_2 (PRESSURE_STRETCH_ORIENTATION/REPLICATE)
       where orientation can be: D, V, 1 (=V), 2 (=D), D1, D2, V1, V2, etc.
    2. Incomplete: 120_D, 80_V (PRESSURE_ORIENTATION, no stretch info)
    3. Pressure only: 120, 80 (just pressure, no stretch info)
    
    Does NOT accept:
    - Typos or spaces: "80 _1_V" (must be perfectly formatted)
    - Sample ID formats: 5993_120
    
    Returns:
        (pressure, stretch) or (pressure, None) or (None, None)
    """
    # NO cleaning - format must be perfect as-is
    
    # Pattern 1: Complete format PRESSURE_STRETCH_* (with optional variant suffix)
    # Matches: 120_2_D, 50_1_1, 120_2_D2, 80_1_V3, etc.
    match = re.match(r'^(\d+)_(\d+)_', folder_name)
    if match:
        pressure = int(match.group(1))
        stretch = int(match.group(2))
        # Validate reasonable ranges
        if 10 <= pressure <= 200 and 1 <= stretch <= 5:
            return pressure, stretch
    
    # Pattern 2: Incomplete format PRESSURE_ORIENTATION (120_D, 80_V)
    match = re.match(r'^(\d+)_[DV]$', folder_name)
    if match:
        pressure = int(match.group(1))
        if 10 <= pressure <= 200:
            return pressure, None  # No stretch info
    
    # Pattern 3: Pressure only (120, 80)
    match = re.match(r'^(\d{2,3})$', folder_name)
    if match:
        pressure = int(match.group(1))
        if 10 <= pressure <= 200:
            return pressure, None  # No stretch info
    
    # No match = not an accepted format
    return None, None


def is_analysis_folder(folder_name: str) -> bool:
    """Check if folder is an analysis/processing folder (not raw data)."""
    analysis_keywords = {
        'collagen', 'elastin', 'orientation', 'thickness', 
        'vol_fraction', 'volume', 'edit', 'front', 'top',
        'bottom', 'left', 'right', 'summary', 'tmp'
    }
    name_lower = folder_name.lower()
    
    # Check for exact matches or keywords
    if name_lower in analysis_keywords:
        return True
    
    # Check for _edit suffix
    if '_edit' in name_lower:
        return True
    
    # Check for date-based names (likely metadata folders)
    if re.match(r'^\d{6,8}_', folder_name):
        return True
    
    return False


def scan_sample_folder(sample_folder_path: Path) -> Tuple[Set[int], Set[int]]:
    """
    Scan a sample folder and extract all unique pressures and stretches.
    Automatically filters out analysis folders.
    
    Returns:
        (set of pressures, set of stretches)
    """
    pressures = set()
    stretches = set()
    
    try:
        for subfolder in sample_folder_path.iterdir():
            if not subfolder.is_dir():
                continue
            
            # Skip analysis/processing folders
            if is_analysis_folder(subfolder.name):
                continue
            
            pressure, stretch = extract_pressure_stretch_from_folder(subfolder.name)
            
            if pressure is not None:
                pressures.add(pressure)
            if stretch is not None:
                stretches.add(stretch)
    
    except (PermissionError, OSError):
        pass
    
    return pressures, stretches


def scan_path(base_path: Path, relative_path: str) -> Tuple[Set[int], Set[int]]:
    """
    Scan all samples in a path and extract pressures and stretches.
    
    Returns:
        (set of pressures, set of stretches)
    """
    if not relative_path or not relative_path.strip():
        return set(), set()
    
    # Normalize path to handle unicode issues
    full_path = base_path / normalize_path(relative_path)
    
    if not full_path.exists() or not full_path.is_dir():
        return set(), set()
    
    all_pressures = set()
    all_stretches = set()
    
    try:
        # Iterate through sample folders
        for sample_folder in full_path.iterdir():
            if not sample_folder.is_dir():
                continue
            
            pressures, stretches = scan_sample_folder(sample_folder)
            all_pressures.update(pressures)
            all_stretches.update(stretches)
    
    except (PermissionError, OSError):
        pass
    
    return all_pressures, all_stretches


def format_values(values: Set[int]) -> str:
    """Format a set of values as comma-separated string."""
    if not values:
        return ""
    return ", ".join(str(v) for v in sorted(values))


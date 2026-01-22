#!/usr/bin/env python3
"""
Create stratified train/test split from enriched dataset.

This script filters the dataset based on a configuration file and creates
a stratified split ensuring balanced representation across key features.
"""

import json
import argparse
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any
from datetime import datetime
import random

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)


def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file."""
    print(f"Loading configuration from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print("  Configuration loaded successfully")
    return config


def load_dataset(input_path: Path) -> Dict:
    """Load dataset JSON file."""
    print(f"Loading dataset from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  Loaded {len(data.get('stacks', []))} stacks")
    return data


def check_filter_match(value: Any, filter_value: Any) -> bool:
    """
    Check if a value matches a filter (supports single values and lists).
    
    Args:
        value: The actual value from stack
        filter_value: The filter criteria (can be single value or list)
    
    Returns:
        True if value matches filter, False otherwise
    """
    # If filter is disabled (None/null in config), accept all values
    if filter_value is None:
        return True
    
    # Convert filter to list if it isn't already
    if not isinstance(filter_value, list):
        filter_list = [filter_value]
    else:
        filter_list = filter_value
    
    # Check if value matches any in the filter list
    return str(value) in [str(f) for f in filter_list]


def filter_stacks(stacks: List[Dict], filters: Dict) -> List[Dict]:
    """
    Filter stacks based on configuration criteria.
    Supports single values and lists for all filters.
    
    Args:
        stacks: List of stack dictionaries
        filters: Dictionary with filter criteria from config
    
    Returns:
        List of filtered stacks
    """
    filtered = []
    
    for stack in stacks:
        if 'infos' not in stack:
            continue
        
        infos = stack['infos']
        keep = True
        
        try:
            # Age filter (range-based)
            if 'age' in filters and filters['age'] is not None:
                age = int(infos.get('Age (wk)', '999'))
                age_config = filters['age']
                
                if age_config.get('min') is not None and age < age_config['min']:
                    keep = False
                if age_config.get('max') is not None and age > age_config['max']:
                    keep = False
            
            # Axial Stretch filter (single value or list)
            if keep and 'axial_stretch' in filters and filters['axial_stretch'] is not None:
                stretch = infos.get('Axial Stretch', 'None')
                if not check_filter_match(stretch, filters['axial_stretch']):
                    keep = False
            
            # Pressure filter (single value or list)
            if keep and 'pressure' in filters and filters['pressure'] is not None:
                pressure = infos.get('Pressure', 'None')
                if not check_filter_match(pressure, filters['pressure']):
                    keep = False
            
            # Region filter (single value or list)
            if keep and 'region' in filters and filters['region'] is not None:
                region = infos.get('Region ', '').strip()  # Note the space in key
                if not check_filter_match(region, filters['region']):
                    keep = False
            
            # Classe filter (single value or list)
            if keep and 'classe' in filters and filters['classe'] is not None:
                classe = infos.get('Classe', '')
                if not check_filter_match(classe, filters['classe']):
                    keep = False
            
            # Genetic filter (single value or list)
            if keep and 'genetic' in filters and filters['genetic'] is not None:
                genetic = infos.get('Genetic', '')
                if not check_filter_match(genetic, filters['genetic']):
                    keep = False
            
            # Sex filter (single value or list)
            if keep and 'sex' in filters and filters['sex'] is not None:
                sex = infos.get('Sex', '')
                if not check_filter_match(sex, filters['sex']):
                    keep = False
            
            # Orientation filter (single value or list)
            if keep and 'orientation' in filters and filters['orientation'] is not None:
                orientation = infos.get('Orientation', '')
                if not check_filter_match(orientation, filters['orientation']):
                    keep = False
            
            if keep:
                filtered.append(stack)
            
        except (ValueError, KeyError) as e:
            continue
    
    return filtered


def create_stratification_key(stack: Dict, stratify_by: List[str]) -> str:
    """
    Create a stratification key from stack metadata.
    
    Args:
        stack: Stack dictionary
        stratify_by: List of keys to stratify by
    """
    infos = stack.get('infos', {})
    
    key_parts = []
    for field in stratify_by:
        value = infos.get(field, 'Unknown')
        key_parts.append(str(value))
    
    return "_".join(key_parts)


def stratified_split(stacks: List[Dict], config: Dict) -> Tuple[List[Dict], List[Dict]]:
    """
    Perform stratified train/test split.
    
    Args:
        stacks: List of stack dictionaries
        config: Split configuration dictionary
    
    Returns:
        Tuple of (train_stacks, test_stacks)
    """
    test_size = config.get('test_size', 0.2)
    random_seed = config.get('random_seed', 42)
    stratify_by = config.get('stratify_by', ['Age (wk)', 'Sex', 'Orientation', 'Genetic'])
    
    random.seed(random_seed)
    
    # Group stacks by stratification key
    strata = defaultdict(list)
    for stack in stacks:
        key = create_stratification_key(stack, stratify_by)
        strata[key].append(stack)
    
    train_stacks = []
    test_stacks = []
    
    # Split each stratum (sort keys for deterministic order)
    for stratum_key in sorted(strata.keys()):
        stratum_stacks = strata[stratum_key]
        # Shuffle within stratum
        random.shuffle(stratum_stacks)
        
        # Calculate split point
        n_test = max(1, int(len(stratum_stacks) * test_size))
        
        # If stratum has only 1 sample, put it in train
        if len(stratum_stacks) == 1:
            train_stacks.extend(stratum_stacks)
        else:
            test_stacks.extend(stratum_stacks[:n_test])
            train_stacks.extend(stratum_stacks[n_test:])
    
    return train_stacks, test_stacks


def compute_statistics(stacks: List[Dict]) -> Dict:
    """Compute distribution statistics for a set of stacks."""
    stats = {
        'total': len(stacks),
        'age': Counter(),
        'sex': Counter(),
        'orientation': Counter(),
        'genetic': Counter(),
        'classe': Counter(),
        'pressure': Counter(),
        'axial_stretch': Counter(),
        'region': Counter()
    }
    
    for stack in stacks:
        infos = stack.get('infos', {})
        stats['age'][infos.get('Age (wk)', 'Unknown')] += 1
        stats['sex'][infos.get('Sex', 'Unknown')] += 1
        stats['orientation'][infos.get('Orientation', 'Unknown')] += 1
        stats['genetic'][infos.get('Genetic', 'Unknown')] += 1
        stats['classe'][infos.get('Classe', 'Unknown')] += 1
        stats['pressure'][infos.get('Pressure', 'Unknown')] += 1
        stats['axial_stretch'][infos.get('Axial Stretch', 'Unknown')] += 1
        stats['region'][infos.get('Region ', 'Unknown').strip()] += 1
    
    return stats


def print_statistics(stats: Dict, label: str):
    """Print statistics in a readable format."""
    print(f"\n{'='*80}")
    print(f"{label.upper()} STATISTICS")
    print(f"{'='*80}")
    print(f"Total samples: {stats['total']}")
    
    for category in ['age', 'sex', 'orientation', 'genetic', 'classe']:
        if stats[category]:
            print(f"\n{category.capitalize()} distribution:")
            total = stats['total']
            for value, count in sorted(stats[category].items(), key=lambda x: -x[1]):
                percentage = (count / total) * 100
                print(f"  {str(value):30s} : {count:4d} ({percentage:5.1f}%)")


def save_split(train_stacks: List[Dict], test_stacks: List[Dict], 
               output_path: Path, config: Dict):
    """Save train/test split to JSON file."""
    
    filters = config.get('filters', {})
    split_config = config.get('split', {})
    output_config = config.get('output', {})
    
    split_data = {
        "metadata": {
            "total_samples": len(train_stacks) + len(test_stacks),
            "train_samples": len(train_stacks),
            "test_samples": len(test_stacks),
            "train_ratio": len(train_stacks) / (len(train_stacks) + len(test_stacks)),
            "test_ratio": len(test_stacks) / (len(train_stacks) + len(test_stacks)),
            "filters_applied": filters,
            "stratification_keys": split_config.get('stratify_by', []),
            "random_seed": split_config.get('random_seed', 42)
        },
        "train": [stack['id'] for stack in train_stacks],
        "test": [stack['id'] for stack in test_stacks]
    }
    
    # Optionally include full stack details
    if output_config.get('save_full_details', False):
        split_data['train_details'] = train_stacks
        split_data['test_details'] = test_stacks
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    indent = output_config.get('indent_json', 2)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(split_data, f, indent=indent)
    
    print(f"\nSplit saved to: {output_path}")


def save_summary(train_stats: Dict, test_stats: Dict, output_path: Path, config: Dict):
    """Save detailed summary to text file."""
    
    filters = config.get('filters', {})
    split_config = config.get('split', {})
    
    lines = []
    lines.append("TRAIN/TEST SPLIT SUMMARY")
    lines.append("="*80)
    lines.append("")
    lines.append("CONFIGURATION:")
    lines.append("")
    lines.append("Filters applied:")
    
    # Age filter
    if 'age' in filters and filters['age'] is not None:
        age_filter = filters['age']
        min_age = age_filter.get('min', 'None')
        max_age = age_filter.get('max', 'None')
        lines.append(f"  Age (wk): min={min_age}, max={max_age}")
    
    # Other filters
    for key in ['axial_stretch', 'pressure', 'region', 'classe', 'genetic', 'sex', 'orientation']:
        if key in filters and filters[key] is not None:
            value = filters[key]
            if isinstance(value, list):
                lines.append(f"  {key}: {', '.join(map(str, value))}")
            else:
                lines.append(f"  {key}: {value}")
    
    lines.append("")
    lines.append("Split configuration:")
    lines.append(f"  Test size: {split_config.get('test_size', 0.2)*100:.0f}%")
    lines.append(f"  Random seed: {split_config.get('random_seed', 42)}")
    lines.append(f"  Stratification: {', '.join(split_config.get('stratify_by', []))}")
    lines.append("")
    
    # Train statistics
    lines.append("="*80)
    lines.append("TRAIN SET STATISTICS")
    lines.append("="*80)
    lines.append(f"Total samples: {train_stats['total']}")
    
    for category in ['age', 'sex', 'orientation', 'genetic', 'classe']:
        if train_stats[category]:
            lines.append(f"\n{category.capitalize()} distribution:")
            total = train_stats['total']
            for value, count in sorted(train_stats[category].items(), key=lambda x: -x[1]):
                percentage = (count / total) * 100
                lines.append(f"  {str(value):30s} : {count:4d} ({percentage:5.1f}%)")
    
    # Test statistics
    lines.append("")
    lines.append("="*80)
    lines.append("TEST SET STATISTICS")
    lines.append("="*80)
    lines.append(f"Total samples: {test_stats['total']}")
    
    for category in ['age', 'sex', 'orientation', 'genetic', 'classe']:
        if test_stats[category]:
            lines.append(f"\n{category.capitalize()} distribution:")
            total = test_stats['total']
            for value, count in sorted(test_stats[category].items(), key=lambda x: -x[1]):
                percentage = (count / total) * 100
                lines.append(f"  {str(value):30s} : {count:4d} ({percentage:5.1f}%)")
    
    # Comparison
    lines.append("")
    lines.append("="*80)
    lines.append("COMPARISON (Train vs Test)")
    lines.append("="*80)
    
    for category in ['age', 'sex', 'orientation', 'genetic', 'classe']:
        lines.append(f"\n{category.capitalize()}:")
        all_values = set(train_stats[category].keys()) | set(test_stats[category].keys())
        for value in sorted(all_values):
            train_pct = (train_stats[category][value] / train_stats['total'] * 100) if value in train_stats[category] else 0
            test_pct = (test_stats[category][value] / test_stats['total'] * 100) if value in test_stats[category] else 0
            diff = abs(train_pct - test_pct)
            lines.append(f"  {str(value):30s} : Train {train_pct:5.1f}% | Test {test_pct:5.1f}% | Diff {diff:5.1f}%")
    
    lines.append("")
    lines.append("="*80)
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"Summary saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create stratified train/test split from enriched dataset using config file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python create_train_test_split.py \\
      --config config/split_config.yaml \\
      --input data_final/dataset_enriched_FINAL.json \\
      --output splits/train_test_split.json
  
  # Specify custom output directory
  python create_train_test_split.py \\
      -c config/split_config.yaml \\
      -i data_final/dataset_enriched_FINAL.json \\
      -o outputs/my_split.json

Configuration file (YAML) should contain:
  - filters: age, pressure, region, etc.
  - split: test_size, random_seed, stratify_by
  - output: generate_summary, summary_filename
"""
    )
    
    # Required arguments
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input enriched dataset JSON"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output split JSON file"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    config_path = Path(args.config).resolve()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_path.parent / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_path.name
    
    # Validate inputs
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("TRAIN/TEST SPLIT CREATION")
    print("="*80)
    print(f"Config:     {config_path}")
    print(f"Input:      {input_path}")
    print(f"Output:     {output_path}")
    print("="*80 + "\n")
    
    # Load configuration
    config = load_config(config_path)
    
    # Load dataset
    dataset = load_dataset(input_path)
    
    # Get config sections
    filters = config.get('filters', {})
    split_config = config.get('split', {})
    output_config = config.get('output', {})
    
    # Display filter configuration
    print("\nFilter configuration:")
    if 'age' in filters and filters['age'] is not None:
        age_config = filters['age']
        print(f"  Age (wk): min={age_config.get('min')}, max={age_config.get('max')}")
    
    for key in ['axial_stretch', 'pressure', 'region', 'classe', 'genetic', 'sex', 'orientation']:
        if key in filters and filters[key] is not None:
            value = filters[key]
            if isinstance(value, list):
                print(f"  {key}: {', '.join(map(str, value))}")
            else:
                print(f"  {key}: {value}")
    
    print(f"\nSplit configuration:")
    print(f"  Test size: {split_config.get('test_size', 0.2)*100:.0f}%")
    print(f"  Random seed: {split_config.get('random_seed', 42)}")
    
    # Filter stacks
    print("\nApplying filters...")
    filtered_stacks = filter_stacks(dataset['stacks'], filters)
    print(f"  Filtered: {len(filtered_stacks)} stacks (from {len(dataset['stacks'])} total)")
    
    if len(filtered_stacks) == 0:
        print("\nERROR: No stacks match the filter criteria!")
        sys.exit(1)
    
    # Create stratified split
    print("\nCreating stratified split...")
    train_stacks, test_stacks = stratified_split(filtered_stacks, split_config)
    print(f"  Train: {len(train_stacks)} stacks ({len(train_stacks)/(len(train_stacks)+len(test_stacks))*100:.1f}%)")
    print(f"  Test:  {len(test_stacks)} stacks ({len(test_stacks)/(len(train_stacks)+len(test_stacks))*100:.1f}%)")
    
    # Compute statistics
    train_stats = compute_statistics(train_stacks)
    test_stats = compute_statistics(test_stacks)
    
    # Print statistics
    print_statistics(train_stats, "TRAIN")
    print_statistics(test_stats, "TEST")
    
    # Save split
    print("\n" + "="*80)
    save_split(train_stacks, test_stacks, output_path, config)
    
    # Save summary if requested
    if output_config.get('generate_summary', True):
        summary_filename = output_config.get('summary_filename', 'split_summary.txt')
        summary_path = output_path.parent / summary_filename
        save_summary(train_stats, test_stats, summary_path, config)
    
    print("\n" + "="*80)
    print("SPLIT CREATION COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

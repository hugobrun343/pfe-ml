#!/usr/bin/env python3
"""
Script to clean paths in dataset.json by removing the storage prefix.
Removes: /storage/simple/users/dudognonm/raw/ds_snapshot_2026-01-11/
"""

import json
import sys
import argparse
import shutil
from pathlib import Path

def clean_paths(data, prefix_to_remove):
    """
    Remove prefix from all source_files paths in the dataset.
    
    Args:
        data: The dataset dictionary
        prefix_to_remove: The prefix string to remove from paths
    
    Returns:
        Modified data with cleaned paths
    """
    count = 0
    
    if 'stacks' in data:
        for stack in data['stacks']:
            if 'source_files' in stack:
                for channel, files in stack['source_files'].items():
                    for i, file_path in enumerate(files):
                        if file_path.startswith(prefix_to_remove):
                            # Remove the prefix
                            stack['source_files'][channel][i] = file_path[len(prefix_to_remove):]
                            count += 1
    
    return data, count


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Clean paths in dataset.json by removing a prefix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean paths with default prefix
  python clean_dataset_paths.py \\
      --input backups/dataset_original.json \\
      --output data_intermediate/dataset_cleaned.json \\
      --prefix "/storage/simple/users/dudognonm/raw/ds_snapshot_2026-01-11/"
  
  # No backup creation
  python clean_dataset_paths.py \\
      --input dataset.json \\
      --output dataset_cleaned.json \\
      --prefix "/some/path/" \\
      --no-backup
"""
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input JSON file"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output JSON file (can be same as input to overwrite)"
    )
    
    parser.add_argument(
        "--prefix", "-p",
        type=str,
        required=True,
        help="Prefix to remove from all file paths"
    )
    
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create a backup file before modifying"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    input_file = Path(args.input).resolve()
    output_file = Path(args.output).resolve()
    prefix = args.prefix
    
    # Validate input exists
    if not input_file.exists():
        print(f"\nERROR ERROR: Input file not found: {input_file}")
        sys.exit(1)
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"CLEANING DATASET PATHS")
    print(f"{'='*80}")
    print(f"Input file:  {input_file}")
    print(f"Output file: {output_file}")
    print(f"Prefix to remove: {prefix}")
    print(f"{'='*80}\n")
    
    # Create backup
    if not args.no_backup and input_file == output_file:
        backup_file = input_file.with_suffix('.json.backup')
        print("Creating backup...")
        shutil.copy2(input_file, backup_file)
        print(f"OK Backup created: {backup_file}\n")
    
    # Load JSON
    print("Loading JSON file...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"OK Loaded {len(data.get('stacks', []))} stacks\n")
    
    # Clean paths
    print("Cleaning paths...")
    data, count = clean_paths(data, prefix)
    print(f"OK Cleaned {count} file paths\n")
    
    # Save modified JSON
    print("Saving modified JSON...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"OK Saved to {output_file}\n")
    
    print(f"{'='*80}")
    print(f"OK DONE!")
    print(f"{'='*80}\n")
    print(f"Modified {count} file paths\n")


if __name__ == "__main__":
    main()

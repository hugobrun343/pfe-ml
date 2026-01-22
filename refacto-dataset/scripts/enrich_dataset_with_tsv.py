#!/usr/bin/env python3
"""
Enrich dataset.json with metadata from Database_Extracted.tsv

This script adds an 'infos' section to each stack in dataset.json containing
all the metadata from the corresponding row in the TSV file.
"""

import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import sys
from datetime import datetime


def load_tsv_data(tsv_file: Path) -> Dict[str, Dict[str, str]]:
    """
    Load TSV data and index it by the directory path.
    
    Returns:
        Dictionary mapping directory paths to row data (as dict of column_name -> value)
    """
    tsv_index = {}
    
    with open(tsv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        
        for row in reader:
            # The 'Path' column contains the directory path
            if 'Path' in row:
                path = row['Path'].strip()
                tsv_index[path] = row
    
    print(f"Loaded {len(tsv_index)} entries from TSV file")
    return tsv_index


def extract_directory_path(source_files: Dict[str, List[str]]) -> Optional[str]:
    """
    Extract the directory path from source_files.
    
    Takes the first file from any channel and removes the filename,
    keeping only the directory path.
    
    Example:
        "A5A2_mgR/ATA/8365_ATA/120_2_D/09-59-28_PMT - PMT [Blue] _C00_z-Stepper Z0000.ome.tif"
        -> "A5A2_mgR/ATA/8365_ATA/120_2_D"
    """
    for channel, files in source_files.items():
        if files and len(files) > 0:
            first_file = files[0]
            # Remove the filename after the last '/'
            parts = first_file.rsplit('/', 1)
            if len(parts) == 2:
                return parts[0]
            elif len(parts) == 1:
                # No directory, just filename
                return ""
    
    return None


def enrich_dataset(dataset_file: Path, tsv_file: Path, output_file: Optional[Path] = None):
    """
    Enrich dataset.json with TSV metadata.
    
    Args:
        dataset_file: Path to dataset.json
        tsv_file: Path to Database_Extracted.tsv
        output_file: Optional output path (defaults to overwriting dataset_file)
    """
    # Create backup
    backup_file = dataset_file.with_suffix('.json.backup_enriched')
    print(f"Creating backup: {backup_file}")
    with open(dataset_file, 'r', encoding='utf-8') as f:
        backup_content = f.read()
    with open(backup_file, 'w', encoding='utf-8') as f:
        f.write(backup_content)
    
    # Load TSV data
    print(f"Loading TSV data from: {tsv_file}")
    tsv_index = load_tsv_data(tsv_file)
    
    # Load dataset.json
    print(f"Loading dataset from: {dataset_file}")
    with open(dataset_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process each stack
    total_stacks = 0
    enriched_count = 0
    not_found_count = 0
    not_found_paths = []
    
    if "stacks" in data:
        total_stacks = len(data["stacks"])
        print(f"Processing {total_stacks} stacks...")
        
        for stack in data["stacks"]:
            if "source_files" in stack:
                # Extract directory path from source files
                dir_path = extract_directory_path(stack["source_files"])
                
                if dir_path:
                    # Look for matching TSV entry
                    if dir_path in tsv_index:
                        # Add 'infos' section with all TSV data
                        stack["infos"] = tsv_index[dir_path]
                        enriched_count += 1
                    else:
                        # Try to find a match by checking if TSV path is contained in dir_path
                        found = False
                        for tsv_path, tsv_data in tsv_index.items():
                            if dir_path.endswith(tsv_path) or tsv_path in dir_path:
                                stack["infos"] = tsv_data
                                enriched_count += 1
                                found = True
                                break
                        
                        if not found:
                            not_found_count += 1
                            not_found_paths.append(dir_path)
    
    # Save enriched dataset
    output_path = output_file if output_file else dataset_file
    print(f"\nSaving enriched dataset to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("ENRICHMENT SUMMARY")
    print("=" * 60)
    print(f"Total stacks processed: {total_stacks}")
    print(f"Stacks enriched with TSV data: {enriched_count}")
    print(f"Stacks without matching TSV entry: {not_found_count}")
    
    if not_found_count > 0 and not_found_count <= 20:
        print("\nPaths without matches (showing first 20):")
        for path in not_found_paths[:20]:
            print(f"  - {path}")
    elif not_found_count > 20:
        print(f"\nPaths without matches (showing first 20 of {not_found_count}):")
        for path in not_found_paths[:20]:
            print(f"  - {path}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Enrich dataset.json with metadata from TSV file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python enrich_dataset_with_tsv.py \\
      --json data_intermediate/dataset_cleaned.json \\
      --tsv data_intermediate/database_extracted_with_measurements.tsv \\
      --output data_final/dataset_enriched_FINAL.json
  
  # Using default paths
  python enrich_dataset_with_tsv.py \\
      --json dataset.json \\
      --tsv output/Database_Extracted.tsv \\
      --output dataset_enriched.json
"""
    )
    
    parser.add_argument(
        "--json", "-j",
        type=str,
        required=True,
        help="Path to input dataset JSON file"
    )
    
    parser.add_argument(
        "--tsv", "-t",
        type=str,
        required=True,
        help="Path to TSV file with metadata (e.g., Database_Extracted.tsv)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output enriched JSON file"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    dataset_file = Path(args.json).resolve()
    tsv_file = Path(args.tsv).resolve()
    output_file = Path(args.output).resolve()
    
    # Check files exist
    if not dataset_file.exists():
        print(f"\nERROR ERROR: Dataset file not found: {dataset_file}")
        sys.exit(1)
    
    if not tsv_file.exists():
        print(f"\nERROR ERROR: TSV file not found: {tsv_file}")
        sys.exit(1)
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("DATASET ENRICHMENT")
    print("=" * 60)
    print(f"JSON input:  {dataset_file}")
    print(f"TSV input:   {tsv_file}")
    print(f"JSON output: {output_file}")
    print("=" * 60 + "\n")
    
    # Run enrichment (modified to take output path)
    enrich_dataset(dataset_file, tsv_file, output_file)
    
    print("\nOK Dataset enrichment complete!")


if __name__ == "__main__":
    main()

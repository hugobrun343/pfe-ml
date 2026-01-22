#!/usr/bin/env python3
"""
Two-Photon Microscopy Data Extraction

This script extracts detailed measurement data from a hierarchical folder structure.
It reads a source TSV file and expands each row into multiple rows, one per measurement.

Structure: Path -> Sample folders -> Measurement folders (e.g., 120_2_D)
Output: One row per measurement with extracted metadata
"""

import sys
import csv
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent))
from utils import extract_pressure_stretch_from_folder, is_analysis_folder


# Column indices in source TSV
COL_CLASSE = 0
COL_GROUP_NAME = 1
COL_PATH = 2
NUM_ORIGINAL_COLUMNS = 14
# Columns to exclude from output (Sample Number, 2-Photon, Pression (A), Axial stretch (B))
COLUMNS_TO_EXCLUDE = [8, 11, 12, 13]  # indices 8, 11, 12, 13


# Special cases that don't follow standard naming patterns
# Format: full_path -> (sample_name, pressure, stretch, orientation, replicate)
# Notes: 'top' and 'front' = Ventral (V), 'a'/'b' = replicate 1/2, 'A'/'B' = replicate 1/2
SPECIAL_CASES = {
    "C1039G/ATA/5993/5993_120": ("5993", 120, 2, None, None),
    "C1039_Het_1Y/ATA/130/120_front": ("130", 120, 2, "V", None),    # front=V
    "C1039_Het_1Y/ATA/130/120a_top": ("130", 120, 2, "D", "1"),      # top=V, a=1
    "C1039_Het_1Y/ATA/130/120b_top": ("130", 120, 2, "D", "2"),      # top=V, b=2
    "C1039_Het_1Y/ATA/130/80_front": ("130", 80, 2, "V", None),      # front=V
    "C1039_Het_1Y/ATA/130/80_top": ("130", 80, 2, "D", None),        # top=V
    "C1039_Het_1Y/ATA/136/120A_front": ("136", 120, 2, "V", "1"),    # front=V, A=1
    "C1039_Het_1Y/ATA/136/120B_front": ("136", 120, 2, "V", "2"),    # front=V, B=2
    "C1039_Het_1Y/ATA/136/120_top": ("136", 120, 2, "D", None),      # top=V
    "C1039_Het_1Y/ATA/136/80_front": ("136", 80, 2, "V", None),      # front=V
}


def parse_orientation_and_replicate(folder_name: str) -> Tuple[str, str]:
    """
    Parse orientation (D/V) and replicate number from folder name.
    
    Examples:
        120_2_D   -> orientation='D', replicate=''
        80_1_V2   -> orientation='V', replicate='2'
        10_2_1    -> orientation='V', replicate=''  # 1 = V
        10_2_2    -> orientation='D', replicate=''  # 2 = D
        50_3_D1   -> orientation='D', replicate='1'
        40_D      -> orientation='D', replicate=''  # No stretch
        80_V      -> orientation='V', replicate=''  # No stretch
    
    Orientation encoding:
        - 1 = V (Ventral)
        - 2 = D (Dorsal)
        - V1, V2, V3... = V with replicate number
        - D1, D2, D3... = D with replicate number
    
    Args:
        folder_name: Measurement folder name (e.g., "120_2_D" or "40_D")
    
    Returns:
        Tuple of (orientation, replicate)
    """
    orientation = ""
    replicate = ""
    
    parts = folder_name.split('_')
    
    # Format: PRESSURE_ORIENTATION (e.g., 40_D, 80_V)
    if len(parts) == 2:
        second_part = parts[1]
        if second_part:
            first_char = second_part[0]
            
            # Case 1: Starts with D or V (e.g., D, V, D1, V2)
            if first_char in ['D', 'V']:
                orientation = first_char
                # Check if there's a replicate number after (e.g., D1, V2)
                if len(second_part) > 1 and second_part[1:].isdigit():
                    replicate = second_part[1:]
    
    # Format: PRESSURE_STRETCH_ORIENTATION (e.g., 120_2_D, 10_2_1)
    elif len(parts) >= 3:
        third_part = parts[2]
        if third_part:
            first_char = third_part[0]
            
            # Case 1: Starts with D or V (e.g., D, V, D1, V2)
            if first_char in ['D', 'V']:
                orientation = first_char
                # Check if there's a replicate number after (e.g., D1, V2)
                if len(third_part) > 1 and third_part[1:].isdigit():
                    replicate = third_part[1:]
            
            # Case 2: Numeric code for orientation
            elif third_part == '1':
                orientation = 'V'  # 1 = Ventral
            elif third_part == '2':
                orientation = 'D'  # 2 = Dorsal
            
            # Case 3: Other digits (edge case, treat as unknown)
            elif first_char.isdigit() and len(third_part) > 1:
                # Could be a replicate without clear orientation
                replicate = third_part
    
    return orientation, replicate


def create_measurement_row(base_row: List[str], detailed_path: str, sample_name: str,
                           pressure: Optional[int], stretch: Optional[int],
                           orientation: str, replicate: str) -> List[str]:
    """
    Create a new row for a single measurement.
    
    Args:
        base_row: Original row from source TSV (first 14 columns)
        detailed_path: Full path to measurement folder
        sample_name: Name of the sample
        pressure: Pressure value
        stretch: Axial stretch value
        orientation: D (Dorsal) or V (Ventral)
        replicate: Replicate number
    
    Returns:
        Complete row with original columns (excluding specified columns) + new columns
    """
    # Copy first 14 columns from original row
    full_row = base_row[:NUM_ORIGINAL_COLUMNS].copy()
    
    # Ensure we have exactly 14 columns
    while len(full_row) < NUM_ORIGINAL_COLUMNS:
        full_row.append("null")
    
    # Normalize "SAINS" to "SAIN" in Classe column
    if len(full_row) > COL_CLASSE and full_row[COL_CLASSE] == "SAINS":
        full_row[COL_CLASSE] = "SAIN"
    
    # Replace Path column with detailed path
    full_row[COL_PATH] = detailed_path
    
    # Filter out excluded columns and replace empty strings with "null"
    new_row = [col if col and col.strip() else "null" 
               for idx, col in enumerate(full_row) if idx not in COLUMNS_TO_EXCLUDE]
    
    # Add new columns: Sample, Pressure, Axial Stretch, Orientation, Replicate
    new_row.extend([
        sample_name if sample_name else "null",
        str(pressure) if pressure is not None else "null",
        str(stretch) if stretch is not None else "null",
        orientation if orientation else "null",
        replicate if replicate else "null"
    ])
    
    return new_row


def extract_measurements_from_path(base_path: Path, relative_path: str, 
                                   original_row: List[str]) -> List[List[str]]:
    """
    Extract all measurements from a given path.
    
    Navigates through: relative_path -> sample folders -> measurement folders
    For each valid measurement folder, creates one row.
    
    Args:
        base_path: Base directory path (e.g., /Volumes/DATA_CAVINATO_3)
        relative_path: Relative path from base (e.g., Maturation/AGE_P21)
        original_row: Original TSV row to use as template
    
    Returns:
        List of rows, one per valid measurement found
    """
    # Validate input
    if not relative_path or not relative_path.strip():
        return []
    
    full_path = base_path / relative_path
    if not full_path.exists() or not full_path.is_dir():
        return []
    
    measurement_rows = []
    
    try:
        # Level 1: Iterate through sample folders (e.g., P21_F5, P21_F6)
        for sample_folder in full_path.iterdir():
            if not sample_folder.is_dir():
                continue
            
            sample_name = sample_folder.name
            
            # Level 2: Iterate through measurement folders (e.g., 120_2_D, 80_1_V)
            try:
                for measurement_folder in sample_folder.iterdir():
                    if not measurement_folder.is_dir():
                        continue
                    
                    measurement_folder_name = measurement_folder.name
                    
                    # Skip non-data folders (e.g., Collagen, Elastin, etc.)
                    if is_analysis_folder(measurement_folder_name):
                        continue
                    
                    # Build complete path
                    detailed_path = f"{relative_path}/{sample_name}/{measurement_folder_name}"
                    
                    # Check if this is a special case with manual values
                    if detailed_path in SPECIAL_CASES:
                        special_sample, pressure, stretch, orientation, replicate = SPECIAL_CASES[detailed_path]
                        final_sample_name = special_sample
                    else:
                        # Extract pressure and stretch from folder name
                        pressure, stretch = extract_pressure_stretch_from_folder(measurement_folder_name)
                        
                        # Parse orientation and replicate
                        orientation, replicate = parse_orientation_and_replicate(measurement_folder_name)
                        final_sample_name = sample_name
                    
                    # Default pressure to 80 if None
                    if pressure is None:
                        pressure = 80
                    
                    # Default axial stretch to 2 if None
                    if stretch is None:
                        stretch = 2
                    
                    # Create measurement row
                    measurement_row = create_measurement_row(
                        base_row=original_row,
                        detailed_path=detailed_path,
                        sample_name=final_sample_name,
                        pressure=pressure,
                        stretch=stretch,
                        orientation=orientation,
                        replicate=replicate
                    )
                    
                    measurement_rows.append(measurement_row)
            
            except (PermissionError, OSError):
                # Skip folders we can't access
                continue
    
    except (PermissionError, OSError):
        # Skip if we can't access the main path
        pass
    
    return measurement_rows


def process_database(base_path: Path, input_tsv_path: Path, output_tsv_path: Path):
    """
    Process entire database and generate detailed output TSV.
    
    Reads source TSV, extracts measurements for each row, and writes output.
    
    Args:
        base_path: Base directory containing all data folders
        input_tsv_path: Path to source TSV file
        output_tsv_path: Path to output TSV file
    """
    output_rows = []
    stats = {
        'rows_processed': 0,
        'rows_with_empty_path': 0,
        'measurements_extracted': 0
    }
    
    # Print header
    print(f"\n{'='*100}")
    print(f"TWO-PHOTON DATA EXTRACTION")
    print(f"{'='*100}")
    print(f"  Database: {input_tsv_path.name}")
    print(f"  Base path: {base_path}")
    print(f"{'='*100}\n")
    
    # Read and process TSV
    with open(input_tsv_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        
        # Read header and add new columns
        original_header = next(reader)
        # Filter out excluded columns from header
        filtered_header = [col for idx, col in enumerate(original_header[:NUM_ORIGINAL_COLUMNS]) 
                          if idx not in COLUMNS_TO_EXCLUDE]
        extended_header = filtered_header + [
            'Sample',
            'Pressure',
            'Axial Stretch',
            'Orientation',
            'Replicate'
        ]
        output_rows.append(extended_header)
        
        # Process each row
        for row_number, row in enumerate(reader, start=2):
            stats['rows_processed'] += 1
            
            # Ensure row has enough columns
            while len(row) < NUM_ORIGINAL_COLUMNS:
                row.append("")
            
            # Extract path and group name for logging
            path = row[COL_PATH].strip() if len(row) > COL_PATH else ""
            group_name = row[COL_GROUP_NAME].strip() if len(row) > COL_GROUP_NAME else ""
            
            # Skip rows with empty path
            if not path:
                stats['rows_with_empty_path'] += 1
                continue
            
            # Extract all measurements for this row
            measurements = extract_measurements_from_path(base_path, path, row)
            
            if measurements:
                output_rows.extend(measurements)
                stats['measurements_extracted'] += len(measurements)
                print(f"  [OK]   Row {row_number:3d} | {group_name:30s} | {path:30s} -> {len(measurements):3d} measurements")
            else:
                print(f"  [SKIP] Row {row_number:3d} | {group_name:30s} | {path:30s} -> No data found")
    
    # Write output file
    with open(output_tsv_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(output_rows)
    
    # Print summary
    print(f"\n{'='*100}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*100}")
    print(f"  Rows processed:         {stats['rows_processed']}")
    print(f"  Rows with empty path:   {stats['rows_with_empty_path']}")
    print(f"  Measurements extracted: {stats['measurements_extracted']}")
    print(f"\n  Output file: {output_tsv_path}")
    print(f"{'='*100}\n")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Extract detailed measurement data from TSV database and folder structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python extract_data.py \\
      --input backups/database_original.tsv \\
      --base-path /path/to/ds_snapshot_2026-01-11 \\
      --output data_intermediate/database_extracted.tsv
  
  # Use default paths from project root
  python extract_data.py \\
      --input Database.tsv \\
      --base-path ds_snapshot_2026-01-11 \\
      --output output/Database_Extracted.tsv
"""
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input TSV file (e.g., backups/database_original.tsv)"
    )
    
    parser.add_argument(
        "--base-path", "-b",
        type=str,
        required=True,
        help="Base path to folder containing image data (e.g., /path/to/ds_snapshot_2026-01-11)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output TSV file (e.g., data_intermediate/database_extracted.tsv)"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    input_tsv_path = Path(args.input).resolve()
    base_path = Path(args.base_path).resolve()
    output_tsv_path = Path(args.output).resolve()
    
    # Create output directory if needed
    output_tsv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Validate input paths exist
    if not base_path.exists():
        print(f"\nERROR: Base path not found: {base_path}")
        print("Please check that the path exists and is accessible.")
        sys.exit(1)
    
    if not input_tsv_path.exists():
        print(f"\nERROR: Input TSV not found: {input_tsv_path}")
        print("Please check that the file exists.")
        sys.exit(1)
    
    # Display configuration
    print("=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"Input TSV:   {input_tsv_path}")
    print(f"Base path:   {base_path}")
    print(f"Output TSV:  {output_tsv_path}")
    print("=" * 80)
    print()
    
    # Run extraction
    process_database(base_path, input_tsv_path, output_tsv_path)


if __name__ == "__main__":
    main()

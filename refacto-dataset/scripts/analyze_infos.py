#!/usr/bin/env python3
"""
Script to analyze and summarize the values in the 'infos' section of dataset.json
"""

import json
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path


def analyze_dataset_infos(json_path: str, output_path: str = None):
    """Analyze infos section from dataset.json and provide summary statistics
    
    Args:
        json_path: Path to JSON file
        output_path: Path to output file (optional, prints to console if None)
    """
    
    # Load JSON
    print(f"> Chargement de {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Columns to exclude from analysis
    excluded_columns = {'Path', 'Mechanics Passive', 'Mechanics Active', 'Sample'}
    
    # Collect all values for each column
    column_values = defaultdict(list)
    total_stacks = 0
    stacks_with_infos = 0
    
    if 'stacks' in dataset:
        total_stacks = len(dataset['stacks'])
        print(f"OK Dataset contient {total_stacks} stacks\n")
        
        for stack in dataset['stacks']:
            if 'infos' in stack:
                stacks_with_infos += 1
                for key, value in stack['infos'].items():
                    if key not in excluded_columns:
                        column_values[key].append(value)
    
    # Build report
    report_lines = []
    report_lines.append("ANALYSE DES DONNÉES - DATASET.JSON")
    report_lines.append("=" * 80 + "\n")
    report_lines.append(f"Total de stacks: {total_stacks}")
    report_lines.append(f"Stacks avec infos: {stacks_with_infos}")
    report_lines.append("=" * 80)
    
    # Analyze each column
    for column_name in sorted(column_values.keys()):
        values = column_values[column_name]
        value_counts = Counter(values)
        total_count = len(values)
        
        report_lines.append(f"\n  Colonne: {column_name}")
        report_lines.append("-" * 80)
        report_lines.append(f"Total d'entrées: {total_count}")
        report_lines.append(f"Valeurs uniques: {len(value_counts)}")
        report_lines.append(f"\nDistribution des valeurs:")
        
        # Sort by count (descending) then by value
        sorted_items = sorted(value_counts.items(), 
                            key=lambda x: (-x[1], str(x[0])))
        
        for value, count in sorted_items:
            percentage = (count / total_count) * 100
            # Display value (truncate if too long)
            display_value = str(value)
            if len(display_value) > 50:
                display_value = display_value[:47] + "..."
            
            report_lines.append(f"  {display_value:50s} : {count:4d} ({percentage:5.1f}%)")
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append("OK Analyse terminée!")
    
    report_text = "\n".join(report_lines)
    
    # Output to file or console
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as out:
            out.write(report_text)
        print(f"OK Résultats écrits dans: {output_path}")
    else:
        print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and summarize values in dataset.json 'infos' section",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Save to file
  python analyze_infos.py \\
      --input data_final/dataset_enriched_FINAL.json \\
      --output reports/infos_summary.txt
  
  # Print to console only
  python analyze_infos.py \\
      --input data_final/dataset_enriched_FINAL.json
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
        default=None,
        help="Path to output report file (optional, prints to console if not specified)"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    json_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve() if args.output else None
    
    # Validate input exists
    if not json_path.exists():
        print(f"\nERROR Erreur: Le fichier {json_path} n'existe pas!")
        sys.exit(1)
    
    # Create output directory if needed
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("ANALYSE DES INFORMATIONS")
    print("=" * 60)
    print(f"Input JSON: {json_path}")
    if output_path:
        print(f"Output:     {output_path}")
    else:
        print(f"Output:     Console (stdout)")
    print("=" * 60 + "\n")
    
    analyze_dataset_infos(json_path, output_path)


if __name__ == "__main__":
    main()

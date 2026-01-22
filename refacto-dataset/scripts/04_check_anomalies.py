#!/usr/bin/env python3
"""
Script to identify naming anomalies and typos in folder structure.
"""

import sys
import re
import unicodedata
import argparse
import csv
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Optional

# Import extraction functions to check formats
sys.path.insert(0, str(Path(__file__).parent))
from utils import extract_pressure_stretch_from_folder, is_analysis_folder


def has_unicode_accent(text: str) -> bool:
    """Check if text contains accented characters."""
    normalized = unicodedata.normalize('NFD', text)
    return any(unicodedata.category(char) == 'Mn' for char in normalized)


def has_extra_spaces(text: str) -> bool:
    """Check if text has extra/unusual spaces."""
    return '  ' in text or text != text.strip() or ' _' in text or '_ ' in text


def check_folder_name_format(name: str) -> List[str]:
    """
    Check for various anomalies in folder name.
    Returns list of issues found.
    """
    issues = []
    
    # Check for accented characters
    if has_unicode_accent(name):
        issues.append(f"ACCENT: Contains accented character(s)")
    
    # Check for extra spaces
    if has_extra_spaces(name):
        issues.append(f"SPACE: Extra or misplaced spaces")
    
    # Check for unusual characters
    if re.search(r'[^a-zA-Z0-9_\-/.]', name):
        unusual = re.findall(r'[^a-zA-Z0-9_\-/.]', name)
        issues.append(f"CHAR: Unusual characters: {set(unusual)}")
    
    # Check for inconsistent naming
    # Note: D2, V2, D3, V3 etc are VALID variants (multiple versions of same pressure-stretch-orientation)
    # So we DON'T flag them as issues anymore
    
    # Check for mixed case in orientation (should be uppercase D/V)
    if re.search(r'_[dv]$', name) or re.search(r'_[dv]\d', name):
        issues.append(f"CASE: Lowercase orientation (should be uppercase D/V)")
    
    return issues


def check_subfolder_format(name: str) -> Optional[str]:
    """
    Check if subfolder has a valid data format.
    Returns issue description if format is invalid, None otherwise.
    """
    # Check if it's an analysis folder (not data)
    if is_analysis_folder(name):
        return f"ANALYSIS: '{name}' is an analysis folder (Collagen, Elastin, etc.)"
    
    # Try to extract pressure/stretch
    pressure, stretch = extract_pressure_stretch_from_folder(name)
    
    # If nothing extracted, it's an invalid format
    if pressure is None:
        # Skip if it's a known non-data folder pattern
        if re.match(r'^\d{8}_', name):  # Date-based folder
            return None
        if name.lower() in ['tmp', 'temp', 'backup', 'old']:
            return None
        
        return f"INVALID_FORMAT: '{name}' doesn't match any accepted format"
    
    return None


def scan_directory_structure(base_path: Path, path_list: List[str]) -> dict:
    """
    Scan all directories in the given paths and find anomalies.
    """
    anomalies = {
        'accents': [],
        'spaces': [],
        'unusual_chars': [],
        'case_issues': [],
        'invalid_formats': [],  # New: subfolders with invalid/non-data formats
        'other': []
    }
    
    for relative_path in path_list:
        if not relative_path or not relative_path.strip():
            continue
        
        full_path = base_path / relative_path
        
        if not full_path.exists():
            continue
        
        # Check the path itself
        path_issues = check_folder_name_format(relative_path)
        if path_issues:
            for issue in path_issues:
                if 'ACCENT' in issue:
                    anomalies['accents'].append((relative_path, None, issue))
                elif 'SPACE' in issue:
                    anomalies['spaces'].append((relative_path, None, issue))
                elif 'CHAR' in issue:
                    anomalies['unusual_chars'].append((relative_path, None, issue))
                elif 'CASE' in issue:
                    anomalies['case_issues'].append((relative_path, None, issue))
                else:
                    anomalies['other'].append((relative_path, None, issue))
        
        # Check sample folders
        try:
            for sample_folder in full_path.iterdir():
                if not sample_folder.is_dir():
                    continue
                
                sample_issues = check_folder_name_format(sample_folder.name)
                if sample_issues:
                    for issue in sample_issues:
                        if 'ACCENT' in issue:
                            anomalies['accents'].append((relative_path, sample_folder.name, issue))
                        elif 'SPACE' in issue:
                            anomalies['spaces'].append((relative_path, sample_folder.name, issue))
                        elif 'CHAR' in issue:
                            anomalies['unusual_chars'].append((relative_path, sample_folder.name, issue))
                        elif 'CASE' in issue:
                            anomalies['case_issues'].append((relative_path, sample_folder.name, issue))
                        else:
                            anomalies['other'].append((relative_path, sample_folder.name, issue))
                
                # Check subfolders (data folders)
                try:
                    for subfolder in sample_folder.iterdir():
                        if not subfolder.is_dir():
                            continue
                        
                        # Check naming anomalies
                        sub_issues = check_folder_name_format(subfolder.name)
                        if sub_issues:
                            for issue in sub_issues:
                                folder_path = f"{relative_path}/{sample_folder.name}/{subfolder.name}"
                                if 'ACCENT' in issue:
                                    anomalies['accents'].append((folder_path, None, issue))
                                elif 'SPACE' in issue:
                                    anomalies['spaces'].append((folder_path, None, issue))
                                elif 'CHAR' in issue:
                                    anomalies['unusual_chars'].append((folder_path, None, issue))
                                elif 'CASE' in issue:
                                    anomalies['case_issues'].append((folder_path, None, issue))
                                else:
                                    anomalies['other'].append((folder_path, None, issue))
                        
                        # Check format validity (NEW)
                        format_issue = check_subfolder_format(subfolder.name)
                        if format_issue:
                            folder_path = f"{relative_path}/{sample_folder.name}"
                            anomalies['invalid_formats'].append((folder_path, subfolder.name, format_issue))
                
                except (PermissionError, OSError):
                    continue
        
        except (PermissionError, OSError):
            continue
    
    return anomalies


def print_report(anomalies: dict, output_file: Path = None):
    """Print and save anomalies report (optional file output)."""
    
    total = sum(len(v) for v in anomalies.values())
    
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("RAPPORT DES ANOMALIES DANS LES NOMS DE DOSSIERS")
    report_lines.append("=" * 100)
    report_lines.append(f"\nTotal d'anomalies trouvées: {total}\n")
    
    # Accents
    if anomalies['accents']:
        report_lines.append("=" * 100)
        report_lines.append(f"[ACCENTS] CARACTÈRES ACCENTUÉS ({len(anomalies['accents'])} occurrences)")
        report_lines.append("=" * 100)
        report_lines.append("Ces caractères peuvent causer des problèmes d'encodage\n")
        
        for path, subfolder, issue in anomalies['accents']:
            if subfolder:
                report_lines.append(f"> {path}")
                report_lines.append(f"   └─ {subfolder}")
                # Show what should be corrected
                clean = unicodedata.normalize('NFKD', subfolder).encode('ASCII', 'ignore').decode('ASCII')
                if clean != subfolder:
                    report_lines.append(f"   >  Correction suggérée: {clean}")
            else:
                report_lines.append(f"> {path}")
                # Extract the problematic part
                parts = path.split('/')
                for part in parts:
                    if has_unicode_accent(part):
                        clean = unicodedata.normalize('NFKD', part).encode('ASCII', 'ignore').decode('ASCII')
                        if clean != part:
                            report_lines.append(f"   >  '{part}' → '{clean}'")
            report_lines.append("")
    
    # Spaces
    if anomalies['spaces']:
        report_lines.append("=" * 100)
        report_lines.append(f"[SPACES] ESPACES PROBLÉMATIQUES ({len(anomalies['spaces'])} occurrences)")
        report_lines.append("=" * 100)
        report_lines.append("Espaces en trop, doubles espaces ou mal placés\n")
        
        for path, subfolder, issue in anomalies['spaces']:
            display_name = subfolder if subfolder else path.split('/')[-1]
            report_lines.append(f"> {path}")
            if subfolder:
                report_lines.append(f"   └─ '{subfolder}'")
            else:
                report_lines.append(f"   └─ '{display_name}'")
            
            # Show spaces visually
            visible = display_name.replace(' ', '·')
            report_lines.append(f"   >  Visible: '{visible}'")
            
            # Suggest correction
            clean = ' '.join(display_name.split())
            if clean != display_name:
                report_lines.append(f"   >  Correction: '{clean}'")
            report_lines.append("")
    
    # Unusual characters
    if anomalies['unusual_chars']:
        report_lines.append("=" * 100)
        report_lines.append(f"WARNING  CARACTÈRES INHABITUELS ({len(anomalies['unusual_chars'])} occurrences)")
        report_lines.append("=" * 100)
        report_lines.append("Caractères spéciaux qui pourraient poser problème\n")
        
        for path, subfolder, issue in anomalies['unusual_chars']:
            report_lines.append(f"> {path}")
            if subfolder:
                report_lines.append(f"   └─ {subfolder}")
            report_lines.append(f"   WARNING  {issue}")
            report_lines.append("")
    
    # Case issues
    if anomalies['case_issues']:
        report_lines.append("=" * 100)
        report_lines.append(f"[CASE] PROBLÈMES DE CASSE ({len(anomalies['case_issues'])} occurrences)")
        report_lines.append("=" * 100)
        report_lines.append("Majuscules/minuscules incohérentes\n")
        
        for path, subfolder, issue in anomalies['case_issues']:
            report_lines.append(f"> {path}")
            if subfolder:
                report_lines.append(f"   └─ {subfolder}")
            report_lines.append(f"   📝 {issue}")
            report_lines.append("")
    
    # Invalid formats (NEW)
    if anomalies['invalid_formats']:
        report_lines.append("=" * 100)
        report_lines.append(f"[INVALID] FORMATS NON-STANDARD ({len(anomalies['invalid_formats'])} occurrences)")
        report_lines.append("=" * 100)
        report_lines.append("Sous-dossiers qui ne suivent pas le format de données attendu\n")
        
        # Group by type
        by_type = defaultdict(list)
        for path, subfolder, issue in anomalies['invalid_formats']:
            issue_type = issue.split(':')[0]
            by_type[issue_type].append((path, subfolder, issue))
        
        # Analysis folders
        if 'ANALYSIS' in by_type:
            report_lines.append(f"  Dossiers d'analyse ({len(by_type['ANALYSIS'])} occurrences)")
            report_lines.append("   Ces dossiers contiennent des résultats d'analyse, pas des données brutes\n")
            # Group by path to avoid repetition
            by_path = defaultdict(list)
            for path, subfolder, issue in by_type['ANALYSIS']:  # NO LIMIT
                by_path[path].append(subfolder)
            
            for path, subfolders in by_path.items():  # SHOW ALL
                report_lines.append(f"   > {path}")
                report_lines.append(f"      Dossiers d'analyse: {', '.join(subfolders)}")
                report_lines.append("")
        
        # Invalid formats
        if 'INVALID_FORMAT' in by_type:
            report_lines.append(f"ERROR Formats invalides ({len(by_type['INVALID_FORMAT'])} occurrences)")
            report_lines.append("   Ces dossiers n'ont pas de format reconnu (ni données, ni analyse)\n")
            for path, subfolder, issue in by_type['INVALID_FORMAT']:  # NO LIMIT - SHOW ALL
                report_lines.append(f"   > {path}")
                report_lines.append(f"      └─ {subfolder}")
                report_lines.append("")
    
    # Other issues
    if anomalies['other']:
        report_lines.append("=" * 100)
        report_lines.append(f"[OTHER] AUTRES ANOMALIES ({len(anomalies['other'])} occurrences)")
        report_lines.append("=" * 100)
        report_lines.append("")
        
        for path, subfolder, issue in anomalies['other']:
            report_lines.append(f"> {path}")
            if subfolder:
                report_lines.append(f"   └─ {subfolder}")
            report_lines.append(f"   [OTHER] {issue}")
            report_lines.append("")
    
    if total == 0:
        report_lines.append("\nOK Aucune anomalie trouvée ! Tous les noms de dossiers sont propres.\n")
    
    report_lines.append("=" * 100)
    
    # Print to console
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # Save to file (if output_file is specified)
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n> Rapport sauvegardé: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check for naming anomalies in folder structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python check_anomalies.py \\
      --base-path /path/to/ds_snapshot_2026-01-11 \\
      --tsv backups/database_original.tsv \\
      --output reports/anomalies_report.txt
  
  # Skip output file (print to console only)
  python check_anomalies.py \\
      --base-path /path/to/ds_snapshot_2026-01-11 \\
      --tsv backups/database_original.tsv
"""
    )
    
    parser.add_argument(
        "--base-path", "-b",
        type=str,
        required=True,
        help="Base path to folder containing image data"
    )
    
    parser.add_argument(
        "--tsv", "-t",
        type=str,
        required=True,
        help="Path to TSV file with paths to check (e.g., Database.tsv)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to output report file (optional)"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    base_path = Path(args.base_path).resolve()
    tsv_file = Path(args.tsv).resolve()
    
    # Validate paths exist
    if not base_path.exists():
        print(f"\nERROR Error: Base path does not exist: {base_path}")
        sys.exit(1)
    
    if not tsv_file.exists():
        print(f"\nERROR Error: TSV file not found: {tsv_file}")
        sys.exit(1)
    
    # Read paths from TSV
    paths = []
    with open(tsv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # Skip header
        for row in reader:
            if len(row) > 2 and row[2].strip():
                paths.append(row[2].strip())
    
    print(f"\n> Scanning {len(paths)} paths for anomalies...")
    print(f"Base path: {base_path}\n")
    
    # Scan for anomalies
    anomalies = scan_directory_structure(base_path, paths)
    
    # Generate report
    if args.output:
        output_file = Path(args.output).resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        print_report(anomalies, output_file)
    else:
        print_report(anomalies, None)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Main entry point for analyzing training results."""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import analysis modules
from scripts.analytics.io import load_results
from scripts.analytics.analysis import analyze_positions, analyze_volumes, analyze_combined
from scripts.analytics.visualizations import plot_position_heatmap, plot_position_bar_chart, plot_volume_analysis
from scripts.analytics.report import save_text_report


def main():
    """Main function to run analysis."""
    parser = argparse.ArgumentParser(
        description='Analyze training results to identify problematic positions and volumes'
    )
    parser.add_argument(
        'results_json',
        type=str,
        help='Path to training_results.json file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save analytics (default: results_json parent / analytics)'
    )
    
    args = parser.parse_args()
    
    results_path = Path(args.results_json)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return 1
    
    # Output directory: use provided or default to subdirectory next to results_json
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_path.parent / 'analytics'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from: {results_path}")
    results = load_results(results_path)
    
    print("Analyzing positions...")
    position_analysis = analyze_positions(results)
    
    print("Analyzing volumes...")
    volume_analysis = analyze_volumes(results)
    
    print("Analyzing combined (volume, position)...")
    combined_analysis = analyze_combined(results)
    
    print("Generating visualizations...")
    plot_position_heatmap(position_analysis, output_dir)
    plot_position_bar_chart(position_analysis, output_dir)
    plot_volume_analysis(volume_analysis, output_dir)
    
    print("Saving text report...")
    save_text_report(position_analysis, volume_analysis, combined_analysis, output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {output_dir}")
    print("Generated files:")
    print("  - analytics/position_analysis_heatmap.png")
    print("  - analytics/position_analysis_barchart.png")
    print("  - analytics/volume_analysis.png")
    print("  - analytics/analysis_report.txt")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

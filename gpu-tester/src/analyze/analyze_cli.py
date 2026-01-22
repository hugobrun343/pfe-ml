#!/usr/bin/env python3
"""
CLI for analyzing grid search results
"""

import sys
import argparse
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import from analyze package (relative import within src)
from analyze.analyzer import ResultsAnalyzer
from analyze.data_loader import get_model_families


def main():
    parser = argparse.ArgumentParser(
        description="Analyze grid search results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all models
  python -m src.analyze.analyze_cli results/test_results.csv
  
  # Analyze only ResNet3D models
  python -m src.analyze.analyze_cli results/test_results.csv --family ResNet3D
  
  # Analyze and save to custom directory
  python -m src.analyze.analyze_cli results/test_results.csv --output my_analysis --family SE-ResNet3D
        """
    )
    
    parser.add_argument(
        'results_file',
        type=str,
        help='Path to results file (CSV or JSON)'
    )
    
    parser.add_argument(
        '--family',
        type=str,
        default=None,
        choices=list(get_model_families().keys()) + [None],
        help='Model family to filter (optional)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for analysis results (default: results/analysis or results/analysis_{family})'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip creating visualizations'
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output is None:
        if args.family:
            args.output = f"results/analysis_{args.family.lower()}"
        else:
            args.output = "results/analysis"
    
    try:
        analyzer = ResultsAnalyzer(args.results_file)
        
        if args.family:
            analyzer.filter_by_family(args.family)
        
        analyzer.analyze(
            output_dir=args.output,
            create_plots=not args.no_plots
        )
        
    except Exception as e:
        print(f"\nERREUR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

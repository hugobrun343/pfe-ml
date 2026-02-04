#!/usr/bin/env python3
"""
CLI for analyzing grid search results.
Run from project root: python scripts/analyze_results.py results/grid_search_results.json
"""

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import argparse
from analyze.analyzer import ResultsAnalyzer
from analyze.data_loader import get_model_families


def main():
    parser = argparse.ArgumentParser(
        description="Analyze grid search results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/analyze_results.py results/grid_search_results.json
  python scripts/analyze_results.py results/grid_search_results.csv --family ResNet3D
  python scripts/analyze_results.py results/grid_search_results.json --output my_analysis --no-plots
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
        help='Output directory for analysis (default: results/analysis or results/analysis_{family})'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip creating visualizations'
    )

    args = parser.parse_args()

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

#!/usr/bin/env python3
"""
Main entry point for VRAM Grid Search
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from grid_search_runner import run_grid_search
from utils import get_gpu_info, print_banner, load_config


def cmd_run(args):
    """Run grid search"""
    print_banner("STARTING GRID SEARCH", width=80)
    
    # Check GPU
    gpu_info = get_gpu_info()
    if not gpu_info['available']:
        print(f"ERROR: {gpu_info['error']}")
        sys.exit(1)
    
    print("GPU Information:")
    print(f"  Device: {gpu_info['device_name']}")
    print(f"  Total VRAM: {gpu_info['total_memory_gb']:.2f} GB")
    print()
    
    # Run grid search
    try:
        results = run_grid_search(
            config_path=args.config,
            output_dir=args.output,
            max_tests=args.max_tests,
            resume=not args.no_resume
        )
        
        print(f"\nGrid search completed! {len(results)} total results")
        print(f"Results saved to: {args.output}/")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_info(args):
    """Show configuration info"""
    print_banner("CONFIGURATION INFO", width=80)
    
    try:
        config = load_config(args.config)
        
        print("\n=== GPU ===")
        print(f"GPU: {config['gpu']['name']}")
        print(f"VRAM: {config['gpu']['vram_gb']} GB")
        
        print("\n=== GRID SEARCH SPACE ===")
        grid = config['grid_search']
        print(f"Spatial resolutions: {len(grid['spatial_resolutions'])} values")
        print(f"Depth sizes: {len(grid['depth_sizes'])} values")
        print(f"Batch sizes: {len(grid['batch_sizes'])} values")
        print(f"Models: {len(grid['models'])} architectures")
        
        total = (
            len(grid['spatial_resolutions']) *
            len(grid['depth_sizes']) *
            len(grid['batch_sizes']) *
            len(grid['models'])
        )
        print(f"\nTotal combinations: {total:,}")
        
        print("\n=== GPU STATUS ===")
        gpu_info = get_gpu_info()
        if gpu_info['available']:
            print(f"GPU Available: {gpu_info['device_name']}")
            print(f"Total VRAM: {gpu_info['total_memory_gb']:.2f} GB")
        else:
            print(f"GPU Not Available: {gpu_info['error']}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="VRAM Grid Search")
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/grid_search_config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Run command
    parser_run = subparsers.add_parser('run', help='Run grid search')
    parser_run.add_argument('--max-tests', type=int, default=None, help='Max tests to run')
    parser_run.add_argument('--no-resume', action='store_true', help='Start from scratch')
    parser_run.set_defaults(func=cmd_run)
    
    # Info command
    parser_info = subparsers.add_parser('info', help='Show configuration info')
    parser_info.set_defaults(func=cmd_info)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()

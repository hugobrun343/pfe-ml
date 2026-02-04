"""
Grid Search Runner - Orchestrates the complete grid search process
Tests all combinations of hyperparameters systematically
"""

import time
from itertools import product
from typing import Dict, List, Any, Optional
from pathlib import Path

from vram_tester import VRAMTester
from utils import (
    load_config,
    setup_logger,
    save_results_json,
    save_results_csv,
    get_gpu_info,
    estimate_total_combinations,
    create_result_entry,
    print_banner,
    format_bytes,
    wait_for_gpu_ready,
    get_model_size_score
)


class GridSearchRunner:
    """
    Orchestrates the complete grid search process
    Tests all combinations and saves results progressively
    """
    
    def __init__(
        self,
        config_path: str = "grid_search_config.yaml",
        output_dir: str = "results",
        resume_from_checkpoint: bool = True
    ):
        """
        Initialize Grid Search Runner
        
        Args:
            config_path: Path to configuration file
            output_dir: Directory for output files
            resume_from_checkpoint: Whether to resume from previous results
        """
        self.config = load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        log_file = self.output_dir / self.config['output']['log_file']
        self.logger = setup_logger(str(log_file))
        
        # Initialize VRAM tester
        sim_config = self.config['simulation']
        self.tester = VRAMTester(
            device="cuda",
            num_warmup_iterations=sim_config['num_warmup_iterations'],
            num_test_iterations=sim_config['num_test_iterations'],
            clear_cache_between_tests=sim_config['clear_cache_between_tests']
        )
        
        # Results storage
        self.results: List[Dict] = []
        self.resume_from_checkpoint = resume_from_checkpoint
        
        # Statistics (initialize BEFORE loading results)
        self.stats = {
            "total_tests": 0,
            "successful_tests": 0,
            "failed_tests": 0,
            "oom_tests": 0,
            "start_time": None,
            "end_time": None
        }
        
        # GPU health monitoring
        self.consecutive_cuda_errors = 0
        # Get max consecutive CUDA errors from config (default: 10)
        self.max_consecutive_cuda_errors = self.config.get('safety', {}).get('max_consecutive_cuda_errors', 10)
        
        # Failed configurations cache (for smart skipping)
        # Format: (model_name, batch_size, depth, spatial_res) -> True if failed
        self.failed_configs: set = set()
        
        # Load existing results if resuming
        if resume_from_checkpoint:
            self._load_existing_results()
            # Build failed configs cache from existing results
            self._build_failed_configs_cache()
    
    def _load_existing_results(self) -> None:
        """Load existing results from checkpoint file if it exists"""
        results_file = self.output_dir / self.config['output']['results_file']
        
        if results_file.exists():
            import json
            with open(results_file, 'r') as f:
                self.results = json.load(f)
            
            self.logger.info(f"Loaded {len(self.results)} existing results from checkpoint")
            
            # Update statistics
            self.stats['total_tests'] = len(self.results)
            self.stats['successful_tests'] = sum(1 for r in self.results if r['success'])
            self.stats['failed_tests'] = sum(1 for r in self.results if not r['success'])
            self.stats['oom_tests'] = sum(
                1 for r in self.results 
                if not r['success'] and r.get('error_message') == 'OOM'
            )
    
    def _build_failed_configs_cache(self) -> None:
        """Build cache of failed configurations from existing results"""
        for result in self.results:
            # Only cache OOM/CUDA failures (not invalid configs)
            if not result['success']:
                error_msg = str(result.get('error_message', '')).upper()
                is_oom_or_cuda = 'OOM' in error_msg or 'CUDA' in error_msg
                
                if is_oom_or_cuda:
                    key = (
                        result['model_name'],
                        result['batch_size'],
                        result['depth'],
                        result['spatial_resolution']
                    )
                    self.failed_configs.add(key)
        
        if self.failed_configs:
            self.logger.info(f"Built failed configs cache: {len(self.failed_configs)} failed combinations")
    
    def _get_model_family(self, model_name: str) -> str:
        """
        Get model family from model name
        
        Args:
            model_name: Name of model (e.g., "ResNet3D-10", "SE-ResNet3D-18")
            
        Returns:
            Model family (e.g., "ResNet3D", "SE-ResNet3D")
        """
        # Special case for SE-ResNet3D - family is "SE-ResNet3D"
        if model_name.startswith("SE-ResNet3D"):
            return "SE-ResNet3D"
        # For others, family is the part before the last "-"
        # ResNet3D-10 -> ResNet3D
        # EfficientNet3D-B0 -> EfficientNet3D
        # ViT3D-Tiny -> ViT3D
        # ConvNeXt3D-Tiny -> ConvNeXt3D
        if '-' in model_name:
            parts = model_name.rsplit('-', 1)
            return parts[0]
        return model_name
    
    def _should_skip_combination(
        self,
        model_name: str,
        batch_size: int,
        depth: int,
        spatial_res: int
    ) -> bool:
        """
        Check if combination should be skipped because a smaller one failed
        Only skips within the same model family
        
        Args:
            model_name: Name of model
            batch_size: Batch size
            depth: Depth dimension
            spatial_res: Spatial resolution
            
        Returns:
            True if should skip (smaller combination failed in same family)
        """
        model_family = self._get_model_family(model_name)
        model_score = get_model_size_score(model_name)
        
        # Check if a smaller combination failed - ONLY within same family
        for (failed_model, failed_batch, failed_depth, failed_spatial) in self.failed_configs:
            # Only compare models from the same family
            failed_model_family = self._get_model_family(failed_model)
            if model_family != failed_model_family:
                continue
            
            failed_model_score = get_model_size_score(failed_model)
            
            # Check if this model is >= failed model (same or bigger)
            # AND all params are >= failed params
            if (model_score >= failed_model_score and
                batch_size >= failed_batch and
                depth >= failed_depth and
                spatial_res >= failed_spatial):
                # Skip - a smaller combination already failed in same family
                return True
        
        return False
    
    def _is_already_tested(
        self,
        spatial_res: int,
        depth: int,
        batch_size: int,
        model_name: str
    ) -> bool:
        """Check if a configuration has already been tested"""
        for result in self.results:
            if (result['spatial_resolution'] == spatial_res and
                result['depth'] == depth and
                result['batch_size'] == batch_size and
                result['model_name'] == model_name):
                return True
        return False
    
    def run(self, max_tests: Optional[int] = None) -> List[Dict]:
        """
        Run the complete grid search
        
        Args:
            max_tests: Maximum number of tests to run (None for all)
            
        Returns:
            List of all test results
        """
        self.stats['start_time'] = time.time()
        
        # Print GPU info
        gpu_info = get_gpu_info()
        print_banner("VRAM GRID SEARCH - L40S", width=80)
        self.logger.info("GPU Information:")
        for key, value in gpu_info.items():
            self.logger.info(f"  {key}: {value}")
        
        # Generate all combinations
        combinations = self._generate_combinations()
        total_combinations = len(combinations)
        
        self.logger.info(f"\nTotal combinations to test: {total_combinations:,}")
        
        if max_tests:
            combinations = combinations[:max_tests]
            self.logger.info(f"Limited to {max_tests} tests for this run")
        
        # Filter out already tested combinations
        if self.resume_from_checkpoint:
            original_count = len(combinations)
            # New format: (model_name, spatial_res, depth, batch_size, model_size)
            combinations = [
                c for c in combinations
                if not self._is_already_tested(c[1], c[2], c[3], c[0])  # spatial, depth, batch, model
            ]
            skipped = original_count - len(combinations)
            if skipped > 0:
                self.logger.info(f"Skipping {skipped} already tested combinations")
        
        # Filter out combinations that should be skipped (smaller one failed within same family)
        original_count = len(combinations)
        combinations = [
            c for c in combinations
            if not self._should_skip_combination(c[0], c[3], c[2], c[1])  # model, batch, depth, spatial
        ]
        smart_skip_count = original_count - len(combinations)
        if smart_skip_count > 0:
            self.logger.info(f"Smart skip: {smart_skip_count} combinations skipped (smaller combination failed in same family)")
        
        if not combinations:
            self.logger.info("All combinations already tested!")
            return self.results
        
        self.logger.info(f"Testing {len(combinations)} new combinations\n")
        
        # Run tests
        # New format: (model_name, spatial_res, depth, batch_size, model_size)
        for idx, (model_name, spatial_res, depth, batch_size, model_size) in enumerate(combinations, 1):
            self._test_and_log(
                model_name, spatial_res, depth, batch_size, model_size,
                idx, len(combinations)
            )
            
            # Check GPU health - stop if too many consecutive CUDA errors
            if self.consecutive_cuda_errors >= self.max_consecutive_cuda_errors:
                self.logger.error(
                    f"\n{'='*80}\n"
                    f"GPU HEALTH ALERT: {self.consecutive_cuda_errors} consecutive CUDA errors detected!\n"
                    f"The GPU appears to be in a broken state.\n"
                    f"Please reset the GPU and restart the grid search.\n"
                    f"Results saved up to test {idx}.\n"
                    f"{'='*80}\n"
                )
                print(f"\n⚠️  GPU appears broken ({self.consecutive_cuda_errors} consecutive CUDA errors)")
                print("   Stopping grid search. Please reset GPU and restart.")
                break
            
            # Save checkpoint every 10 tests
            if idx % 10 == 0:
                self._save_checkpoint()
        
        # Final save
        self._save_checkpoint()
        
        # Print final statistics
        self._print_final_stats()
        
        self.stats['end_time'] = time.time()
        
        return self.results
    
    def _generate_combinations(self) -> List[tuple]:
        """
        Generate all combinations to test
        ORDER: Model first, then spatial_res, depth, batch_size (as requested)
        
        Returns:
            List of tuples (model_name, spatial_res, depth, batch_size, model_size)
        """
        grid = self.config['grid_search']
        
        spatial_resolutions = grid['spatial_resolutions']
        depth_sizes = grid['depth_sizes']
        batch_sizes = grid['batch_sizes']
        models = grid['models']
        
        combinations = []
        
        # Loop by MODEL FIRST (as requested)
        # Support both list and dict format for models
        if isinstance(models, list):
            # New format: list of model names
            for model_name in models:
                for spatial_res, depth, batch_size in product(
                    spatial_resolutions, depth_sizes, batch_sizes
                ):
                    combinations.append((
                        model_name, spatial_res, depth, batch_size, model_name
                    ))
        else:
            # Old format: dict with model_size -> model_info
            for model_size, model_info in models.items():
                model_name = model_info['name']
                for spatial_res, depth, batch_size in product(
                    spatial_resolutions, depth_sizes, batch_sizes
                ):
                    combinations.append((
                        model_name, spatial_res, depth, batch_size, model_size
                    ))
        
        return combinations
    
    def _test_and_log(
        self,
        model_name: str,
        spatial_res: int,
        depth: int,
        batch_size: int,
        model_size: str,
        test_idx: int,
        total_tests: int
    ) -> None:
        """
        Test a single configuration and log results
        
        Args:
            model_name: Model name
            spatial_res: Spatial resolution
            depth: Depth dimension
            batch_size: Batch size
            model_size: Model size category
            test_idx: Current test index
            total_tests: Total number of tests
        """
        # Check if should skip (double-check in case cache was updated)
        if self._should_skip_combination(model_name, batch_size, depth, spatial_res):
            model_family = self._get_model_family(model_name)
            self.logger.info(
                f"[{test_idx}/{total_tests}] "
                f"⏭ SKIPPED: {model_name} | {batch_size}x{depth}x{spatial_res}x{spatial_res} "
                f"(smaller combination failed in {model_family} family)"
            )
            # Create a failed result entry for skipped combinations
            skipped_result = {
                'success': False,
                'status': 'skipped',
                'error_message': 'SKIPPED_SMALLER_COMBINATION_FAILED',
                'vram_peak_bytes': None,
                'vram_used_bytes': None,
                'duration_seconds': 0
            }
            # Create result entry
            result = create_result_entry(
                spatial_res=spatial_res,
                depth=depth,
                batch_size=batch_size,
                model_name=model_name,
                model_size=model_size,
                success=skipped_result['success'],
                vram_used_bytes=skipped_result['vram_used_bytes'],
                vram_peak_bytes=skipped_result['vram_peak_bytes'],
                error_message=skipped_result['error_message'],
                duration_seconds=skipped_result['duration_seconds']
            )
            # Add to results
            self.results.append(result)
            # Update statistics (count as skipped, not failed)
            self.stats['total_tests'] += 1
            error_msg = result['error_message'] or 'Smaller combination failed'
            self.logger.info(f"  ⊘ SKIPPED - {error_msg}")
            # Print progress
            progress = (test_idx / total_tests) * 100
            skipped_count = sum(1 for r in self.results if 'SKIPPED' in str(r.get('error_message', '')))
            valid_tests = self.stats['total_tests'] - skipped_count
            if valid_tests > 0:
                success_rate = (self.stats['successful_tests'] / valid_tests) * 100
            else:
                success_rate = 0.0
            self.logger.info(
                f"  Progress: {progress:.1f}% | "
                f"Success rate: {success_rate:.1f}% "
                f"({self.stats['successful_tests']}/{valid_tests} valid, {skipped_count} skipped)\n"
            )
            return
        
        # Log test start
        config_str = (
            f"[{test_idx}/{total_tests}] "
            f"Testing: {model_name} | "
            f"{batch_size}x{depth}x{spatial_res}x{spatial_res}"
        )
        self.logger.info(config_str)
        
        # Run test with error recovery
        try:
            test_result = self.tester.test_configuration(
                spatial_res=spatial_res,
                depth=depth,
                batch_size=batch_size,
                model_name=model_name
            )
            
            # Reset consecutive CUDA errors on success
            if test_result['success']:
                self.consecutive_cuda_errors = 0
            elif 'CUDA' in str(test_result.get('error_message', '')) or 'OOM' in str(test_result.get('error_message', '')):
                self.consecutive_cuda_errors += 1
                # Brief wait for GPU to recover (short timeout - GPU usually recovers fast)
                self.logger.info("Waiting for GPU to recover...")
                gpu_ready = wait_for_gpu_ready(timeout_seconds=5.0, check_interval=0.5)
                if gpu_ready:
                    self.logger.info("✓ GPU recovered, continuing")
                # Always continue - let next test decide if GPU is really dead
            else:
                self.consecutive_cuda_errors = 0
            
            # Add failed configs to cache (for smart skipping)
            if not test_result['success']:
                error_msg = str(test_result.get('error_message', '')).upper()
                is_oom_or_cuda = 'OOM' in error_msg or 'CUDA' in error_msg
                
                if is_oom_or_cuda:
                    key = (model_name, batch_size, depth, spatial_res)
                    self.failed_configs.add(key)
                
        except Exception as e:
            # If test completely crashes, create a failure result
            import traceback
            error_str = str(e)
            is_cuda_error = 'CUDA' in error_str or 'cuda' in error_str.lower()
            
            # Get full traceback for CUDA errors
            if is_cuda_error:
                tb_str = traceback.format_exc()
                # Extract relevant lines (last 5 lines usually show the issue)
                tb_lines = tb_str.strip().split('\n')
                relevant_tb = '\n'.join(tb_lines[-10:])  # Last 10 lines
                error_msg = f"CUDA_ERROR: {error_str}\n{relevant_tb}"
            else:
                error_msg = f"CRITICAL_ERROR: {error_str}"
            
            test_result = {
                'success': False,
                'status': 'failed',
                'error_message': error_msg[:500],  # Limit length
                'vram_peak_bytes': 0,
                'vram_used_bytes': 0,
                'duration_seconds': 0
            }
            self.logger.error(f"Critical error during test: {e}")
            if is_cuda_error:
                self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
            
            # Track consecutive CUDA errors
            if is_cuda_error:
                self.consecutive_cuda_errors += 1
                # Brief wait for GPU to recover (short timeout - GPU usually recovers fast)
                self.logger.info("Waiting for GPU to recover...")
                gpu_ready = wait_for_gpu_ready(timeout_seconds=5.0, check_interval=0.5)
                if gpu_ready:
                    self.logger.info("✓ GPU recovered, continuing")
                # Always continue - let next test decide if GPU is really dead
            else:
                self.consecutive_cuda_errors = 0
            
            # Add failed configs to cache (for smart skipping) - also in exception case
            error_msg_upper = error_msg.upper()
            is_oom_or_cuda = 'OOM' in error_msg_upper or 'CUDA' in error_msg_upper
            if is_oom_or_cuda:
                key = (model_name, batch_size, depth, spatial_res)
                self.failed_configs.add(key)
            
            # Try to recover GPU (minimal cleanup)
            try:
                import torch
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except:
                        pass  # GPU might be broken
            except:
                pass
        
        # Create result entry
        result = create_result_entry(
            spatial_res=spatial_res,
            depth=depth,
            batch_size=batch_size,
            model_name=model_name,
            model_size=model_size,
            success=test_result['success'],
            vram_used_bytes=test_result['vram_used_bytes'],
            vram_peak_bytes=test_result['vram_peak_bytes'],
            error_message=test_result['error_message'],
            duration_seconds=test_result['duration_seconds']
        )
        
        # Add to results
        self.results.append(result)
        
        # Update statistics
        self.stats['total_tests'] += 1
        
        # Check if skipped (invalid config)
        is_skipped = 'SKIPPED' in str(test_result.get('error_message', ''))
        
        if result['success']:
            self.stats['successful_tests'] += 1
            vram_gb = result['vram_peak_gb']
            self.logger.info(f"  ✓ SUCCESS - VRAM: {vram_gb:.2f} GB / 48 GB")
        elif is_skipped:
            # Don't count skipped as failed
            error_msg = result['error_message'] or 'Invalid config'
            self.logger.info(f"  ⊘ SKIPPED - {error_msg}")
        else:
            self.stats['failed_tests'] += 1
            if result['error_message'] == 'OOM':
                self.stats['oom_tests'] += 1
            error_msg = result['error_message'] or 'Unknown error'
            self.logger.info(f"  ✗ FAILED - {error_msg}")
        
        # Print progress
        progress = (test_idx / total_tests) * 100
        # Calculate success rate excluding skipped
        skipped_count = sum(1 for r in self.results if 'SKIPPED' in str(r.get('error_message', '')))
        valid_tests = self.stats['total_tests'] - skipped_count
        if valid_tests > 0:
            success_rate = (self.stats['successful_tests'] / valid_tests) * 100
        else:
            success_rate = 0.0
        self.logger.info(
            f"  Progress: {progress:.1f}% | "
            f"Success rate: {success_rate:.1f}% "
            f"({self.stats['successful_tests']}/{valid_tests} valid, {skipped_count} skipped)\n"
        )
    
    def _save_checkpoint(self) -> None:
        """Save current results to checkpoint files"""
        results_json = self.output_dir / self.config['output']['results_file']
        results_csv = self.output_dir / self.config['output']['results_csv']
        
        save_results_json(self.results, str(results_json))
        save_results_csv(self.results, str(results_csv))
        
        self.logger.info(f"Checkpoint saved: {len(self.results)} results")
    
    def _print_final_stats(self) -> None:
        """Print final statistics"""
        print_banner("GRID SEARCH COMPLETE", width=80)
        
        total_duration = time.time() - self.stats['start_time']
        hours = int(total_duration // 3600)
        minutes = int((total_duration % 3600) // 60)
        seconds = int(total_duration % 60)
        
        # Count skipped
        skipped_count = sum(1 for r in self.results if 'SKIPPED' in str(r.get('error_message', '')))
        valid_tests = self.stats['total_tests'] - skipped_count
        
        self.logger.info("Final Statistics:")
        self.logger.info(f"  Total tests: {self.stats['total_tests']}")
        self.logger.info(f"  Successful: {self.stats['successful_tests']}")
        self.logger.info(f"  Failed: {self.stats['failed_tests']}")
        self.logger.info(f"  Skipped (invalid config): {skipped_count}")
        self.logger.info(f"  OOM errors: {self.stats['oom_tests']}")
        self.logger.info(f"  Duration: {hours}h {minutes}m {seconds}s")
        
        if self.stats['successful_tests'] > 0 and valid_tests > 0:
            success_rate = (self.stats['successful_tests'] / valid_tests) * 100
            self.logger.info(f"  Success rate: {success_rate:.1f}% (excluding {skipped_count} skipped)")
            
            # Find configuration with max VRAM that succeeded
            successful = [r for r in self.results if r['success']]
            max_vram_config = max(successful, key=lambda x: x['vram_peak_gb'])
            
            self.logger.info("\nConfiguration with highest VRAM usage that fits:")
            self.logger.info(f"  Model: {max_vram_config['model_name']}")
            self.logger.info(f"  Shape: {max_vram_config['patch_shape']}")
            self.logger.info(f"  VRAM: {max_vram_config['vram_peak_gb']:.2f} GB / 48 GB")


def run_grid_search(
    config_path: str = "config/grid_search_config.yaml",
    output_dir: str = "results",
    max_tests: Optional[int] = None,
    resume: bool = True
) -> List[Dict]:
    """
    Convenience function to run grid search
    
    Args:
        config_path: Path to configuration file
        output_dir: Directory for output files
        max_tests: Maximum number of tests (None for all)
        resume: Whether to resume from checkpoint
        
    Returns:
        List of all test results
    """
    runner = GridSearchRunner(
        config_path=config_path,
        output_dir=output_dir,
        resume_from_checkpoint=resume
    )
    
    results = runner.run(max_tests=max_tests)
    
    return results


if __name__ == "__main__":
    # Run grid search
    results = run_grid_search()

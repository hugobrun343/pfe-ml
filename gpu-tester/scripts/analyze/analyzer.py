"""
Main Analyzer - Orchestrates analysis workflow
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional

from .data_loader import (
    load_results, filter_by_model_family,
    get_statistics, get_model_families
)
from .best_configs import (
    get_all_best_combinations, get_failure_analysis
)


class ResultsAnalyzer:
    """Main analyzer class for grid search results"""
    
    def __init__(self, results_file: str):
        """
        Initialize analyzer
        
        Args:
            results_file: Path to results file (CSV or JSON)
        """
        self.results_file = results_file
        self.df = load_results(results_file)
        self.filtered_df = self.df.copy()
        self.model_family = None
        
    def filter_by_family(self, family: Optional[str] = None):
        """
        Filter results by model family
        
        Args:
            family: Model family name (None for all families)
        """
        if family is None:
            self.filtered_df = self.df.copy()
            self.model_family = None
        else:
            self.filtered_df = filter_by_model_family(self.df, family)
            self.model_family = family
    
    def analyze(self, output_dir: str = "results/analysis", create_plots: bool = True):
        """
        Run complete analysis
        
        Args:
            output_dir: Directory to save analysis results
            create_plots: Whether to create visualizations
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Analyzing results from: {self.results_file}")
        if self.model_family:
            print(f"Model family filter: {self.model_family}")
        
        # Statistics
        print("\n" + "="*80)
        print("STATISTIQUES GÉNÉRALES")
        print("="*80)
        stats = get_statistics(self.filtered_df)
        print(f"Total de tests: {stats['total']}")
        print(f"Tests réussis: {stats['successful']}")
        print(f"Tests échoués: {stats['failed']}")
        print(f"Taux de succès: {stats['success_rate']:.2f}%")
        
        if 'vram' in stats:
            print(f"\nVRAM Peak:")
            print(f"  Moyenne: {stats['vram']['mean']:.2f} GB")
            print(f"  Médiane: {stats['vram']['median']:.2f} GB")
            print(f"  Max: {stats['vram']['max']:.2f} GB")
            print(f"  Min: {stats['vram']['min']:.2f} GB")
        
        # Helper function for JSON serialization
        def convert_to_native(obj):
            """Convert numpy types to native Python types for JSON serialization"""
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        # Save statistics
        with open(output_path / "statistics.json", 'w') as f:
            json.dump(convert_to_native(stats), f, indent=2)
        
        # Best configurations
        print("\n" + "="*80)
        print("MEILLEURES CONFIGURATIONS")
        print("="*80)
        
        best_configs = get_all_best_combinations(self.filtered_df, n_per_category=50)
        
        # Top by VRAM
        if 'top_by_vram' in best_configs and len(best_configs['top_by_vram']) > 0:
            print(f"\nTop 20 par VRAM:")
            print(best_configs['top_by_vram'].head(20).to_string(index=False))
            best_configs['top_by_vram'].to_csv(output_path / "top_by_vram.csv", index=False)
        
        # Top by efficiency
        if 'top_by_efficiency' in best_configs and len(best_configs['top_by_efficiency']) > 0:
            print(f"\nTop 20 par efficacité:")
            print(best_configs['top_by_efficiency'].head(20).to_string(index=False))
            best_configs['top_by_efficiency'].to_csv(output_path / "top_by_efficiency.csv", index=False)
        
        # Save all best configs
        for name, df_config in best_configs.items():
            if len(df_config) > 0:
                df_config.to_csv(output_path / f"{name}.csv", index=False)
        
        # Failure analysis
        print("\n" + "="*80)
        print("ANALYSE DES ÉCHECS")
        print("="*80)
        failure_analysis = get_failure_analysis(self.filtered_df)
        print(f"Total échecs: {failure_analysis.get('total_failures', 0)}")
        print(f"OOM: {failure_analysis.get('oom_count', 0)}")
        print(f"Skipped: {failure_analysis.get('skipped_count', 0)}")
        
        if 'failures_by_model' in failure_analysis:
            print(f"\nÉchecs par modèle:")
            for model, count in sorted(failure_analysis['failures_by_model'].items(), 
                                      key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {model}: {count}")
        
        # Save failure analysis
        with open(output_path / "failure_analysis.json", 'w') as f:
            json.dump(convert_to_native(failure_analysis), f, indent=2)
        
        # Create visualizations
        if create_plots:
            try:
                from .visualizations import create_all_plots
                print("\n" + "="*80)
                print("CRÉATION DES VISUALISATIONS")
                print("="*80)
                plots_dir = output_path / "plots"
                create_all_plots(self.filtered_df, str(plots_dir))
            except ImportError as e:
                print(f"\nImpossible de créer les visualisations: {e}")
                print("Installez matplotlib et seaborn: pip install matplotlib seaborn")
        
        print(f"\n✓ Analyse complète sauvegardée dans: {output_dir}")

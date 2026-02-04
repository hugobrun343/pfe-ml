"""
Results Analyzer Package
"""

from .analyzer import ResultsAnalyzer
from .data_loader import load_results, filter_by_model_family, get_model_families
from .best_configs import get_all_best_combinations, get_failure_analysis

# Optional import for visualizations
try:
    from .visualizations import create_all_plots
    __all__ = [
        'ResultsAnalyzer',
        'load_results',
        'filter_by_model_family',
        'get_model_families',
        'get_all_best_combinations',
        'get_failure_analysis',
        'create_all_plots',
    ]
except ImportError:
    __all__ = [
        'ResultsAnalyzer',
        'load_results',
        'filter_by_model_family',
        'get_model_families',
        'get_all_best_combinations',
        'get_failure_analysis',
    ]

"""Run analytics — validation split integrity, stack ranking, aggregated metrics."""

from .io import load_run, load_results
from .extract import extract_validation_identifiers
from .validate import validate
from .report import print_report
from .stack_ranking import rank_problematic_stacks
from .aggregated_metrics import analyze_runs, get_run_summary

__all__ = [
    "load_run",
    "load_results",
    "extract_validation_identifiers",
    "validate",
    "print_report",
    "rank_problematic_stacks",
    "analyze_runs",
    "get_run_summary",
]

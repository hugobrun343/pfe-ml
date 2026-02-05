"""Text report generation for training results analysis."""

from pathlib import Path
from typing import Dict, Tuple, Optional


def save_text_report(
    position_analysis: Dict, 
    volume_analysis: Dict, 
    combined_analysis: Dict, 
    output_dir: Path,
    patch_index_analysis: Optional[Dict] = None
) -> None:
    """
    Save text report with detailed statistics.
    
    Args:
        position_analysis: Dictionary mapping (i, j) -> stats
        volume_analysis: Dictionary mapping stack_id -> stats
        combined_analysis: Dictionary mapping (stack_id, (i, j)) -> stats
        output_dir: Directory to save the report
    """
    report_path = output_dir / 'analysis_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINING RESULTS ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Position analysis (mode 'max') or patch index analysis (mode 'top_n')
        if position_analysis:
            f.write("WORST PERFORMING GRID POSITIONS (Mode: max)\n")
            f.write("-" * 80 + "\n")
            sorted_positions = sorted(
                position_analysis.items(),
                key=lambda x: (x[1]['failure_rate'], -x[1]['total']),
                reverse=True
            )
            f.write(f"{'Position':<15} {'Total':<10} {'Failures':<10} {'Failure Rate':<15}\n")
            f.write("-" * 80 + "\n")
            for (i, j), stats in sorted_positions[:20]:
                f.write(f"({i:2d},{j:2d}):{'':<6} {stats['total']:<10} "
                       f"{stats['failures']:<10} {stats['failure_rate']*100:>6.2f}%\n")
        elif patch_index_analysis:
            f.write("WORST PERFORMING PATCH INDICES (Mode: top_n)\n")
            f.write("-" * 80 + "\n")
            sorted_indices = sorted(
                patch_index_analysis.items(),
                key=lambda x: (x[1]['failure_rate'], -x[1]['total']),
                reverse=True
            )
            f.write(f"{'Patch Index':<15} {'Total':<10} {'Failures':<10} {'Failure Rate':<15}\n")
            f.write("-" * 80 + "\n")
            for idx, stats in sorted_indices[:20]:
                f.write(f"Patch {idx:3d}:{'':<6} {stats['total']:<10} "
                       f"{stats['failures']:<10} {stats['failure_rate']*100:>6.2f}%\n")
        else:
            f.write("No position data available.\n")
        f.write("\n\n")
        
        # Volume analysis
        f.write("WORST PERFORMING VOLUMES\n")
        f.write("-" * 80 + "\n")
        if volume_analysis:
            sorted_volumes = sorted(
                volume_analysis.items(),
                key=lambda x: (x[1]['failure_rate'], -x[1]['total']),
                reverse=True
            )
            f.write(f"{'Volume ID':<20} {'Total':<10} {'Failures':<10} {'Failure Rate':<15}\n")
            f.write("-" * 80 + "\n")
            for vol_id, stats in sorted_volumes[:30]:
                f.write(f"{vol_id:<20} {stats['total']:<10} "
                       f"{stats['failures']:<10} {stats['failure_rate']*100:>6.2f}%\n")
        else:
            f.write("No volume data available.\n")
        f.write("\n\n")
        
        # Combined analysis
        f.write("WORST PERFORMING (VOLUME, POSITION) COMBINATIONS\n")
        f.write("-" * 80 + "\n")
        if combined_analysis:
            sorted_combined = sorted(
                combined_analysis.items(),
                key=lambda x: (x[1]['failure_rate'], -x[1]['total']),
                reverse=True
            )
            f.write(f"{'Volume ID':<20} {'Position':<15} {'Total':<10} "
                   f"{'Failures':<10} {'Failure Rate':<15}\n")
            f.write("-" * 80 + "\n")
            for (vol_id, (i, j)), stats in sorted_combined[:30]:
                f.write(f"{vol_id:<20} ({i:2d},{j:2d}):{'':<6} {stats['total']:<10} "
                       f"{stats['failures']:<10} {stats['failure_rate']*100:>6.2f}%\n")
        else:
            f.write("No combined data available.\n")

"""Stats constants and default workers."""

import os

STATS_FILENAME_GLOBAL = "global_intensity.json"
STATS_FILENAME_P1P99 = "stack_p1p99.json"
STATS_FILENAME_P5P95 = "stack_p5p95.json"


def _default_stats_workers() -> int:
    return min(16, (os.cpu_count() or 4))

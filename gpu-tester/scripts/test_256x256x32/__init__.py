"""VRAM test 256x256x32x3 — subprocess per family to avoid GPU cascade."""

from .const import BATCH_SIZES, DEFAULT_FAMILIES, DEPTH, IN_CHANNELS, SPATIAL
from .orchestrate import run_all_families
from .run_single import run_single_family

__all__ = [
    "BATCH_SIZES",
    "DEFAULT_FAMILIES",
    "DEPTH",
    "IN_CHANNELS",
    "SPATIAL",
    "run_all_families",
    "run_single_family",
]

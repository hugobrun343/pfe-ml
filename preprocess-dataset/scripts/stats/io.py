"""Load stats JSON and lookup by stack."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .paths import _nii_name


def load_global_intensity(path: Path) -> Dict[str, Any]:
    """Load global intensity stats: {channels: [{channel, lo, hi}, ...]}."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_stats_map(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load per-stack stats JSON (list of {file, channels}) -> map file -> record."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, Dict[str, Any]] = {}
    for rec in data:
        fn = rec.get("file")
        if not fn:
            raise ValueError("Stats record missing 'file'")
        out[fn] = rec
        base = fn.replace(".nii.gz", "").replace(".nii", "")
        if base != fn:
            out[base] = rec
    return out


def get_stats_record_for_stack(
    stats_map: Dict[str, Dict[str, Any]],
    stack: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Lookup per-stack stats by nii filename."""
    name = _nii_name(stack)
    return stats_map.get(name) or stats_map.get(name.replace(".nii.gz", ""))

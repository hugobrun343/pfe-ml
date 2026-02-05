"""Orchestrate subprocesses per family and merge results."""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .const import SPATIAL, DEPTH, IN_CHANNELS, BATCH_SIZES


def run_all_families(
    model_families: list,
    script_path: Path,
    root: Path,
    results_dir: Path,
) -> int:
    """Run each family in subprocess, merge, save, print summary. Returns exit code."""
    all_results = []
    all_models = []

    print("=" * 70)
    print("  VRAM test: 256x256x32x3 (one subprocess per family)")
    print("=" * 70)
    print(f"  Families: {model_families}")
    print("=" * 70)

    for fam in model_families:
        partial = results_dir / f"test_256x256x32_{fam}.json"
        print(f"\n>>> Subprocess: {fam}")
        ret = subprocess.run(
            [sys.executable, str(script_path), "--models", fam, "--output", str(partial)],
            cwd=str(root),
        )
        if ret.returncode != 0:
            print(f"  Warning: {fam} subprocess exited with {ret.returncode}")
        if partial.exists():
            with open(partial, encoding="utf-8") as f:
                data = json.load(f)
            all_results.extend(data["results"])
            all_models.extend(data["config"]["models"])
            partial.unlink()

    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "spatial_resolution": SPATIAL,
            "depth": DEPTH,
            "in_channels": IN_CHANNELS,
            "models": all_models,
            "batch_sizes": BATCH_SIZES,
        },
        "results": all_results,
    }
    final_path = results_dir / "test_256x256x32.json"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"  Results saved to: {final_path}")
    print("=" * 70)

    print("\n  Summary (batch_max per model):")
    by_model = {}
    for r in all_results:
        m = r["model_name"]
        if r["success"] and (m not in by_model or r["batch_size"] > by_model[m]["batch_size"]):
            by_model[m] = r
    for m in sorted(by_model.keys()):
        r = by_model[m]
        vram = r.get("vram_peak_formatted", f"{r.get('vram_peak_gb', 0):.2f} GB")
        print(f"    {m}: batch_max={r['batch_size']}  VRAM={vram}")

    return 0

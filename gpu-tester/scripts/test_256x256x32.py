#!/usr/bin/env python3
"""
Test VRAM for 256x256x32x3 — SE-ResNet3D, ViT3D, ConvNeXt3D.

Fixed input: spatial=256, depth=32, in_channels=3.
Tests multiple model sizes and batch sizes, no skip logic.
Output: results/test_256x256x32.json (format aligned with grid_search_results)
"""

import json
import sys
from datetime import datetime
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from vram_tester import VRAMTester
from models import get_available_models
from utils import create_result_entry

# Fixed input
SPATIAL = 256
DEPTH = 32
IN_CHANNELS = 3
NUM_CLASSES = 2  # for CrossEntropyLoss

# Models to test (only SE-ResNet3D, ViT3D, ConvNeXt3D)
MODEL_FAMILIES = ["SE-ResNet3D", "ViT3D", "ConvNeXt3D"]
BATCH_SIZES = [4, 8, 12, 16, 20, 24, 32, 48, 64]


def main():
    all_models = get_available_models()
    models = [m for m in all_models if any(m.startswith(fam) for fam in MODEL_FAMILIES)]

    if not models:
        print("No models found for SE-ResNet3D, ViT3D, ConvNeXt3D")
        return 1

    print("=" * 70)
    print("  VRAM test: 256x256x32x3")
    print("=" * 70)
    print(f"  Models: {len(models)}")
    print(f"  Batch sizes: {BATCH_SIZES}")
    print("=" * 70)

    tester = VRAMTester(
        device="cuda",
        num_warmup_iterations=3,
        num_test_iterations=5,
        clear_cache_between_tests=True,
    )

    results = []
    for i, model_name in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] {model_name}")
        for batch_size in BATCH_SIZES:
            try:
                r = tester.test_configuration(
                    spatial_res=SPATIAL,
                    depth=DEPTH,
                    batch_size=batch_size,
                    model_name=model_name,
                    in_channels=IN_CHANNELS,
                    num_classes=NUM_CLASSES,
                )
            except Exception as e:
                r = {
                    "success": False,
                    "error_message": str(e)[:200],
                    "vram_used_bytes": None,
                    "vram_peak_bytes": None,
                    "duration_seconds": 0,
                }

            entry = create_result_entry(
                spatial_res=SPATIAL,
                depth=DEPTH,
                batch_size=batch_size,
                model_name=model_name,
                model_size=model_name,
                success=r.get("success", False),
                vram_used_bytes=r.get("vram_used_bytes"),
                vram_peak_bytes=r.get("vram_peak_bytes"),
                error_message=r.get("error_message"),
                duration_seconds=r.get("duration_seconds", 0),
            )
            entry["in_channels"] = IN_CHANNELS  # extra field for our config
            results.append(entry)

            status = "✓" if entry["success"] else "✗"
            vram_str = entry.get("vram_peak_formatted", "-")
            err = entry.get("error_message") or ""
            err_str = f" ({err[:50]}...)" if len(err) > 50 else (f" ({err})" if err else "")
            print(f"  batch={batch_size:2d} {status}  VRAM={vram_str}{err_str}")

    # Save
    out_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "test_256x256x32.json"

    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "spatial_resolution": SPATIAL,
            "depth": DEPTH,
            "in_channels": IN_CHANNELS,
            "models": models,
            "batch_sizes": BATCH_SIZES,
        },
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"  Results saved to: {out_path}")
    print("=" * 70)

    # Summary
    print("\n  Summary (batch_max per model):")
    by_model = {}
    for r in results:
        m = r["model_name"]
        if r["success"] and (m not in by_model or r["batch_size"] > by_model[m]["batch_size"]):
            by_model[m] = r
    for m in sorted(by_model.keys()):
        r = by_model[m]
        vram = r.get("vram_peak_formatted", f"{r.get('vram_peak_gb', 0):.2f} GB")
        print(f"    {m}: batch_max={r['batch_size']}  VRAM={vram}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

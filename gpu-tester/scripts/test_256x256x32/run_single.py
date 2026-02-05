"""Run VRAM tests for one or more model families in a single process."""

import json
from datetime import datetime
from pathlib import Path

from .const import SPATIAL, DEPTH, IN_CHANNELS, NUM_CLASSES, BATCH_SIZES


def run_single_family(model_families: list, output_path: Path) -> int:
    """Run VRAM tests for given families (single process). Returns exit code."""
    from vram_tester import VRAMTester
    from models import get_available_models
    from utils import create_result_entry, wait_for_gpu_ready

    all_models = get_available_models()
    models = [m for m in all_models if any(m.startswith(fam) for fam in model_families)]

    if not models:
        print(f"No models found for {model_families}")
        return 1

    print("=" * 70)
    print(f"  VRAM test: 256x256x32x3 — {model_families}")
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
                error_str = str(e).upper()
                if "CUDA" in error_str or "OOM" in error_str:
                    print("  Waiting for GPU to recover...")
                    if wait_for_gpu_ready(timeout_seconds=5.0, check_interval=0.5):
                        print("  ✓ GPU recovered, continuing")
                    try:
                        import gc
                        import torch
                        gc.collect()
                        if torch.cuda.is_available():
                            try:
                                torch.cuda.empty_cache()
                            except RuntimeError:
                                pass
                    except Exception:
                        pass

            if not r.get("success") and r.get("error_message"):
                err_upper = str(r["error_message"]).upper()
                if "CUDA" in err_upper or "OOM" in err_upper:
                    print("  Waiting for GPU to recover...")
                    if wait_for_gpu_ready(timeout_seconds=5.0, check_interval=0.5):
                        print("  ✓ GPU recovered, continuing")

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
            entry["in_channels"] = IN_CHANNELS
            results.append(entry)

            status = "✓" if entry["success"] else "✗"
            vram_str = entry.get("vram_peak_formatted", "-")
            err = entry.get("error_message") or ""
            err_str = f" ({err[:50]}...)" if len(err) > 50 else (f" ({err})" if err else "")
            print(f"  batch={batch_size:2d} {status}  VRAM={vram_str}{err_str}")

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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return 0

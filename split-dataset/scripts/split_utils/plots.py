"""Generate distribution histograms for dataset splits."""

from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    import numpy as np
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

from .stats import CATEGORIES, pretty_name


def plot_distributions(
    stats_dict: Dict[str, Dict],
    out_dir: Path,
    stats_dict_pct: Optional[Dict[str, Dict]] = None,
) -> None:
    """Generate distribution plots for each metadata category.

    For each category, produces:
      - ``{cat}.png``      — absolute counts
      - ``{cat}_pct.png``  — percentages
      - (contributed to) ``all_pct.png`` — all categories on one page

    Parameters
    ----------
    stats_dict : {"Label": stats, ...}
        Data used for the **counts** plots.
    out_dir : Path
        Directory where PNGs are saved (created if needed).
    stats_dict_pct : dict or None
        If provided, used for the **percentage** plots instead of
        *stats_dict*.  Useful when the groups differ (e.g. holdout
        alone for counts, holdout vs CV pool for percentages).
    """
    if not _HAS_MPL:
        print("  [WARNING] matplotlib not installed — skipping plots.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    sd_pct = stats_dict_pct or stats_dict
    labels_cnt = list(stats_dict.keys())
    labels_pct = list(sd_pct.keys())

    # Collect per-category data for the big summary PNG
    pct_subplots: List[dict] = []

    for cat in CATEGORIES:
        # Union of all values across both dicts
        all_vals: set = set()
        for st in stats_dict.values():
            all_vals |= set(st[cat].keys())
        for st in sd_pct.values():
            all_vals |= set(st[cat].keys())
        values = sorted(all_vals, key=str)
        if not values:
            continue

        # Counts
        counts = _build_rows_count(stats_dict, labels_cnt, cat, values)
        _save_single(values, labels_cnt, counts, cat,
                     out_dir / f"{cat}.png", ylabel="Count", fmt="d",
                     title_suffix="absolute counts")

        # Percentages
        pcts = _build_rows_pct(sd_pct, labels_pct, cat, values)
        _save_single(values, labels_pct, pcts, cat,
                     out_dir / f"{cat}_pct.png", ylabel="Percentage (%)",
                     fmt=".1f", title_suffix="percentage")

        pct_subplots.append({
            "values": values, "labels": labels_pct,
            "data": pcts, "cat": cat,
        })

    # Big summary PNG with all categories
    if pct_subplots:
        _save_summary(pct_subplots, out_dir / "all_pct.png")

    print(f"  Plots saved: {out_dir}/")


# -- data helpers ------------------------------------------------------------

def _build_rows_count(sd, labels, cat, values):
    return [[sd[lb][cat].get(v, 0) for v in values] for lb in labels]


def _build_rows_pct(sd, labels, cat, values):
    rows = []
    for lb in labels:
        total = max(sd[lb]["total"], 1)
        rows.append([sd[lb][cat].get(v, 0) / total * 100 for v in values])
    return rows


# -- drawing -----------------------------------------------------------------

def _save_single(
    values: list, labels: List[str], data: List[list],
    cat: str, path: Path, ylabel: str, fmt: str, title_suffix: str,
) -> None:
    """One bar chart, one PNG."""
    n_groups = len(labels)
    n_vals = len(values)
    bar_w = 0.8 / max(n_groups, 1)
    x = np.arange(n_vals)

    fig_w = max(8, n_vals * n_groups * 0.45 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    ann_fs = 9 if n_vals < 5 else (7 if n_vals < 10 else 5)
    for i, label in enumerate(labels):
        offset = (i - n_groups / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, data[i], bar_w, label=label)
        _annotate(ax, bars, fmt=fmt, fontsize=ann_fs)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{pretty_name(cat)} — {title_suffix}", fontsize=14)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_xticks(x)
    xtick_fs = 11 if n_vals < 5 else (9 if n_vals < 10 else 7)
    ax.set_xticklabels([str(v) for v in values], rotation=45,
                       ha="right", fontsize=xtick_fs)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_summary(subplots: List[dict], path: Path) -> None:
    """All percentage charts on one big page (4x2 grid)."""
    n = len(subplots)
    cols = 2
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for idx, sp in enumerate(subplots):
        ax = axes[idx]
        values = sp["values"]
        labels = sp["labels"]
        data = sp["data"]
        cat = sp["cat"]
        n_groups = len(labels)
        bar_w = 0.8 / max(n_groups, 1)
        x = np.arange(len(values))

        nv = len(values)
        ann_fs = 7 if nv < 5 else (5 if nv < 10 else 4)
        for i, label in enumerate(labels):
            offset = (i - n_groups / 2 + 0.5) * bar_w
            bars = ax.bar(x + offset, data[i], bar_w, label=label)
            _annotate(ax, bars, fmt=".1f", fontsize=ann_fs)

        ax.set_ylabel("(%)", fontsize=11)
        ax.set_title(pretty_name(cat), fontsize=13)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xticks(x)
        xtick_fs = 10 if nv < 5 else (8 if nv < 10 else 6)
        ax.set_xticklabels([str(v) for v in values], rotation=45,
                           ha="right", fontsize=xtick_fs)

    # Hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Distribution — all categories (%)", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _annotate(ax, bars, fmt: str = "d", fontsize: int = 6) -> None:
    """Put value labels above bars."""
    for bar in bars:
        h = bar.get_height()
        if h == 0:
            continue
        text = f"{h:{fmt}}" if fmt == "d" else f"{h:{fmt}}"
        ax.annotate(
            text,
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=fontsize,
        )

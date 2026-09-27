# -*- coding: utf-8 -*-
"""
Aggregate predictions of all systems on the test video and generate report figures:
A. full-cycle phase timeline comparison (GT vs systems)
B. frame accuracy / boundary match / boundary MAE bar charts
C. v3 per-frame probability timeline (with GT boundary markers)
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 7-stage palette
STAGE_COLORS = np.array([
    [91, 155, 213],   # fill
    [153, 204, 255],  # post-fill idle
    [237, 125, 49],   # anoxic
    [112, 173, 71],   # aerobic
    [165, 105, 189],  # settling
    [255, 99, 97],    # decant
    [200, 160, 200],  # post-decant idle
]) / 255.0

SYSTEMS = [
    ("baseline", "hard-coded baseline", "sm"),
    ("v1", "pre-incremental model v1", "sm"),
    ("v2", "incremental v2 (leak-free)", "sm"),
    ("v3", "incremental v3 (full)", "sm"),
]


def load(tag):
    return np.load(os.path.join(C.RESULT_DIR, f"test_pred_{tag}.npz"))


def chart_a(times, gt):
    """Full-cycle phase timeline comparison: GT + decoded outputs, horizontal bands."""
    rows = [("Ground truth", gt)]
    for tag, name, key in SYSTEMS:
        z = load(tag)
        rows.append((name, z[key]))

    fig, ax = plt.subplots(figsize=(14, 4.8))
    tmin = times / 60.0
    for ri, (name, stages) in enumerate(rows):
        y = len(rows) - 1 - ri
        i = 0
        while i < len(stages):
            j = i
            while j + 1 < len(stages) and stages[j + 1] == stages[i]:
                j += 1
            ax.fill_between([tmin[i], tmin[min(j + 1, len(tmin) - 1)]],
                            y - 0.38, y + 0.38,
                            color=STAGE_COLORS[stages[i]], linewidth=0)
            i = j + 1
        ax.text(-1.5, y, name, ha="right", va="center", fontsize=11,
                fontweight="bold")
    for s, e, c in C.TEST_GT_INTERVALS[1:6]:
        ax.axvline(s / 60.0, color="k", linestyle="--", alpha=0.35, linewidth=1)
        ax.text(s / 60.0, len(rows) - 0.35, f"{s//60:.0f}:{s%60:02d}",
                ha="center", fontsize=8, color="k")
    ax.set_xlim(-12, tmin[-1])
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_yticks([])
    ax.set_xlabel("video time (min)")
    ax.set_title("Full-cycle test video (94.9 min) phase recognition: GT vs system outputs")
    legend = [Patch(facecolor=STAGE_COLORS[i], label=C.CLASSES[i])
              for i in range(C.NUM_CLASSES)]
    ax.legend(handles=legend, ncol=7, fontsize=9, loc="lower center",
              bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    fig.savefig(os.path.join(C.RESULT_DIR, "report_timeline_compare.png"), dpi=160,
                bbox_inches="tight")
    plt.close(fig)


def chart_b():
    """Three-metric comparison bar charts."""
    names, accs, bounds_ratio, bounds_mae = [], [], [], []
    for tag, name, _ in SYSTEMS:
        with open(os.path.join(C.RESULT_DIR, f"test_eval_{tag}.json"),
                  encoding="utf-8") as f:
            r = json.load(f)
        sm = r["state_machine"]
        names.append(name)
        accs.append(sm["frame_acc"] * 100)
        m, tot = sm["boundary_matched"].split("/")
        bounds_ratio.append(int(m) / int(tot) * 100)
        bounds_mae.append(sm["boundary_mae_s"] if sm["boundary_mae_s"] is not None else np.nan)

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, vals, title, fmt, ylim in [
        (axes[0], accs, "frame accuracy (%)", "{:.1f}", 105),
        (axes[1], bounds_ratio, "boundary detection rate (%)", "{:.0f}", 115),
        (axes[2], bounds_mae, "boundary MAE (s)", "{:.0f}", None),
    ]:
        colors = ["#c00000", "#ed7d31", "#70ad47", "#2e75b6"]
        bars = ax.bar(x, vals, color=colors, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9, rotation=12)
        ax.set_title(title)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                        ha="center", va="bottom", fontsize=10, fontweight="bold")
        if ylim:
            ax.set_ylim(0, ylim)
    fig.suptitle("Hard-coded baseline vs models before/after incremental training", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(C.RESULT_DIR, "report_metrics_compare.png"), dpi=160,
                bbox_inches="tight")
    plt.close(fig)


def chart_c(times, gt):
    """v3 per-frame probability timeline + GT boundaries."""
    z = load("v3")
    probs = z["probs"].astype(np.float32)
    fig, ax = plt.subplots(figsize=(14, 4.2))
    tmin = times / 60.0
    for c in range(C.NUM_CLASSES):
        ax.plot(tmin, probs[:, c], label=C.CLASSES[c], linewidth=1.1,
                color=STAGE_COLORS[c])
    for s, e, c in C.TEST_GT_INTERVALS[1:6]:
        ax.axvline(s / 60.0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("video time (min)")
    ax.set_ylabel("phase probability")
    ax.set_ylim(0, 1.03)
    ax.set_title("v3 per-frame phase probabilities on the full-cycle test video (dashed = GT boundaries)")
    ax.legend(ncol=7, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(os.path.join(C.RESULT_DIR, "report_v3_probs.png"), dpi=160,
                bbox_inches="tight")
    plt.close(fig)


def main():
    z0 = load("v3")
    times, gt = z0["times"], z0["gt"]
    chart_a(times, gt)
    chart_b()
    chart_c(times, gt)
    print("[INFO] report figures generated")


if __name__ == "__main__":
    main()

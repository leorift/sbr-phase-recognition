# -*- coding: utf-8 -*-
"""Paper figure generation (English labels): outputs to paper/figs/"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

ROOT = C.ROOT
RES = os.path.join(ROOT, "outputs", "results")
FIGS = os.path.join(ROOT, "paper", "figs")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

STAGES_EN = ["Fill", "Post-fill idle", "Anoxic", "Aerobic", "Settling", "Decant", "Post-decant idle"]
COLORS = ["#4C9F70", "#A8D08D", "#F4B183", "#C00000", "#9DC3E6", "#2E75B6", "#BFBFBF"]


def band(ax, times, stages, y, h=0.8):
    for c in range(7):
        m = stages == c
        if not m.any():
            continue
        idx = np.where(m)[0]
        splits = np.where(np.diff(idx) > 1)[0] + 1
        for seg in np.split(idx, splits):
            ax.fill_between(times[seg] / 3600.0, y - h / 2, y + h / 2,
                            color=COLORS[c], lw=0)


# ---------- Fig A: 5h video timeline (GT / v7 raw / v7 SM) ----------
d = np.load(os.path.join(RES, "pred2_cycle5h_raw_v7.npz"), allow_pickle=True)
times, gt, raw, sm = d["times"], d["gt"], d["raw"], d["sm"]
fig, ax = plt.subplots(figsize=(10, 3.2))
band(ax, times, gt, 2.2)
band(ax, times, raw, 1.1)
band(ax, times, sm, 0.0)
# calibration window markers
cal = json.load(open(os.path.join(ROOT, "excl_cycle5h_raw.json"), encoding="utf-8"))
for s, e in cal:
    ax.axvspan(s / 3600, e / 3600, ymin=0.72, ymax=0.78, color="k", alpha=0.55)
for t in [367, 638, 4285, 18690]:
    ax.axvline(t / 3600, color="k", ls="--", lw=0.8, alpha=0.6)
ax.set_yticks([2.2, 1.1, 0.0])
ax.set_yticklabels(["Ground truth", "v7 raw", "v7 decoded"])
ax.set_xlabel("Time (h)")
ax.set_xlim(0, times[-1] / 3600)
ax.set_ylim(-0.6, 2.8)
handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[i]) for i in [0, 1, 2, 3, 4]]
ax.legend(handles, [STAGES_EN[i] for i in [0, 1, 2, 3, 4]], ncol=5,
          loc="upper center", bbox_to_anchor=(0.5, 1.28), frameon=False, fontsize=8)
fig.savefig(os.path.join(FIGS, "fig_5h_timeline.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig_5h_timeline.png")

# ---------- Fig B: sparse calibration protocol schematic ----------
fig, ax = plt.subplots(figsize=(8, 2.2))
s, e = 0.0, 10.0
ax.axvspan(s, e, ymin=0.55, ymax=0.85, color="#9DC3E6", alpha=0.8)
ax.text(5, 0.95, "Stage interval (e.g., Anoxic, 60.8 min)", ha="center", fontsize=9)
k, w = 8, 0.25
starts = np.linspace(s, e - w, k)
for ws in starts:
    ax.axvspan(ws, ws + w, ymin=0.15, ymax=0.45, color="#C00000")
ax.text(5, 0.02, "K short calibration windows (W = 15 s each, uniformly spread)", ha="center", fontsize=9)
ax.annotate("", xy=(starts[0] + w / 2, 0.47), xytext=(starts[0] + w / 2, 0.53),
            arrowprops=dict(arrowstyle="->", lw=1))
ax.set_xlim(-0.3, 10.3); ax.set_ylim(-0.1, 1.2)
ax.axis("off")
fig.savefig(os.path.join(FIGS, "fig_calib_protocol.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig_calib_protocol.png")

# ---------- Fig C: CM v3 vs v4 ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, tag, ttl in [(axes[0], "v3", "v3 (no oversampling)"), (axes[1], "v4", "v4 (4x oversampling)")]:
    m = json.load(open(os.path.join(RES, f"train_metrics_{tag}.json"), encoding="utf-8"))
    cm = np.array(m["confusion_matrix"], dtype=float)
    cmn = cm / (cm.sum(1, keepdims=True) + 1e-9)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    short = ["Fill", "Idle-1", "Anox", "Aer", "Set", "Dec", "Idle-2"]
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_title(f"{ttl}\nacc={m['val_frame_acc']:.3f}, macro-F1={m['val_macro_f1']:.3f}", fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
fig.savefig(os.path.join(FIGS, "fig_cm_v3v4.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig_cm_v3v4.png")

# ---------- Fig D: ablation bar chart ----------
fig, ax = plt.subplots(figsize=(7.5, 3.4))
labels = ["DET-only\n(12d)", "CNN-only\n(1280d)", "RF-6\nlayers", "MLP\n(no temporal)", "BiLSTM", "Full model\n(v3)"]
vals = [41.0, 47.3, 47.3, 47.4, 47.4, 93.8]
cols = ["#BFBFBF", "#BFBFBF", "#BFBFBF", "#BFBFBF", "#BFBFBF", "#C00000"]
bars = ax.bar(labels, vals, color=cols)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=9)
ax.set_ylabel("Frame accuracy (%)")
ax.set_ylim(0, 105)
ax.set_title("Ablations and temporal baselines on the 94.9-min full-cycle test video", fontsize=10)
ax.grid(axis="y", alpha=0.3)
fig.savefig(os.path.join(FIGS, "fig_ablation.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig_ablation.png")

# ---------- Fig E: local view timeline sep25 ----------
d = np.load(os.path.join(RES, "pred2_local_sep25_v7.npz"), allow_pickle=True)
times, gt, raw, sm = d["times"], d["gt"], d["raw"], d["sm"]
fig, ax = plt.subplots(figsize=(9, 2.4))
band(ax, times, gt, 2.2)
band(ax, times, raw, 1.1)
band(ax, times, sm, 0.0)
ax.axvline(360 / 3600, color="k", ls="--", lw=0.8)
ax.set_yticks([2.2, 1.1, 0.0])
ax.set_yticklabels(["Ground truth", "v7 raw", "v7 decoded"])
ax.set_xlabel("Time (h)")
ax.set_xlim(0, times[-1] / 3600)
ax.set_ylim(-0.6, 2.8)
handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[i]) for i in [3, 4]]
ax.legend(handles, [STAGES_EN[3], STAGES_EN[4]], ncol=2,
          loc="upper center", bbox_to_anchor=(0.5, 1.35), frameon=False, fontsize=8)
fig.savefig(os.path.join(FIGS, "fig_local_sep25.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print("fig_local_sep25.png")
print("[DONE] figs ->", FIGS)

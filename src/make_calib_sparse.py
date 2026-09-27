# -*- coding: utf-8 -*-
"""
Sparse temporal calibration clip generation: uniformly spread K short windows
of W seconds within each stage interval of the test video, cut from extracted
features, written to FEATURE_DIR for incremental training.
Compared with the "first-30% continuous prefix" protocol, sparse sampling covers
the appearance evolution across the whole stage, mitigating intra-session drift.
Use the generated calib_sparse_ranges.json to mask all sampled windows during
evaluation (anti-leakage).
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

FEAT2 = os.path.join(C.ROOT, "outputs", "features_test")

# (source feature name, clip prefix, {class id: (K windows, W window length s)})
JOBS = [
    ("cycle5h_raw", "calib5s", {0: (6, 10), 1: (6, 10), 2: (8, 15), 3: (10, 20), 4: (6, 10)}),
    ("local_sep25", "calibs25", {3: (4, 10), 4: (6, 10)}),
    ("local_sep21", "calibs21", {3: (6, 10)}),
]


def sparse_windows(s, e, k, w):
    """Uniformly spread k windows of length w inside [s, e]; return [(ws, we), ...]"""
    if e - s <= w:
        return []
    k = max(1, min(k, int((e - s) // (w * 2))))
    starts = np.linspace(s, e - w, k)
    return [(float(ws), float(ws + w)) for ws in starts]


def main():
    ranges = {}
    for src, prefix, plan in JOBS:
        z = np.load(os.path.join(FEAT2, f"{src}.npz"))
        times, cnn, det = z["times"], z["cnn"], z["det"]
        intervals = C.TEST_VIDEOS2[src]["intervals"]
        ranges[src] = []
        for s, e, cls in intervals:
            k, w = plan.get(cls, (6, 10))
            wins = sparse_windows(s, e, k, w)
            for i, (ws, we) in enumerate(wins):
                mask = (times >= ws) & (times < we)
                if mask.sum() < 8:
                    continue
                name = f"{prefix}_{C.CLASSES[cls]}_{i}"
                np.savez_compressed(os.path.join(C.FEATURE_DIR, f"{name}.npz"),
                                    cnn=cnn[mask], det=det[mask], label=cls,
                                    sample_fps=C.SAMPLE_FPS)
                ranges[src].append([ws, we])
            print(f"  {prefix}_{C.CLASSES[cls]}: {len(wins)} windows x {w}s "
                  f"(interval {s:.0f}-{e:.0f}s, {sum(int((we-ws)*C.SAMPLE_FPS) for ws,we in wins)} frames)")
    out = os.path.join(FEAT2, "calib_sparse_ranges.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(ranges, f, ensure_ascii=False, indent=2)
    print("[INFO] sparse calibration ranges ->", out)


if __name__ == "__main__":
    main()

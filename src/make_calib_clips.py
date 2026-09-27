# -*- coding: utf-8 -*-
"""
On-site calibration clip generation (prefix protocol): from extracted test-video
features, cut the first 30% of each ground-truth stage interval as calibration
clips, written to FEATURE_DIR for incremental training (the
"deploy -> on-site calibration -> incremental adaptation" workflow in the paper).
Use --exclude to mask calibration time ranges during evaluation (anti-leakage).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

CALIB_FRAC = 0.30
FEAT2 = os.path.join(C.ROOT, "outputs", "features_test")

# (source feature name, clip name prefix) - intervals from config TEST_VIDEOS2
JOBS = [
    ("cycle5h_raw", "calib5h"),
    ("local_sep25", "calibsep25"),
    ("local_sep21", "calibsep21"),
]

import json


def main():
    ranges = {}
    for src, prefix in JOBS:
        z = np.load(os.path.join(FEAT2, f"{src}.npz"))
        times, cnn, det = z["times"], z["cnn"], z["det"]
        intervals = C.TEST_VIDEOS2[src]["intervals"]
        ranges[src] = []
        for s, e, cls in intervals:
            ce = s + (e - s) * CALIB_FRAC
            mask = (times >= s) & (times < ce)
            if mask.sum() < 16:
                print(f"  skip too-short segment {prefix}_{C.CLASSES[cls]}")
                continue
            name = f"{prefix}_{C.CLASSES[cls]}"
            np.savez_compressed(os.path.join(C.FEATURE_DIR, f"{name}.npz"),
                                cnn=cnn[mask], det=det[mask], label=cls,
                                sample_fps=C.SAMPLE_FPS)
            ranges[src].append([float(s), float(ce)])
            print(f"  calib clip {name}: {s:.0f}-{ce:.0f}s ({mask.sum()} frames) class={C.CLASSES[cls]}")
    with open(os.path.join(FEAT2, "calib_ranges.json"), "w", encoding="utf-8") as f:
        json.dump(ranges, f, ensure_ascii=False, indent=2)
    print("[INFO] calibration ranges ->", os.path.join(FEAT2, "calib_ranges.json"))


if __name__ == "__main__":
    main()

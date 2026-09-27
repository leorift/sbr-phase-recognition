# -*- coding: utf-8 -*-
"""
Generate a phase-transition highlight demo video: take a window around each GT
boundary of the test video, overlaying the GT phase and the v3 model (decoded)
predicted phase.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from infer import draw_overlay_text

OUT_FPS = 8
PRE_S, POST_S = 25, 35   # 25 s before / 35 s after each boundary


def gt_at(t):
    for s, e, c in C.TEST_GT_INTERVALS:
        if s <= t < e:
            return c
    return C.TEST_GT_INTERVALS[-1][2]


def get_roi(t):
    for s, e, box in C.TEST_ROI_SEGMENTS:
        if s <= t < e:
            return box
    return C.TEST_ROI_SEGMENTS[-1][2]


def main():
    z = np.load(os.path.join(C.RESULT_DIR, "test_pred_v3.npz"))
    times, sm = z["times"], z["sm"]

    boundaries = [iv[0] for iv in C.TEST_GT_INTERVALS[1:6]]  # 5 transitions
    out_path = os.path.join(C.RESULT_DIR, "demo_boundaries_v3.avi")

    cap = cv2.VideoCapture(C.TEST_VIDEO)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / OUT_FPS)))
    writer = None

    for bi, bt in enumerate(boundaries):
        f_start = int((bt - PRE_S) * src_fps)
        f_end = int((bt + POST_S) * src_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
        idx = f_start
        while idx < f_end:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                t = idx / src_fps
                x1, y1, x2, y2 = get_roi(t)
                roi = frame[y1:y2, x1:x2]
                roi = cv2.resize(roi, (480, 640))
                si = int(np.argmin(np.abs(times - t)))
                pred, gt = int(sm[si]), gt_at(t)
                match = pred == gt
                canvas = np.zeros((720, 480, 3), dtype=np.uint8)
                canvas[60:700, :] = roi
                gt_text = f"GT: {C.CLASSES[gt]}"
                pred_text = f"Pred: {C.CLASSES[pred]}"
                canvas = draw_overlay_text(canvas, gt_text, (20, 12),
                                           (255, 255, 255))
                canvas = draw_overlay_text(
                    canvas, pred_text, (240, 12),
                    (60, 220, 60) if match else (60, 60, 255))
                if writer is None:
                    writer = cv2.VideoWriter(
                        out_path, cv2.VideoWriter_fourcc(*"XVID"),
                        OUT_FPS, (480, 720))
                writer.write(canvas)
            idx += 1
        print(f"  boundary {bi+1}/5 ({int(bt//60)}:{int(bt%60):02d}) done")
    cap.release()
    writer.release()
    print("[INFO] demo video ->", out_path)


if __name__ == "__main__":
    main()

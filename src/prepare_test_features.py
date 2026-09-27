# -*- coding: utf-8 -*-
"""
Full-cycle test video feature extraction (segmented ROI + resumable chunks):
sample at 2 fps, apply per-segment ROI crops, extract EfficientNet-B0 + YOLO
features frame by frame, and generate per-frame labels from GT intervals.
Output: outputs/features_test/full_cycle.npz  {cnn, det, times, gt}
"""
import os
import sys
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from extract_features import build_cnn, cnn_forward, det_features

OUT_DIR = os.path.join(C.ROOT, "outputs", "features_test")
CHUNK_SEC = 300  # one chunk per 5 min, resumable


def get_roi(t_sec):
    for s, e, box in C.TEST_ROI_SEGMENTS:
        if s <= t_sec < e:
            return box
    return C.TEST_ROI_SEGMENTS[-1][2]


def gt_at(t_sec):
    for s, e, c in C.TEST_GT_INTERVALS:
        if s <= t_sec < e:
            return c
    return C.TEST_GT_INTERVALS[-1][2]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = C.DEVICE if torch.cuda.is_available() else "cpu"
    cnn_model = build_cnn(device)
    yolo_model = YOLO(C.YOLO_WEIGHTS)

    cap = cv2.VideoCapture(C.TEST_VIDEO)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps
    step = max(1, int(round(src_fps / C.SAMPLE_FPS)))
    print(f"[INFO] test video {duration/60:.1f}min, step {step} frames")

    t0 = time.time()
    chunk_start = 0
    while chunk_start < duration:
        chunk_path = os.path.join(OUT_DIR, f"chunk_{chunk_start:05d}.npz")
        if os.path.exists(chunk_path):
            chunk_start += CHUNK_SEC
            continue
        cnn_feats, det_feats, times = [], [], []
        buf = []
        f_start = int(chunk_start * src_fps)
        f_end = int(min(chunk_start + CHUNK_SEC, duration) * src_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_start)
        idx = f_start
        while idx < f_end:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                t_sec = idx / src_fps
                x1, y1, x2, y2 = get_roi(t_sec)
                buf.append(frame[y1:y2, x1:x2])
                times.append(t_sec)
                if len(buf) >= C.EXTRACT_BATCH:
                    _flush(buf, cnn_model, yolo_model, device, cnn_feats, det_feats)
            idx += 1
        if buf:
            _flush(buf, cnn_model, yolo_model, device, cnn_feats, det_feats)
        cnn = np.concatenate(cnn_feats, axis=0).astype(np.float16)
        det = np.concatenate(det_feats, axis=0).astype(np.float32)
        np.savez_compressed(chunk_path, cnn=cnn, det=det,
                            times=np.array(times, dtype=np.float32))
        print(f"  chunk {chunk_start}-{chunk_start+CHUNK_SEC}s done "
              f"({cnn.shape[0]} frames, {time.time()-t0:.0f}s total)")
        chunk_start += CHUNK_SEC
    cap.release()

    # merge chunks
    cnns, dets, times_all = [], [], []
    t = 0
    while True:
        p = os.path.join(OUT_DIR, f"chunk_{t:05d}.npz")
        if not os.path.exists(p):
            break
        z = np.load(p)
        cnns.append(z["cnn"]); dets.append(z["det"]); times_all.append(z["times"])
        t += CHUNK_SEC
    cnn = np.concatenate(cnns, axis=0)
    det = np.concatenate(dets, axis=0)
    times_arr = np.concatenate(times_all, axis=0)
    gt = np.array([gt_at(t) for t in times_arr], dtype=int)
    np.savez_compressed(os.path.join(OUT_DIR, "full_cycle.npz"),
                        cnn=cnn, det=det, times=times_arr, gt=gt)
    print(f"[INFO] merged: {cnn.shape[0]} frames ->",
          os.path.join(OUT_DIR, "full_cycle.npz"))


def _flush(buf, cnn_model, yolo_model, device, cnn_feats, det_feats):
    cnn_feats.append(cnn_forward(cnn_model, buf, device))
    results = yolo_model.predict(buf, device=device, verbose=False, conf=C.YOLO_CONF)
    det_feats.append(np.stack(
        [det_features(r, buf[0].shape[0], C.YOLO_CONF) for r in results]))
    buf.clear()


if __name__ == "__main__":
    main()

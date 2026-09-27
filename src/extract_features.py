# -*- coding: utf-8 -*-
"""
Feature extraction: sample each labeled video at SAMPLE_FPS and extract
per-frame EfficientNet-B0 spatial features (1280d) + YOLO structured
detection features (12d). Saved to outputs/features/<video>.npz
"""
import os
import sys
import time

import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

# ImageNet normalization
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_cnn(device):
    """EfficientNet-B0 (ImageNet pretrained), classifier head removed, GAP -> 1280d."""
    model = torchvision.models.efficientnet_b0(weights="DEFAULT")
    model.classifier = torch.nn.Identity()
    model.eval().to(device)
    return model


@torch.no_grad()
def cnn_forward(model, frames_bgr, device):
    """frames_bgr: list of np.ndarray(H,W,3) BGR -> [N,1280]"""
    x = np.stack([
        cv2.resize(f, (C.CNN_INPUT_SIZE, C.CNN_INPUT_SIZE))[:, :, ::-1]  # BGR->RGB
        for f in frames_bgr
    ]).astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    x = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)  # [N,3,224,224]
    feat = model.features(x)                 # [N,1280,7,7]
    feat = feat.mean(dim=(2, 3))             # GAP -> [N,1280]
    return feat.cpu().numpy()


def pick_best_box(result, cls_id, conf_thr):
    """Highest-confidence box of a given class -> (conf, y1, y2) or None."""
    best = None
    if result.boxes is None:
        return None
    for box in result.boxes:
        if int(box.cls.item()) != cls_id:
            continue
        conf = float(box.conf.item())
        if conf < conf_thr:
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        if best is None or conf > best[0]:
            best = (conf, float(y1), float(y2))
    return best


def det_features(result, frame_h, conf_thr):
    """
    12-d structured detection features per frame (normalized):
    [mud present, mud conf, mud height, mud top-y,
     water present, water conf, water height, water top-y,
     bubble present, bubble count, bubble max conf, mud/water height ratio]
    """
    feat = np.zeros(C.DET_FEAT_DIM, dtype=np.float32)

    mud = pick_best_box(result, C.ID_MUD, conf_thr)
    if mud is not None:
        conf, y1, y2 = mud
        feat[0] = 1.0
        feat[1] = conf
        feat[2] = abs(y2 - y1) / frame_h
        feat[3] = y1 / frame_h

    water = pick_best_box(result, C.ID_WATER, conf_thr)
    if water is not None:
        conf, y1, y2 = water
        feat[4] = 1.0
        feat[5] = conf
        feat[6] = abs(y2 - y1) / frame_h
        feat[7] = y1 / frame_h

    n_bubble, max_conf = 0, 0.0
    if result.boxes is not None:
        for box in result.boxes:
            if int(box.cls.item()) == C.ID_BUBBLE:
                conf = float(box.conf.item())
                if conf >= conf_thr:
                    n_bubble += 1
                    max_conf = max(max_conf, conf)
    if n_bubble > 0:
        feat[8] = 1.0
        feat[9] = min(n_bubble, 20) / 20.0
        feat[10] = max_conf

    if mud is not None and water is not None and feat[6] > 1e-4:
        feat[11] = min(feat[2] / feat[6], 3.0) / 3.0

    return feat


def extract_one_video(video_path, label, cnn_model, yolo_model, device):
    """Process one video -> cnn[T,1280](float16), det[T,12](float32)"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(src_fps / C.SAMPLE_FPS)))

    cnn_feats, det_feats = [], []
    buf = []
    idx = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            buf.append(frame)
            if len(buf) >= C.EXTRACT_BATCH:
                _flush(buf, cnn_model, yolo_model, device, frame_h, cnn_feats, det_feats)
        idx += 1
    if buf:
        _flush(buf, cnn_model, yolo_model, device, frame_h, cnn_feats, det_feats)
    cap.release()

    cnn = np.concatenate(cnn_feats, axis=0).astype(np.float16)
    det = np.concatenate(det_feats, axis=0).astype(np.float32)
    print(f"  {os.path.basename(video_path)}: {total} frames -> {cnn.shape[0]} sampled "
          f"({time.time()-t0:.1f}s) label={C.CLASSES[label]}")
    return cnn, det


def _flush(buf, cnn_model, yolo_model, device, frame_h, cnn_feats, det_feats):
    cnn_feats.append(cnn_forward(cnn_model, buf, device))
    results = yolo_model.predict(buf, device=device, verbose=False, conf=C.YOLO_CONF)
    det_feats.append(np.stack([det_features(r, frame_h, C.YOLO_CONF) for r in results]))
    buf.clear()


def main():
    os.makedirs(C.FEATURE_DIR, exist_ok=True)
    device = C.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")
    cnn_model = build_cnn(device)
    yolo_model = YOLO(C.YOLO_WEIGHTS)

    jobs = [(C.VIDEO_DIR, C.FILENAME_TO_CLASS)]
    if os.path.isdir(C.NEW_VIDEO_DIR):
        jobs.append((C.NEW_VIDEO_DIR, C.NEW_FILENAME_TO_CLASS))
    for vdir, mapping in jobs:
        for name, label in mapping.items():
            out_path = os.path.join(C.FEATURE_DIR, f"{name}.npz")
            if os.path.exists(out_path):
                print(f"  skip (exists): {name}")
                continue
            video_path = os.path.join(vdir, f"{name}.mp4")
            if not os.path.exists(video_path):
                print(f"  warning: missing {video_path}")
                continue
            cnn, det = extract_one_video(video_path, label, cnn_model, yolo_model, device)
            np.savez_compressed(out_path, cnn=cnn, det=det, label=label,
                                sample_fps=C.SAMPLE_FPS)
    print("[INFO] all features extracted ->", C.FEATURE_DIR)


if __name__ == "__main__":
    main()

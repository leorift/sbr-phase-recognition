# -*- coding: utf-8 -*-
"""
General test-video evaluation pipeline (multi-video validation for the paper):
  python eval_video.py --name cycle5h_roi --ckpt <weights> [--tag output_tag]
Flow: chunked feature extraction (300 s chunks, resumable) -> merge ->
      model inference -> metrics (frame acc / per-stage / boundary error with
      +/-30 s tolerance / segments / illegal transitions) -> JSON + npz
"""
import argparse
import json
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
from model import MSTCNPP
from state_machine import smooth_offline

FEAT2_DIR = os.path.join(C.ROOT, "outputs", "features_test")
CHUNK_SEC = 300


def gt_at(intervals, t):
    for s, e, c in intervals:
        if s <= t < e:
            return c
    return -1


def extract(video_path, name, roi=None):
    """Chunked feature extraction (optional fixed ROI crop), returns npz path."""
    out_npz = os.path.join(FEAT2_DIR, f"{name}.npz")
    if os.path.exists(out_npz):
        print(f"[INFO] features exist: {out_npz}")
        return out_npz
    os.makedirs(FEAT2_DIR, exist_ok=True)
    device = C.DEVICE if torch.cuda.is_available() else "cpu"
    cnn_model = build_cnn(device)
    yolo_model = YOLO(C.YOLO_WEIGHTS)

    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps
    step = max(1, int(round(src_fps / C.SAMPLE_FPS)))
    print(f"[INFO] {name}: {duration/60:.1f}min, step {step}")

    t0 = time.time()
    chunk_start = 0
    while chunk_start < duration:
        chunk_path = os.path.join(FEAT2_DIR, f"{name}_chunk_{chunk_start:06d}.npz")
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
            if idx % step == 0:
                ret, frame = cap.read()
                if not ret:
                    break
                if roi is not None:
                    x1, y1, x2, y2 = roi
                    frame = frame[y1:y2, x1:x2]
                buf.append(frame)
                times.append(idx / src_fps)
                if len(buf) >= C.EXTRACT_BATCH:
                    _flush(buf, cnn_model, yolo_model, device, cnn_feats, det_feats)
            else:
                # grab() skips pixel decoding for non-sampled frames (much faster)
                if not cap.grab():
                    break
            idx += 1
        if buf:
            _flush(buf, cnn_model, yolo_model, device, cnn_feats, det_feats)
        np.savez_compressed(chunk_path,
                            cnn=np.concatenate(cnn_feats, axis=0).astype(np.float16),
                            det=np.concatenate(det_feats, axis=0).astype(np.float32),
                            times=np.array(times, dtype=np.float32))
        print(f"  chunk {chunk_start}s done ({len(times)} frames, {time.time()-t0:.0f}s total)")
        chunk_start += CHUNK_SEC
    cap.release()

    cnns, dets, times_all = [], [], []
    t = 0
    while True:
        p = os.path.join(FEAT2_DIR, f"{name}_chunk_{t:06d}.npz")
        if not os.path.exists(p):
            break
        z = np.load(p)
        cnns.append(z["cnn"]); dets.append(z["det"]); times_all.append(z["times"])
        t += CHUNK_SEC
    np.savez_compressed(out_npz, cnn=np.concatenate(cnns, axis=0),
                        det=np.concatenate(dets, axis=0),
                        times=np.concatenate(times_all, axis=0))
    print(f"[INFO] features merged -> {out_npz}")
    return out_npz


def _flush(buf, cnn_model, yolo_model, device, cnn_feats, det_feats):
    cnn_feats.append(cnn_forward(cnn_model, buf, device))
    results = yolo_model.predict(buf, device=device, verbose=False, conf=C.YOLO_CONF)
    det_feats.append(np.stack(
        [det_features(r, buf[0].shape[0], C.YOLO_CONF) for r in results]))
    buf.clear()


def count_illegal(stages):
    """Count raw transitions violating the cyclic order (hold / +1 step are legal)."""
    n, illegal = 0, 0
    for i in range(1, len(stages)):
        if stages[i] != stages[i - 1]:
            n += 1
            if stages[i] != (stages[i - 1] + 1) % C.NUM_CLASSES:
                illegal += 1
    return n, illegal


def boundary_errors_tol(times, stages, intervals, tol=30.0):
    """Boundary matching with tolerance: a predicted transition with matching
    direction and |error| <= tol counts as a hit."""
    gt_trans = [(iv[0], iv[2]) for iv in intervals[1:] if iv[2] >= 0]
    pred_trans = []
    for i in range(1, len(stages)):
        if stages[i] != stages[i - 1]:
            pred_trans.append((float(times[i]), int(stages[i - 1]), int(stages[i])))
    results, used = [], set()
    for gt_t, gt_to in gt_trans:
        best, best_j = None, -1
        for j, (pt, pf, pto) in enumerate(pred_trans):
            if j in used or pto != gt_to:
                continue
            err = pt - gt_t
            if best is None or abs(err) < abs(best):
                best, best_j = err, j
        matched = best is not None and abs(best) <= tol
        if matched:
            used.add(best_j)
        results.append({"true_t": gt_t, "to_stage": C.CLASSES[gt_to],
                        "error_s": round(best, 1) if best is not None else None,
                        "matched": matched})
    hit = [b for b in results if b["matched"]]
    mae = float(np.mean([abs(b["error_s"]) for b in hit])) if hit else None
    return results, len(hit), mae


def evaluate_ckpt(feat_npz, intervals, ckpt_path, tag, name, exclude=None):
    z = np.load(feat_npz)
    feat = np.concatenate([z["cnn"].astype(np.float32), z["det"].astype(np.float32)], axis=1)
    times = z["times"]
    gt = np.array([gt_at(intervals, t) for t in times], dtype=int)
    valid = gt >= 0
    if exclude:  # mask calibration windows (anti-leakage)
        excl = np.zeros(len(times), dtype=bool)
        for s, e in exclude:
            excl |= (times >= s) & (times < e)
        valid &= ~excl

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    device = C.DEVICE if torch.cuda.is_available() else "cpu"
    model = MSTCNPP(input_dim=ckpt["input_dim"], num_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    feat_n = (feat - ckpt["feat_mean"]) / ckpt["feat_std"]
    with torch.no_grad():
        x = torch.from_numpy(feat_n).unsqueeze(0).to(device)
        probs = torch.softmax(model(x)[-1], dim=1)[0].permute(1, 0).cpu().numpy()

    raw = probs.argmax(1)
    sm = smooth_offline(probs)
    n_trans_raw, illegal_raw = count_illegal(raw)
    n_trans_sm, illegal_sm = count_illegal(sm)

    def metrics(stages):
        acc = float((stages[valid] == gt[valid]).mean())
        per_stage = {}
        for c in range(C.NUM_CLASSES):
            m = valid & (gt == c)
            if m.sum() > 0:
                per_stage[C.CLASSES[c]] = round(float((stages[m] == c).mean()), 4)
        bounds, hit, mae = boundary_errors_tol(times, stages, intervals)
        seg = int(1 + np.sum(stages[1:] != stages[:-1]))
        return {"frame_acc": round(acc, 4), "per_stage_acc": per_stage,
                "segments": seg, "boundaries": bounds,
                "boundary_recall": f"{hit}/{len(bounds)}",
                "boundary_mae_s": round(mae, 1) if mae is not None else None}

    result = {"video": name, "ckpt": os.path.basename(ckpt_path), "tag": tag,
              "raw": metrics(raw), "state_machine": metrics(sm),
              "transitions": {"raw_total": n_trans_raw, "raw_illegal": illegal_raw,
                              "decoded_total": n_trans_sm, "decoded_illegal": illegal_sm}}
    with open(os.path.join(C.RESULT_DIR, f"eval2_{name}_{tag}.json"), "w",
              encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    np.savez_compressed(os.path.join(C.RESULT_DIR, f"pred2_{name}_{tag}.npz"),
                        times=times, gt=gt, raw=raw, sm=sm,
                        probs=probs.astype(np.float16))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=list(C.TEST_VIDEOS2.keys()))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--exclude", default=None,
                    help='masked time ranges JSON, e.g. "[[0,110],[367,448]]"')
    ap.add_argument("--skip-extract", action="store_true")
    a = ap.parse_args()
    spec = C.TEST_VIDEOS2[a.name]
    npz = (os.path.join(FEAT2_DIR, f"{a.name}.npz") if a.skip_extract
           else extract(spec["file"], a.name, spec.get("roi")))
    exclude = json.loads(a.exclude) if a.exclude else None
    evaluate_ckpt(npz, spec["intervals"], a.ckpt, a.tag or os.path.basename(a.ckpt),
                  a.name, exclude=exclude)

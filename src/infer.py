# -*- coding: utf-8 -*-
"""
End-to-end inference pipeline:
video -> (EfficientNet-B0 + YOLO) per-frame features -> MS-TCN++ per-frame
phase probabilities -> cycle-constrained state-machine decoding ->
annotated video + phase timeline figure + phase log.

Usage:
  python infer.py video.mp4 [video2.mp4 ...]   # specific videos
  python infer.py --all                        # all videos under videos/
"""
import os
import sys
import time
from datetime import datetime

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from extract_features import build_cnn, cnn_forward, det_features
from model import MSTCNPP
from state_machine import smooth_offline

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# display color per stage (BGR)
STAGE_COLORS = [(255, 160, 60), (200, 200, 80), (160, 80, 255),
                (60, 200, 60), (80, 80, 220), (60, 60, 255), (200, 120, 200)]

# cv2.putText does not support CJK; render text with PIL instead
from PIL import Image, ImageDraw, ImageFont
_FONT = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 34)


def draw_overlay_text(frame_bgr, text, pos, color_bgr):
    """Draw text on a BGR frame with PIL, return the drawn frame."""
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    draw.text(pos, text, font=_FONT, fill=color_bgr[::-1])  # BGR->RGB
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


class SBRPhaseRecognizer:
    """End-to-end phase recognizer: load models once, reuse across videos."""

    def __init__(self, ckpt_path=None, device=None):
        self.device = device or (C.DEVICE if torch.cuda.is_available() else "cpu")
        ckpt_path = ckpt_path or os.path.join(C.CKPT_DIR, "ms_tcnpp_best.pt")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.mean = ckpt["feat_mean"]
        self.std = ckpt["feat_std"]
        self.classes = ckpt["classes"]

        self.cnn = build_cnn(self.device)
        self.yolo = YOLO(C.YOLO_WEIGHTS)
        self.tcn = MSTCNPP(input_dim=ckpt["input_dim"],
                           num_classes=len(self.classes)).to(self.device)
        self.tcn.load_state_dict(ckpt["model_state"])
        self.tcn.eval()
        print(f"[INFO] recognizer ready (device={self.device}, "
              f"ckpt epoch={ckpt.get('epoch')}, val_acc={ckpt.get('val_acc'):.4f})")

    @torch.no_grad()
    def predict_video(self, video_path):
        """Return stages[T], probs[T,N], sampling step, source fps."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video: {video_path}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        step = max(1, int(round(src_fps / C.SAMPLE_FPS)))

        cnn_feats, det_feats, buf = [], [], []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                buf.append(frame)
                if len(buf) >= C.EXTRACT_BATCH:
                    self._flush(buf, frame_h, cnn_feats, det_feats)
            idx += 1
        if buf:
            self._flush(buf, frame_h, cnn_feats, det_feats)
        cap.release()

        feat = np.concatenate([np.concatenate(cnn_feats, axis=0).astype(np.float32),
                               np.concatenate(det_feats, axis=0)], axis=1)
        feat = (feat - self.mean) / self.std
        x = torch.from_numpy(feat).unsqueeze(0).to(self.device)  # [1,T,F]
        probs = torch.softmax(self.tcn(x)[-1], dim=1)[0].permute(1, 0).cpu().numpy()
        stages = smooth_offline(probs)
        return stages, probs, step, src_fps

    def _flush(self, buf, frame_h, cnn_feats, det_feats):
        cnn_feats.append(cnn_forward(self.cnn, buf, self.device))
        results = self.yolo.predict(buf, device=self.device, verbose=False,
                                    conf=C.YOLO_CONF)
        det_feats.append(np.stack(
            [det_features(r, frame_h, C.YOLO_CONF) for r in results]))
        buf.clear()


def seg_count(stages):
    """Number of segments in the prediction (ideal: 1 for a single-phase video)."""
    return int(1 + np.sum(stages[1:] != stages[:-1])) if len(stages) else 0


def render_outputs(video_path, stages, probs, step, src_fps, out_dir, tag=""):
    """Generate annotated video, phase timeline figure and phase log."""
    name = os.path.splitext(os.path.basename(video_path))[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(out_dir, exist_ok=True)

    # ---- timeline figure ----
    t_sec = np.arange(len(stages)) * step / src_fps
    fig, ax = plt.subplots(figsize=(12, 3.2))
    for c in range(probs.shape[1]):
        ax.plot(t_sec, probs[:, c], label=C.CLASSES[c], alpha=0.65, linewidth=1)
    for t in range(1, len(stages)):
        if stages[t] != stages[t - 1]:
            ax.axvline(t_sec[t], color="k", linestyle="--", alpha=0.5)
    ax.set_xlabel("time (s)"); ax.set_ylabel("phase probability")
    ax.set_ylim(0, 1.02)
    ax.legend(ncol=7, fontsize=8, loc="upper center")
    ax.set_title(f"{name} - decoded phases: "
                 + " / ".join(sorted({C.CLASSES[s] for s in stages})))
    fig.tight_layout()
    timeline_path = os.path.join(out_dir, f"{name}_timeline_{ts}.png")
    fig.savefig(timeline_path, dpi=150)
    plt.close(fig)

    # ---- phase log ----
    log_path = os.path.join(out_dir, f"{name}_stagelog_{ts}.txt")
    changes = []
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("SBR phase recognition result (CNN + MS-TCN++ + cycle-constrained decoder)\n")
        f.write(f"input: {video_path}\nsample rate: {C.SAMPLE_FPS} fps\n")
        f.write("=" * 60 + "\n")
        f.write(f"00:00:00  initial phase: {C.CLASSES[stages[0]]}\n")
        for t in range(1, len(stages)):
            if stages[t] != stages[t - 1]:
                sec = t * step / src_fps
                hms = f"{int(sec//3600):02d}:{int(sec%3600//60):02d}:{int(sec%60):02d}"
                line = f"{hms}  transition: {C.CLASSES[stages[t-1]]} -> {C.CLASSES[stages[t]]}"
                f.write(line + "\n")
                changes.append(line)
        f.write("=" * 60 + "\n")
        f.write(f"segments: {seg_count(stages)} (ideal=1 for single-phase video)\n")

    # ---- annotated video (limited fps/resolution to control encoding cost) ----
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = min(fps, 8.0)
    frame_skip = max(1, int(round(fps / out_fps)))
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, 1280.0 / w0)
    w, h = int(w0 * scale), int(h0 * scale)
    video_out = os.path.join(out_dir, f"{name}_detected_{ts}.avi")
    writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"XVID"), out_fps, (w, h))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_skip != 0:
            idx += 1
            continue
        if scale < 1.0:
            frame = cv2.resize(frame, (w, h))
        si = min(idx // step, len(stages) - 1)
        stage = stages[si]
        conf = probs[si, stage]
        color = STAGE_COLORS[stage % len(STAGE_COLORS)]
        label = f"Stage: {C.CLASSES[stage]} ({conf:.2f})"
        sec = idx / fps
        info = f"Time: {int(sec//60):02d}:{int(sec%60):02d}"
        cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
        frame = draw_overlay_text(frame, label, (20, 12), color)
        frame = draw_overlay_text(frame, info, (w - 260, 12), (255, 255, 255))
        writer.write(frame)
        idx += 1
    cap.release()
    writer.release()
    return {"timeline": timeline_path, "log": log_path, "video": video_out,
            "segments": seg_count(stages), "changes": changes}


def main():
    args = sys.argv[1:]
    if not args or args[0] == "--all":
        videos = [os.path.join(C.VIDEO_DIR, f)
                  for f in sorted(os.listdir(C.VIDEO_DIR)) if f.endswith(".mp4")]
    else:
        videos = args

    recognizer = SBRPhaseRecognizer()
    summary = []
    for vp in videos:
        name = os.path.basename(vp)
        t0 = time.time()
        stages, probs, step, src_fps = recognizer.predict_video(vp)
        outs = render_outputs(vp, stages, probs, step, src_fps, C.RESULT_DIR)
        gt = os.path.splitext(name)[0]
        gt_label = C.FILENAME_TO_CLASS.get(gt)
        agree = float(np.mean(stages == gt_label)) if gt_label is not None else None
        summary.append({
            "video": name, "gt": C.CLASSES[gt_label] if gt_label is not None else "?",
            "major_pred": C.CLASSES[int(np.bincount(stages).argmax())],
            "frame_agree": round(agree, 4) if agree is not None else None,
            "segments": outs["segments"], "elapsed_s": round(time.time() - t0, 1),
        })
        print(f"[DONE] {name}: segments={outs['segments']} "
              f"frame_agree={summary[-1]['frame_agree']} elapsed={summary[-1]['elapsed_s']}s")

    print("\n===== inference summary =====")
    for s in summary:
        print(s)
    import json
    with open(os.path.join(C.RESULT_DIR, "inference_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

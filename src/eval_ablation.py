# -*- coding: utf-8 -*-
"""
Ablation model evaluation: supports feature slicing (cnn/det) and
receptive-field variants (stage1_layers). Evaluates full_cycle or
TEST_VIDEOS2 videos with the same metrics as eval_video.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from model import MSTCNPP
from state_machine import smooth_offline
from eval_video import gt_at, boundary_errors_tol, count_illegal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", default="full_cycle")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--subsample", type=int, default=1, help="frame subsampling (fps ablation)")
    a = ap.parse_args()

    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    device = C.DEVICE if torch.cuda.is_available() else "cpu"
    model = MSTCNPP(input_dim=ckpt["input_dim"], num_classes=len(ckpt["classes"]),
                    stage1_layers=ckpt.get("stage1_layers", C.TCN_STAGE1_LAYERS))
    model.load_state_dict(ckpt["model_state"]); model.eval().to(device)
    feat_mode = ckpt.get("feat_mode", "full")

    if a.name == "full_cycle":
        feat_npz = os.path.join(C.ROOT, "outputs", "features_test", "full_cycle.npz")
        intervals = C.TEST_GT_INTERVALS
    else:
        feat_npz = os.path.join(C.ROOT, "outputs", "features_test", f"{a.name}.npz")
        intervals = C.TEST_VIDEOS2[a.name]["intervals"]
    z = np.load(feat_npz)
    feat = np.concatenate([z["cnn"].astype(np.float32), z["det"].astype(np.float32)], 1)
    times = z["times"]
    if a.subsample > 1:
        feat, times = feat[::a.subsample], times[::a.subsample]
    if feat_mode == "cnn":
        feat = feat[:, :1280]
    elif feat_mode == "det":
        feat = feat[:, 1280:]
    gt = np.array([gt_at(intervals, t) for t in times], dtype=int)
    valid = gt >= 0
    if a.exclude:
        for s, e in json.loads(a.exclude):
            valid &= ~((times >= s) & (times < e))

    feat_n = (feat - ckpt["feat_mean"]) / ckpt["feat_std"]
    with torch.no_grad():
        x = torch.from_numpy(feat_n).unsqueeze(0).float().to(device)
        probs = torch.softmax(model(x)[-1], dim=1)[0].permute(1, 0).cpu().numpy()

    raw = probs.argmax(1)
    sm = smooth_offline(probs)
    nr, ir = count_illegal(raw); ns, ism = count_illegal(sm)

    def metrics(st):
        acc = float((st[valid] == gt[valid]).mean())
        per = {}
        for c in range(C.NUM_CLASSES):
            m = valid & (gt == c)
            if m.sum() > 0:
                per[C.CLASSES[c]] = round(float((st[m] == c).mean()), 4)
        bounds, hit, mae = boundary_errors_tol(times, st, intervals)
        seg = int(1 + np.sum(st[1:] != st[:-1]))
        return {"frame_acc": round(acc, 4), "per_stage_acc": per, "segments": seg,
                "boundaries": bounds, "boundary_recall": f"{hit}/{len(bounds)}",
                "boundary_mae_s": round(mae, 1) if mae is not None else None}

    res = {"video": a.name, "tag": a.tag, "subsample": a.subsample,
           "raw": metrics(raw), "state_machine": metrics(sm),
           "transitions": {"raw_total": nr, "raw_illegal": ir,
                           "decoded_total": ns, "decoded_illegal": ism}}
    out = os.path.join(C.RESULT_DIR, f"eval2_{a.name}_{a.tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    np.savez_compressed(os.path.join(C.RESULT_DIR, f"pred2_{a.name}_{a.tag}.npz"),
                        times=times, gt=gt, raw=raw, sm=sm,
                        probs=probs.astype(np.float16))
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

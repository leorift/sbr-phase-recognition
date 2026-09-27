# -*- coding: utf-8 -*-
"""
Evaluate any MS-TCN++ checkpoint on full-cycle test video features:
frame accuracy / per-stage accuracy / boundary localization error /
segment count (over-segmentation metric).
Outputs JSON metrics + prediction npz (for comparison plots).

Usage: python evaluate_test.py --ckpt <path> --tag <name>
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

TEST_FEAT = os.path.join(C.ROOT, "outputs", "features_test", "full_cycle.npz")


def boundary_errors(times, stages, gt_intervals):
    """
    For each ground-truth transition, find the nearest predicted transition
    with matching direction (prev stage -> next stage).
    Returns list of dict(true_t, pred_t, error_s, matched).
    """
    gt_trans = []
    for i in range(1, len(gt_intervals)):
        gt_trans.append((gt_intervals[i][0], gt_intervals[i - 1][2], gt_intervals[i][2]))

    pred_trans = []  # (t, from, to)
    for i in range(1, len(stages)):
        if stages[i] != stages[i - 1]:
            pred_trans.append((float(times[i]), int(stages[i - 1]), int(stages[i])))

    results, used = [], set()
    for gt_t, f_from, f_to in gt_trans:
        best, best_j = None, -1
        for j, (pt, pf, pto) in enumerate(pred_trans):
            if j in used or pto != f_to:
                continue
            err = pt - gt_t
            if best is None or abs(err) < abs(best):
                best, best_j = err, j
        if best is not None:
            used.add(best_j)
            results.append({"true_t": gt_t, "to_stage": C.CLASSES[f_to],
                            "error_s": round(best, 1), "matched": True})
        else:
            results.append({"true_t": gt_t, "to_stage": C.CLASSES[f_to],
                            "error_s": None, "matched": False})
    return results, len(pred_trans)


def evaluate(ckpt_path, tag):
    z = np.load(TEST_FEAT)
    feat = np.concatenate([z["cnn"].astype(np.float32), z["det"].astype(np.float32)], axis=1)
    times, gt = z["times"], z["gt"]

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

    def metrics(stages):
        acc = float((stages == gt).mean())
        per_stage = {}
        for c in range(C.NUM_CLASSES):
            mask = gt == c
            if mask.sum() > 0:
                per_stage[C.CLASSES[c]] = round(float((stages[mask] == c).mean()), 4)
        bounds, n_seg = boundary_errors(times, stages, C.TEST_GT_INTERVALS)
        matched = [b for b in bounds if b["matched"]]
        mae = float(np.mean([abs(b["error_s"]) for b in matched])) if matched else None
        return {
            "frame_acc": round(acc, 4),
            "per_stage_acc": per_stage,
            "segments": n_seg,
            "boundaries": bounds,
            "boundary_mae_s": round(mae, 1) if mae is not None else None,
            "boundary_matched": f"{len(matched)}/{len(bounds)}",
        }

    result = {"tag": tag, "ckpt": os.path.basename(ckpt_path),
              "raw": metrics(raw), "state_machine": metrics(sm)}
    out_json = os.path.join(C.RESULT_DIR, f"test_eval_{tag}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    np.savez_compressed(os.path.join(C.RESULT_DIR, f"test_pred_{tag}.npz"),
                        times=times, gt=gt, raw=raw, sm=sm, probs=probs.astype(np.float16))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    evaluate(a.ckpt, a.tag)

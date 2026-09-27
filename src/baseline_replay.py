# -*- coding: utf-8 -*-
"""
Offline replay of the hard-coded baseline (SBR detective.py):
faithfully ports its decision logic (initial-stage judgment + trend-triggered
transitions + first-transition correction), fed by YOLO detection quantities
extracted from the test video (aggregated to 1 Hz, matching the original
one-decision-per-second design).
Outputs metrics JSON + prediction npz in the same schema as evaluate_test.
"""
import json
import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from evaluate_test import boundary_errors

TEST_FEAT = os.path.join(C.ROOT, "outputs", "features_test", "full_cycle.npz")

# ===== parameters identical to the baseline program =====
TREND_WINDOW_SECONDS = 15
BUBBLE_RECENT_SECONDS = 5
BUBBLE_BASELINE_SECONDS = 10
BUBBLE_SIGNIFICANT_RATIO = 2.0
INITIAL_DETECTION_SECONDS = 5
STABLE_STD_THRESHOLD = 8.0
SIMILAR_HEIGHT_THRESHOLD = 12.0
MUD_LOWER_MARGIN = 15.0
INITIAL_BUBBLE_FREQUENCY_THRESHOLD = 0.5
SETTLING_RATIO_THRESHOLD = 3.0
IDLE_RATIO_THRESHOLD = 2.5

STAGE_ORDER = ["fill", "anoxic", "aerobic", "settling", "decant"]
BASELINE_TO_CLASS = {"fill": 0, "anoxic": 2, "aerobic": 3, "settling": 4, "decant": 5, "idle": 6}

STAGE_TRANSITION_LOGIC = {
    "fill": {"next_stage": "anoxic", "trigger": "mud_rise"},
    "anoxic": {"next_stage": "aerobic", "trigger": "bubble_surge"},
    "aerobic": {"next_stage": "settling", "trigger": "mud_fall"},
    "settling": {"next_stage": "decant", "trigger": "water_fall"},
    "decant": {"next_stage": "fill", "trigger": "water_rise"},
}
TRIGGER_TO_STAGE = {"water_rise": "fill", "mud_rise": "anoxic", "bubble_surge": "aerobic",
                    "mud_fall": "settling", "water_fall": "decant"}


def consecutive_trend(values, window, direction):
    if len(values) < window + 1:
        return False
    valid = [v for v in values[-(window + 1):] if v is not None]
    if len(valid) < window + 1:
        return False
    diffs = [valid[i] - valid[i - 1] for i in range(1, len(valid))]
    if direction == "up":
        return all(d > 0 for d in diffs[-window:])
    return all(d < 0 for d in diffs[-window:])


def bubble_frequency_surge(flags, recent_sec, baseline_sec, ratio):
    if len(flags) < baseline_sec + recent_sec:
        return False
    recent = list(flags)[-recent_sec:]
    baseline = list(flags)[-(baseline_sec + recent_sec):-recent_sec]
    r_rate = sum(recent) / len(recent) if recent else 0
    b_rate = sum(baseline) / len(baseline) if baseline else 0
    if b_rate == 0:
        return r_rate > 0
    return (r_rate / b_rate) >= ratio


def is_stable(values, threshold=STABLE_STD_THRESHOLD):
    valid = [v for v in values if v is not None]
    return len(valid) >= 3 and float(np.std(valid)) < threshold


def determine_initial_stage(mud_h, water_h, bubble_h):
    mud_v = [v for v in mud_h if v is not None]
    water_v = [v for v in water_h if v is not None]
    if not mud_v or not water_v:
        return STAGE_ORDER[0]
    avg_mud, avg_water = sum(mud_v) / len(mud_v), sum(water_v) / len(water_v)
    if consecutive_trend(mud_v, min(len(mud_v) - 1, TREND_WINDOW_SECONDS), "up"):
        return "fill"
    if consecutive_trend(water_v, min(len(water_v) - 1, TREND_WINDOW_SECONDS), "down"):
        return "decant"
    if is_stable(mud_v) and is_stable(water_v):
        diff = avg_water - avg_mud
        if abs(diff) < SIMILAR_HEIGHT_THRESHOLD:
            rate = sum(bubble_h) / len(bubble_h) if bubble_h else 0
            return "aerobic" if rate >= INITIAL_BUBBLE_FREQUENCY_THRESHOLD else "anoxic"
        elif diff > MUD_LOWER_MARGIN:
            ratio = avg_water / avg_mud if avg_mud > 0 else float("inf")
            if ratio >= SETTLING_RATIO_THRESHOLD:
                return "settling"
            elif ratio <= IDLE_RATIO_THRESHOLD:
                return "idle"
            return "settling"
    return "anoxic"


def main():
    z = np.load(TEST_FEAT)
    times, det, gt = z["times"], z["det"], z["gt"]

    # aggregate to 1 Hz: mud/water pixel heights (rescaled by ROI height), bubble flag
    sec_max = int(times[-1]) + 1
    mud_1hz = [None] * sec_max
    water_1hz = [None] * sec_max
    bubble_1hz = [0] * sec_max
    for i, t in enumerate(times):
        s = int(t)
        crop_h = 920 if 1665 <= t < 2960 else 1280  # per-segment ROI heights
        if det[i][0] > 0:
            h = det[i][2] * crop_h
            mud_1hz[s] = h if mud_1hz[s] is None else (mud_1hz[s] + h) / 2
        if det[i][4] > 0:
            h = det[i][6] * crop_h
            water_1hz[s] = h if water_1hz[s] is None else (water_1hz[s] + h) / 2
        if det[i][8] > 0:
            bubble_1hz[s] = 1

    present = {"mud": float(np.mean([m is not None for m in mud_1hz])),
               "water": float(np.mean([w is not None for w in water_1hz])),
               "bubble_sec_ratio": float(np.mean(bubble_1hz))}
    print("[diag] 1Hz detection availability:", {k: round(v, 3) for k, v in present.items()})

    # ===== replay baseline main loop (one decision per second) =====
    hist_len = TREND_WINDOW_SECONDS + BUBBLE_BASELINE_SECONDS + 10
    mud_hist = deque(maxlen=hist_len)
    water_hist = deque(maxlen=hist_len)
    bubble_hist = deque(maxlen=hist_len)

    stage_order = list(STAGE_ORDER)
    current = stage_order[0]
    initial_done = False
    first_corrected = False
    stage_per_sec = []

    for s in range(sec_max):
        mud_hist.append(mud_1hz[s])
        water_hist.append(water_1hz[s])
        bubble_hist.append(bubble_1hz[s])

        if not initial_done:
            if len(mud_hist) >= INITIAL_DETECTION_SECONDS:
                current = determine_initial_stage(list(mud_hist), list(water_hist),
                                                  list(bubble_hist))
                if current == "idle" and "idle" not in stage_order:
                    stage_order.append("idle")
                initial_done = True
        else:
            logic = STAGE_TRANSITION_LOGIC.get(current)
            if logic is None and current == "idle":
                logic = {"next_stage": "fill", "trigger": "water_rise"}
            if logic is not None:
                trig = logic["trigger"]
                if trig == "water_rise":
                    fired = consecutive_trend(list(water_hist), TREND_WINDOW_SECONDS, "up")
                elif trig == "mud_rise":
                    fired = consecutive_trend(list(mud_hist), TREND_WINDOW_SECONDS, "up")
                elif trig == "bubble_surge":
                    fired = bubble_frequency_surge(bubble_hist, BUBBLE_RECENT_SECONDS,
                                                   BUBBLE_BASELINE_SECONDS,
                                                   BUBBLE_SIGNIFICANT_RATIO)
                elif trig == "mud_fall":
                    fired = consecutive_trend(list(mud_hist), TREND_WINDOW_SECONDS, "down")
                else:
                    fired = consecutive_trend(list(water_hist), TREND_WINDOW_SECONDS, "down")
                if fired:
                    nxt = logic["next_stage"]
                    if not first_corrected:
                        target = TRIGGER_TO_STAGE.get(trig, nxt)
                        if target not in stage_order or current not in stage_order or \
                           stage_order.index(target) != (stage_order.index(current) + 1) % len(stage_order):
                            if target in stage_order:
                                idx = stage_order.index(target)
                                current = stage_order[(idx - 1) % len(stage_order)]
                        first_corrected = True
                        nxt = target
                    current = nxt
        stage_per_sec.append(current)

    # map back to the 2 fps frame-level sequence
    stage_cls = np.array([BASELINE_TO_CLASS[s] for s in stage_per_sec], dtype=int)
    frames = np.array([stage_cls[min(int(t), sec_max - 1)] for t in times], dtype=int)

    acc = float((frames == gt).mean())
    per_stage = {}
    for c in range(C.NUM_CLASSES):
        mask = gt == c
        if mask.sum() > 0:
            per_stage[C.CLASSES[c]] = round(float((frames[mask] == c).mean()), 4)
    bounds, n_seg = boundary_errors(times, frames, C.TEST_GT_INTERVALS)
    matched = [b for b in bounds if b["matched"]]
    mae = float(np.mean([abs(b["error_s"]) for b in matched])) if matched else None

    result = {
        "tag": "baseline", "ckpt": "hard-coded rules (SBR detective.py replay)",
        "detection_availability": {k: round(v, 3) for k, v in present.items()},
        "state_machine": {
            "frame_acc": round(acc, 4), "per_stage_acc": per_stage,
            "segments": n_seg, "boundaries": bounds,
            "boundary_mae_s": round(mae, 1) if mae is not None else None,
            "boundary_matched": f"{len(matched)}/{len(bounds)}",
        },
    }
    with open(os.path.join(C.RESULT_DIR, "test_eval_baseline.json"), "w",
              encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    np.savez_compressed(os.path.join(C.RESULT_DIR, "test_pred_baseline.npz"),
                        times=times, gt=gt, sm=frames)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Comparison baselines: with the same features and training protocol as MS-TCN++
  1) frame-wise MLP classifier (no temporal modeling)
  2) BiLSTM (2-layer bidirectional, temporal baseline)
Training protocol matches train.py (same data/split/class weights/optimizer);
evaluation reuses eval_video metrics (frame acc / boundary +/-30s / segments /
illegal transitions).
Usage:
  python baselines.py --arch mlp    --tag mlp --exclude-test-clips
  python baselines.py --arch bilstm --tag bilstm --exclude-test-clips
  python baselines.py --eval-only --arch bilstm --ckpt ... --name full_cycle
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from dataset import load_all_features, compute_norm_stats, SBRClipDataset


class FrameMLP(nn.Module):
    """Frame-wise classifier: each frame independently 1292 -> 256 -> 7 (no temporal context)"""

    def __init__(self, input_dim, num_classes=C.NUM_CLASSES, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes))

    def forward(self, x):           # x: [B, T, F]
        B, T, F = x.shape
        return self.net(x.reshape(B * T, F)).reshape(B, T, -1).permute(0, 2, 1)  # [B,C,T]


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM temporal classifier: projection -> 2-layer BiLSTM -> per-step head"""

    def __init__(self, input_dim, num_classes=C.NUM_CLASSES, proj=256, hidden=128, dropout=0.3):
        super().__init__()
        self.proj = nn.Linear(input_dim, proj)
        self.lstm = nn.LSTM(proj, hidden, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.head = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):           # x: [B, T, F]
        h = torch.relu(self.proj(x))
        o, _ = self.lstm(h)
        return self.head(o).permute(0, 2, 1)  # [B,C,T]


def build(arch, input_dim):
    return FrameMLP(input_dim) if arch == "mlp" else BiLSTMClassifier(input_dim)


def infer_probs(model, feat_n, device, chunk=4096):
    """Chunked inference over long sequences, returns (T, C) probabilities.
    BiLSTM chunking cuts context; use overlapping windows (256 frames context
    on each side) to mitigate boundary effects."""
    model.eval()
    ctx = 256
    outs = []
    with torch.no_grad():
        for s in range(0, len(feat_n), chunk):
            e = min(s + chunk, len(feat_n))
            s2, e2 = max(0, s - ctx), min(len(feat_n), e + ctx)
            x = torch.from_numpy(feat_n[s2:e2]).unsqueeze(0).float().to(device)
            p = torch.softmax(model(x), dim=1)[0].permute(1, 0).cpu().numpy()
            outs.append(p[s - s2: p.shape[0] - (e2 - e)])
    return np.concatenate(outs, 0)


def train(args):
    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    os.makedirs(C.CKPT_DIR, exist_ok=True)
    device = C.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}  arch: {args.arch}")

    items = load_all_features()
    if args.exclude_test_clips:
        items = [it for it in items if it["name"] not in C.NEW_CLIPS_FROM_TEST]
    if args.exclude_prefix:
        prefixes = tuple(p for p in args.exclude_prefix.split(",") if p)
        items = [it for it in items if not it["name"].startswith(prefixes)]
    print(f"[INFO] {len(items)} feature clips")

    mean, std = compute_norm_stats(items)
    input_dim = items[0]["feat"].shape[1]
    train_ds = SBRClipDataset(items, mean, std, split="train")
    val_ds = SBRClipDataset(items, mean, std, split="val")
    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=C.BATCH_SIZE, shuffle=False, num_workers=0)

    counts = np.zeros(C.NUM_CLASSES)
    for it in items:
        counts[it["label"]] += int(it["T"] * C.TRAIN_RATIO)
    weights = torch.tensor((counts.sum() / (counts + 1e-9)) / C.NUM_CLASSES,
                           dtype=torch.float32).to(device)

    model = build(args.arch, input_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] params: {n_params/1e6:.2f}M")
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1, best_epoch = -1.0, -1
    ckpt_path = os.path.join(C.CKPT_DIR, f"{args.arch}_{args.tag}.pt")
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)   # x:[B,T,F] y:[B,T]
            logits = model(x)                    # [B,C,T]
            loss = criterion(logits, y)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot += loss.item(); nb += 1
        scheduler.step()
        # validation
        model.eval(); ps, ts = [], []
        with torch.no_grad():
            for x, y in val_loader:
                ps.append(model(x.to(device)).argmax(1).cpu().numpy().ravel())
                ts.append(y.numpy().ravel())
        yp, yt = np.concatenate(ps), np.concatenate(ts)
        acc = float((yp == yt).mean())
        f1s = []
        for c in range(C.NUM_CLASSES):
            tp = ((yp == c) & (yt == c)).sum(); fp = ((yp == c) & (yt != c)).sum()
            fn = ((yp != c) & (yt == c)).sum()
            pr, rc = tp / (tp + fp + 1e-9), tp / (tp + fn + 1e-9)
            f1s.append(2 * pr * rc / (pr + rc + 1e-9))
        mf1 = float(np.mean(f1s))
        print(f"[E{epoch:02d}] loss={tot/nb:.4f} val_acc={acc:.4f} val_macroF1={mf1:.4f}")
        if mf1 > best_f1:
            best_f1, best_epoch = mf1, epoch
            torch.save({"arch": args.arch, "model_state": model.state_dict(),
                        "input_dim": input_dim, "feat_mean": mean, "feat_std": std,
                        "classes": C.CLASSES, "epoch": epoch,
                        "val_acc": acc, "val_macro_f1": mf1,
                        "n_params": n_params}, ckpt_path)
    print(f"[INFO] best epoch={best_epoch} val_macroF1={best_f1:.4f} -> {ckpt_path}")


def evaluate(args):
    from eval_video import gt_at, boundary_errors_tol, count_illegal
    from state_machine import smooth_offline
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    device = C.DEVICE if torch.cuda.is_available() else "cpu"
    model = build(ckpt["arch"], ckpt["input_dim"])
    model.load_state_dict(ckpt["model_state"]); model.eval().to(device)

    if args.name == "full_cycle":
        feat_npz = os.path.join(C.ROOT, "outputs", "features_test", "full_cycle.npz")
        intervals = C.TEST_GT_INTERVALS
    else:
        feat_npz = os.path.join(C.ROOT, "outputs", "features_test", f"{args.name}.npz")
        intervals = C.TEST_VIDEOS2[args.name]["intervals"]
    z = np.load(feat_npz)
    feat = np.concatenate([z["cnn"].astype(np.float32), z["det"].astype(np.float32)], 1)
    times = z["times"]
    gt = np.array([gt_at(intervals, t) for t in times], dtype=int)
    valid = gt >= 0
    if args.exclude:
        for s, e in json.loads(args.exclude):
            valid &= ~((times >= s) & (times < e))
    feat_n = (feat - ckpt["feat_mean"]) / ckpt["feat_std"]
    probs = infer_probs(model, feat_n, device)

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

    res = {"video": args.name, "arch": ckpt["arch"], "tag": args.tag,
           "raw": metrics(raw), "state_machine": metrics(sm),
           "transitions": {"raw_total": nr, "raw_illegal": ir,
                           "decoded_total": ns, "decoded_illegal": ism}}
    out = os.path.join(C.RESULT_DIR, f"eval2_{args.name}_{args.tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    np.savez_compressed(os.path.join(C.RESULT_DIR, f"pred2_{args.name}_{args.tag}.npz"),
                        times=times, gt=gt, raw=raw, sm=sm,
                        probs=probs.astype(np.float16))
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["mlp", "bilstm"], default="mlp")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--exclude-test-clips", action="store_true")
    ap.add_argument("--exclude-prefix", default=None)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--name", default="full_cycle")
    ap.add_argument("--exclude", default=None)
    a = ap.parse_args()
    if a.eval_only:
        evaluate(a)
    else:
        train(a)

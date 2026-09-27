# -*- coding: utf-8 -*-
"""
Train MS-TCN++: load pre-extracted features -> train -> validate
(frame accuracy / macro-F1 / confusion matrix) -> save best weights and
normalization stats -> output training curves and confusion matrix figure.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from dataset import load_all_features, compute_norm_stats, SBRClipDataset
from model import MSTCNPP, SBRPhaseLoss

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def evaluate(model, loader, device):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x)[-1]                    # final refinement stage [B,C,T]
            all_pred.append(preds.argmax(1).cpu().numpy().ravel())
            all_true.append(y.numpy().ravel())
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)

    acc = (y_pred == y_true).mean()
    f1s = []
    for c in range(C.NUM_CLASSES):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1s.append(2 * prec * rec / (prec + rec + 1e-9))
    macro_f1 = float(np.mean(f1s))

    cm = np.zeros((C.NUM_CLASSES, C.NUM_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return float(acc), macro_f1, f1s, cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="", help="checkpoint name suffix, e.g. v2")
    parser.add_argument("--exclude-test-clips", action="store_true",
                        help="exclude clips cut from test videos (anti-leakage)")
    parser.add_argument("--oversample-idle", type=int, default=1,
                        help="oversampling factor for the scarce post-fill idle class")
    parser.add_argument("--exclude-prefix", default=None,
                        help="exclude feature clips with given prefixes, comma separated")
    parser.add_argument("--epochs", type=int, default=C.EPOCHS)
    parser.add_argument("--init-from", default=None,
                        help="initialize from a checkpoint (resume / fine-tune)")
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--feat-mode", choices=["full", "cnn", "det"], default="full",
                        help="feature ablation: full=1292d / cnn=1280d only / det=12d only")
    parser.add_argument("--rf-layers", type=int, default=None,
                        help="receptive-field ablation: overrides TCN_STAGE1_LAYERS")
    args = parser.parse_args()

    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    os.makedirs(C.CKPT_DIR, exist_ok=True)
    os.makedirs(C.RESULT_DIR, exist_ok=True)
    device = C.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")

    ckpt_name = f"ms_tcnpp_{args.tag}.pt" if args.tag else "ms_tcnpp_best.pt"
    metrics_name = f"train_metrics_{args.tag}.json" if args.tag else "train_metrics.json"

    items = load_all_features()
    if args.exclude_test_clips:
        items = [it for it in items if it["name"] not in C.NEW_CLIPS_FROM_TEST]
    if args.exclude_prefix:
        prefixes = tuple(p for p in args.exclude_prefix.split(",") if p)
        items = [it for it in items if not it["name"].startswith(prefixes)]
    print(f"[INFO] loaded {len(items)} feature clips")

    if args.feat_mode != "full":  # feature ablation slicing
        for it in items:
            it["feat"] = it["feat"][:, :1280] if args.feat_mode == "cnn" \
                else it["feat"][:, 1280:]
        print(f"[INFO] feature ablation: {args.feat_mode} (dim={items[0]['feat'].shape[1]})")
    if args.rf_layers:
        C.TCN_STAGE1_LAYERS = args.rf_layers
        print(f"[INFO] RF ablation: TCN_STAGE1_LAYERS={args.rf_layers}")

    mean, std = compute_norm_stats(items)
    input_dim = items[0]["feat"].shape[1]

    train_ds = SBRClipDataset(items, mean, std, split="train",
                              class_oversample={1: args.oversample_idle})
    val_ds = SBRClipDataset(items, mean, std, split="val")
    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=C.BATCH_SIZE, shuffle=False,
                            num_workers=0)
    print(f"[INFO] train clips/epoch: {len(train_ds)}, val clips: {len(val_ds)}")

    # class-balanced weights (inverse frame frequency over train split)
    counts = np.zeros(C.NUM_CLASSES)
    for it in items:
        counts[it["label"]] += int(it["T"] * C.TRAIN_RATIO)
    weights = torch.tensor((counts.sum() / (counts + 1e-9)) / C.NUM_CLASSES,
                           dtype=torch.float32)

    model = MSTCNPP(input_dim=input_dim,
                    stage1_layers=C.TCN_STAGE1_LAYERS).to(device)
    criterion = SBRPhaseLoss(class_weights=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.LR,
                                  weight_decay=C.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    for _ in range(args.start_epoch):
        scheduler.step()

    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state"])
        print(f"[INFO] initialized from {args.init_from} (epoch={ck.get('epoch')})")

    history = {"loss": [], "val_acc": [], "val_f1": []}
    best_f1, best_epoch = -1.0, -1
    best_cm, best_f1s = None, None

    for epoch in range(args.start_epoch + 1, args.epochs + 1):
        model.train()
        epoch_loss, nb = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = criterion(preds, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            nb += 1
        scheduler.step()

        acc, macro_f1, f1s, cm = evaluate(model, val_loader, device)
        history["loss"].append(epoch_loss / nb)
        history["val_acc"].append(acc)
        history["val_f1"].append(macro_f1)
        print(f"[E{epoch:02d}] loss={epoch_loss/nb:.4f} "
              f"val_acc={acc:.4f} val_macroF1={macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1, best_epoch = macro_f1, epoch
            best_cm, best_f1s = cm, f1s
            torch.save({
                "model_state": model.state_dict(),
                "input_dim": input_dim,
                "feat_mean": mean, "feat_std": std,
                "classes": C.CLASSES,
                "epoch": epoch, "val_acc": acc, "val_macro_f1": macro_f1,
                "feat_mode": args.feat_mode,
                "stage1_layers": C.TCN_STAGE1_LAYERS,
            }, os.path.join(C.CKPT_DIR, ckpt_name))

    print(f"[INFO] best: epoch={best_epoch} val_macroF1={best_f1:.4f}")

    # ---- training curves ----
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(history["loss"], label="train loss", color="tab:blue")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(history["val_acc"], label="val acc", color="tab:green")
    ax2.plot(history["val_f1"], label="val macro-F1", color="tab:orange")
    ax2.set_ylabel("Metric")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right")
    plt.title("MS-TCN++ training curves")
    fig.tight_layout()
    fig.savefig(os.path.join(C.RESULT_DIR, f"training_curve{'_' + args.tag if args.tag else ''}.png"), dpi=150)
    plt.close(fig)

    # ---- confusion matrix ----
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(best_cm, cmap="Blues")
    ax.set_xticks(range(C.NUM_CLASSES)); ax.set_yticks(range(C.NUM_CLASSES))
    ax.set_xticklabels(C.CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(C.CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(C.NUM_CLASSES):
        for j in range(C.NUM_CLASSES):
            ax.text(j, i, best_cm[i, j], ha="center", va="center",
                    color="white" if best_cm[i, j] > best_cm.max() / 2 else "black",
                    fontsize=8)
    plt.title(f"Validation confusion matrix (epoch {best_epoch})")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(C.RESULT_DIR, f"confusion_matrix{'_' + args.tag if args.tag else ''}.png"), dpi=150)
    plt.close(fig)

    # ---- metrics summary ----
    report = {
        "best_epoch": best_epoch,
        "val_frame_acc": round(float(history["val_acc"][best_epoch - 1]), 4),
        "val_macro_f1": round(best_f1, 4),
        "per_class_f1": {C.CLASSES[i]: round(float(best_f1s[i]), 4)
                         for i in range(C.NUM_CLASSES)},
        "confusion_matrix": best_cm.tolist(),
    }
    with open(os.path.join(C.RESULT_DIR, metrics_name), "w",
              encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("[INFO] weights ->", os.path.join(C.CKPT_DIR, ckpt_name))


if __name__ == "__main__":
    main()

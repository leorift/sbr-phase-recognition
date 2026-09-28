# SBR Phase Recognition

Vision-based phase recognition for Sequencing Batch Reactors (SBR) using a
hierarchical temporal segmentation architecture.

## Overview

This repository contains the source code accompanying the manuscript
*"Vision-based phase recognition for sequencing batch reactors via hierarchical
temporal segmentation with cycle-constrained decoding"* (under submission).

The system recognizes the operational phases of an SBR (fill, post-fill idle,
anoxic, aerobic, settling, decant, post-decant idle) from a fixed camera view
of the reactor, without any in-tank instrumentation.

**Pipeline:**

1. **Per-frame features** — EfficientNet-B0 (ImageNet-pretrained, GAP) 1280-d
   spatial features fused with 12-d structured detection features from a YOLO
   detector (mud column / water column / bubbles), giving a 1292-d vector per
   sampled frame (2 fps).
2. **Temporal segmentation** — MS-TCN++ (multi-stage temporal convolutional
   network) produces per-frame phase probabilities.
3. **Cycle-constrained decoding** — a finite-state machine enforcing the SBR
   cyclic phase order removes illegal transitions and over-segmentation.
4. **On-site adaptation** — a sparse calibration protocol (K short windows
   uniformly spread per phase, ~2.7% of frames) supports incremental
   adaptation to a new reactor/session.

## Key results

| Evaluation | Accuracy |
|---|---|
| 94.9-min full-cycle test (zero-shot, session A) | 93.8% frame acc, 5/5 boundaries |
| 5.2-h cross-session video with sparse calibration | 97.2% frame acc, boundary MAE 7.3 s |
| Local / partial-view videos (zero-shot) | 100% frame acc |
| Hard-coded rule baseline (same detector inputs) | 4.8% frame acc |

Feature extraction runs at ~49.5 fps (20.2 ms/frame) on an RTX 5070 Ti;
full-cycle TCN inference takes 8.2 ms.

## Repository layout

```
src/
  config.py                # paths, hyperparameters, class definitions, test-video specs
  model.py                 # MS-TCN++ architecture and loss
  dataset.py               # feature loading, normalization, clip dataset
  state_machine.py         # cycle-constrained offline decoding
  extract_features.py      # CNN + YOLO feature extraction for training clips
  prepare_test_features.py # feature extraction for the full-cycle test video
  train.py                 # MS-TCN++ training (ablation flags included)
  baselines.py             # MLP / BiLSTM temporal baselines
  infer.py                 # end-to-end inference: video -> annotated outputs
  evaluate_test.py         # evaluation on the full-cycle test video
  eval_video.py            # general multi-video evaluation pipeline
  eval_ablation.py         # ablation evaluation (feature slice / RF / fps)
  make_calib_clips.py      # prefix (first-30%) calibration protocol
  make_calib_sparse.py     # sparse-window calibration protocol
```

## Environment

- Python 3.10+, PyTorch with CUDA, torchvision, ultralytics, opencv-python,
  numpy, matplotlib, pillow
- Install: `pip install -r requirements.txt`
- A YOLO detector trained on mud column / water column / bubbles is required
  (weights path configured in `src/config.py`).

## Usage

```bash
# 1. extract features from labeled single-phase clips
python src/extract_features.py

# 2. train MS-TCN++
python src/train.py --tag v3

# 3. evaluate on a test video
python src/eval_video.py --name cycle5h_raw --ckpt outputs/ckpt/ms_tcnpp_v3.pt

# 4. end-to-end inference with annotated video output
python src/infer.py path/to/video.mp4
```

## Citation

Manuscript under submission; citation information will be added upon
publication.

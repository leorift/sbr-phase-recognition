# -*- coding: utf-8 -*-
"""
数据集：加载预提取特征 npz，按时间顺序切分训练/验证，
训练时以片段采样 + 时序增强（时间抖动/速度扰动/帧丢弃/特征噪声）。
"""
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

import config as C


def load_all_features():
    """返回 list of dict(name, feat[T,F], label, T)。"""
    items = []
    for path in sorted(glob.glob(os.path.join(C.FEATURE_DIR, "*.npz"))):
        z = np.load(path)
        cnn = z["cnn"].astype(np.float32)
        det = z["det"].astype(np.float32)
        feat = np.concatenate([cnn, det], axis=1)  # [T, 1280+12]
        items.append({
            "name": os.path.basename(path).replace(".npz", ""),
            "feat": feat,
            "label": int(z["label"]),
            "T": feat.shape[0],
        })
    return items


def compute_norm_stats(items, train_ratio=C.TRAIN_RATIO):
    """仅用各视频训练段计算标准化统计量（防验证集信息泄漏）。"""
    train_parts = [it["feat"][: int(it["T"] * train_ratio)] for it in items]
    all_train = np.concatenate(train_parts, axis=0)
    mean = all_train.mean(axis=0)
    std = all_train.std(axis=0) + 1e-6
    return mean, std


class SBRClipDataset(Dataset):
    """
    每个视频按时间顺序切分：前 train_ratio 为训练段，其余为验证段。
    训练模式：随机采样片段 + 时序增强；验证模式：顺序不重叠切分。
    """

    def __init__(self, items, mean, std, split="train",
                 clip_length=C.CLIP_LENGTH, train_ratio=C.TRAIN_RATIO,
                 clips_per_video=C.CLIPS_PER_VIDEO_PER_EPOCH, seed=C.SEED,
                 class_oversample=None):
        self.split = split
        self.clip_length = clip_length
        self.rng = np.random.RandomState(seed)
        self.samples = []  # (feat_segment_normalized, label, name)

        for it in items:
            T = it["T"]
            cut = int(T * train_ratio)
            if split == "train":
                seg = it["feat"][:cut]
            else:
                seg = it["feat"][cut:]
            if len(seg) < 16:
                continue
            seg = (seg - mean) / std
            self.samples.append((seg.astype(np.float32), it["label"], it["name"]))

        if split == "train":
            class_oversample = class_oversample or {}
            self.train_index = []
            for si, (_, label, _) in enumerate(self.samples):
                mult = class_oversample.get(label, 1)
                self.train_index.extend([si] * mult)
        else:
            self.epoch_plan = []
            for si, (seg, label, name) in enumerate(self.samples):
                for s in range(0, max(1, len(seg) - clip_length + 1), clip_length):
                    e = min(s + clip_length, len(seg))
                    if e - s >= 16:
                        self.epoch_plan.append((si, s, e))

    def __len__(self):
        if self.split == "train":
            return len(self.train_index) * C.CLIPS_PER_VIDEO_PER_EPOCH
        return len(self.epoch_plan)

    def _augment(self, clip):
        """时序增强：速度扰动 -> 时间抖动 -> 帧丢弃 -> 特征噪声。"""
        T = clip.shape[0]
        speed = self.rng.uniform(0.8, 1.2)
        src_idx = np.clip((np.arange(T) * speed).astype(int), 0, T - 1)
        clip = clip[src_idx]
        shift = self.rng.randint(-2, 3)
        clip = np.roll(clip, shift, axis=0)
        drop = self.rng.rand(T) < 0.02
        if drop.any() and not drop.all():
            idx = np.arange(T)
            valid = idx[~drop]
            clip[drop] = clip[np.searchsorted(valid, idx[drop]).clip(0, len(valid) - 1)]
        clip = clip + self.rng.normal(0, 0.02, clip.shape).astype(np.float32)
        return clip

    def __getitem__(self, i):
        if self.split == "train":
            si = self.train_index[i % len(self.train_index)]
            seg, label, name = self.samples[si]
            if len(seg) <= self.clip_length:
                s = 0
                clip = seg
            else:
                s = self.rng.randint(0, len(seg) - self.clip_length + 1)
                clip = seg[s:s + self.clip_length]
            clip = self._augment(clip.copy())
        else:
            si, s, e = self.epoch_plan[i]
            seg, label, name = self.samples[si]
            clip = seg[s:e]

        if clip.shape[0] < self.clip_length:
            pad = np.repeat(clip[-1:], self.clip_length - clip.shape[0], axis=0)
            clip = np.concatenate([clip, pad], axis=0)

        x = torch.from_numpy(clip)
        y = torch.full((self.clip_length,), label, dtype=torch.long)
        return x, y
